import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pycocotools import mask as mask_util
import tempfile
import shutil
from typing import Optional, Dict, List, Tuple

import tree_crown_pipeline
import orthomosaic_pipeline

# Page configuration
st.set_page_config(
    page_title="Intreeligent Prototype v1",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'config' not in st.session_state:
    st.session_state.config = None

# ============================================================================
# Helper Functions
# ============================================================================

def safe_hex_to_rgb(color_str: str) -> Tuple[int, int, int]:
    """Convert hex or rgb() color to RGB tuple"""
    try:
        color_clean = color_str.strip()
        if color_str.startswith('rgb('):
            rgb_str = color_clean.replace('rgb(', '').replace(')', '')
            r, g, b = map(int, rgb_str.split(','))
            return (r, g, b)
        hex_clean = color_clean.lstrip('#')
        if len(hex_clean) != 6:
            return (128, 128, 128)
        return tuple(int(hex_clean[i:i+2], 16) for i in (0, 2, 4))
    except:
        return (128, 128, 128)

def create_tile_thumbnail(tile_rgb, detections, cluster_info, size=256):
    """Create thumbnail of tile with mask overlays"""
    overlay = tile_rgb.copy()
    
    for det in detections:
        cluster_id = det['cluster_id']
        color_rgb = safe_hex_to_rgb(cluster_info[str(cluster_id)]['color'])
        
        # Decode mask
        rle = det['segmentation']
        mask = mask_util.decode(rle)
        
        # Resize mask if needed
        if mask.shape != tile_rgb.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (tile_rgb.shape[1], tile_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Apply overlay
        overlay[mask > 0] = (
            overlay[mask > 0] * 0.5 + np.array(color_rgb) * 0.5
        ).astype(np.uint8)
    
    # Resize for display
    thumbnail = cv2.resize(overlay, (size, size))
    return thumbnail

# ============================================================================
# Main Application
# ============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🌳 Intreeligent Prototype v1</div>', 
                unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode selection
        mode = st.radio(
            "Processing Mode",
            ["Single Image (1024×1024)", "Orthomosaic (Large Scale)"],
            help="Choose single image for testing or orthomosaic for production"
        )
        
        st.divider()
        
        # File uploads
        st.subheader("📁 Upload Files")
        
        image_file = st.file_uploader(
            "Image File" if mode.startswith("Single") else "Orthomosaic File",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
            help="Upload your tree crown image"
        )
        
        maskrcnn_file = st.file_uploader(
            "Mask R-CNN Checkpoint (.ckpt)",
            type=['ckpt'],
            help="Your trained Mask R-CNN model"
        )
        
        autoencoder_file = st.file_uploader(
            "Autoencoder Weights (.pth)",
            type=['pth', 'pt'],
            help="Your trained autoencoder model"
        )
        
        st.divider()
        
        # Detection parameters
        st.subheader("🎯 Detection Settings")
        
        score_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Lower values detect more crowns (including false positives)"
        )
        
        mask_threshold = st.slider(
            "Mask Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        st.divider()
        
        # Clustering parameters
        st.subheader("🔍 Clustering Settings")
        
        clustering_method = st.selectbox(
            "Algorithm",
            ["HDBSCAN", "DBSCAN"],
            help="HDBSCAN is more robust for varying densities"
        )
        
        if clustering_method == "HDBSCAN":
            min_cluster_size = st.slider(
                "Min Cluster Size",
                min_value=2,
                max_value=50,
                value=5,
                help="Minimum crowns needed to form a cluster"
            )
            min_samples = st.slider(
                "Min Samples",
                min_value=1,
                max_value=20,
                value=3
            )
        else:
            eps = st.slider(
                "EPS (Distance)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5
            )
            min_samples = st.slider(
                "Min Samples",
                min_value=2,
                max_value=20,
                value=5
            )
        
        # Orthomosaic-specific settings
        if mode.startswith("Orthomosaic"):
            st.divider()
            st.subheader("🗺️ Tiling Settings")
            
            tile_size = st.selectbox(
                "Tile Size",
                [512, 1024, 2048],
                index=1
            )
            
            overlap_percent = st.slider(
                "Overlap Percentage",
                min_value=10,
                max_value=30,
                value=20,
                step=5
            )
            
            iou_threshold = st.slider(
                "IoU Threshold (Duplicate Merging)",
                min_value=0.3,
                max_value=0.8,
                value=0.5,
                step=0.05
            )
        
        st.divider()
        
        # Run button
        run_button = st.button(
            "🚀 Run Analysis",
            type="primary",
            disabled=not (image_file and maskrcnn_file and autoencoder_file),
            use_container_width=True
        )
    
    # Main content area
    if not (image_file and maskrcnn_file and autoencoder_file):
        st.info("👈 Please upload all required files in the sidebar to begin")
        
        # Show example/instructions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📸 Step 1: Upload Image")
            st.markdown("Upload your tree crown orthomosaic or test image")
        with col2:
            st.markdown("### 🤖 Step 2: Upload Models")
            st.markdown("Provide your trained Mask R-CNN and autoencoder")
        with col3:
            st.markdown("### ⚙️ Step 3: Configure & Run")
            st.markdown("Adjust parameters and start the analysis")
        
        return
    
    # Run pipeline
    if run_button:
        with st.spinner("🔄 Processing... This may take a few minutes"):
            try:
                # Save uploaded files to temp directory
                temp_dir = Path(tempfile.mkdtemp())
                
                image_path = temp_dir / image_file.name
                maskrcnn_path = temp_dir / maskrcnn_file.name
                autoencoder_path = temp_dir / autoencoder_file.name
                
                with open(image_path, 'wb') as f:
                    f.write(image_file.read())
                with open(maskrcnn_path, 'wb') as f:
                    f.write(maskrcnn_file.read())
                with open(autoencoder_path, 'wb') as f:
                    f.write(autoencoder_file.read())
                
                # Build config
                if mode.startswith("Single"):
                    from tree_crown_pipeline import PipelineConfig, run_pipeline
                    
                    config = PipelineConfig(
                        image_path=str(image_path),
                        maskrcnn_checkpoint=str(maskrcnn_path),
                        autoencoder_checkpoint=str(autoencoder_path),
                        output_dir=str(temp_dir / "outputs"),
                        score_threshold=score_threshold,
                        mask_threshold=mask_threshold,
                        clustering_method=clustering_method
                    )
                    
                    # Build param grid
                    if clustering_method == "HDBSCAN":
                        param_grid = {
                            'min_cluster_size': [min_cluster_size],
                            'min_samples': [min_samples]
                        }
                    else:
                        param_grid = {
                            'eps': [eps],
                            'min_samples': [min_samples]
                        }
                    
                    # Initialize autoencoder (you'll need to import your model class)
                    # autoencoder_model = YourAutoencoderClass(...)
                    
                    # Run pipeline
                    # results_json, fig = run_pipeline(config, autoencoder_model, param_grid)
                    
                    # For now, show placeholder
                    st.error("⚠️ Please import and initialize your autoencoder model class")
                    return
                
                else:  # Orthomosaic
                    from orthomosaic_pipeline import OrthomosaicConfig, run_orthomosaic_pipeline
                    
                    config = OrthomosaicConfig(
                        orthomosaic_path=str(image_path),
                        maskrcnn_checkpoint=str(maskrcnn_path),
                        autoencoder_checkpoint=str(autoencoder_path),
                        output_dir=str(temp_dir / "outputs"),
                        tile_size=tile_size,
                        tile_overlap_percent=overlap_percent,
                        score_threshold=score_threshold,
                        mask_threshold=mask_threshold,
                        clustering_method=clustering_method,
                        iou_threshold=iou_threshold
                    )
                    
                    # Build param grid
                    if clustering_method == "HDBSCAN":
                        param_grid = {
                            'min_cluster_size': [min_cluster_size],
                            'min_samples': [min_samples]
                        }
                    else:
                        param_grid = {
                            'eps': [eps],
                            'min_samples': [min_samples]
                        }
                    
                    # Run pipeline
                    # results_json, fig = run_orthomosaic_pipeline(config, autoencoder_model, param_grid)
                    
                    st.error("⚠️ Please import and initialize your autoencoder model class")
                    return
                
                # Store results
                st.session_state.results = results_json
                st.session_state.config = config
                st.session_state.processing_complete = True
                
                st.success("✅ Processing complete!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)
    
    # Display results
    if st.session_state.processing_complete and st.session_state.results:
        display_results(st.session_state.results, st.session_state.config)

# ============================================================================
# Results Display
# ============================================================================

def display_results(results: Dict, config):
    """Display analysis results with interactive visualizations"""
    
    st.success("✅ Analysis Complete!")
    
    # Summary metrics
    st.header("📊 Summary Statistics")
    
    metadata = results['metadata']
    total_crowns = len(results['annotations'])
    n_clusters = len([k for k in results['clusters'].keys() if int(k) != -1])
    noise_count = results['clusters'].get('-1', {}).get('count', 0)
    noise_pct = (noise_count / total_crowns * 100) if total_crowns > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tree Crowns", total_crowns)
    with col2:
        st.metric("Clusters Found", n_clusters)
    with col3:
        st.metric("Noise Detections", f"{noise_count} ({noise_pct:.1f}%)")
    with col4:
        st.metric("Silhouette Score", f"{metadata['silhouette_score']:.3f}")
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Cluster Analysis",
        "🗺️ Tile Gallery",
        "📋 Data Table",
        "💾 Export"
    ])
    
    # Tab 1: Cluster Analysis
    with tab1:
        st.subheader("Cluster Distribution")
        
        # Create bar chart
        cluster_data = []
        for cluster_id, info in results['clusters'].items():
            cluster_data.append({
                'Cluster': f"Cluster {cluster_id}" if int(cluster_id) != -1 else "Noise",
                'Count': info['count'],
                'Color': info['color']
            })
        
        df_clusters = pd.DataFrame(cluster_data)
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=df_clusters['Cluster'],
                y=df_clusters['Count'],
                marker_color=df_clusters['Color'].apply(lambda c: 
                    c if not c.startswith('rgb(') else 
                    '#' + ''.join(f'{int(x):02x}' for x in c.replace('rgb(','').replace(')','').split(','))
                ),
                text=df_clusters['Count'],
                textposition='auto'
            )
        ])
        
        fig_bar.update_layout(
            title="Tree Crowns per Cluster",
            xaxis_title="Cluster",
            yaxis_title="Number of Crowns",
            height=400
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Statistics by cluster
        st.subheader("Cluster Statistics")
        
        df = pd.DataFrame(results['annotations'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Average Confidence by Cluster**")
            conf_stats = df.groupby('cluster_id')['confidence'].mean().sort_values(ascending=False)
            st.dataframe(conf_stats, use_container_width=True)
        
        with col2:
            st.markdown("**Average Area by Cluster**")
            area_stats = df.groupby('cluster_id')['area'].mean().sort_values(ascending=False)
            st.dataframe(area_stats, use_container_width=True)
    
    # Tab 2: Tile Gallery
    with tab2:
        st.subheader("Tile Visualizations")
        
        if 'tile_id' in df.columns:
            # Get unique tiles
            tiles = df['tile_id'].unique()
            
            # Sorting options
            sort_option = st.selectbox(
                "Sort by",
                ["Most Detections", "Tile ID", "Cluster Diversity"]
            )
            
            # Filter options
            show_tiles = st.slider(
                "Number of tiles to display",
                min_value=1,
                max_value=min(len(tiles), 100),
                value=min(25, len(tiles))
            )
            
            # Display tiles in grid
            if sort_option == "Most Detections":
                tile_counts = df.groupby('tile_id').size().sort_values(ascending=False)
            elif sort_option == "Cluster Diversity":
                tile_counts = df.groupby('tile_id')['cluster_id'].nunique().sort_values(ascending=False)
            else:
                tile_counts = df.groupby('tile_id').size().sort_index()
            
            selected_tiles = tile_counts.head(show_tiles).index
            
            # Grid display
            cols_per_row = 5
            rows_needed = int(np.ceil(len(selected_tiles) / cols_per_row))
            
            st.info(f"📌 Showing {len(selected_tiles)} tiles with mask overlays")
            
            # Note: Actual tile rendering would require access to the OrthomosaicReader
            # For now, show tile metadata
            for tile_id in selected_tiles:
                tile_dets = df[df['tile_id'] == tile_id]
                with st.expander(f"{tile_id} - {len(tile_dets)} detections"):
                    st.dataframe(tile_dets[['cluster_id', 'confidence', 'area']])
        else:
            st.info("Single image mode - no tiles to display")
    
    # Tab 3: Data Table
    with tab3:
        st.subheader("Detection Data")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            selected_clusters = st.multiselect(
                "Filter by Cluster",
                options=sorted(df['cluster_id'].unique()),
                default=sorted(df['cluster_id'].unique())
            )
        
        with col2:
            min_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05
            )
        
        # Filter dataframe
        filtered_df = df[
            (df['cluster_id'].isin(selected_clusters)) &
            (df['confidence'] >= min_confidence)
        ]
        
        st.dataframe(
            filtered_df[['cluster_id', 'confidence', 'area']],
            use_container_width=True,
            height=400
        )
        
        st.metric("Filtered Detections", len(filtered_df))
    
    # Tab 4: Export
    with tab4:
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export JSON
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="📥 Download Full Results (JSON)",
                data=json_str,
                file_name="tree_crown_results.json",
                mime="application/json",
                use_container_width=True
            )
            
            # Export CSV
            csv_str = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Detections (CSV)",
                data=csv_str,
                file_name="tree_crown_detections.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export cluster summary
            cluster_summary = pd.DataFrame([
                {
                    'Cluster ID': k,
                    'Count': v['count'],
                    'Percentage': f"{v['count']/total_crowns*100:.1f}%"
                }
                for k, v in results['clusters'].items()
            ])
            
            st.download_button(
                label="📥 Download Cluster Summary (CSV)",
                data=cluster_summary.to_csv(index=False),
                file_name="cluster_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.info("💡 All results are also saved in the output directory")

# ============================================================================
# Run App
# ============================================================================

if __name__ == "__main__":
    main()