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
from autoencoders import TreeCrownResNet34, TreeCrownDINO, TreeCrownResNet50
from orthomosaic_pipeline import OrthomosaicConfig, run_orthomosaic_pipeline


#===========================================================================
# Model Registry Configuration
#===========================================================================

MODEL_REGISTRY = {
    "maskrcnn_models": {
        "500x500": "models/instance500x500.ckpt",
        "1024x1024": "models/instance1024x1024.ckpt"
    },
    "autoencoder_models": {
        "resnet34": {
            "model": "models/resnet34.pth",  # Fixed: was a set, now a string
            "class": TreeCrownResNet34,
            "latent_dim": 256,
            "default_freeze": True  # Fixed: typo "defualt" → "default"
        },
        "resnet50": {
            "model": "models/resnet50.pth",
            "class": TreeCrownResNet50,
            "latent_dim": 256,
            "default_freeze": True
        },
        "dinov2": {
            "model": "models/dino.pth",
            "class": TreeCrownDINO,
            "latent_dim": 256,
            "default_freeze": True
        }
    }
}


USE_LOCAL_MODELS = True
#===========================================================================

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
def build_param_grid(clustering_method, param_mode, **kwargs):
    """
    Build parameter grid for clustering based on mode
    
    Args:
        clustering_method: "HDBSCAN" or "DBSCAN"
        param_mode: "Automatic (Grid Search)" or "Manual"
        **kwargs: Parameters from UI
    
    Returns:
        Dictionary with parameter grid
    """
    if param_mode == "Manual":
        # Single parameter set
        if clustering_method == "HDBSCAN":
            return {
                'min_cluster_size': [kwargs['min_cluster_size']],
                'min_samples': [kwargs['min_samples']]
            }
        else:  # DBSCAN
            return {
                'eps': [kwargs['eps']],
                'min_samples': [kwargs['min_samples']]
            }
    
    else:  # Automatic Grid Search
        if clustering_method == "HDBSCAN":
            min_cluster_size_range = kwargs['min_cluster_size_range']
            cluster_size_step = kwargs['cluster_size_step']
            min_samples_range = kwargs['min_samples_range']
            samples_step = kwargs['samples_step']
            
            return {
                'min_cluster_size': list(range(
                    min_cluster_size_range[0],
                    min_cluster_size_range[1] + 1,
                    cluster_size_step
                )),
                'min_samples': list(range(
                    min_samples_range[0],
                    min_samples_range[1] + 1,
                    samples_step
                ))
            }
        
        else:  # DBSCAN
            eps_range = kwargs['eps_range']
            eps_step = kwargs['eps_step']
            min_samples_range = kwargs['min_samples_range']
            samples_step = kwargs['samples_step']
            
            return {
                'eps': list(np.arange(
                    eps_range[0],
                    eps_range[1] + eps_step,
                    eps_step
                )),
                'min_samples': list(range(
                    min_samples_range[0],
                    min_samples_range[1] + 1,
                    samples_step
                ))
            }

def initialize_autoencoder(autoencoder_type: str, checkpoint_path: str, device: torch.device):
    """
    Initialize autoencoder model based on architecture type
    
    Args:
        autoencoder_type: One of "resnet34", "resnet50", "dinov2"
        checkpoint_path: Path to model weights
        device: torch device
    
    Returns:
        Initialized and loaded autoencoder model in eval mode
    """
    try:
        # Get architecture configuration
        arch_config = MODEL_REGISTRY["autoencoder_models"][autoencoder_type]
        
        # Get the model class and configuration
        ModelClass = arch_config["class"]
        latent_dim = arch_config["latent_dim"]
        freeze_backbone = arch_config["default_freeze"]
        
        # Initialize model
        autoencoder_model = ModelClass(
            freeze_backbone=freeze_backbone,
            latent_dim=latent_dim
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                autoencoder_model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                autoencoder_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                autoencoder_model.load_state_dict(checkpoint)
        else:
            autoencoder_model.load_state_dict(checkpoint)
        
        # Move to device and set to eval mode
        autoencoder_model.to(device)
        autoencoder_model.eval()
        
        return autoencoder_model
        
    except KeyError as e:
        raise ValueError(f"Unknown autoencoder type: {autoencoder_type}. Available types: {list(MODEL_REGISTRY['autoencoder_models'].keys())}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Error initializing {autoencoder_type} autoencoder: {str(e)}")


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

        st.divider()
        st.subheader("🤖 Model Selection")

        if USE_LOCAL_MODELS:
            
            maskrcnn_choice = st.selectbox(
                "Mask R-CNN Model",
                options = list(MODEL_REGISTRY["maskrcnn_models"].keys()),
                help="Select your trained Mask R-CNN model"
            )
            maskrcnn_path = MODEL_REGISTRY["maskrcnn_models"][maskrcnn_choice]

            autoencoder_type = st.selectbox(
                "Autoencoder Architecture",
                options=list(MODEL_REGISTRY["autoencoder_models"].keys()),
                format_func=lambda x: x.upper().replace("RESNET", "ResNet ").replace("DINOV2", "DINOv2"),
                help="Select autoencoder backbone architecture"
            )

            autoencoder_config = MODEL_REGISTRY["autoencoder_models"][autoencoder_type] 
            autoencoder_path = autoencoder_config["model"]
        
            with st.expander("📍 Model Paths"):
                st.code(f"Mask R-CNN: {maskrcnn_path}", language="text")
                st.code(f"Autoencoder: {autoencoder_path}", language="text")
                st.info(
                    f"**Architecture:** {autoencoder_type.upper()}\n\n"
                    f"**Latent Dimension:** {autoencoder_config['latent_dim']}\n\n"
                    f"**Backbone Frozen:** {autoencoder_config['default_freeze']}"
                )

            maskrcnn_valid = Path(maskrcnn_path).exists()
            autoencoder_valid = Path(autoencoder_path).exists()

            if not maskrcnn_valid:
                st.error(f"❌ Mask R-CNN model not found at {maskrcnn_path}")
            if not autoencoder_valid:
                st.error(f"❌ Autoencoder model not found at {autoencoder_path}")

            models_ready = maskrcnn_valid and autoencoder_valid

        else:
            maskrcnn_file = st.file_uploader(
                "Mask R-CNN Checkpoint (.ckpt)",
                type=['ckpt'],
                help="Your trained Mask R-CNN model"
            )
        
            autoencoder_type = st.selectbox(
                "Autoencoder Architecture",
                options=list(MODEL_REGISTRY["autoencoder_models"].keys()),
                format_func=lambda x: x.upper().replace("RESNET", "ResNet ").replace("DINOV2", "DINOv2"),
                help="Select the architecture type of your uploaded model"
            )
        
            arch_config = MODEL_REGISTRY["autoencoder_models"][autoencoder_type]


            autoencoder_file = st.file_uploader(
                f"Autoencoder Weights - {autoencoder_type.upper()} (.pth)",
                type=['pth', 'pt'],
                hhelp=f"Upload {autoencoder_type.upper()} autoencoder weights"
            )

            models_ready = maskrcnn_file is not None and autoencoder_file is not None

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

        param_mode = st.radio(
            "Parameter Selection Mode",
            ["Automatic (Grid Search)", "Manual"],
            help="Automatic mode searches for optimal parameters"
        )
        
        if param_mode == "Manual":
            # Manual parameter selection
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
            else:  # DBSCAN
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
        else:
            # Automatic grid search configuration
            st.info("🤖 Grid search will test multiple parameter combinations")
    
            if clustering_method == "HDBSCAN":
                with st.expander("🔧 Grid Search Ranges", expanded=True):
                    col1, col2 = st.columns(2)
            
                with col1:
                    min_cluster_size_range = st.slider(
                        "Min Cluster Size Range",
                        min_value=2,
                        max_value=50,
                        value=(3, 15),
                        help="Range to search for min_cluster_size"
                    )
                    cluster_size_step = st.number_input(
                        "Step Size",
                        min_value=1,
                        max_value=10,
                        value=2
                    )
            
                with col2:
                    min_samples_range = st.slider(
                        "Min Samples Range",
                        min_value=1,
                        max_value=20,
                        value=(1, 10),
                        help="Range to search for min_samples"
                    )
                    samples_step = st.number_input(
                        "Step Size ",
                        min_value=1,
                        max_value=5,
                        value=2
                    )
            
                # Calculate grid size
                n_cluster_sizes = len(range(min_cluster_size_range[0], min_cluster_size_range[1] + 1, cluster_size_step))
                n_samples = len(range(min_samples_range[0], min_samples_range[1] + 1, samples_step))
                total_combinations = n_cluster_sizes * n_samples
            
                st.metric("Total Combinations to Test", total_combinations)
                if total_combinations > 50:
                    st.warning("⚠️ Large grid size may take longer to process")
    
            else:  # DBSCAN
                with st.expander("🔧 Grid Search Ranges", expanded=True):
                    col1, col2 = st.columns(2)
            
                    with col1:
                        eps_range = st.slider(
                            "EPS Range",
                            min_value=0.5,
                            max_value=5.0,
                            value=(0.5, 3.0),
                            step=0.5,
                            help="Range to search for eps distance"
                        )
                    eps_step = st.number_input(
                        "EPS Step",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.5,
                        step=0.1
                    )
            
                    with col2:
                        min_samples_range = st.slider(
                        "Min Samples Range",
                        min_value=2,
                        max_value=20,
                        value=(2, 10),
                        help="Range to search for min_samples"
                    )
                    samples_step = st.number_input(
                        "Samples Step",
                        min_value=1,
                        max_value=5,
                        value=2
                    )
            
                # Calculate grid size
                n_eps = len(np.arange(eps_range[0], eps_range[1] + eps_step, eps_step))
                n_samples = len(range(min_samples_range[0], min_samples_range[1] + 1, samples_step))
                total_combinations = n_eps * n_samples
            
                st.metric("Total Combinations to Test", total_combinations)
                if total_combinations > 50:
                    st.warning("⚠️ Large grid size may take longer to process")
    
        # Scoring metric selection
        scoring_metric = st.selectbox(
            "Optimization Metric",
            ["silhouette", "calinski_harabasz", "davies_bouldin"],
            help="Metric to optimize (silhouette is most common)"
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
            disabled=not (image_file and models_ready),
            use_container_width=True
        )
    
    # Main content area
    if not image_file:
        st.info("👈 Please upload all required files in the sidebar to begin")
        return
    
    if not models_ready:
        st.info("👈 Please upload/select valid models in the sidebar to proceed")
        return

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
            
                # Handle image
                image_path = temp_dir / image_file.name
                with open(image_path, 'wb') as f:
                    f.write(image_file.read())
            
                # Handle model paths
                if USE_LOCAL_MODELS:
                    # Use local model paths directly
                    maskrcnn_checkpoint = maskrcnn_path
                    autoencoder_checkpoint = autoencoder_path
                else:
                    # Save uploaded models to temp
                    maskrcnn_checkpoint = temp_dir / maskrcnn_file.name
                    autoencoder_checkpoint = temp_dir / autoencoder_file.name
                
                    with open(maskrcnn_checkpoint, 'wb') as f:
                        f.write(maskrcnn_file.read())
                    with open(autoencoder_checkpoint, 'wb') as f:
                        f.write(autoencoder_file.read())
            
                # Initialize autoencoder with selected architecture
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
                with st.spinner(f"🔧 Initializing {autoencoder_type.upper()} autoencoder..."):
                    autoencoder_model = initialize_autoencoder(
                        autoencoder_type=autoencoder_type,
                        checkpoint_path=str(autoencoder_checkpoint),
                        device=device
                    )
            
                st.info(f"✅ {autoencoder_type.upper()} autoencoder loaded successfully!")

                if param_mode == "Manual":
                    if clustering_method == "HDBSCAN":
                        param_grid = build_param_grid(
                            clustering_method,
                            param_mode,
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples
                        )
                    else:
                        param_grid = build_param_grid(
                            clustering_method,
                            param_mode,
                            eps=eps,
                            min_samples=min_samples
                        )

                    st.info(f"🔧 Using manual parameters")

                else:
                    if clustering_method == "HDBSCAN":
                        param_grid = build_param_grid(
                            clustering_method,
                            param_mode,
                            min_cluster_size_range=min_cluster_size_range,
                            cluster_size_step=cluster_size_step,
                            min_samples_range=min_samples_range,
                            samples_step=samples_step
                        )
                    else:
                        param_grid = build_param_grid(
                            clustering_method,
                            param_mode,
                            eps_range=eps_range,
                            eps_step=eps_step,
                            min_samples_range=min_samples_range,
                            samples_step=samples_step
                        )

                    total_combinations = np.prod([len(v) for v in param_grid.values()])
                    st.info(f"🔧 Running grid search with {total_combinations} parameter combinations")
                    st.info(f"🔧 Optimizing for {scoring_metric} metric")

                # Build config
                if mode.startswith("Single"):
                    from tree_crown_pipeline import PipelineConfig, run_pipeline
                
                    config = PipelineConfig(
                        image_path=str(image_path),
                        maskrcnn_checkpoint=str(maskrcnn_checkpoint),
                        autoencoder_checkpoint=str(autoencoder_checkpoint),
                        output_dir=str(temp_dir / "outputs"),
                        score_threshold=score_threshold,
                        mask_threshold=mask_threshold,
                        clustering_method=clustering_method,
                        subset_ratio=0.25
                        #scoring_metric=scoring_metric if param_mode == "Automatic (Grid Search)" else None
                    )

                    results_json, fig = run_pipeline(config, autoencoder_model, param_grid)
            
                else:  # Orthomosaic
                
                    config = OrthomosaicConfig(
                        image_path=str(image_path),
                        maskrcnn_checkpoint=str(maskrcnn_checkpoint),
                        autoencoder_checkpoint=str(autoencoder_checkpoint),
                        output_dir=str(temp_dir / "outputs"),
                        score_threshold=score_threshold,
                        mask_threshold=mask_threshold,
                        clustering_method=clustering_method,
                        #scoring_metric=scoring_metric if param_mode == "Automatic (Grid Search)" else None
                    )
                
                    # Run pipeline
                    results_json, fig = run_orthomosaic_pipeline(config, autoencoder_model, param_grid)
            
                # Store results
                st.session_state.results = results_json
                st.session_state.config = config
                st.session_state.fig = fig
                st.session_state.processing_complete = True
                st.session_state.output_dir = str(temp_dir / "outputs")
            
                st.success("✅ Processing complete!")
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                st.exception(e)


    # Display results
    if st.session_state.processing_complete and st.session_state.results:
        display_results(
            st.session_state.results, 
            st.session_state.config,
            st.session_state.fig,
            st.session_state.output_dir
        )

# ============================================================================
# Results Display
# ============================================================================

def display_results(results: Dict, config, fig, output_dir:str):
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

    st.header("🖼️ Visualizations")

    is_single_image = 'tile_id' not in pd.DataFrame(results['annotations']).columns
    
    if is_single_image:
        # Single Image Mode - Show main overlay and clustering visualization
        
        st.subheader("Clustered Detections")

        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        overlay_path = Path(output_dir) / "visualization_overlay.png"
        if overlay_path.exists():
            overlay_img = Image.open(overlay_path)

            from io import BytesIO
            buf = BytesIO()
            overlay_img.save(buf, format='PNG')
            buf.seek(0)

            st. download_button(
                label="📥 Download Overlay Image",
                data=buf,
                file_name="tree_crown_overlay.png",
                mime="image/png",
                use_container_width=True
            )
            

    else:
        # Orthomosaic Mode - Show tile gallery preview
        st.subheader("📍 Tile Overview")
    
        # Display main figure if available
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        
            # Try to save and offer download
            try:
                from io import BytesIO
                buf = BytesIO()
                fig.write_image(buf, format='png', width=1920, height=1080)
                buf.seek(0)
            
                st.download_button(
                    label="📥 Download Overview Visualization",
                    data=buf,
                    file_name="orthomosaic_overview.png",
                    mime="image/png",
                    use_container_width=True
                )
            except:
                st.info("Install kaleido for image export: pip install kaleido")
    
        # Quick stats
        df = pd.DataFrame(results['annotations'])
        st.info(f"📊 Processed {df['tile_id'].nunique()} tiles with {len(df)} total detections")

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
                ["Tile ID", "Most Detections", "Cluster Diversity"]
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