# --- Standard Library Imports ---
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

# --- Third-Party Library Imports ---
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision
import pytorch_lightning as pl

from pycocotools import mask as mask_util
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import hdbscan

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Pipeline Configuration
@dataclass
class PipelineConfig:
    """Configuration for the tree crown pipeline"""
    # Paths
    image_path: str
    maskrcnn_checkpoint: str
    autoencoder_checkpoint: str
    output_dir: str = "outputs"
    
    # Image sizes
    original_size: int = 1024
    maskrcnn_size: int = 500
    autoencoder_size: int = 512
    
    # Detection thresholds
    mask_threshold: float = 0.5
    score_threshold: float = 0.5
    
    # Clustering
    clustering_method: str = "HDBSCAN"  # "DBSCAN" or "HDBSCAN"
    subset_ratio: float = 0.25  # Use 25% for parameter optimization
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# <h4>Model Loading</h4>

# In[51]:


# Model Loading
class PredictionProcessor:
    """Process Mask R-CNN predictions"""
    def __init__(self, mask_threshold=0.5, score_threshold=0.5):
        self.mask_threshold = mask_threshold
        self.score_threshold = score_threshold
    
    def __call__(self, predictions):
        """Filter predictions by score threshold"""
        if isinstance(predictions, list):
            predictions = predictions[0]
        
        # Filter by score
        keep = predictions['scores'] > self.score_threshold
        
        filtered = {
            'boxes': predictions['boxes'][keep],
            'masks': predictions['masks'][keep],
            'scores': predictions['scores'][keep],
            'labels': predictions['labels'][keep]
        }
        
        # Threshold masks
        filtered['masks'] = (filtered['masks'] > self.mask_threshold).squeeze(1)
        
        return filtered


# In[52]:


class LitMaskRCNN(pl.LightningModule):
    """PyTorch Lightning Mask R-CNN Module"""
    def __init__(self, lr=1e-3, num_classes=2, mask_threshold=0.5, score_threshold=0.5):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pretrained Mask R-CNN with ResNet50 FPN backbone
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        
        # Replace box predictor head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = \
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor head
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = \
            torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                in_features_mask, hidden_layer, num_classes
            )
        
        # Prediction post-processor
        self.pred_processor = PredictionProcessor(
            mask_threshold=mask_threshold,
            score_threshold=score_threshold
        )
    
    def forward(self, images, targets=None):
        return self.model(images, targets)


# In[53]:


def load_maskrcnn(checkpoint_path: str, config: PipelineConfig) -> LitMaskRCNN:
    """Load Mask R-CNN model from checkpoint"""
    print(f"Loading Mask R-CNN from {checkpoint_path}...")
    model = LitMaskRCNN.load_from_checkpoint(
        checkpoint_path,
        mask_threshold=config.mask_threshold,
        score_threshold=config.score_threshold
    )
    model.eval()
    model.to(config.device)
    print("✓ Mask R-CNN loaded successfully")
    return model


# In[54]:


def load_autoencoder(checkpoint_path: str, config: PipelineConfig, model_class):
    """Load autoencoder model from .pth file"""
    print(f"Loading Autoencoder from {checkpoint_path}...")
    model = model_class  # Initialize your TreeCrownResNet34 or similar
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
    model.eval()
    model.to(config.device)
    print("✓ Autoencoder loaded successfully")
    return model


# <h4>Image Processing and Detection</h4>

# In[55]:


def load_and_preprocess_image(image_path: str, target_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load image and prepare for Mask R-CNN inference
    
    Returns:
        torch_image: (C, H, W) tensor for model input
        original_np: (H, W, C) numpy array for visualization
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_np = np.array(image)
    
    # Resize to target size
    image_resized = image.resize((target_size, target_size), Image.BILINEAR)
    
    # Convert to tensor and normalize
    image_tensor = torchvision.transforms.functional.to_tensor(image_resized)
    
    return image_tensor, original_np


# In[56]:


def run_detection(model: LitMaskRCNN, image_tensor: torch.Tensor, config: PipelineConfig) -> Dict:
    """Run Mask R-CNN detection on image"""
    print("Running Mask R-CNN detection...")
    
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).to(config.device)
        predictions = model(image_batch)
        predictions_processed = model.pred_processor(predictions)
    
    # Move to CPU
    for key in predictions_processed:
        predictions_processed[key] = predictions_processed[key].cpu()
    
    num_detections = len(predictions_processed['scores'])
    print(f"✓ Detected {num_detections} tree crowns")
    
    return predictions_processed


# In[57]:


def scale_predictions(predictions: Dict, from_size: int, to_size: int) -> Dict:
    """Scale prediction coordinates from one size to another"""
    scale_factor = to_size / from_size
    
    scaled_preds = predictions.copy()
    scaled_preds['boxes'] = predictions['boxes'] * scale_factor
    
    # Scale masks
    masks = predictions['masks'].numpy()
    scaled_masks = []
    for mask in masks:
        scaled_mask = cv2.resize(
            mask.astype(np.uint8),
            (to_size, to_size),
            interpolation=cv2.INTER_NEAREST
        )
        scaled_masks.append(scaled_mask)
    
    scaled_preds['masks'] = torch.from_numpy(np.array(scaled_masks))
    
    return scaled_preds


# <h4>Mask Cropping and Feature Extraction</h4>

# In[58]:


def crop_masks(image_np: np.ndarray, predictions: Dict, config: PipelineConfig) -> List[np.ndarray]:
    """
    Crop individual tree crown masks with transparent background
    
    Returns:
        List of RGBA images (H, W, 4) with transparent backgrounds
    """
    print("Cropping individual tree crown masks...")
    crops = []
    
    boxes = predictions['boxes'].numpy()
    masks = predictions['masks'].numpy()
    
    for i, (box, mask) in enumerate(zip(boxes, masks)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Clip to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image_np.shape[1], x2)
        y2 = min(image_np.shape[0], y2)
        
        # Crop image and mask
        crop_rgb = image_np[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        
        # Create RGBA with transparent background
        crop_rgba = np.zeros((*crop_rgb.shape[:2], 4), dtype=np.uint8)
        crop_rgba[:, :, :3] = crop_rgb
        crop_rgba[:, :, 3] = (crop_mask * 255).astype(np.uint8)
        
        crops.append(crop_rgba)
    
    print(f"✓ Created {len(crops)} mask crops")
    return crops


# In[59]:


def resize_crops(crops: List[np.ndarray], target_size: int) -> List[np.ndarray]:
    """Resize crops to target size for autoencoder"""
    resized = []
    for crop in crops:
        resized_crop = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        resized.append(resized_crop)
    return resized


# In[60]:


def extract_features(autoencoder, crops: List[np.ndarray], config: PipelineConfig) -> np.ndarray:
    """
    Extract latent features from crops using autoencoder
    
    Returns:
        features: (N, latent_dim) numpy array
    """
    print("Extracting latent features from autoencoder...")
    
    features_list = []
    
    # Process in batches
    batch_size = 16
    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i + batch_size]
        
        # Convert to tensor (take RGB channels only)
        batch_tensors = []
        for crop in batch_crops:
            # Convert RGBA to RGB and normalize
            rgb = crop[:, :, :3].astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1)  # (C, H, W)
            batch_tensors.append(tensor)
        
        batch = torch.stack(batch_tensors).to(config.device)
        
        # Extract features
        with torch.no_grad():
            latent = autoencoder.encode(batch)
            features_list.append(latent.cpu().numpy())
    
    features = np.vstack(features_list)
    print(f"✓ Extracted features with shape {features.shape}")
    
    return features


# In[61]:


def optimize_clustering_parameters(
    features: np.ndarray,
    config: PipelineConfig,
    param_grid: Optional[Dict] = None
) -> Tuple[Dict, float]:
    """
    Optimize clustering parameters using grid search on subset
    
    Returns:
        best_params: Dictionary of best parameters
        best_score: Best silhouette score achieved
    """
    print("\nOptimizing clustering parameters...")
    
    # Sample subset
    n_samples = len(features)
    subset = features
    
    print(f"Using {n_samples} samples for optimization")
    
    # Default parameter grids
    if param_grid is None:
        if config.clustering_method == "DBSCAN":
            param_grid = {
                'eps': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                'min_samples': [3, 5, 10, 15, 20]
            }
        else:  # HDBSCAN
            param_grid = {
                'min_cluster_size': [10, 15, 20, 30, 50],
                'min_samples': [3, 5, 10, 15]
            }
    
    best_score = -1
    best_params = None
    results = []
    
    # Grid search
    if config.clustering_method == "DBSCAN":
        for eps in param_grid['eps']:
            for min_samples in param_grid['min_samples']:
                clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                labels = clusterer.fit_predict(subset)
                
                # Calculate metrics (only if we have more than 1 cluster)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    # Filter out noise for scoring
                    mask = labels != -1
                    if mask.sum() > 1:
                        score = silhouette_score(subset[mask], labels[mask])
                        noise_ratio = (labels == -1).sum() / len(labels)
                        
                        results.append({
                            'eps': eps,
                            'min_samples': min_samples,
                            'silhouette': score,
                            'n_clusters': n_clusters,
                            'noise_ratio': noise_ratio
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = {'eps': eps, 'min_samples': min_samples}
    
    else:  # HDBSCAN
        for min_cluster_size in param_grid['min_cluster_size']:
            for min_samples in param_grid['min_samples']:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_method='leaf'
                )
                labels = clusterer.fit_predict(subset)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    mask = labels != -1
                    if mask.sum() > 1:
                        score = silhouette_score(subset[mask], labels[mask])
                        noise_ratio = (labels == -1).sum() / len(labels)
                        
                        results.append({
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'silhouette': score,
                            'n_clusters': n_clusters,
                            'noise_ratio': noise_ratio
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'min_cluster_size': min_cluster_size,
                                'min_samples': min_samples
                            }
    
    if best_params is None:
        print("⚠ No valid clustering found during optimization. Using default parameters.")
    if config.clustering_method == "DBSCAN":
        best_params = {'eps': 2.0, 'min_samples': 2}
        best_score = 0.0
    else:  # HDBSCAN
        best_params = {'min_cluster_size': 2, 'min_samples': 1}
        best_score = 0.0

    print(f"✓ Best parameters: {best_params}")
    print(f"✓ Best silhouette score: {best_score:.3f}")
    
    return best_params, best_score


# In[62]:


def cluster_features(features: np.ndarray, config: PipelineConfig, params: Dict) -> np.ndarray:
    """
    Cluster features using optimized parameters
    
    Returns:
        labels: Cluster assignments for each sample
    """
    print(f"\nClustering all {len(features)} samples...")
    
    if config.clustering_method == "DBSCAN":
        clusterer = DBSCAN(**params, metric='euclidean')
    else:
        clusterer = hdbscan.HDBSCAN(**params, metric='euclidean')
    
    labels = clusterer.fit_predict(features)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    noise_ratio = n_noise / len(labels) * 100
    
    print(f"✓ Found {n_clusters} clusters")
    print(f"✓ Noise samples: {n_noise} ({noise_ratio:.1f}%)")
    
    return labels


# <h4>JSON Export (COCO Format)</h4>

# In[63]:


def predictions_to_coco_json(
    predictions: Dict,
    labels: np.ndarray,
    image_path: str,
    config: PipelineConfig,
    clustering_params: Dict,
    silhouette: float
) -> Dict:
    """
    Convert predictions and cluster labels to COCO format JSON
    """
    print("\nCreating COCO format JSON...")
    
    # Generate cluster colors
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Use a color palette
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set3
    cluster_colors = {}
    for i, label in enumerate(unique_labels):
        if label == -1:
            cluster_colors[-1] = "#808080"  # Grey for noise
        else:
            cluster_colors[label] = colors[i % len(colors)]
    
    # Count crowns per cluster
    cluster_counts = {int(label): int((labels == label).sum()) for label in unique_labels}
    
    # Build annotations
    annotations = []
    boxes = predictions['boxes'].numpy()
    masks = predictions['masks'].numpy()
    scores = predictions['scores'].numpy()
    
    for i, (box, mask, score, label) in enumerate(zip(boxes, masks, scores, labels)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Convert mask to RLE
        mask_fortran = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_util.encode(mask_fortran)
        rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string
        
        annotation = {
            "id": int(i),
            "tile_position": {"row": 0, "col": 0},  # Single image, no tiling yet
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "segmentation": rle,
            "area": float(mask.sum()),
            "confidence": float(score),
            "cluster_id": int(label)
        }
        annotations.append(annotation)
    
    # Build full COCO structure
    coco_output = {
        "metadata": {
            "image_file": str(image_path),
            "image_dimensions": [config.original_size, config.original_size],
            "tile_size": config.original_size,
            "tile_overlap_percent": 0,  # Single image
            "total_tiles": 1,
            "clustering_algorithm": config.clustering_method,
            "clustering_params": clustering_params,
            "silhouette_score": float(silhouette)
        },
        "annotations": annotations,
        "clusters": {
            str(label): {
                "count": cluster_counts[label],
                "color": cluster_colors[label]
            }
            for label in unique_labels
        }
    }
    
    print(f"✓ Created {len(annotations)} annotations")
    return coco_output


# In[64]:


def save_json(data: Dict, output_path: str):
    """Save data to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved JSON to {output_path}")


# <h4>Visualize Results</h4>

# In[65]:


def visualize_results(
    image_np: np.ndarray,
    predictions: Dict,
    labels: np.ndarray,
    cluster_info: Dict,
    config: PipelineConfig
) -> go.Figure:
    """
    Create interactive Plotly visualization
    """
    print("\nCreating interactive visualization...")
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Clustered Tree Crowns", "Cluster Statistics"),
        specs=[[{"type": "image"}, {"type": "bar"}]],
        column_widths=[0.7, 0.3]
    )
    
    # Prepare overlay image
    overlay = image_np.copy()
    masks = predictions['masks'].numpy()
    
    # Get cluster colors from cluster_info
    cluster_colors_hex = {int(k): v['color'] for k, v in cluster_info.items()}
    
    # Overlay masks with cluster colors
    for mask, label in zip(masks, labels):
        color_hex = cluster_colors_hex[int(label)]
        # Convert hex to RGB
        color_rgb = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Apply colored mask with transparency
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask > 0] = color_rgb
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
    
    # Add image
    fig.add_trace(
        go.Image(z=overlay),
        row=1, col=1
    )
    
    # Add cluster statistics bar chart
    cluster_counts = [v['count'] for k, v in sorted(cluster_info.items()) if int(k) != -1]
    cluster_labels_list = [f"Cluster {k}" for k in sorted(cluster_info.keys()) if int(k) != -1]
    cluster_colors_list = [v['color'] for k, v in sorted(cluster_info.items()) if int(k) != -1]
    
    # Add noise separately if it exists
    if '-1' in cluster_info:
        cluster_labels_list.append("Noise")
        cluster_counts.append(cluster_info['-1']['count'])
        cluster_colors_list.append(cluster_info['-1']['color'])
    
    fig.add_trace(
        go.Bar(
            x=cluster_labels_list,
            y=cluster_counts,
            marker_color=cluster_colors_list,
            text=cluster_counts,
            textposition='auto',
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Cluster", row=1, col=2)
    fig.update_yaxes(title_text="Number of Crowns", row=1, col=2)
    
    # Add title with statistics
    total_crowns = len(labels)
    n_clusters = len([k for k in cluster_info.keys() if int(k) != -1])
    noise_pct = (cluster_info.get('-1', {}).get('count', 0) / total_crowns) * 100 if '-1' in cluster_info else 0
    
    title_text = (
        f"Tree Crown Segmentation Results<br>"
        f"<sub>Total Crowns: {total_crowns} | Clusters: {n_clusters} | "
        f"Noise: {noise_pct:.1f}%</sub>"
        f"Mask Threshold: {config.mask_threshold}"
    )
    
    fig.update_layout(
        title_text=title_text,
        height=600,
        showlegend=False
    )
    
    print("✓ Visualization created")
    return fig


# <h4>Main Pipeline</h4>

# In[66]:


def run_pipeline(
    config: PipelineConfig,
    autoencoder_model,  # Pass your initialized TreeCrownResNet34 here
    param_grid: Optional[Dict] = None
) -> Tuple[Dict, go.Figure]:
    """
    Run the complete tree crown segmentation and clustering pipeline
    
    Args:
        config: Pipeline configuration
        autoencoder_model: Your initialized autoencoder model class
        param_grid: Optional custom parameter grid for clustering
    
    Returns:
        coco_json: COCO format dictionary with results
        fig: Plotly figure for visualization
    """
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("TREE CROWN SEGMENTATION & CLUSTERING PIPELINE")
    print("="*70)
    
    # 1. Load models
    maskrcnn = load_maskrcnn(config.maskrcnn_checkpoint, config)
    autoencoder = load_autoencoder(config.autoencoder_checkpoint, config, autoencoder_model)
    
    # 2. Load and preprocess image
    print(f"\nLoading image from {config.image_path}...")
    image_tensor, image_np = load_and_preprocess_image(config.image_path, config.maskrcnn_size)
    print(f"✓ Image loaded: {image_np.shape}")
    
    # 3. Run detection
    predictions = run_detection(maskrcnn, image_tensor, config)
    
    if len(predictions['scores']) == 0:
        print("⚠ No tree crowns detected!")
        return None, None
    
    # 4. Scale predictions back to original size
    predictions_scaled = scale_predictions(
        predictions,
        from_size=config.maskrcnn_size,
        to_size=config.original_size
    )
    
    # Reload original image at full resolution
    image_full = np.array(Image.open(config.image_path).convert('RGB'))
    
    # 5. Crop masks
    crops = crop_masks(image_full, predictions_scaled, config)
    
    # 6. Resize crops for autoencoder
    crops_resized = resize_crops(crops, config.autoencoder_size)
    
    # 7. Extract features
    features = extract_features(autoencoder, crops_resized, config)
    
    # 8. Optimize clustering parameters
    best_params, best_score = optimize_clustering_parameters(features, config, param_grid)
    
    # 9. Cluster all features
    labels = cluster_features(features, config, best_params)
    
    # 10. Export to JSON
    coco_json = predictions_to_coco_json(
        predictions_scaled,
        labels,
        config.image_path,
        config,
        best_params,
        best_score
    )
    
    output_json_path = Path(config.output_dir) / "results.json"
    save_json(coco_json, str(output_json_path))
    
    # 11. Visualize
    fig = visualize_results(
        image_full,
        predictions_scaled,
        labels,
        coco_json['clusters'],
        config
    )
    
    # Save visualization
    output_html_path = Path(config.output_dir) / "visualization.html"
    fig.write_html(str(output_html_path))
    print(f"✓ Saved interactive visualization to {output_html_path}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    
    return coco_json, fig