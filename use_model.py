#!/usr/bin/env python3
"""
Script to use the trained tree segmentation model via Modal
"""

import modal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import time
import glob
import random
import io
import base64

app = modal.App("tree-crown-segmentation-visualization")

# Modal image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "torchvision",
        "pandas",
        "numpy",
        "pillow",
        "opencv-python-headless",
        "scikit-learn",
        "scikit-image",
        "matplotlib",
        "tqdm"
    ])
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
)

# Create volume for dataset
volume = modal.Volume.from_name("treesdataset", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="T4",
    timeout=600
)
def predict_volume_image_tiled(image_path_in_volume, tile_size=512, target_size=256, overlap=64, model_path="/data/best_model.pth"):
    """Predict segmentation using tiled inference for high-resolution images"""
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # For headless environment
    import io
    import base64
    import time
    
    # Define the same UNet classes as in training
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels, dropout=0.1):
            super(DoubleConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=2, features=[64, 128, 256, 512, 1024]):
            super(UNet, self).__init__()
            self.ups = nn.ModuleList()
            self.downs = nn.ModuleList()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Down part of UNET
            for feature in features:
                self.downs.append(DoubleConv(in_channels, feature))
                in_channels = feature
                
            # Up part of UNET
            for feature in reversed(features):
                self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
                self.ups.append(DoubleConv(feature*2, feature))
                
            self.bottleneck = DoubleConv(features[-1], features[-1]*2)
            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
            
        def forward(self, x):
            skip_connections = []
            
            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)
                
            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]
            
            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx//2]
                
                if x.shape != skip_connection.shape:
                    x = transforms.functional.resize(x, size=skip_connection.shape[2:])
                    
                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx+1](concat_skip)
                
            return self.final_conv(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = UNet(in_channels=3, out_channels=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        model_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'val_loss': float(checkpoint.get('val_loss', 0)),
            'train_loss': float(checkpoint.get('train_loss', 0))
        }
        print(f"Loaded model from epoch {model_info['epoch']} with val_loss: {model_info['val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        model_info = {'epoch': 'final', 'val_loss': 'unknown', 'train_loss': 'unknown'}
    
    model.eval()
    
    # Load high-resolution image directly from Modal volume
    print(f"Loading image from volume: {image_path_in_volume}")
    
    try:
        image = Image.open(image_path_in_volume).convert('RGB')
        print(f"✅ Successfully loaded image from volume")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        raise
    
    original_size = image.size
    width, height = original_size
    print(f"Processing high-res image: {width}x{height}")
    
    # Calculate number of tiles needed
    stride = tile_size - overlap
    tiles_x = (width - overlap + stride - 1) // stride
    tiles_y = (height - overlap + stride - 1) // stride
    print(f"Creating {tiles_x}x{tiles_y} = {tiles_x * tiles_y} tiles (tile_size={tile_size}, overlap={overlap})")
    
    # Prepare transform for individual tiles (resize to match training data!)
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),  # Resize tile to model's expected input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize full-size prediction arrays
    full_prediction = np.zeros((height, width), dtype=np.float32)
    overlap_count = np.zeros((height, width), dtype=np.float32)
    
    start_time = time.time()
    
    print("🔄 Processing tiles...")
    with torch.no_grad():
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                # Calculate tile coordinates
                start_x = tile_x * stride
                start_y = tile_y * stride
                end_x = min(start_x + tile_size, width)
                end_y = min(start_y + tile_size, height)
                
                # Extract larger tile from image
                tile = image.crop((start_x, start_y, end_x, end_y))
                
                # Pad tile to tile_size if needed (for edge tiles)
                if tile.size != (tile_size, tile_size):
                    padded_tile = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                    padded_tile.paste(tile, (0, 0))
                    tile = padded_tile
                
                # Process tile (transform will resize to target_size for model)
                tile_tensor = transform(tile).unsqueeze(0).to(device)
                outputs = model(tile_tensor)
                tile_probs = torch.softmax(outputs, dim=1)[0, 1].cpu().numpy()  # Tree probability
                
                # Resize prediction back to original tile size
                tile_probs_resized = np.array(Image.fromarray(tile_probs).resize((tile_size, tile_size), Image.LANCZOS))
                
                # Handle padding for edge tiles
                actual_height = end_y - start_y
                actual_width = end_x - start_x
                tile_probs = tile_probs_resized[:actual_height, :actual_width]
                
                # Add to full prediction with overlap handling
                full_prediction[start_y:end_y, start_x:end_x] += tile_probs
                overlap_count[start_y:end_y, start_x:end_x] += 1
                
                if (tile_y * tiles_x + tile_x + 1) % 10 == 0:
                    print(f"  📊 Processed {tile_y * tiles_x + tile_x + 1}/{tiles_x * tiles_y} tiles")
    
    # Average overlapping predictions
    overlap_count[overlap_count == 0] = 1  # Avoid division by zero
    full_prediction = full_prediction / overlap_count
    
    # Convert probabilities to binary mask (higher threshold to reduce false positives)
    mask = (full_prediction > 0.8).astype(np.uint8)
    
    # Apply post-processing to clean up the mask
    import cv2
    # Remove small noise with morphological opening
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small connected components (less than 1000 pixels)
    from skimage import measure, morphology
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=1000).astype(np.uint8)
    
    print(f"🧹 Applied post-processing to clean up segmentation")
    
    inference_time = time.time() - start_time
    print(f"✅ Processed all tiles in {inference_time:.2f}s")
    
    # Calculate performance metrics for high-resolution result
    metrics = {
        'model_info': model_info,
        'original_size': original_size,
        'inference_time_ms': inference_time * 1000,
        'tiles_processed': tiles_x * tiles_y,
        'tile_size': tile_size,
        'overlap': overlap,
        'tree_pixels': int(np.sum(mask == 1)),
        'background_pixels': int(np.sum(mask == 0)),
        'tree_coverage_percent': float(np.sum(mask == 1) / mask.size * 100),
        'max_tree_probability': float(np.max(full_prediction)),
        'min_tree_probability': float(np.min(full_prediction)),
        'avg_tree_probability': float(np.mean(full_prediction[mask == 1])) if np.any(mask == 1) else 0.0,
        'prediction_confidence': float(np.mean(full_prediction))
    }
    
    # Create visualization with full-resolution results
    # For display, create reasonable-sized versions
    display_width = 800
    aspect_ratio = height / width
    display_height = int(display_width * aspect_ratio)
    
    # Resize for visualization (maintain aspect ratio)
    display_image = image.resize((display_width, display_height), Image.LANCZOS)
    display_mask = Image.fromarray((mask * 255).astype(np.uint8)).resize((display_width, display_height), Image.NEAREST)
    display_mask_array = np.array(display_mask) / 255.0
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original high-resolution image (downsampled for display)
    axes[0].imshow(display_image)
    axes[0].set_title(f"Original Image\n{width}x{height} → {display_width}x{display_height}", fontsize=12)
    axes[0].axis('off')
    
    # High-resolution segmentation mask
    axes[1].imshow(display_mask_array, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_title(f"Tree Segmentation (Tiled)\n{tiles_x * tiles_y} tiles processed", fontsize=12)
    axes[1].axis('off')
    
    # Tree overlay on original image
    overlay = np.array(display_image)
    tree_mask_display = display_mask_array > 0.5
    overlay[tree_mask_display] = overlay[tree_mask_display] * 0.6 + np.array([0, 255, 0]) * 0.4
    
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title(f"High-Res Tree Overlay\nCoverage: {metrics['tree_coverage_percent']:.2f}%", fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization to volume and return as bytes
    plt.savefig('/data/user_prediction_result.png', dpi=150, bbox_inches='tight')
    
    # Convert plot to bytes for returning
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_bytes = buf.getvalue()
    buf.close()
    plt.close()
    
    return {
        "success": True,
        "metrics": metrics,
        "visualization_bytes": base64.b64encode(img_bytes).decode('utf-8'),
        "message": "Segmentation completed successfully"
    }

@app.function(
    image=image,
    volumes={"/data": volume}
)
def list_volume_images():
    """List available images in the Modal volume"""
    import os
    import glob
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF']
    all_images = []
    
    # Search in common directories
    search_dirs = ['/data/train', '/data/test', '/data/evaluation', '/data']
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for ext in image_extensions:
                pattern = os.path.join(search_dir, '**', ext)
                found_images = glob.glob(pattern, recursive=True)
                all_images.extend(found_images)
    
    # Remove duplicates and sort
    all_images = sorted(list(set(all_images)))
    
    print(f"Found {len(all_images)} images in volume")
    for img in all_images[:10]:  # Show first 10
        print(f"  📁 {img}")
    if len(all_images) > 10:
        print(f"  ... and {len(all_images) - 10} more")
    
    return {
        "images": all_images,
        "count": len(all_images)
    }

def select_volume_image():
    """Select an image from the Modal volume"""
    print("\n🖼️  Image Selection from Modal Volume")
    print("=" * 40)
    
    # Get list of available images from volume
    print("📋 Fetching available images from volume...")
    try:
        result = list_volume_images.remote()
        available_images = result["images"]
        
        if not available_images:
            print("❌ No images found in volume")
            return None
            
        print(f"✅ Found {len(available_images)} images")
        
        # Show options
        print("\n1. Select random image")
        print("2. Choose specific image")
        print("3. List all images")
        
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == '1':
            # Random selection
            selected = random.choice(available_images)
            print(f"🎲 Randomly selected: {selected}")
            return selected
            
        elif choice == '2':
            # Show first 20 images with numbers
            print(f"\n📁 Available images (showing first 20):")
            display_images = available_images[:20]
            
            for i, img_path in enumerate(display_images, 1):
                img_name = img_path.split('/')[-1]  # Just filename
                print(f"  {i:2d}. {img_name}")
            
            if len(available_images) > 20:
                print(f"     ... and {len(available_images) - 20} more")
            
            try:
                selection = int(input(f"\nEnter number (1-{min(20, len(available_images))}): "))
                if 1 <= selection <= min(20, len(available_images)):
                    selected = display_images[selection - 1]
                    print(f"✅ Selected: {selected}")
                    return selected
                else:
                    print("❌ Invalid selection")
                    return None
            except ValueError:
                print("❌ Please enter a valid number")
                return None
                
        elif choice == '3':
            # List all images
            print(f"\n📁 All {len(available_images)} images:")
            for img_path in available_images:
                print(f"  📄 {img_path}")
            return select_volume_image()  # Recurse to make selection
            
        else:
            print("❌ Invalid choice")
            return None
            
    except Exception as e:
        print(f"❌ Error accessing volume: {e}")
        return None

def main():
    """Main function - Load image from Modal volume and visualize segmentation"""
    print("🌳 Tree Crown Segmentation - Volume Image Demo")
    print("=" * 55)
    
    # Select image from Modal volume
    volume_image_path = select_volume_image()
    if not volume_image_path:
        print("❌ No image selected. Exiting.")
        return
    
    image_name = volume_image_path.split('/')[-1]
    print(f"\n🚀 Processing volume image: {image_name}")
    
    try:
        # Run tiled segmentation on volume image
        print("🧠 Running high-resolution tiled segmentation on Modal GPU...")
        result = predict_volume_image_tiled.remote(volume_image_path, tile_size=512, target_size=256, overlap=64)
        
        if result["success"]:
            metrics = result["metrics"]
            
            # Display results
            print(f"\n✨ HIGH-RESOLUTION RESULTS:")
            print(f"   🖼️  Original Size: {metrics['original_size'][0]}x{metrics['original_size'][1]}")
            print(f"   🔲 Tiles Processed: {metrics['tiles_processed']} ({metrics['tile_size']}x{metrics['tile_size']}, overlap={metrics['overlap']})")
            print(f"   🌳 Tree Coverage: {metrics['tree_coverage_percent']:.2f}%")
            print(f"   🎯 Average Confidence: {metrics['prediction_confidence']:.3f}")
            print(f"   ⚡ Total Processing: {metrics['inference_time_ms']:.1f}ms")
            print(f"   📊 Tree Pixels: {metrics['tree_pixels']:,}/{metrics['tree_pixels'] + metrics['background_pixels']:,}")
            print(f"   📦 Model: Epoch {metrics['model_info']['epoch']}, Val Loss {metrics['model_info']['val_loss']}")
            
            # Save visualization locally
            print("\n💾 Saving visualization...")
            viz_data = base64.b64decode(result["visualization_bytes"])
            
            output_path = f"volume_segmentation_{image_name.split('.')[0]}.png"
            with open(output_path, 'wb') as f:
                f.write(viz_data)
            
            print(f"✅ Saved: {output_path}")
            
            # Try to display the image
            try:
                img = Image.open(io.BytesIO(viz_data))
                img.show()
                print("🖼️  Visualization opened!")
            except:
                print("💡 Open the saved PNG file to view results")
                
            print(f"\n🎉 {result['message']}")
            
        else:
            print("❌ Segmentation failed")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Make sure Modal is properly configured and volume is accessible")

if __name__ == "__main__":
    main();