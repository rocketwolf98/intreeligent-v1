#!/usr/bin/env python3
"""
Script to load and use the downloaded tree segmentation model
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

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

def load_model(model_path="./models/best_model.pth"):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model architecture
    model = UNet(in_channels=3, out_channels=2).to(device)
    
    # Load weights
    if model_path.endswith('best_model.pth'):
        # Load from checkpoint format
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} with val_loss: {checkpoint['val_loss']:.4f}")
    else:
        # Load from state dict format
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    
    model.eval()
    return model, device

def preprocess_image(image_path, size=(256, 256)):
    """Preprocess image for model input"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, original_size

def predict_segmentation(model, image_tensor, device):
    """Run segmentation prediction"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        predictions = torch.argmax(outputs, dim=1)
        probabilities = torch.softmax(outputs, dim=1)
    
    return predictions.cpu().numpy()[0], probabilities.cpu().numpy()[0]

def visualize_results(image_path, mask, probabilities, save_path=None):
    """Visualize segmentation results"""
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    original_image = original_image.resize((256, 256))
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Tree Segmentation')
    axes[1].axis('off')
    
    # Tree probability
    axes[2].imshow(probabilities[1], cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Tree Probability')
    axes[2].axis('off')
    
    # Overlay
    overlay = np.array(original_image)
    tree_mask = mask == 1
    overlay[tree_mask] = overlay[tree_mask] * 0.7 + np.array([0, 255, 0]) * 0.3
    axes[3].imshow(overlay.astype(np.uint8))
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results saved to {save_path}")
    
    plt.show()

def main():
    """Main function to demonstrate model usage"""
    # Check if model exists
    model_path = "./models/best_model.pth"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("Please run the training pipeline first to download the model.")
        return
    
    # Load model
    print("Loading trained model...")
    model, device = load_model(model_path)
    
    # Example usage - replace with your image path
    image_path = "path/to/your/image.jpg"
    
    if Path(image_path).exists():
        print(f"Processing image: {image_path}")
        
        # Preprocess
        image_tensor, original_size = preprocess_image(image_path)
        
        # Predict
        mask, probabilities = predict_segmentation(model, image_tensor, device)
        
        # Calculate statistics
        tree_pixels = np.sum(mask == 1)
        total_pixels = mask.size
        tree_percentage = (tree_pixels / total_pixels) * 100
        
        print(f"Tree coverage: {tree_percentage:.1f}%")
        print(f"Tree pixels: {tree_pixels}/{total_pixels}")
        
        # Visualize
        visualize_results(image_path, mask, probabilities, 
                         save_path=f"segmentation_result_{Path(image_path).stem}.png")
    else:
        print(f"❌ Image not found at {image_path}")
        print("Please provide a valid image path.")

if __name__ == "__main__":
    main()