#import packages
import modal
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import segmentation_models_pytorch as smp
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set up Modal App for intreeligent
app = modal.App("intreeligent-v1")

# Set up Modal image
image = (
    modal.Image.debian_slim(python_version="3.13.7")
    .pip_install(
        "torch",
        "torchvision",
        "segmentation-models-pytorch",
        "opencv-python-headless",
        "pillow",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "et-xmlfile"
    )
    # .apt_install("libgl1-mesa-glx" "libglib2.0-0")  # for cv2
)

# Create volume for data
volume = modal.Volume.from_name("treesdataset", create_if_missing=True)

# Converting XML annotations to masks
def xml_to_mask(xml_path, img_shape):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    mask = np.zeros(img_shape[:2], dtype=np.uint8)

    for obj in root.findall('.//object'):
        tree_element = obj.find('tree')
        if tree_element is not None:
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                center_x = (xmin + xmax) // 2
                center_y = (ymin + ymax) // 2
                width = (xmax - xmin) // 2
                height = (ymax - ymin) // 2

                cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 1, -1)

    return mask

class TreeDataset(Dataset):
    def __init__(self, image_dir, xml_dir, transform=None):
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.tif', '.tiff', '.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        xml_name = img_name.replace('.tif', '.xml').replace('.tiff', '.xml').replace('.jpg', '.xml').replace('.png', '.xml')
        xml_path = os.path.join(self.xml_dir, xml_name)

        image_array = np.array(image)
        mask = xml_to_mask(xml_path, image_array.shape)

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(mask).long()
        return image, mask
    
def extract_tree_features(image, mask):
    """Extract features for each tree in the mask."""
    features_list = []
    contours, _ = cv2.findContours(mask.numpy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 50:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        tree_region = image[y:y+h, x:x+w]
        tree_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.filPoly(tree_mask, contour - [x, y], 1)

        features = []

        for channel in range(3):
            channel_pixels = tree_region[:, :, channel][tree_mask > 0]
            if len(channel_pixels) > 0:
                features.append(np.mean(channel_pixels))
                features.append(np.std(channel_pixels))
            else:
                features.extend([0,0])

        area = cv2.contourArea(contour)
        features.append(area)
        features.append(w/h)

        features_list.append(features)

    return np.array(features_list) 

# We will execute with Modal functions
@app.function(
    image=image,
    gpu="A100",
    memory=32768,
    timeout=3600,
    volumes={"/data": volume}
)

def upload_data():
    print("Please upload your dataset to /data/ using Modal CLI")
    print(" Modal volume put treesdataset local/path/to/train /data/train")
    print(" Modal volume put treesdataset local/path/to/annotations /data/annotations") 

@app.function(
    image=image,
    gpu="A100",
    memory=32768,
    timeout=7200,
    volumes={"/data": volume}
)

def train_model():
    print("Starting model training...")

    IMAGE_DIR = "/data/train"
    XML_DIR = "/data/annotations"

    if not os.path.exists(IMAGE_DIR) or not os.path.exists(XML_DIR):
        raise FileNotFoundError("Image or XML directory not found in /data/. Please upload your dataset.")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = TreeDataset(IMAGE_DIR, XML_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches")

    model = smp.Unet(encoder_name="resnet34", 
                     encoder_weights="imagenet", 
                     in_channels=3, 
                     classes=2).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(3):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/3")

        for batch_idx, (images, masks) in enumerate(progress_bar):
            images, masks = images.cuda(), masks.cuda()

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({
                'Loss' : f'{loss.item():.4f}',
                'Avg Loss' : f'{(total_loss/(batch_idx+1)):.4f}'
            })

            torch.cuda.empty_cache()
        
        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "/data/tree_segmentation_model.pth")
    print("Model training completed and saved to /data/tree_segmentation_model.pth")

    return "Training completed successfully."

# Test Execution
@app.local_entrypoint()
def main():
    """Main execution pipeline"""

    print("🌲 Modal Tree Crown Segmentation Pipeline")
    print("=" * 50)
    
    # Step 1: Upload data (run this first manually)
    print("1. Upload your data using Modal CLI:")
    print("   modal volume put tree-data-volume ./train /data/train")  
    print("   modal volume put tree-data-volume ./annotations /data/annotations")
    print()
    
    # Step 2: Train model
    print("2. Training model...")
    train_result = train_model.remote()
    print(f"Training result: {train_result}")

if __name__ == "__main__":
    main()