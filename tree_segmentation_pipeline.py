import modal

app = modal.App("tree-crown-segmentation")

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
        "matplotlib",
        "tqdm"
    ])
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
)

# Create volume for dataset
volume = modal.Volume.from_name("tree-dataset", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="T4",
    timeout=3600
)
def train_model(epochs=50, batch_size=8, learning_rate=1e-4):
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    import cv2
    from pathlib import Path
    from tqdm import tqdm
    
    class TreeCrownDataset(Dataset):
        def __init__(self, image_dir, csv_path, transform=None, mask_size=(256, 256)):
            self.image_dir = Path(image_dir)
            self.annotations = pd.read_csv(csv_path)
            self.transform = transform
            self.mask_size = mask_size
            
            # Clean up any unnamed columns
            self.annotations = self.annotations.loc[:, ~self.annotations.columns.str.contains('^Unnamed')]
            
            # Simplify all tree labels to just "tree" for binary segmentation
            if 'label' in self.annotations.columns:
                self.annotations['label'] = 'tree'
                
            print(f"Dataset loaded: {len(self.annotations)} tree annotations")
            
        def __len__(self):
            return len(self.annotations)
        
        def __getitem__(self, idx):
            row = self.annotations.iloc[idx]
            
            # Load image
            img_path = self.image_dir / row['image_path']
            image = Image.open(img_path).convert('RGB')
            
            # Get bounding box coordinates
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            # Create segmentation mask from bounding box
            mask = self.create_mask_from_bbox(image.size, xmin, ymin, xmax, ymax)
            
            # Resize to standard size
            image = image.resize(self.mask_size, Image.LANCZOS)
            mask = Image.fromarray(mask).resize(self.mask_size, Image.NEAREST)
            mask = np.array(mask)
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
                
            # Convert mask to tensor
            mask = torch.from_numpy(mask).long()
            
            return image, mask
        
        def create_mask_from_bbox(self, img_size, xmin, ymin, xmax, ymax):
            """Create circular/elliptical mask from bounding box for tree segmentation
            Returns binary mask: 0 = background, 1 = tree"""
            width, height = img_size
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Calculate center and dimensions
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            radius_x = (xmax - xmin) // 2
            radius_y = (ymax - ymin) // 2
            
            # Create elliptical mask for tree crown (label = 1 for tree)
            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 1, -1)
            
            return mask

    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(DoubleConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=2, features=[64, 128, 256, 512]):
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
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TreeCrownDataset("/data/train", "/data/train/train.csv", transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize model
    model = UNet(in_channels=3, out_channels=2).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"/data/checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), "/data/tree_segmentation_model.pth")
    
    return {
        "message": "Training completed successfully",
        "final_loss": train_losses[-1],
        "epochs_trained": epochs
    }

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="T4"
)
def evaluate_model(model_path="/data/tree_segmentation_model.pth"):
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    import cv2
    from pathlib import Path
    from sklearn.metrics import accuracy_score, jaccard_score
    from tqdm import tqdm
    
    # Import classes (same as in training)
    class TreeCrownDataset(Dataset):
        def __init__(self, image_dir, csv_path, transform=None, mask_size=(256, 256)):
            self.image_dir = Path(image_dir)
            self.annotations = pd.read_csv(csv_path)
            self.transform = transform
            self.mask_size = mask_size
            
            # Clean up any unnamed columns
            self.annotations = self.annotations.loc[:, ~self.annotations.columns.str.contains('^Unnamed')]
            
            # Simplify all tree labels to just "tree" for binary segmentation
            if 'label' in self.annotations.columns:
                self.annotations['label'] = 'tree'
                
            print(f"Dataset loaded: {len(self.annotations)} tree annotations")
            
        def __len__(self):
            return len(self.annotations)
        
        def __getitem__(self, idx):
            row = self.annotations.iloc[idx]
            
            # Load image
            img_path = self.image_dir / row['image_path']
            image = Image.open(img_path).convert('RGB')
            
            # Get bounding box coordinates
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            # Create segmentation mask from bounding box
            mask = self.create_mask_from_bbox(image.size, xmin, ymin, xmax, ymax)
            
            # Resize to standard size
            image = image.resize(self.mask_size, Image.LANCZOS)
            mask = Image.fromarray(mask).resize(self.mask_size, Image.NEAREST)
            mask = np.array(mask)
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
                
            # Convert mask to tensor
            mask = torch.from_numpy(mask).long()
            
            return image, mask
        
        def create_mask_from_bbox(self, img_size, xmin, ymin, xmax, ymax):
            """Create circular/elliptical mask from bounding box for tree segmentation
            Returns binary mask: 0 = background, 1 = tree"""
            width, height = img_size
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Calculate center and dimensions
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            radius_x = (xmax - xmin) // 2
            radius_y = (ymax - ymin) // 2
            
            # Create elliptical mask for tree crown (label = 1 for tree)
            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 1, -1)
            
            return mask

    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(DoubleConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=2, features=[64, 128, 256, 512]):
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
    
    # Load model
    model = UNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TreeCrownDataset("/data/test", "/data/test/test.csv", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    iou = jaccard_score(all_targets, all_predictions, average='weighted')
    
    return {
        "accuracy": float(accuracy),
        "iou": float(iou),
        "num_samples": len(test_dataset)
    }

@app.function(image=image, volumes={"/data": volume})
def upload_dataset():
    """Upload local dataset to Modal volume"""
    import shutil
    import os
    
    # Copy train, test, and evaluation folders
    for folder in ["train", "test", "evaluation"]:
        if os.path.exists(folder):
            print(f"Uploading {folder}...")
            shutil.copytree(f"./{folder}", f"/data/{folder}", dirs_exist_ok=True)
            print(f"✓ {folder} uploaded successfully")
        else:
            print(f"⚠ {folder} not found locally")
    
    return "Dataset upload completed"

@app.local_entrypoint()
def main():
    """Main pipeline execution"""
    print("🌲 Tree Crown Segmentation Pipeline")
    print("=" * 40)
    
    # Step 1: Upload dataset
    print("1. Uploading dataset...")
    upload_result = upload_dataset.remote()
    print(upload_result)
    
    # Step 2: Train model
    print("\n2. Training model...")
    train_result = train_model.remote(epochs=30, batch_size=8, learning_rate=1e-4)
    print(train_result)
    
    # Step 3: Evaluate model
    print("\n3. Evaluating model...")
    eval_result = evaluate_model.remote()
    print(f"Evaluation Results:")
    print(f"  Accuracy: {eval_result['accuracy']:.4f}")
    print(f"  IoU: {eval_result['iou']:.4f}")
    print(f"  Test samples: {eval_result['num_samples']}")
    
    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()