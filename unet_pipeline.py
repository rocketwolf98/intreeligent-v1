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
volume = modal.Volume.from_name("treesdataset", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100",
    timeout=3600
)
def train_model(epochs=75, batch_size=16, learning_rate=1e-3):
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
    
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = TreeCrownDataset("/data/train", "/data/train/train.csv")
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=42)
    
    # Create separate datasets with different transforms
    train_dataset = TreeCrownDataset("/data/train", "/data/train/train.csv", transform=train_transform)
    val_dataset = TreeCrownDataset("/data/train", "/data/train/train.csv", transform=val_transform)
    
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    # Initialize model
    model = UNet(in_channels=3, out_channels=2).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update scheduler
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, "/data/best_model.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, f"/data/checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), "/data/tree_segmentation_model.pth")
    
    return {
        "message": "Training completed successfully",
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses)
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
def download_model(model_name="tree_segmentation_model.pth", local_path="./"):
    """Download trained model from Modal volume to local filesystem"""
    import shutil
    import os
    from pathlib import Path
    
    # Available models in the volume
    model_files = [
        "tree_segmentation_model.pth",  # Final model
        "best_model.pth",              # Best validation model
    ]
    
    # Add checkpoint files if they exist
    checkpoint_files = []
    for i in range(10, 80, 10):  # Check for checkpoints every 10 epochs
        checkpoint_file = f"checkpoint_epoch_{i}.pth"
        if os.path.exists(f"/data/{checkpoint_file}"):
            checkpoint_files.append(checkpoint_file)
    
    print(f"Available models: {model_files + checkpoint_files}")
    
    # Download specified model
    remote_path = f"/data/{model_name}"
    local_file_path = Path(local_path) / model_name
    
    if os.path.exists(remote_path):
        shutil.copy(remote_path, local_file_path)
        print(f"✓ Downloaded {model_name} to {local_file_path}")
        print(f"Model size: {os.path.getsize(local_file_path) / 1024 / 1024:.2f} MB")
        return f"Model downloaded successfully to {local_file_path}"
    else:
        print(f"❌ Model {model_name} not found in volume")
        return f"Model {model_name} not found"

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
    train_result = train_model.remote(epochs=75, batch_size=16, learning_rate=1e-3)
    print(train_result)
    
    # Step 3: Evaluate model
    print("\n3. Evaluating model...")
    eval_result = evaluate_model.remote()
    print(f"Evaluation Results:")
    print(f"  Accuracy: {eval_result['accuracy']:.4f}")
    print(f"  IoU: {eval_result['iou']:.4f}")
    print(f"  Test samples: {eval_result['num_samples']}")
    
    # Step 4: Download models
    print("\n4. Downloading trained models...")
    
    # Download best model (recommended)
    best_model_result = download_model.remote(model_name="best_model.pth", local_path="./models/")
    print(best_model_result)
    
    # Download final model
    final_model_result = download_model.remote(model_name="tree_segmentation_model.pth", local_path="./models/")
    print(final_model_result)
    
    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()