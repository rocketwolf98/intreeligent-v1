from torchvision.models import resnet34, ResNet34_Weights, resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from torchvision import models

#ResNet34 Pipeline
class TreeCrownResNet34(nn.Module):
    """Fine-tuned ResNet34 for 512x512 tree crown masks"""

    def __init__(self, freeze_backbone=True, latent_dim=256, use_pretrained=True):
        super().__init__()
        
        # Load pretrained weights or random init
        weights = ResNet34_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = models.resnet34(weights=weights)

        # Grab feature dimension before fc
        in_features = self.backbone.fc.in_features
        
        # Remove the final classifier head
        self.backbone.fc = nn.Identity()
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, in_features)
        )

        # Store preprocessing from weights (for transforms)
        self.preprocess = weights.transforms() if use_pretrained else None

    def encode(self, x):
        features = self.backbone(x)   # (B, 512)
        return self.projection(features)

    def forward(self, x):
        features = self.backbone(x)
        latent = self.projection(features)
        reconstructed = self.decoder(latent)
        return latent, reconstructed, features
    
class TreeCrownResNet50(nn.Module):
    """Fine-tuned ResNet34 for 512x512 tree crown masks"""

    def __init__(self, freeze_backbone=True, latent_dim=256, use_pretrained=True):
        super().__init__()
        
        # Load pretrained weights or random init
        weights = ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Grab feature dimension before fc
        in_features = self.backbone.fc.in_features
        
        # Remove the final classifier head
        self.backbone.fc = nn.Identity()
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, in_features)
        )

        # Store preprocessing from weights (for transforms)
        self.preprocess = weights.transforms() if use_pretrained else None

    def encode(self, x):
        features = self.backbone(x)   # (B, 512)
        return self.projection(features)

    def forward(self, x):
        features = self.backbone(x)
        latent = self.projection(features)
        reconstructed = self.decoder(latent)
        return latent, reconstructed, features
    

#DINO Pipeline
class TreeCrownDINO(nn.Module):
    """Fine-tuned DINOv2 for 512x512 tree crown masks"""
    
    def __init__(self, freeze_backbone=True, latent_dim=256):
        super().__init__()
        
        # Load pre-trained DINOv2
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head for tree crown features
        self.projection = nn.Sequential(
            nn.Linear(384, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, latent_dim)
        )
        
        # Decoder for reconstruction loss
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 384)
        )
    
    def encode(self, x):
        """Extract latent features"""
        features = self.backbone(x)
        return self.projection(features)
    
    def forward(self, x):
        features = self.backbone(x)
        latent = self.projection(features)
        reconstructed = self.decoder(latent)
        return latent, reconstructed, features