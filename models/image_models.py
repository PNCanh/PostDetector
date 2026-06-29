import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Extractor(nn.Module):
    """
    Standard ResNet50 feature extractor.
    """
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # Remove final FC layer
        self.proj = nn.Linear(2048, hidden_dim) # ResNet50 outputs 2048-d
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1) # Flatten (B, 2048)
        out = self.proj(features) # (B, hidden_dim)
        return out

class ResNet50Transformer(nn.Module):
    """
    ResNet50 followed by Transformer encoder layers for richer spatial feature extraction.
    """
    def __init__(self, hidden_dim: int = 768, num_layers: int = 2):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Keep features up to the last conv layer to preserve spatial dimensions
        # resnet.children()[:-2] removes AvgPool and FC
        self.features = nn.Sequential(*list(resnet.children())[:-2]) 
        
        self.conv_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.features(x) # (B, 2048, H', W')
        features = self.conv_proj(features) # (B, hidden_dim, H', W')
        
        B, C, H, W = features.shape
        # Flatten spatial dimensions -> (B, H*W, C) for transformer
        features = features.flatten(2).permute(0, 2, 1) 
        
        # Pass through transformer
        transformed_features = self.transformer(features)
        
        # Mean pooling over spatial tokens
        out = transformed_features.mean(dim=1) # (B, hidden_dim)
        
        return out
