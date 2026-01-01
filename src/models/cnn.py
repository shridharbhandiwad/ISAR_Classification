"""Custom CNN architecture for ISAR image classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with SE attention."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True
    ):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        
        out += identity
        out = self.activation(out)
        
        return out


class ISARClassifierCNN(nn.Module):
    """
    Custom CNN architecture optimized for ISAR image classification.
    
    Features:
    - Squeeze-and-Excitation attention blocks
    - Residual connections
    - Spatial pyramid pooling
    - Dropout regularization
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        base_channels: int = 64,
        dropout: float = 0.3,
        use_se: bool = True
    ):
        """
        Initialize the CNN classifier.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            base_channels: Base number of channels
            dropout: Dropout rate
            use_se: Whether to use SE attention
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Initial convolution
        self.stem = nn.Sequential(
            ConvBlock(in_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual stages
        self.stage1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels, use_se=use_se),
            ResidualBlock(base_channels, base_channels, use_se=use_se)
        )
        
        self.stage2 = nn.Sequential(
            ResidualBlock(base_channels, base_channels * 2, stride=2, use_se=use_se),
            ResidualBlock(base_channels * 2, base_channels * 2, use_se=use_se)
        )
        
        self.stage3 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 4, stride=2, use_se=use_se),
            ResidualBlock(base_channels * 4, base_channels * 4, use_se=use_se)
        )
        
        self.stage4 = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 8, stride=2, use_se=use_se),
            ResidualBlock(base_channels * 8, base_channels * 8, use_se=use_se)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """Get intermediate feature maps for visualization."""
        features = {}
        
        x = self.stem(x)
        features['stem'] = x
        
        x = self.stage1(x)
        features['stage1'] = x
        
        x = self.stage2(x)
        features['stage2'] = x
        
        x = self.stage3(x)
        features['stage3'] = x
        
        x = self.stage4(x)
        features['stage4'] = x
        
        return features


class ISARClassifierLightCNN(nn.Module):
    """Lightweight CNN for faster inference."""
    
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
