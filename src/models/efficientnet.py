"""EfficientNet-based models for ISAR image classification."""

import torch
import torch.nn as nn
from typing import Optional


class ISAREfficientNet(nn.Module):
    """
    EfficientNet-based classifier adapted for ISAR images.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        variant: str = 'b0',
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize EfficientNet classifier.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            variant: EfficientNet variant ('b0', 'b1', 'b2', etc.)
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Load base model
        try:
            from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2
            from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights
            
            if variant == 'b0':
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                self.base_model = efficientnet_b0(weights=weights)
                fc_in_features = 1280
            elif variant == 'b1':
                weights = EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
                self.base_model = efficientnet_b1(weights=weights)
                fc_in_features = 1280
            elif variant == 'b2':
                weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
                self.base_model = efficientnet_b2(weights=weights)
                fc_in_features = 1408
            else:
                raise ValueError(f"Unknown variant: {variant}")
            
        except ImportError:
            # Fallback to custom implementation
            self.base_model = self._create_custom_efficientnet(variant)
            fc_in_features = 1280
        
        # Modify first conv layer for grayscale input
        if in_channels != 3:
            original_conv = self.base_model.features[0][0]
            self.base_model.features[0][0] = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            if pretrained:
                with torch.no_grad():
                    self.base_model.features[0][0].weight.data = \
                        original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Replace classifier
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_in_features, num_classes)
        )
        
        self.feature_dim = fc_in_features
    
    def _create_custom_efficientnet(self, variant: str) -> nn.Module:
        """Create a custom EfficientNet-like architecture."""
        # Simplified version if torchvision doesn't have EfficientNet
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            MBConvBlock(32, 16, 1),
            MBConvBlock(16, 24, 2, expand_ratio=6),
            MBConvBlock(24, 40, 2, expand_ratio=6),
            MBConvBlock(40, 80, 2, expand_ratio=6),
            MBConvBlock(80, 112, 1, expand_ratio=6),
            MBConvBlock(112, 192, 2, expand_ratio=6),
            MBConvBlock(192, 320, 1, expand_ratio=6),
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.base_model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer."""
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze backbone layers."""
        for name, param in self.base_model.features.named_parameters():
            param.requires_grad = not freeze


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 1,
        se_ratio: float = 0.25
    ):
        super().__init__()
        
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1,
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        
        # SE block
        se_channels = max(1, int(in_channels * se_ratio))
        layers.append(SEBlock(hidden_dim, se_channels))
        
        # Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels: int, se_channels: int):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, se_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y
