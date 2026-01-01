"""ResNet-based models for ISAR image classification."""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights
from typing import Optional


class ISARResNet(nn.Module):
    """
    ResNet-based classifier adapted for ISAR images.
    
    Supports grayscale input and various ResNet variants.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        variant: str = 'resnet18',
        pretrained: bool = True,
        dropout: float = 0.3,
        feature_dim: int = 512
    ):
        """
        Initialize ResNet classifier.
        
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels
            variant: ResNet variant ('resnet18', 'resnet34', 'resnet50')
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate
            feature_dim: Feature dimension before classifier
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.variant = variant
        
        # Load base model
        if variant == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.base_model = resnet18(weights=weights)
            fc_in_features = 512
        elif variant == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.base_model = resnet34(weights=weights)
            fc_in_features = 512
        elif variant == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.base_model = resnet50(weights=weights)
            fc_in_features = 2048
        else:
            raise ValueError(f"Unknown variant: {variant}")
        
        # Modify first conv layer for grayscale input
        if in_channels != 3:
            original_conv = self.base_model.conv1
            self.base_model.conv1 = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Initialize with averaged weights from pretrained
            if pretrained:
                with torch.no_grad():
                    self.base_model.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Replace classifier
        self.base_model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_in_features, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Store feature extractor
        self.feature_dim = fc_in_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.base_model(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer."""
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze backbone layers."""
        for name, param in self.base_model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = not freeze
    
    def get_layer_groups(self):
        """Get parameter groups for differential learning rates."""
        return [
            {'params': self.base_model.conv1.parameters()},
            {'params': self.base_model.bn1.parameters()},
            {'params': self.base_model.layer1.parameters()},
            {'params': self.base_model.layer2.parameters()},
            {'params': self.base_model.layer3.parameters()},
            {'params': self.base_model.layer4.parameters()},
            {'params': self.base_model.fc.parameters()},
        ]


def resnet18_isar(
    num_classes: int = 5,
    in_channels: int = 1,
    pretrained: bool = True,
    **kwargs
) -> ISARResNet:
    """Create ResNet18 model for ISAR classification."""
    return ISARResNet(
        num_classes=num_classes,
        in_channels=in_channels,
        variant='resnet18',
        pretrained=pretrained,
        **kwargs
    )


def resnet34_isar(
    num_classes: int = 5,
    in_channels: int = 1,
    pretrained: bool = True,
    **kwargs
) -> ISARResNet:
    """Create ResNet34 model for ISAR classification."""
    return ISARResNet(
        num_classes=num_classes,
        in_channels=in_channels,
        variant='resnet34',
        pretrained=pretrained,
        **kwargs
    )


class ISARResNetWithAttention(nn.Module):
    """ResNet with additional attention mechanisms."""
    
    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 1,
        variant: str = 'resnet18',
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.base = ISARResNet(
            num_classes=num_classes,
            in_channels=in_channels,
            variant=variant,
            pretrained=pretrained,
            dropout=dropout
        )
        
        # Add CBAM-style attention
        self.channel_attention = ChannelAttention(self.base.feature_dim)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features before final pooling
        x = self.base.base_model.conv1(x)
        x = self.base.base_model.bn1(x)
        x = self.base.base_model.relu(x)
        x = self.base.base_model.maxpool(x)
        
        x = self.base.base_model.layer1(x)
        x = self.base.base_model.layer2(x)
        x = self.base.base_model.layer3(x)
        x = self.base.base_model.layer4(x)
        
        # Apply attention
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        
        x = self.base.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.base_model.fc(x)
        
        return x


class ChannelAttention(nn.Module):
    """Channel attention module."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))
