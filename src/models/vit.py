"""Vision Transformer (ViT) model for ISAR image classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""
    
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 768
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, embed_dim, num_patches_h, num_patches_w)
        x = self.projection(x)
        # Flatten: (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        qkv_bias: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_weights = attn
        attn = self.attn_dropout(attn)
        
        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x, attn_weights


class MLP(nn.Module):
    """Feed-forward MLP block."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim, num_heads,
            dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class ISARViT(nn.Module):
    """
    Vision Transformer for ISAR image classification.
    
    Adapted for grayscale radar images with configurable architecture.
    """
    
    def __init__(
        self,
        image_size: int = 128,
        patch_size: int = 16,
        in_channels: int = 1,
        num_classes: int = 5,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        classifier_dropout: float = 0.3
    ):
        """
        Initialize ViT classifier.
        
        Args:
            image_size: Input image size
            patch_size: Size of image patches
            in_channels: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            classifier_dropout: Classifier dropout rate
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, mlp_ratio,
                dropout, attention_dropout
            )
            for _ in range(depth)
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x)
        
        # Normalize
        x = self.norm(x)
        
        # Classify using class token
        x = x[:, 0]
        x = self.classifier(x)
        
        return x
    
    def get_attention_maps(self, x: torch.Tensor) -> list:
        """Get attention maps from all layers."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Get attention from each block
        attention_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attention_maps.append(attn)
        
        return attention_maps
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification layer."""
        B = x.shape[0]
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        for block in self.blocks:
            x, _ = block(x)
        
        x = self.norm(x)
        return x[:, 0]


def vit_tiny_isar(num_classes: int = 5, **kwargs) -> ISARViT:
    """Create Tiny ViT model."""
    return ISARViT(
        num_classes=num_classes,
        embed_dim=192,
        depth=4,
        num_heads=3,
        **kwargs
    )


def vit_small_isar(num_classes: int = 5, **kwargs) -> ISARViT:
    """Create Small ViT model."""
    return ISARViT(
        num_classes=num_classes,
        embed_dim=384,
        depth=6,
        num_heads=6,
        **kwargs
    )


def vit_base_isar(num_classes: int = 5, **kwargs) -> ISARViT:
    """Create Base ViT model."""
    return ISARViT(
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )
