"""Model factory for creating ISAR classifiers."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .cnn import ISARClassifierCNN, ISARClassifierLightCNN
from .resnet import ISARResNet, resnet18_isar, resnet34_isar
from .efficientnet import ISAREfficientNet
from .vit import ISARViT, vit_tiny_isar, vit_small_isar, vit_base_isar


# Registry of available models
MODEL_REGISTRY = {
    'cnn': ISARClassifierCNN,
    'cnn_light': ISARClassifierLightCNN,
    'resnet18': lambda **kwargs: resnet18_isar(**kwargs),
    'resnet34': lambda **kwargs: resnet34_isar(**kwargs),
    'resnet50': lambda **kwargs: ISARResNet(variant='resnet50', **kwargs),
    'efficientnet_b0': lambda **kwargs: ISAREfficientNet(variant='b0', **kwargs),
    'efficientnet_b1': lambda **kwargs: ISAREfficientNet(variant='b1', **kwargs),
    'efficientnet_b2': lambda **kwargs: ISAREfficientNet(variant='b2', **kwargs),
    'vit_tiny': vit_tiny_isar,
    'vit_small': vit_small_isar,
    'vit_base': vit_base_isar,
    'vit': lambda **kwargs: ISARViT(**kwargs),
}


def get_available_models() -> list:
    """Get list of available model architectures."""
    return list(MODEL_REGISTRY.keys())


def create_model(
    architecture: str,
    num_classes: int = 5,
    in_channels: int = 1,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Create a model based on architecture name.
    
    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        in_channels: Number of input channels
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model-specific arguments
        
    Returns:
        Instantiated model
    """
    if architecture not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available: {get_available_models()}"
        )
    
    model_fn = MODEL_REGISTRY[architecture]
    
    # Build model kwargs
    model_kwargs = {
        'num_classes': num_classes,
        'in_channels': in_channels,
    }
    
    # Add pretrained only for supported models
    if architecture.startswith('resnet') or architecture.startswith('efficientnet'):
        model_kwargs['pretrained'] = pretrained
    
    # Merge with additional kwargs
    model_kwargs.update(kwargs)
    
    return model_fn(**model_kwargs)


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Create a model from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Instantiated model
    """
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    architecture = model_config.get('architecture', 'resnet18')
    num_classes = model_config.get('num_classes', 5)
    in_channels = data_config.get('channels', 1)
    pretrained = model_config.get('pretrained', True)
    dropout = model_config.get('dropout', 0.3)
    
    return create_model(
        architecture=architecture,
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        dropout=dropout
    )


def load_model(
    checkpoint_path: str,
    architecture: str,
    num_classes: int,
    device: Optional[torch.device] = None,
    **kwargs
) -> nn.Module:
    """
    Load a model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        architecture: Model architecture
        num_classes: Number of classes
        device: Device to load model to
        **kwargs: Additional model arguments
        
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,
        **kwargs
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


class EnsembleModel(nn.Module):
    """Ensemble of multiple models."""
    
    def __init__(
        self,
        models: list,
        weights: Optional[list] = None,
        mode: str = 'average'
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of models
            weights: Optional weights for each model
            mode: Ensemble mode ('average', 'vote', 'weighted')
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.mode = mode
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        outputs = []
        
        for model in self.models:
            out = model(x)
            outputs.append(out)
        
        if self.mode == 'average':
            return torch.stack(outputs).mean(dim=0)
        elif self.mode == 'weighted':
            weighted = sum(w * o for w, o in zip(self.weights, outputs))
            return weighted
        elif self.mode == 'vote':
            # Hard voting
            preds = [o.argmax(dim=1) for o in outputs]
            stacked = torch.stack(preds)
            voted = torch.mode(stacked, dim=0).values
            # Return one-hot-ish output
            return torch.nn.functional.one_hot(
                voted, num_classes=outputs[0].shape[1]
            ).float()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
