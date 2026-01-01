"""Model explainability and interpretability tools."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import cv2


class GradCAMExplainer:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Generates visual explanations for CNN predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None
    ):
        """
        Initialize Grad-CAM explainer.
        
        Args:
            model: CNN model
            target_layer: Target layer for CAM (default: last conv layer)
        """
        self.model = model
        self.model.eval()
        
        # Find target layer if not specified
        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)
        
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        """Find the last convolutional layer in the model."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is None:
            raise ValueError("No convolutional layer found in model")
        
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image: Input image tensor (1, C, H, W)
            target_class: Target class index (default: predicted class)
            normalize: Whether to normalize output
            
        Returns:
            Tuple of (heatmap, predicted_class, confidence)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(image)
        
        # Get prediction
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        # Use predicted class if target not specified
        if target_class is None:
            target_class = predicted_class
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, (weight, activation) in enumerate(zip(weights, activations)):
            cam += weight * activation
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Convert to numpy
        cam = cam.cpu().numpy()
        
        # Resize to input image size
        cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
        
        # Normalize
        if normalize and cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, predicted_class, confidence
    
    def overlay_cam(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay CAM heatmap on image.
        
        Args:
            image: Original image (H, W) or (H, W, C)
            cam: CAM heatmap
            alpha: Overlay transparency
            colormap: OpenCV colormap
            
        Returns:
            Image with CAM overlay
        """
        # Ensure image is 3-channel
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[0] in [1, 3]:
            image = image.transpose(1, 2, 0)
            if image.shape[-1] == 1:
                image = np.concatenate([image] * 3, axis=-1)
        
        # Normalize image to 0-255
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        
        # Apply colormap to CAM
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlaid = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        
        return overlaid


class AttentionVisualizer:
    """Visualize attention maps from Vision Transformers."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention visualizer.
        
        Args:
            model: Vision Transformer model with get_attention_maps method
        """
        self.model = model
        self.model.eval()
    
    def get_attention_maps(
        self,
        image: torch.Tensor
    ) -> List[np.ndarray]:
        """
        Get attention maps from all layers.
        
        Args:
            image: Input image tensor
            
        Returns:
            List of attention maps (one per layer)
        """
        if not hasattr(self.model, 'get_attention_maps'):
            raise ValueError("Model does not support attention visualization")
        
        with torch.no_grad():
            attention_maps = self.model.get_attention_maps(image)
        
        return [attn.cpu().numpy() for attn in attention_maps]
    
    def visualize_cls_attention(
        self,
        image: torch.Tensor,
        layer_idx: int = -1,
        head_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Visualize attention from CLS token to patches.
        
        Args:
            image: Input image tensor
            layer_idx: Layer index to visualize
            head_idx: Head index (None for average)
            
        Returns:
            Attention heatmap
        """
        attention_maps = self.get_attention_maps(image)
        
        # Get specified layer
        attn = attention_maps[layer_idx]  # (B, H, N, N)
        
        # Get CLS attention (first row)
        cls_attn = attn[0, :, 0, 1:]  # (H, num_patches)
        
        # Average over heads or select specific head
        if head_idx is None:
            cls_attn = cls_attn.mean(axis=0)
        else:
            cls_attn = cls_attn[head_idx]
        
        # Reshape to 2D
        num_patches = int(np.sqrt(cls_attn.shape[0]))
        attention_map = cls_attn.reshape(num_patches, num_patches)
        
        # Resize to image size
        attention_map = cv2.resize(
            attention_map,
            (image.shape[3], image.shape[2]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Normalize
        attention_map = (attention_map - attention_map.min()) / (
            attention_map.max() - attention_map.min() + 1e-8
        )
        
        return attention_map


class IntegratedGradients:
    """Integrated Gradients for feature attribution."""
    
    def __init__(self, model: nn.Module):
        """
        Initialize Integrated Gradients.
        
        Args:
            model: Model to explain
        """
        self.model = model
        self.model.eval()
    
    def attribute(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> np.ndarray:
        """
        Compute integrated gradients attribution.
        
        Args:
            image: Input image tensor
            target_class: Target class (default: predicted)
            baseline: Baseline image (default: black)
            steps: Number of integration steps
            
        Returns:
            Attribution map
        """
        if baseline is None:
            baseline = torch.zeros_like(image)
        
        # Get prediction if target not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()
        
        # Compute scaled inputs
        scaled_inputs = [
            baseline + (i / steps) * (image - baseline)
            for i in range(steps + 1)
        ]
        scaled_inputs = torch.cat(scaled_inputs, dim=0)
        scaled_inputs.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(scaled_inputs)
        
        # Backward pass
        target_outputs = outputs[:, target_class].sum()
        target_outputs.backward()
        
        # Get gradients
        gradients = scaled_inputs.grad  # (steps+1, C, H, W)
        
        # Average gradients
        avg_gradients = gradients.mean(dim=0)  # (C, H, W)
        
        # Compute integrated gradients
        integrated_gradients = (image - baseline).squeeze(0) * avg_gradients
        
        # Sum over channels
        attribution = integrated_gradients.sum(dim=0).abs()
        
        # Normalize
        attribution = attribution.cpu().numpy()
        attribution = (attribution - attribution.min()) / (
            attribution.max() - attribution.min() + 1e-8
        )
        
        return attribution
