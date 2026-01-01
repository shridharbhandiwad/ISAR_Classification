"""Image visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict
import torch
import cv2


def visualize_samples(
    images: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    num_samples: int = 16,
    ncols: int = 4,
    figsize: Tuple[int, int] = (12, 12),
    title: str = "Sample Images",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize a grid of sample images.
    
    Args:
        images: Image tensor (N, C, H, W)
        labels: Optional label tensor (N,)
        class_names: List of class names
        num_samples: Number of samples to display
        ncols: Number of columns in grid
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    num_samples = min(num_samples, images.shape[0])
    nrows = (num_samples + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Get image
        img = images[idx].numpy()
        
        # Handle channels
        if img.shape[0] == 1:
            img = img.squeeze(0)
            ax.imshow(img, cmap='gray')
        elif img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
            ax.imshow(img)
        else:
            img = img[0]  # Show first channel
            ax.imshow(img, cmap='gray')
        
        # Add label
        if labels is not None:
            label = labels[idx].item() if isinstance(labels[idx], torch.Tensor) else labels[idx]
            label_name = class_names[label] if class_names else str(label)
            ax.set_title(label_name, fontsize=10)
        
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_predictions(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    probabilities: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    num_samples: int = 16,
    ncols: int = 4,
    figsize: Tuple[int, int] = (14, 14),
    show_errors_only: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize predictions with true labels.
    
    Args:
        images: Image tensor (N, C, H, W)
        true_labels: True label tensor
        pred_labels: Predicted label tensor
        probabilities: Optional probability tensor
        class_names: List of class names
        num_samples: Number of samples to display
        ncols: Number of columns
        figsize: Figure size
        show_errors_only: Only show misclassified samples
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Filter for errors if requested
    if show_errors_only:
        error_mask = true_labels != pred_labels
        indices = torch.where(error_mask)[0]
    else:
        indices = torch.arange(len(images))
    
    num_samples = min(num_samples, len(indices))
    indices = indices[:num_samples]
    
    nrows = (num_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Get image
        img = images[idx].numpy()
        if img.shape[0] == 1:
            img = img.squeeze(0)
            ax.imshow(img, cmap='gray')
        elif img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
            ax.imshow(img)
        else:
            ax.imshow(img[0], cmap='gray')
        
        # Get labels
        true_label = true_labels[idx].item()
        pred_label = pred_labels[idx].item()
        
        true_name = class_names[true_label] if class_names else str(true_label)
        pred_name = class_names[pred_label] if class_names else str(pred_label)
        
        # Create title
        correct = true_label == pred_label
        color = 'green' if correct else 'red'
        
        title_str = f"True: {true_name}\nPred: {pred_name}"
        if probabilities is not None:
            conf = probabilities[idx, pred_label].item()
            title_str += f"\nConf: {conf:.2%}"
        
        ax.set_title(title_str, fontsize=9, color=color)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    title = "Prediction Errors" if show_errors_only else "Predictions"
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_gradcam(
    images: torch.Tensor,
    cams: List[np.ndarray],
    labels: Optional[torch.Tensor] = None,
    predictions: Optional[List[int]] = None,
    class_names: Optional[List[str]] = None,
    ncols: int = 4,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize Grad-CAM heatmaps.
    
    Args:
        images: Image tensor (N, C, H, W)
        cams: List of CAM heatmaps
        labels: Optional true labels
        predictions: Optional predictions
        class_names: List of class names
        ncols: Number of columns
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    num_samples = len(cams)
    nrows = (num_samples + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=figsize)
    axes = axes.reshape(-1, 2) if nrows > 1 or ncols > 1 else [[axes[0], axes[1]]]
    
    for i in range(num_samples):
        row = i // ncols
        col = i % ncols
        
        if nrows == 1 and ncols == 1:
            ax_img, ax_cam = axes[0]
        else:
            ax_idx = row * ncols + col
            if ax_idx >= len(axes):
                break
            ax_img = axes.flatten()[i * 2]
            ax_cam = axes.flatten()[i * 2 + 1]
        
        # Original image
        img = images[i].numpy()
        if img.shape[0] == 1:
            img = img.squeeze(0)
            ax_img.imshow(img, cmap='gray')
            # For overlay
            img_rgb = np.stack([img] * 3, axis=-1)
        elif img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
            ax_img.imshow(img)
            img_rgb = img
        else:
            img = img[0]
            ax_img.imshow(img, cmap='gray')
            img_rgb = np.stack([img] * 3, axis=-1)
        
        # Normalize image
        if img_rgb.max() <= 1:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        else:
            img_rgb = img_rgb.astype(np.uint8)
        
        # CAM overlay
        cam = cams[i]
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = (0.5 * heatmap + 0.5 * img_rgb).astype(np.uint8)
        ax_cam.imshow(overlay)
        
        # Titles
        title = ""
        if labels is not None:
            true_label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]
            true_name = class_names[true_label] if class_names else str(true_label)
            title += f"True: {true_name}"
        
        if predictions is not None:
            pred_name = class_names[predictions[i]] if class_names else str(predictions[i])
            title += f"\nPred: {pred_name}"
        
        ax_img.set_title("Original" + (f"\n{title}" if title else ""), fontsize=9)
        ax_cam.set_title("Grad-CAM", fontsize=9)
        
        ax_img.axis('off')
        ax_cam.axis('off')
    
    plt.suptitle("Grad-CAM Visualization", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_sample_grid(
    images: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True
) -> np.ndarray:
    """
    Create a grid of images.
    
    Args:
        images: Image tensor (N, C, H, W)
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize values
        
    Returns:
        Grid image as numpy array
    """
    from torchvision.utils import make_grid
    
    grid = make_grid(images, nrow=nrow, padding=padding, normalize=normalize)
    grid = grid.numpy().transpose(1, 2, 0)
    
    if grid.shape[-1] == 1:
        grid = grid.squeeze(-1)
    
    return grid


def plot_augmentation_samples(
    original: torch.Tensor,
    augmented: List[torch.Tensor],
    augmentation_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 3),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize original vs augmented images.
    
    Args:
        original: Original image tensor (C, H, W)
        augmented: List of augmented image tensors
        augmentation_names: Names of augmentations
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_aug = len(augmented)
    fig, axes = plt.subplots(1, n_aug + 1, figsize=figsize)
    
    # Original
    img = original.numpy()
    if img.shape[0] == 1:
        axes[0].imshow(img.squeeze(0), cmap='gray')
    else:
        axes[0].imshow(img.transpose(1, 2, 0))
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis('off')
    
    # Augmented
    for i, aug_img in enumerate(augmented):
        img = aug_img.numpy()
        if img.shape[0] == 1:
            axes[i + 1].imshow(img.squeeze(0), cmap='gray')
        else:
            axes[i + 1].imshow(img.transpose(1, 2, 0))
        
        title = augmentation_names[i] if augmentation_names else f"Aug {i + 1}"
        axes[i + 1].set_title(title, fontsize=10)
        axes[i + 1].axis('off')
    
    plt.suptitle("Augmentation Examples", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
