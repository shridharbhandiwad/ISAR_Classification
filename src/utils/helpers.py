"""Helper utility functions."""

import os
import random
import numpy as np
import torch
from typing import Tuple, Optional
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Specific device string ('cuda', 'cpu', 'mps')
        
    Returns:
        torch.device object
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_model_size(model: torch.nn.Module) -> Tuple[float, str]:
    """
    Get the size of a model in memory.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (size, unit)
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    if total_size >= 1e9:
        return total_size / 1e9, "GB"
    elif total_size >= 1e6:
        return total_size / 1e6, "MB"
    elif total_size >= 1e3:
        return total_size / 1e3, "KB"
    else:
        return total_size, "B"


def ensure_dir(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: dict,
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Dictionary of metrics
        path: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    ensure_dir(os.path.dirname(path))
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None
) -> dict:
    """
    Load a training checkpoint.
    
    Args:
        model: PyTorch model
        path: Path to checkpoint
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint
