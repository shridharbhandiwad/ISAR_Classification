"""Learning rate schedulers for training."""

import math
import torch
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR,
    CosineAnnealingLR, CosineAnnealingWarmRestarts,
    ReduceLROnPlateau, OneCycleLR, LambdaLR
)
from typing import Optional


class WarmupScheduler:
    """Learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: torch.optim.lr_scheduler._LRScheduler,
        warmup_factor: float = 0.1
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            base_scheduler: Base scheduler to use after warmup
            warmup_factor: Initial LR factor during warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_factor = warmup_factor
        self.current_epoch = 0
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch: Optional[int] = None):
        """Update learning rate."""
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.current_epoch / self.warmup_epochs
            factor = self.warmup_factor + (1.0 - self.warmup_factor) * alpha
            
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * factor
        else:
            self.base_scheduler.step()
        
        self.current_epoch += 1
    
    def state_dict(self):
        """Return scheduler state."""
        return {
            'current_epoch': self.current_epoch,
            'base_scheduler': self.base_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.current_epoch = state_dict['current_epoch']
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])


class CosineWarmupScheduler(LambdaLR):
    """Cosine annealing with warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr_factor: float = 0.01
    ):
        """
        Initialize cosine warmup scheduler.
        
        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            max_epochs: Maximum number of epochs
            min_lr_factor: Minimum LR as factor of initial LR
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr_factor = min_lr_factor
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return epoch / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
                return min_lr_factor + (1 - min_lr_factor) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )
        
        super().__init__(optimizer, lr_lambda)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int = 100,
    steps_per_epoch: int = 1,
    step_size: int = 30,
    gamma: float = 0.1,
    patience: int = 10,
    warmup_epochs: int = 0,
    min_lr: float = 1e-6
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler
        epochs: Total number of epochs
        steps_per_epoch: Steps per epoch (for OneCycleLR)
        step_size: Step size for StepLR
        gamma: Decay factor
        patience: Patience for ReduceLROnPlateau
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'multistep':
        milestones = [int(epochs * 0.5), int(epochs * 0.75)]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    
    elif scheduler_type == 'cosine_warmup':
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_epochs, epochs, min_lr_factor=min_lr
        )
        return scheduler
    
    elif scheduler_type == 'cosine_warm_restarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=epochs // 4, T_mult=2, eta_min=min_lr
        )
    
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=gamma,
            patience=patience, min_lr=min_lr
        )
        return scheduler  # No warmup for plateau
    
    elif scheduler_type == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch
        )
        return scheduler  # OneCycleLR handles warmup internally
    
    elif scheduler_type == 'none':
        # Constant learning rate
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Wrap with warmup if specified
    if warmup_epochs > 0:
        scheduler = WarmupScheduler(optimizer, warmup_epochs, scheduler)
    
    return scheduler
