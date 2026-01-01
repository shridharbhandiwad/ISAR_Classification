"""Loss functions for ISAR classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights (optional)
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing.
    
    Smoothed targets: y_smooth = (1 - epsilon) * y + epsilon / K
    """
    
    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Initialize label smoothing loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing parameter (epsilon)
            reduction: Reduction method
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
            
        Returns:
            Label smoothing loss value
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -smooth_targets * log_probs
        
        if self.reduction == 'mean':
            return loss.sum(dim=-1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss.sum(dim=-1)


class ClassBalancedLoss(nn.Module):
    """
    Class-balanced loss based on effective number of samples.
    
    CB(p, y) = (1 - beta^n_y) / (1 - beta) * L(p, y)
    """
    
    def __init__(
        self,
        samples_per_class: list,
        beta: float = 0.9999,
        loss_type: str = 'focal',
        gamma: float = 2.0
    ):
        """
        Initialize class-balanced loss.
        
        Args:
            samples_per_class: Number of samples per class
            beta: Effective number parameter
            loss_type: Base loss type ('cross_entropy' or 'focal')
            gamma: Focal loss gamma
        """
        super().__init__()
        
        # Compute effective numbers and weights
        effective_num = 1.0 - torch.tensor(beta) ** torch.tensor(samples_per_class, dtype=torch.float)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples_per_class)
        
        self.register_buffer('weights', weights)
        self.loss_type = loss_type
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute class-balanced loss."""
        if self.loss_type == 'focal':
            return FocalLoss(
                alpha=self.weights,
                gamma=self.gamma
            )(inputs, targets)
        else:
            return F.cross_entropy(inputs, targets, weight=self.weights)


class ContrastiveLoss(nn.Module):
    """Supervised contrastive loss for representation learning."""
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature scaling parameter
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Feature embeddings (B, D)
            labels: Class labels (B,)
            
        Returns:
            Contrastive loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove self-similarity
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        
        # Compute log-softmax
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        # Loss
        loss = -mean_log_prob_pos.mean()
        
        return loss


def get_loss_function(
    loss_type: str = 'cross_entropy',
    num_classes: int = 5,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
    samples_per_class: Optional[list] = None
) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: Type of loss function
        num_classes: Number of classes
        label_smoothing: Label smoothing factor
        focal_gamma: Focal loss gamma
        class_weights: Optional class weights
        samples_per_class: Samples per class for balanced loss
        
    Returns:
        Loss function module
    """
    if loss_type == 'cross_entropy':
        if label_smoothing > 0:
            return LabelSmoothingLoss(num_classes, label_smoothing)
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(num_classes, label_smoothing or 0.1)
    
    elif loss_type == 'class_balanced':
        if samples_per_class is None:
            raise ValueError("samples_per_class required for class_balanced loss")
        return ClassBalancedLoss(samples_per_class)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
