"""Metrics tracking and computation for training."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)


class MetricsTracker:
    """
    Track and compute metrics during training/evaluation.
    
    Supports:
    - Accuracy
    - Precision, Recall, F1 (per-class and weighted)
    - Confusion matrix
    - ROC-AUC
    - Precision-Recall curves
    """
    
    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        average: str = 'weighted'
    ):
        """
        Initialize metrics tracker.
        
        Args:
            class_names: List of class names
            average: Averaging method for metrics
        """
        self.class_names = class_names
        self.average = average
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and targets."""
        self.all_targets = []
        self.all_predictions = []
        self.all_probabilities = []
    
    def update(
        self,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ):
        """
        Update with batch of predictions.
        
        Args:
            targets: Ground truth labels
            predictions: Predicted labels
            probabilities: Class probabilities (optional)
        """
        self.all_targets.extend(targets.numpy().tolist())
        self.all_predictions.extend(predictions.numpy().tolist())
        
        if probabilities is not None:
            self.all_probabilities.extend(probabilities.numpy().tolist())
    
    def get_accuracy(self) -> float:
        """Get current accuracy."""
        if len(self.all_targets) == 0:
            return 0.0
        return accuracy_score(self.all_targets, self.all_predictions)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metrics
        """
        if len(self.all_targets) == 0:
            return {}
        
        targets = np.array(self.all_targets)
        predictions = np.array(self.all_predictions)
        
        metrics = {
            'precision': precision_score(
                targets, predictions, 
                average=self.average, zero_division=0
            ),
            'recall': recall_score(
                targets, predictions,
                average=self.average, zero_division=0
            ),
            'f1': f1_score(
                targets, predictions,
                average=self.average, zero_division=0
            )
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(self.all_targets, self.all_predictions)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        return classification_report(
            self.all_targets,
            self.all_predictions,
            target_names=self.class_names,
            zero_division=0
        )
    
    def get_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for each class."""
        targets = np.array(self.all_targets)
        predictions = np.array(self.all_predictions)
        
        precision = precision_score(
            targets, predictions,
            average=None, zero_division=0
        )
        recall = recall_score(
            targets, predictions,
            average=None, zero_division=0
        )
        f1 = f1_score(
            targets, predictions,
            average=None, zero_division=0
        )
        
        per_class = {}
        for i, class_name in enumerate(self.class_names or range(len(precision))):
            per_class[str(class_name)] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i])
            }
        
        return per_class
    
    def get_roc_auc(self) -> Dict[str, float]:
        """
        Compute ROC-AUC scores.
        
        Returns:
            Dictionary with ROC-AUC scores
        """
        if len(self.all_probabilities) == 0:
            return {}
        
        targets = np.array(self.all_targets)
        probabilities = np.array(self.all_probabilities)
        
        n_classes = probabilities.shape[1]
        
        # One-hot encode targets
        targets_onehot = np.zeros((len(targets), n_classes))
        targets_onehot[np.arange(len(targets)), targets] = 1
        
        try:
            # Micro and macro average
            roc_auc_micro = roc_auc_score(
                targets_onehot.ravel(),
                probabilities.ravel(),
                average='micro'
            )
            roc_auc_macro = roc_auc_score(
                targets_onehot,
                probabilities,
                average='macro',
                multi_class='ovr'
            )
            
            return {
                'roc_auc_micro': roc_auc_micro,
                'roc_auc_macro': roc_auc_macro
            }
        except ValueError:
            return {}
    
    def get_roc_curves(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get ROC curves for each class.
        
        Returns:
            Dictionary mapping class names to (fpr, tpr) tuples
        """
        if len(self.all_probabilities) == 0:
            return {}
        
        targets = np.array(self.all_targets)
        probabilities = np.array(self.all_probabilities)
        
        n_classes = probabilities.shape[1]
        
        roc_curves = {}
        for i in range(n_classes):
            binary_targets = (targets == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_targets, probabilities[:, i])
            
            class_name = self.class_names[i] if self.class_names else str(i)
            roc_curves[class_name] = (fpr, tpr)
        
        return roc_curves
    
    def get_pr_curves(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get Precision-Recall curves for each class.
        
        Returns:
            Dictionary mapping class names to (precision, recall) tuples
        """
        if len(self.all_probabilities) == 0:
            return {}
        
        targets = np.array(self.all_targets)
        probabilities = np.array(self.all_probabilities)
        
        n_classes = probabilities.shape[1]
        
        pr_curves = {}
        for i in range(n_classes):
            binary_targets = (targets == i).astype(int)
            precision, recall, _ = precision_recall_curve(
                binary_targets, probabilities[:, i]
            )
            
            class_name = self.class_names[i] if self.class_names else str(i)
            pr_curves[class_name] = (precision, recall)
        
        return pr_curves
    
    def get_all_metrics(self) -> Dict[str, any]:
        """Get comprehensive metrics dictionary."""
        metrics = {
            'accuracy': self.get_accuracy(),
            **self.compute_metrics(),
            'confusion_matrix': self.get_confusion_matrix().tolist(),
            'per_class': self.get_per_class_metrics(),
            **self.get_roc_auc()
        }
        
        return metrics


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """Update with new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f'{self.name}: {self.val:.4f} (avg: {self.avg:.4f})'


class EMAMeter:
    """Exponential moving average meter."""
    
    def __init__(self, name: str = '', alpha: float = 0.9):
        self.name = name
        self.alpha = alpha
        self.reset()
    
    def reset(self):
        """Reset EMA value."""
        self.val = None
    
    def update(self, val: float):
        """Update EMA with new value."""
        if self.val is None:
            self.val = val
        else:
            self.val = self.alpha * self.val + (1 - self.alpha) * val
    
    def __str__(self) -> str:
        return f'{self.name}: {self.val:.4f}'
