"""Model evaluation utilities."""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from datetime import datetime

from ..training.metrics import MetricsTracker
from ..utils.helpers import get_device


class Evaluator:
    """
    Comprehensive model evaluator for ISAR classifiers.
    
    Provides:
    - Test set evaluation
    - Per-class metrics
    - Confusion matrix analysis
    - Error analysis
    - Inference timing
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to use
            class_names: List of class names
        """
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = class_names
    
    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            verbose: Whether to show progress
            
        Returns:
            Dictionary of evaluation results
        """
        metrics_tracker = MetricsTracker(class_names=self.class_names)
        
        all_targets = []
        all_predictions = []
        all_probabilities = []
        all_features = []
        inference_times = []
        
        iterator = tqdm(test_loader, desc="Evaluating") if verbose else test_loader
        
        for images, targets in iterator:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Measure inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if start_time:
                start_time.record()
            
            # Forward pass
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                inference_times.append(start_time.elapsed_time(end_time))
            
            # Store results
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update metrics
            metrics_tracker.update(
                targets.cpu(),
                predictions.cpu(),
                probabilities.cpu()
            )
            
            # Extract features if model supports it
            if hasattr(self.model, 'get_features'):
                features = self.model.get_features(images)
                all_features.extend(features.cpu().numpy())
        
        # Compile results
        results = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(all_targets),
            'accuracy': metrics_tracker.get_accuracy(),
            'metrics': metrics_tracker.compute_metrics(),
            'per_class_metrics': metrics_tracker.get_per_class_metrics(),
            'confusion_matrix': metrics_tracker.get_confusion_matrix().tolist(),
            'classification_report': metrics_tracker.get_classification_report(),
            'roc_auc': metrics_tracker.get_roc_auc(),
        }
        
        # Add inference timing if available
        if inference_times:
            results['inference'] = {
                'mean_time_ms': np.mean(inference_times),
                'std_time_ms': np.std(inference_times),
                'total_time_ms': np.sum(inference_times)
            }
        
        # Store raw results for further analysis
        results['raw'] = {
            'targets': all_targets,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
        }
        
        if all_features:
            results['features'] = np.array(all_features)
        
        return results
    
    def analyze_errors(
        self,
        results: Dict[str, Any],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze model errors.
        
        Args:
            results: Evaluation results
            top_k: Number of top errors to return
            
        Returns:
            Error analysis dictionary
        """
        targets = np.array(results['raw']['targets'])
        predictions = np.array(results['raw']['predictions'])
        probabilities = np.array(results['raw']['probabilities'])
        
        # Find misclassifications
        errors_mask = targets != predictions
        error_indices = np.where(errors_mask)[0]
        
        # Analyze error patterns
        error_analysis = {
            'total_errors': int(errors_mask.sum()),
            'error_rate': float(errors_mask.mean()),
            'errors_by_class': {},
            'confusion_pairs': [],
            'high_confidence_errors': []
        }
        
        # Errors by class
        for i, class_name in enumerate(self.class_names or range(len(np.unique(targets)))):
            class_mask = targets == i
            class_errors = (errors_mask & class_mask).sum()
            error_analysis['errors_by_class'][str(class_name)] = {
                'count': int(class_errors),
                'rate': float(class_errors / class_mask.sum()) if class_mask.sum() > 0 else 0
            }
        
        # Most common confusion pairs
        confusion_pairs = {}
        for idx in error_indices:
            true_label = int(targets[idx])
            pred_label = int(predictions[idx])
            pair = (true_label, pred_label)
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        for (true_label, pred_label), count in sorted_pairs[:top_k]:
            true_name = self.class_names[true_label] if self.class_names else str(true_label)
            pred_name = self.class_names[pred_label] if self.class_names else str(pred_label)
            error_analysis['confusion_pairs'].append({
                'true_class': true_name,
                'predicted_class': pred_name,
                'count': count
            })
        
        # High confidence errors
        for idx in error_indices:
            confidence = probabilities[idx, predictions[idx]]
            if confidence > 0.8:  # High confidence threshold
                error_analysis['high_confidence_errors'].append({
                    'index': int(idx),
                    'true_class': self.class_names[targets[idx]] if self.class_names else int(targets[idx]),
                    'predicted_class': self.class_names[predictions[idx]] if self.class_names else int(predictions[idx]),
                    'confidence': float(confidence)
                })
        
        # Sort by confidence
        error_analysis['high_confidence_errors'] = sorted(
            error_analysis['high_confidence_errors'],
            key=lambda x: x['confidence'],
            reverse=True
        )[:top_k]
        
        return error_analysis
    
    def get_predictions_with_confidence(
        self,
        images: torch.Tensor,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get predictions with confidence scores.
        
        Args:
            images: Input images
            top_k: Number of top predictions to return
            
        Returns:
            List of prediction dictionaries
        """
        self.model.eval()
        images = images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(images)
            probabilities = torch.softmax(outputs, dim=1)
        
        results = []
        for i in range(images.shape[0]):
            probs = probabilities[i].cpu().numpy()
            top_indices = np.argsort(probs)[-top_k:][::-1]
            
            predictions = []
            for idx in top_indices:
                class_name = self.class_names[idx] if self.class_names else str(idx)
                predictions.append({
                    'class': class_name,
                    'class_id': int(idx),
                    'confidence': float(probs[idx])
                })
            
            results.append({
                'predictions': predictions,
                'top_class': predictions[0]['class'],
                'top_confidence': predictions[0]['confidence']
            })
        
        return results
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_path: str,
        include_raw: bool = False
    ):
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
            output_path: Path to save results
            include_raw: Whether to include raw predictions
        """
        # Create a serializable copy
        save_results = {k: v for k, v in results.items() if k != 'raw' and k != 'features'}
        
        if include_raw:
            save_results['raw'] = {
                'targets': [int(x) for x in results['raw']['targets']],
                'predictions': [int(x) for x in results['raw']['predictions']],
            }
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(save_results, f, indent=2)


def compare_models(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models on the same test set.
    
    Args:
        models: Dictionary of model name -> model
        test_loader: Test data loader
        class_names: List of class names
        device: Device to use
        
    Returns:
        Dictionary of model name -> evaluation results
    """
    comparison = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        evaluator = Evaluator(model, device, class_names)
        results = evaluator.evaluate(test_loader)
        comparison[name] = {
            'accuracy': results['accuracy'],
            'precision': results['metrics']['precision'],
            'recall': results['metrics']['recall'],
            'f1': results['metrics']['f1'],
            'inference_time': results.get('inference', {}).get('mean_time_ms', None)
        }
    
    return comparison
