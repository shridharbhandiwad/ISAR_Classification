#!/usr/bin/env python3
"""
ISAR Image Analysis - Evaluation Script
========================================

Command-line script for evaluating trained models on ISAR images.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --data_dir data/raw
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.utils.config import load_config
from src.utils.helpers import set_seed, get_device
from src.utils.logger import setup_logger
from src.models import create_model, get_available_models
from src.data import create_data_loaders
from src.evaluation import Evaluator
from src.visualization import (
    plot_confusion_matrix, plot_roc_curves,
    plot_precision_recall_curves, plot_feature_embeddings
)
from src.reports import ReportGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate ISAR image classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_dir', type=str, default='data/raw',
        help='Data directory for evaluation'
    )
    
    # Model arguments
    parser.add_argument('--architecture', type=str, default='resnet18',
                       choices=get_available_models(), help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--save_plots', action='store_true', help='Save visualization plots')
    parser.add_argument('--generate_report', action='store_true', help='Generate evaluation report')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup
    set_seed(args.seed)
    logger = setup_logger(log_dir='logs')
    device = get_device(args.device)
    
    logger.info("=" * 60)
    logger.info("ISAR Image Analysis - Evaluation")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Get configuration from checkpoint if available
    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']
        # Use checkpoint config for model if not overridden
        if 'architecture' not in ckpt_config:
            architecture = args.architecture
        else:
            architecture = args.architecture
    else:
        architecture = args.architecture
    
    # Create data loaders
    logger.info("Loading data...")
    _, _, test_loader, class_names = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    num_classes = len(class_names)
    logger.info(f"Classes: {class_names}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info(f"Creating model: {architecture}")
    model = create_model(
        architecture=architecture,
        num_classes=num_classes,
        in_channels=1,
        pretrained=False
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create evaluator
    evaluator = Evaluator(model=model, device=device, class_names=class_names)
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate(test_loader, verbose=True)
    
    # Print results
    logger.info("=" * 60)
    logger.info("Evaluation Results:")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Precision: {results['metrics']['precision']:.4f}")
    logger.info(f"Recall: {results['metrics']['recall']:.4f}")
    logger.info(f"F1 Score: {results['metrics']['f1']:.4f}")
    
    if 'inference' in results:
        logger.info(f"Mean Inference Time: {results['inference']['mean_time_ms']:.2f} ms")
    
    # Per-class results
    logger.info("\nPer-Class Performance:")
    for class_name, metrics in results['per_class_metrics'].items():
        logger.info(
            f"  {class_name}: P={metrics['precision']:.3f}, "
            f"R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
        )
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info(results['classification_report'])
    
    # Error analysis
    logger.info("\nError Analysis:")
    error_analysis = evaluator.analyze_errors(results)
    logger.info(f"Total errors: {error_analysis['total_errors']}")
    logger.info(f"Error rate: {error_analysis['error_rate']:.2%}")
    
    if error_analysis['confusion_pairs']:
        logger.info("\nMost common confusion pairs:")
        for pair in error_analysis['confusion_pairs'][:5]:
            logger.info(f"  {pair['true_class']} -> {pair['predicted_class']}: {pair['count']}")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    evaluator.save_results(results, results_path, include_raw=False)
    logger.info(f"\nResults saved to: {results_path}")
    
    # Generate plots
    if args.save_plots:
        logger.info("\nGenerating visualizations...")
        
        # Confusion matrix
        cm = np.array(results['confusion_matrix'])
        fig = plot_confusion_matrix(cm, class_names)
        fig.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=150)
        logger.info("Saved confusion_matrix.png")
        
        # Feature embeddings
        if 'features' in results:
            targets = np.array(results['raw']['targets'])
            fig = plot_feature_embeddings(
                results['features'], targets, class_names
            )
            fig.savefig(os.path.join(args.output_dir, 'feature_embeddings.png'), dpi=150)
            logger.info("Saved feature_embeddings.png")
    
    # Generate report
    if args.generate_report:
        logger.info("\nGenerating evaluation report...")
        report_gen = ReportGenerator(output_dir='reports')
        report_path = report_gen.generate_evaluation_report(
            eval_results=results,
            model_name=architecture,
            class_names=class_names,
            format='both'
        )
        logger.info(f"Report saved to: {report_path}")
    
    logger.info("\nDone!")


if __name__ == '__main__':
    main()
