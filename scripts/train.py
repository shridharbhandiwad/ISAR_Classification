#!/usr/bin/env python3
"""
ISAR Image Analysis - Training Script
======================================

Command-line script for training deep learning models on ISAR images.

Usage:
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --architecture resnet18 --epochs 100 --batch_size 32
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.utils.config import load_config
from src.utils.helpers import set_seed, get_device
from src.utils.logger import setup_logger
from src.models import create_model, get_available_models
from src.data import create_data_loaders, generate_synthetic_isar_data
from src.training import Trainer, TrainingConfig
from src.reports import ReportGenerator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ISAR image classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument(
        '--config', type=str, default='config/config.yaml',
        help='Path to configuration file'
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--image_size', type=int, default=None, help='Image size')
    parser.add_argument('--generate_data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--samples_per_class', type=int, default=200, help='Samples per class for synthetic data')
    
    # Model arguments
    parser.add_argument('--architecture', type=str, default=None, 
                       choices=get_available_models(), help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--no_pretrained', action='store_true', help='Do not use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default=None, 
                       choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default=None,
                       choices=['cosine', 'step', 'plateau', 'none'], help='LR scheduler')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--generate_report', action='store_true', help='Generate training report')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load base configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = {}
    
    # Override config with command line arguments
    if args.data_dir:
        config.setdefault('data', {})['data_dir'] = args.data_dir
    if args.image_size:
        config.setdefault('data', {})['image_size'] = args.image_size
    if args.architecture:
        config.setdefault('model', {})['architecture'] = args.architecture
    if args.num_classes:
        config.setdefault('model', {})['num_classes'] = args.num_classes
    if args.no_pretrained:
        config.setdefault('model', {})['pretrained'] = False
    elif args.pretrained:
        config.setdefault('model', {})['pretrained'] = True
    if args.epochs:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.learning_rate:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
    if args.optimizer:
        config.setdefault('training', {})['optimizer'] = args.optimizer
    if args.scheduler:
        config.setdefault('training', {}).setdefault('scheduler', {})['type'] = args.scheduler
    
    # Set defaults
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    data_dir = data_config.get('data_dir', 'data/raw')
    image_size = data_config.get('image_size', 128)
    architecture = model_config.get('architecture', 'resnet18')
    num_classes = model_config.get('num_classes', 5)
    pretrained = model_config.get('pretrained', True)
    dropout = model_config.get('dropout', 0.3)
    epochs = training_config.get('epochs', 100)
    batch_size = training_config.get('batch_size', 32)
    learning_rate = training_config.get('learning_rate', 0.001)
    
    # Setup
    set_seed(args.seed)
    logger = setup_logger(log_dir='logs')
    device = get_device(args.device)
    
    logger.info("=" * 60)
    logger.info("ISAR Image Analysis - Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Architecture: {architecture}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    
    # Generate synthetic data if requested
    if args.generate_data:
        logger.info("Generating synthetic ISAR data...")
        generate_synthetic_isar_data(
            output_dir=data_dir,
            num_samples_per_class=args.samples_per_class,
            image_size=image_size
        )
    
    # Check data directory
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Use --generate_data to create synthetic data")
        sys.exit(1)
    
    # Create data loaders
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        train_ratio=data_config.get('train_ratio', 0.7),
        val_ratio=data_config.get('val_ratio', 0.15),
        test_ratio=data_config.get('test_ratio', 0.15),
        num_workers=args.workers
    )
    
    logger.info(f"Classes: {class_names}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Update num_classes based on data
    num_classes = len(class_names)
    
    # Create model
    logger.info(f"Creating model: {architecture}")
    model = create_model(
        architecture=architecture,
        num_classes=num_classes,
        in_channels=1,
        pretrained=pretrained,
        dropout=dropout
    )
    
    # Training configuration
    train_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=training_config.get('optimizer', 'adamw'),
        weight_decay=training_config.get('weight_decay', 0.0001),
        scheduler_type=training_config.get('scheduler', {}).get('type', 'cosine'),
        early_stopping_enabled=training_config.get('early_stopping', {}).get('enabled', True),
        early_stopping_patience=training_config.get('early_stopping', {}).get('patience', 15),
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        class_names=class_names
    )
    
    # Train
    logger.info("Starting training...")
    results = trainer.train()
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    logger.info(f"Total epochs: {results['epochs_trained']}")
    logger.info("=" * 60)
    
    # Generate report if requested
    if args.generate_report:
        logger.info("Generating training report...")
        report_gen = ReportGenerator(output_dir='reports')
        report_path = report_gen.generate_training_report(
            training_results=results,
            config=config,
            model_name=architecture,
            format='both'
        )
        logger.info(f"Report saved to: {report_path}")
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
