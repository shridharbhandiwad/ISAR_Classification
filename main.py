#!/usr/bin/env python3
"""
ISAR Image Analysis - Main Entry Point
=======================================

This is the main entry point for the ISAR Image Analysis platform.
Use this script to quickly start different components of the system.

Usage:
    python main.py gui          # Launch GUI application
    python main.py train        # Train model
    python main.py evaluate     # Evaluate model
    python main.py inference    # Run inference
    python main.py generate     # Generate synthetic data
"""

import argparse
import sys
import os


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ISAR Image Analysis Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  gui        Launch the Streamlit GUI application
  train      Train a model (use --help for options)
  evaluate   Evaluate a trained model
  inference  Run inference on images
  generate   Generate synthetic ISAR data

Examples:
  python main.py gui
  python main.py train --architecture resnet18 --epochs 100
  python main.py evaluate --checkpoint checkpoints/best_model.pt
  python main.py generate --samples_per_class 200
        """
    )
    
    parser.add_argument(
        'command',
        choices=['gui', 'train', 'evaluate', 'inference', 'generate'],
        help='Command to run'
    )
    
    # Parse only the command first
    args, remaining = parser.parse_known_args()
    
    if args.command == 'gui':
        # Launch Streamlit GUI
        import subprocess
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app/app.py'] + remaining)
    
    elif args.command == 'train':
        # Run training script
        import subprocess
        subprocess.run([sys.executable, 'scripts/train.py'] + remaining)
    
    elif args.command == 'evaluate':
        # Run evaluation script
        import subprocess
        subprocess.run([sys.executable, 'scripts/evaluate.py'] + remaining)
    
    elif args.command == 'inference':
        # Run inference script
        import subprocess
        subprocess.run([sys.executable, 'scripts/inference.py'] + remaining)
    
    elif args.command == 'generate':
        # Generate synthetic data
        from src.data import generate_synthetic_isar_data
        
        gen_parser = argparse.ArgumentParser()
        gen_parser.add_argument('--output_dir', default='data/raw')
        gen_parser.add_argument('--samples_per_class', type=int, default=200)
        gen_parser.add_argument('--image_size', type=int, default=128)
        gen_parser.add_argument('--noise_level', type=float, default=0.1)
        
        gen_args = gen_parser.parse_args(remaining)
        
        print("Generating synthetic ISAR data...")
        output_path, stats = generate_synthetic_isar_data(
            output_dir=gen_args.output_dir,
            num_samples_per_class=gen_args.samples_per_class,
            image_size=gen_args.image_size,
            noise_level=gen_args.noise_level
        )
        
        print(f"\nGenerated {stats['total_samples']} samples")
        print(f"Classes: {stats['classes']}")
        print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
