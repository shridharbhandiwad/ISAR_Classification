#!/usr/bin/env python3
"""
ISAR Image Analysis - Inference Script
=======================================

Command-line script for making predictions on ISAR images.

Usage:
    python scripts/inference.py --checkpoint checkpoints/best_model.pt --image image.png
    python scripts/inference.py --checkpoint checkpoints/best_model.pt --input_dir images/
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from src.utils.helpers import get_device
from src.models import create_model, get_available_models
from src.evaluation import GradCAMExplainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference on ISAR images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    
    # Input arguments
    parser.add_argument('--image', type=str, default=None, help='Single image path')
    parser.add_argument('--input_dir', type=str, default=None, help='Directory of images')
    
    # Model arguments
    parser.add_argument('--architecture', type=str, default='resnet18',
                       choices=get_available_models(), help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--class_names', type=str, nargs='+', default=None, help='Class names')
    
    # Processing arguments
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--top_k', type=int, default=3, help='Top-k predictions to show')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--gradcam', action='store_true', help='Generate Grad-CAM visualizations')
    
    # Misc
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    return parser.parse_args()


def load_and_preprocess_image(image_path: str, image_size: int) -> tuple:
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('L')
    
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    
    tensor = transform(image).unsqueeze(0)
    return image, tensor


def main():
    """Main inference function."""
    args = parse_args()
    
    # Validate inputs
    if args.image is None and args.input_dir is None:
        print("Error: Either --image or --input_dir must be specified")
        sys.exit(1)
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    print(f"Creating model: {args.architecture}")
    model = create_model(
        architecture=args.architecture,
        num_classes=args.num_classes,
        in_channels=1,
        pretrained=False
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get class names
    if args.class_names:
        class_names = args.class_names
    else:
        # Default class names
        class_names = [f"Class_{i}" for i in range(args.num_classes)]
    
    # Create Grad-CAM explainer if requested
    explainer = None
    if args.gradcam:
        explainer = GradCAMExplainer(model)
    
    # Collect images to process
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.input_dir:
        input_dir = Path(args.input_dir)
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
            image_paths.extend(input_dir.glob(ext))
    
    print(f"\nProcessing {len(image_paths)} image(s)...")
    print("=" * 60)
    
    # Process each image
    results = []
    for img_path in image_paths:
        print(f"\nImage: {img_path}")
        
        # Load and preprocess
        original_image, img_tensor = load_and_preprocess_image(str(img_path), args.image_size)
        img_tensor = img_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, args.top_k)
        
        # Print results
        print(f"Predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            print(f"  {i+1}. {class_name}: {prob.item():.2%}")
        
        # Store results
        result = {
            'image': str(img_path),
            'predictions': [
                {'class': class_names[idx] if idx < len(class_names) else f"Class_{idx}",
                 'probability': prob.item()}
                for prob, idx in zip(top_probs, top_indices)
            ]
        }
        results.append(result)
        
        # Generate Grad-CAM if requested
        if explainer and args.output_dir:
            import cv2
            
            cam, pred_class, conf = explainer.generate_cam(img_tensor)
            
            # Create overlay
            img_np = np.array(original_image)
            overlay = explainer.overlay_cam(img_np, cam)
            
            # Save
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"{Path(img_path).stem}_gradcam.png"
            Image.fromarray(overlay).save(output_path)
            print(f"  Grad-CAM saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    
    # Save results if output directory specified
    if args.output_dir:
        import json
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / 'predictions.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
