"""Synthetic ISAR data generator for testing and demonstration."""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from PIL import Image
import json


# Vehicle class definitions for synthetic ISAR data
VEHICLE_CLASSES = {
    'sedan': {
        'length_range': (4.0, 5.0),  # meters
        'width_range': (1.7, 1.9),
        'height_range': (1.4, 1.6),
        'scatterers': 15
    },
    'suv': {
        'length_range': (4.5, 5.5),
        'width_range': (1.8, 2.0),
        'height_range': (1.6, 1.9),
        'scatterers': 18
    },
    'truck': {
        'length_range': (5.0, 8.0),
        'width_range': (2.0, 2.5),
        'height_range': (2.5, 4.0),
        'scatterers': 25
    },
    'motorcycle': {
        'length_range': (2.0, 2.5),
        'width_range': (0.6, 0.9),
        'height_range': (1.0, 1.3),
        'scatterers': 8
    },
    'bus': {
        'length_range': (10.0, 14.0),
        'width_range': (2.4, 2.6),
        'height_range': (3.0, 3.5),
        'scatterers': 35
    }
}


def generate_synthetic_isar_data(
    output_dir: str,
    num_samples_per_class: int = 200,
    image_size: int = 128,
    noise_level: float = 0.1,
    seed: int = 42
) -> Tuple[str, dict]:
    """
    Generate synthetic ISAR-like images for automotive targets.
    
    This function creates realistic synthetic ISAR images by simulating
    radar returns from vehicle scattering points.
    
    Args:
        output_dir: Directory to save generated data
        num_samples_per_class: Number of samples per vehicle class
        image_size: Size of generated images
        noise_level: Level of speckle noise to add
        seed: Random seed
        
    Returns:
        Tuple of (output directory path, dataset statistics)
    """
    np.random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'total_samples': 0,
        'samples_per_class': {},
        'image_size': image_size,
        'classes': list(VEHICLE_CLASSES.keys())
    }
    
    for class_name, class_params in VEHICLE_CLASSES.items():
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        print(f"Generating {num_samples_per_class} samples for class: {class_name}")
        
        for i in range(num_samples_per_class):
            # Generate ISAR image
            isar_image = _generate_isar_image(
                class_params,
                image_size,
                noise_level
            )
            
            # Save image
            img_path = class_dir / f"{class_name}_{i:04d}.png"
            save_image(isar_image, str(img_path))
            
            stats['total_samples'] += 1
        
        stats['samples_per_class'][class_name] = num_samples_per_class
    
    # Save dataset metadata
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset generated successfully!")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Output directory: {output_path}")
    
    return str(output_path), stats


def _generate_isar_image(
    class_params: dict,
    image_size: int,
    noise_level: float
) -> np.ndarray:
    """
    Generate a single synthetic ISAR image.
    
    Args:
        class_params: Vehicle class parameters
        image_size: Output image size
        noise_level: Noise level
        
    Returns:
        Generated ISAR image
    """
    # Initialize image
    image = np.zeros((image_size, image_size), dtype=np.float32)
    
    # Get vehicle dimensions (with random variation)
    length = np.random.uniform(*class_params['length_range'])
    width = np.random.uniform(*class_params['width_range'])
    height = np.random.uniform(*class_params['height_range'])
    num_scatterers = class_params['scatterers']
    
    # Scale factor (pixels per meter)
    scale = image_size / max(length, width) * 0.7
    
    # Center offset
    cx, cy = image_size // 2, image_size // 2
    
    # Random rotation angle (target aspect angle)
    rotation = np.random.uniform(0, 360)
    rot_rad = np.radians(rotation)
    
    # Generate scattering points
    scatterers = _generate_scatterers(
        length, width, height, num_scatterers
    )
    
    # Render scatterers to image
    for sx, sy, sz, amplitude in scatterers:
        # Rotate
        rx = sx * np.cos(rot_rad) - sy * np.sin(rot_rad)
        ry = sx * np.sin(rot_rad) + sy * np.cos(rot_rad)
        
        # Scale and translate to image coordinates
        px = int(cx + rx * scale)
        py = int(cy + ry * scale)
        
        # Check bounds
        if 0 <= px < image_size and 0 <= py < image_size:
            # Create point spread function (PSF)
            _add_psf(image, px, py, amplitude, image_size)
    
    # Add radar-specific effects
    image = _add_radar_effects(image, noise_level)
    
    # Normalize to 0-255
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = (image * 255).astype(np.uint8)
    
    return image


def _generate_scatterers(
    length: float,
    width: float,
    height: float,
    num_scatterers: int
) -> List[Tuple[float, float, float, float]]:
    """
    Generate scattering points for a vehicle.
    
    Args:
        length: Vehicle length
        width: Vehicle width
        height: Vehicle height
        num_scatterers: Number of scattering points
        
    Returns:
        List of (x, y, z, amplitude) tuples
    """
    scatterers = []
    
    # Body corners (strong reflectors)
    corners = [
        (-length/2, -width/2, 0, 1.0),
        (-length/2, width/2, 0, 1.0),
        (length/2, -width/2, 0, 1.0),
        (length/2, width/2, 0, 1.0),
    ]
    scatterers.extend(corners)
    
    # Wheel wells (moderate reflectors)
    wheel_positions = [
        (-length/2 + length*0.2, -width/2, 0, 0.7),
        (-length/2 + length*0.2, width/2, 0, 0.7),
        (length/2 - length*0.2, -width/2, 0, 0.7),
        (length/2 - length*0.2, width/2, 0, 0.7),
    ]
    scatterers.extend(wheel_positions)
    
    # Random body scatterers
    remaining = num_scatterers - len(scatterers)
    for _ in range(remaining):
        x = np.random.uniform(-length/2, length/2)
        y = np.random.uniform(-width/2, width/2)
        z = np.random.uniform(0, height)
        amplitude = np.random.uniform(0.3, 0.8)
        scatterers.append((x, y, z, amplitude))
    
    return scatterers


def _add_psf(
    image: np.ndarray,
    px: int,
    py: int,
    amplitude: float,
    image_size: int
) -> None:
    """
    Add point spread function at a location.
    
    Args:
        image: Target image
        px: X coordinate
        py: Y coordinate
        amplitude: Point amplitude
        image_size: Image size
    """
    # PSF parameters
    sigma = 2.0
    size = 7
    
    for dx in range(-size//2, size//2 + 1):
        for dy in range(-size//2, size//2 + 1):
            nx, ny = px + dx, py + dy
            if 0 <= nx < image_size and 0 <= ny < image_size:
                # Gaussian PSF
                value = amplitude * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                image[ny, nx] += value


def _add_radar_effects(image: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Add radar-specific effects to image.
    
    Args:
        image: Input image
        noise_level: Noise level
        
    Returns:
        Image with radar effects
    """
    # Add speckle noise (multiplicative)
    speckle = np.random.rayleigh(1.0, image.shape)
    image = image * (1 + noise_level * (speckle - 1))
    
    # Add thermal noise (additive)
    thermal = np.random.normal(0, noise_level * 0.1, image.shape)
    image = image + thermal
    
    # Apply mild blur to simulate finite resolution
    from scipy.ndimage import gaussian_filter
    image = gaussian_filter(image, sigma=0.5)
    
    return image


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save image to file.
    
    Args:
        image: Image array
        path: Output path
    """
    img = Image.fromarray(image)
    img.save(path)


def load_ieee_dataset(
    dataset_path: str,
    output_dir: Optional[str] = None
) -> str:
    """
    Load and organize the IEEE DataPort ISAR dataset.
    
    The IEEE dataset contains simulated ISAR images of automotive targets.
    This function organizes the data into the expected directory structure.
    
    Args:
        dataset_path: Path to downloaded dataset
        output_dir: Output directory for organized data
        
    Returns:
        Path to organized dataset
    """
    dataset_path = Path(dataset_path)
    
    if output_dir is None:
        output_dir = dataset_path.parent / 'processed'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Common patterns in IEEE ISAR datasets
    # The actual structure depends on the specific dataset format
    
    print(f"Processing IEEE dataset from: {dataset_path}")
    print(f"Output directory: {output_path}")
    
    # Look for image files
    extensions = ['.png', '.jpg', '.mat', '.npy']
    files_found = []
    
    for ext in extensions:
        files_found.extend(list(dataset_path.rglob(f'*{ext}')))
    
    print(f"Found {len(files_found)} image files")
    
    # Try to infer class structure from filenames or directories
    # This is dataset-specific and may need adjustment
    
    return str(output_path)


if __name__ == "__main__":
    # Generate sample dataset
    generate_synthetic_isar_data(
        output_dir="data/raw",
        num_samples_per_class=100,
        image_size=128
    )
