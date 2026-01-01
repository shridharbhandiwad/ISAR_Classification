"""ISAR Dataset implementation."""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Union
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split


class ISARDataset(Dataset):
    """
    Dataset class for ISAR (Inverse Synthetic Aperture Radar) images.
    
    Supports loading images from directory structure where each subdirectory
    represents a class (automotive target type).
    
    Directory structure:
        data_dir/
            class1/
                image1.png
                image2.png
            class2/
                image3.png
                ...
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size: int = 128,
        grayscale: bool = True,
        file_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.mat', '.npy')
    ):
        """
        Initialize the ISAR dataset.
        
        Args:
            data_dir: Root directory containing class subdirectories
            transform: Optional transform to apply to images
            target_transform: Optional transform to apply to labels
            image_size: Target image size
            grayscale: Whether to load images as grayscale
            file_extensions: Supported file extensions
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.grayscale = grayscale
        self.file_extensions = file_extensions
        
        # Load dataset
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load dataset from directory structure."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Get all class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        if not class_dirs:
            # If no subdirectories, treat all files as single class
            self.classes = ['unknown']
            self.class_to_idx = {'unknown': 0}
            self._load_files_from_dir(self.data_dir, 0)
        else:
            # Sort for consistent ordering
            class_dirs = sorted(class_dirs, key=lambda x: x.name)
            
            for idx, class_dir in enumerate(class_dirs):
                class_name = class_dir.name
                self.classes.append(class_name)
                self.class_to_idx[class_name] = idx
                self._load_files_from_dir(class_dir, idx)
        
        if len(self.samples) == 0:
            print(f"Warning: No samples found in {self.data_dir}")
    
    def _load_files_from_dir(self, directory: Path, class_idx: int) -> None:
        """Load all valid files from a directory."""
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.file_extensions:
                self.samples.append((str(file_path), class_idx))
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image tensor, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image based on file type
        image = self._load_image(img_path)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label
    
    def _load_image(self, path: str) -> Union[np.ndarray, torch.Tensor]:
        """
        Load an image from file.
        
        Args:
            path: Path to image file
            
        Returns:
            Image as numpy array or tensor
        """
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix == '.npy':
            # NumPy array
            image = np.load(path)
        elif suffix == '.mat':
            # MATLAB file
            try:
                from scipy.io import loadmat
                mat_data = loadmat(path)
                # Try common variable names
                for key in ['image', 'data', 'isar', 'img']:
                    if key in mat_data:
                        image = mat_data[key]
                        break
                else:
                    # Use first non-metadata key
                    for key, value in mat_data.items():
                        if not key.startswith('__'):
                            image = value
                            break
            except Exception as e:
                raise RuntimeError(f"Failed to load MATLAB file: {path}") from e
        else:
            # Standard image formats
            if self.grayscale:
                image = Image.open(path).convert('L')
            else:
                image = Image.open(path).convert('RGB')
            image = np.array(image)
        
        # Ensure proper shape
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
        
        # Resize if necessary
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image.squeeze())
            pil_img = pil_img.resize((self.image_size, self.image_size), PILImage.Resampling.BILINEAR)
            image = np.array(pil_img)
            if len(image.shape) == 2:
                image = image[..., np.newaxis]
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image.astype(np.float32)
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Convert to tensor format (C, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of samples across classes."""
        distribution = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            distribution[self.classes[label]] += 1
        return distribution
    
    def get_sample_weights(self) -> torch.Tensor:
        """Calculate sample weights for balanced sampling."""
        class_counts = [0] * len(self.classes)
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Calculate weights (inverse of frequency)
        weights = []
        for _, label in self.samples:
            weights.append(1.0 / class_counts[label])
        
        return torch.tensor(weights, dtype=torch.float32)


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 128,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    augmentation: Optional[Callable] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        image_size: Target image size
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        num_workers: Number of worker processes
        transform: Base transform to apply
        augmentation: Augmentation transform for training
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    # Create base dataset
    full_dataset = ISARDataset(
        data_dir=data_dir,
        transform=transform,
        image_size=image_size
    )
    
    # Get class names
    class_names = full_dataset.classes
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_names
