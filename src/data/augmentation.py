"""Data augmentation for ISAR images."""

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Optional, Tuple, List, Callable
import random


class ISARAugmentation:
    """
    Specialized augmentation pipeline for ISAR images.
    
    Includes radar-specific augmentations that preserve physical characteristics.
    """
    
    def __init__(
        self,
        rotation_range: float = 15.0,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        noise_factor: float = 0.05,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        translate_range: Tuple[float, float] = (0.1, 0.1),
        enable_mixup: bool = False,
        mixup_alpha: float = 0.2
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            horizontal_flip: Whether to apply horizontal flip
            vertical_flip: Whether to apply vertical flip
            brightness_range: Range for brightness adjustment
            noise_factor: Standard deviation of Gaussian noise
            scale_range: Range for scaling
            translate_range: Range for translation (fraction of image size)
            enable_mixup: Whether to enable mixup augmentation
            mixup_alpha: Alpha parameter for mixup
        """
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness_range = brightness_range
        self.noise_factor = noise_factor
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.enable_mixup = enable_mixup
        self.mixup_alpha = mixup_alpha
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to an image.
        
        Args:
            image: Input image tensor (C, H, W)
            
        Returns:
            Augmented image tensor
        """
        # Random rotation
        if self.rotation_range > 0 and random.random() > 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = TF.rotate(image, angle)
        
        # Horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            image = TF.hflip(image)
        
        # Vertical flip
        if self.vertical_flip and random.random() > 0.5:
            image = TF.vflip(image)
        
        # Brightness adjustment
        if self.brightness_range[0] != 1.0 or self.brightness_range[1] != 1.0:
            if random.random() > 0.5:
                factor = random.uniform(*self.brightness_range)
                image = image * factor
                image = torch.clamp(image, 0, 1)
        
        # Add Gaussian noise
        if self.noise_factor > 0 and random.random() > 0.5:
            noise = torch.randn_like(image) * self.noise_factor
            image = image + noise
            image = torch.clamp(image, 0, 1)
        
        # Random scaling
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            if random.random() > 0.5:
                scale = random.uniform(*self.scale_range)
                _, h, w = image.shape
                new_h, new_w = int(h * scale), int(w * scale)
                image = TF.resize(image, [new_h, new_w])
                # Crop or pad to original size
                if scale > 1:
                    image = TF.center_crop(image, [h, w])
                else:
                    padding = [(w - new_w) // 2, (h - new_h) // 2]
                    image = TF.pad(image, padding)
                    image = TF.center_crop(image, [h, w])
        
        return image
    
    def apply_speckle_noise(self, image: torch.Tensor, intensity: float = 0.1) -> torch.Tensor:
        """
        Apply speckle noise (multiplicative noise typical in radar).
        
        Args:
            image: Input image tensor
            intensity: Noise intensity
            
        Returns:
            Image with speckle noise
        """
        noise = 1 + torch.randn_like(image) * intensity
        return torch.clamp(image * noise, 0, 1)
    
    def apply_phase_shift(self, image: torch.Tensor, max_shift: float = 0.2) -> torch.Tensor:
        """
        Simulate phase shift effect in radar images.
        
        Args:
            image: Input image tensor
            max_shift: Maximum shift as fraction of image size
            
        Returns:
            Phase-shifted image
        """
        _, h, w = image.shape
        shift_h = int(random.uniform(-max_shift, max_shift) * h)
        shift_w = int(random.uniform(-max_shift, max_shift) * w)
        return torch.roll(image, shifts=(shift_h, shift_w), dims=(1, 2))


def get_augmentation_transforms(
    config: dict,
    is_training: bool = True
) -> Callable:
    """
    Get augmentation transforms based on configuration.
    
    Args:
        config: Configuration dictionary
        is_training: Whether transforms are for training
        
    Returns:
        Transform function
    """
    aug_config = config.get('data', {}).get('augmentation', {})
    image_size = config.get('data', {}).get('image_size', 128)
    
    if is_training and aug_config.get('enabled', True):
        return T.Compose([
            T.RandomRotation(aug_config.get('rotation_range', 15)),
            T.RandomHorizontalFlip(p=0.5 if aug_config.get('horizontal_flip', True) else 0),
            T.RandomVerticalFlip(p=0.5 if aug_config.get('vertical_flip', False) else 0),
            T.ColorJitter(brightness=0.2),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            T.Resize([image_size, image_size]),
        ])
    else:
        return T.Compose([
            T.Resize([image_size, image_size]),
        ])


class MixupAugmentation:
    """Mixup augmentation for training."""
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize mixup augmentation.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a batch.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
            
        Returns:
            Mixed images, labels_a, labels_b, lambda
        """
        batch_size = images.shape[0]
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        return mixed_images, labels, labels[index], lam


class CutmixAugmentation:
    """Cutmix augmentation for training."""
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize cutmix augmentation.
        
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply cutmix to a batch.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
            
        Returns:
            Mixed images, labels_a, labels_b, lambda
        """
        batch_size = images.shape[0]
        _, _, h, w = images.shape
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Get random box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return mixed_images, labels, labels[index], lam
