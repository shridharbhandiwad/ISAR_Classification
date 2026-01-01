"""ISAR image preprocessing utilities."""

import numpy as np
import torch
from typing import Optional, Tuple, Union
from scipy import ndimage
from scipy.signal import windows
import cv2


class ISARPreprocessor:
    """
    Preprocessing pipeline for ISAR images.
    
    Includes specialized processing for radar images:
    - Windowing functions
    - Noise reduction
    - Normalization
    - Range-Doppler processing
    """
    
    def __init__(
        self,
        image_size: int = 128,
        normalize: bool = True,
        denoise: bool = True,
        enhance_contrast: bool = True,
        window_function: Optional[str] = 'hamming'
    ):
        """
        Initialize the preprocessor.
        
        Args:
            image_size: Target image size
            normalize: Whether to normalize images
            denoise: Whether to apply denoising
            enhance_contrast: Whether to enhance contrast
            window_function: Window function for ISAR processing
        """
        self.image_size = image_size
        self.normalize = normalize
        self.denoise = denoise
        self.enhance_contrast = enhance_contrast
        self.window_function = window_function
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Apply preprocessing pipeline to an image.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        
        # Ensure float
        image = image.astype(np.float32)
        
        # Apply windowing
        if self.window_function:
            image = self._apply_window(image)
        
        # Denoise
        if self.denoise:
            image = self._denoise(image)
        
        # Enhance contrast
        if self.enhance_contrast:
            image = self._enhance_contrast(image)
        
        # Resize
        image = self._resize(image)
        
        # Normalize
        if self.normalize:
            image = self._normalize(image)
        
        # Convert to tensor
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]
        elif image.shape[-1] in [1, 3]:
            image = image.transpose(2, 0, 1)
        
        return torch.from_numpy(image).float()
    
    def _apply_window(self, image: np.ndarray) -> np.ndarray:
        """Apply windowing function to reduce sidelobes."""
        h, w = image.shape[:2]
        
        # Get window function
        if self.window_function == 'hamming':
            win_h = windows.hamming(h)
            win_w = windows.hamming(w)
        elif self.window_function == 'hanning':
            win_h = windows.hann(h)
            win_w = windows.hann(w)
        elif self.window_function == 'blackman':
            win_h = windows.blackman(h)
            win_w = windows.blackman(w)
        elif self.window_function == 'kaiser':
            win_h = windows.kaiser(h, beta=14)
            win_w = windows.kaiser(w, beta=14)
        else:
            return image
        
        # Create 2D window
        window_2d = np.outer(win_h, win_w)
        
        # Apply window
        if len(image.shape) == 3:
            window_2d = window_2d[..., np.newaxis]
        
        return image * window_2d
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to reduce speckle noise."""
        # Normalize to 0-255 for cv2
        img_normalized = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply bilateral filter (preserves edges)
        if len(img_normalized.shape) == 2:
            denoised = cv2.bilateralFilter(img_normalized, 5, 75, 75)
        else:
            denoised = cv2.bilateralFilter(img_normalized, 5, 75, 75)
        
        # Normalize back
        denoised = denoised.astype(np.float32) / 255.0
        
        return denoised
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        # Normalize to 0-255
        img_normalized = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        if len(img_normalized.shape) == 2:
            enhanced = clahe.apply(img_normalized)
        else:
            # Apply to each channel
            enhanced = np.zeros_like(img_normalized)
            for i in range(img_normalized.shape[-1]):
                enhanced[..., i] = clahe.apply(img_normalized[..., i])
        
        # Normalize back
        enhanced = enhanced.astype(np.float32) / 255.0
        
        return enhanced
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            if len(image.shape) == 2:
                image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            else:
                image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return image
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values."""
        # Z-score normalization
        mean = image.mean()
        std = image.std() + 1e-8
        image = (image - mean) / std
        return image
    
    @staticmethod
    def compute_range_doppler(raw_signal: np.ndarray) -> np.ndarray:
        """
        Compute Range-Doppler image from raw radar signal.
        
        Args:
            raw_signal: Raw radar signal (2D complex array)
            
        Returns:
            Range-Doppler image (magnitude)
        """
        # Apply 2D FFT
        rd_image = np.fft.fftshift(np.fft.fft2(raw_signal))
        
        # Get magnitude in dB
        magnitude = 20 * np.log10(np.abs(rd_image) + 1e-10)
        
        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        return magnitude
    
    @staticmethod
    def apply_motion_compensation(
        isar_image: np.ndarray,
        phase_error: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply motion compensation to ISAR image.
        
        Args:
            isar_image: Input ISAR image (complex)
            phase_error: Estimated phase error (optional)
            
        Returns:
            Motion-compensated image
        """
        if phase_error is None:
            # Estimate phase error using dominant scatterer
            # Simple autofocus algorithm
            profile = np.sum(np.abs(isar_image), axis=0)
            dominant_idx = np.argmax(profile)
            phase_error = np.angle(isar_image[:, dominant_idx])
        
        # Compensate
        compensation = np.exp(-1j * phase_error)
        compensated = isar_image * compensation[:, np.newaxis]
        
        return compensated


def preprocess_batch(
    batch: torch.Tensor,
    preprocessor: ISARPreprocessor
) -> torch.Tensor:
    """
    Preprocess a batch of images.
    
    Args:
        batch: Batch of images (B, C, H, W)
        preprocessor: Preprocessor instance
        
    Returns:
        Preprocessed batch
    """
    processed = []
    for i in range(batch.shape[0]):
        img = batch[i].numpy()
        if img.shape[0] in [1, 3]:  # C, H, W format
            img = img.transpose(1, 2, 0).squeeze()
        processed.append(preprocessor(img))
    
    return torch.stack(processed)
