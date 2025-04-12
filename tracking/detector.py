"""
Particle detection module for SPT Analysis.

This module provides various detection algorithms for identifying particles in microscopy images,
with a focus on high-density, noisy environments.
"""

import numpy as np
import numpy.typing as npt
import cv2
from scipy import ndimage
from skimage.feature import blob_dog, blob_log, peak_local_max
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, TypeVar
import logging

logger = logging.getLogger(__name__)

# Type definitions
ImageArray = TypeVar('ImageArray', bound=npt.NDArray[np.float_])
CoordinateArray = TypeVar('CoordinateArray', bound=npt.NDArray[np.float_])

class UNetDetector:
    """
    U-Net based deep learning detector for particles in noisy images.
    
    This detector uses a U-Net architecture to segment particles even in challenging
    conditions with low SNR and high particle density.
    
    Parameters
    ----------
    model_path : str, optional
        Path to pretrained model weights
    device : str, optional
        Computation device ('cpu' or 'cuda'), by default 'cpu'
    threshold : float, optional
        Detection confidence threshold, by default 0.5
        
    Examples
    --------
    >>> detector = UNetDetector(model_path='weights.pth')
    >>> coordinates = detector.detect(image)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu', 
                 threshold: float = 0.5) -> None:
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be a number between 0 and 1")
        if device not in ['cpu', 'cuda']:
            raise ValueError("Device must be either 'cpu' or 'cuda'")
            
        self.device = device
        self.threshold = threshold
        self._build_model()
        
        if model_path:
            self.load_weights(model_path)
    
    def _build_model(self) -> None:
        """Build U-Net model architecture"""
        self.model = UNet(1, 1)  # 1 input channel, 1 output channel
        self.model.to(self.device)
        self.model.eval()
    
    def load_weights(self, model_path: str) -> None:
        """
        Load pretrained model weights.
        
        Parameters
        ----------
        model_path : str
            Path to model weights file
            
        Raises
        ------
        FileNotFoundError
            If the model weights file is not found
        torch.serialization.PyTorchPickleError
            If the file format is invalid
        """
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model weights from {model_path}")
        except FileNotFoundError:
            logger.error(f"Model weights file not found: {model_path}")
            raise
        except torch.serialization.PyTorchPickleError:
            logger.error(f"Invalid model weights file format: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
            raise
    
    def detect(self, image: ImageArray) -> CoordinateArray:
        """
        Detect particles in an image.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image (2D array)
            
        Returns
        -------
        numpy.ndarray
            Array of particle coordinates with shape (n_particles, 2)
            
        Raises
        ------
        TypeError
            If input is not a numpy array or contains non-numeric values
        ValueError
            If input is not 2-dimensional
        """
        try:
            # Input validation
            if not isinstance(image, np.ndarray):
                raise TypeError("Input image must be a numpy array")
            if image.ndim != 2:
                raise ValueError("Input image must be 2-dimensional")
            if not np.issubdtype(image.dtype, np.number):
                raise TypeError("Input image must contain numeric values")
            
            # Ensure image is normalized
            if image.max() > 1.0:
                image = image / 255.0
            
            # Convert to tensor
            x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            x = x.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                prediction = self.model(x)
                prediction = torch.sigmoid(prediction)
            
            # Convert to numpy
            probability_map = prediction.squeeze().cpu().numpy()
            
            # Clear CUDA memory if using GPU
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            # Threshold and find peaks
            binary_mask = probability_map > self.threshold
            labeled_mask, num_features = ndimage.label(binary_mask)
            
            # Get centroids
            coordinates = []
            for label in range(1, num_features + 1):
                y, x = ndimage.center_of_mass(binary_mask, labeled_mask, label)
                coordinates.append([y, x])
            
            return np.array(coordinates)
            
        except Exception as e:
            logger.error(f"Error in particle detection: {str(e)}")
            raise


class UNet(nn.Module):
    """
    U-Net architecture for particle detection.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UNet, self).__init__()
        
        if not isinstance(in_channels, int) or not isinstance(out_channels, int):
            raise TypeError("Channel counts must be integers")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel counts must be positive")
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bridge
        self.bridge = self._conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self._upconv_block(1024, 512)
        self.dec3 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256, 128)
        self.dec1 = self._upconv_block(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bridge
        bridge = self.bridge(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.dec4(bridge)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self._conv_block(dec4.shape[1], 512)(dec4)
        
        dec3 = self.dec3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self._conv_block(dec3.shape[1], 256)(dec3)
        
        dec2 = self.dec2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self._conv_block(dec2.shape[1], 128)(dec2)
        
        dec1 = self.dec1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self._conv_block(dec1.shape[1], 64)(dec1)
        
        # Final layer
        return self.final(dec1)


class WaveletDetector:
    """
    Wavelet-based particle detector.
    
    Parameters
    ----------
    min_sigma : float, optional
        Minimum standard deviation for Gaussian kernel, by default 1.0
    max_sigma : float, optional
        Maximum standard deviation for Gaussian kernel, by default 10.0
    num_sigma : int, optional
        Number of intermediate sigma values, by default 10
    threshold : float, optional
        Detection threshold, by default 0.1
    overlap : float, optional
        Maximum allowed overlap between particles, by default 0.5
    cache_size : int, optional
        Maximum number of cached results, by default 1000
    """
    
    def __init__(self, min_sigma: float = 1.0, max_sigma: float = 10.0, 
                 num_sigma: int = 10, threshold: float = 0.1, 
                 overlap: float = 0.5, cache_size: int = 1000) -> None:
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.cache_size = cache_size
        self._cache: Dict[int, np.ndarray] = {}
    
    def detect(self, image: ImageArray) -> CoordinateArray:
        """
        Detect particles using wavelet transform.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image
            
        Returns
        -------
        numpy.ndarray
            Array of particle coordinates with shape (n_particles, 2)
        """
        try:
            # Check cache
            image_hash = hash(image.tobytes())
            if image_hash in self._cache:
                return self._cache[image_hash].copy()
            
            # Use Laplacian of Gaussian (LoG) to detect blobs
            blobs = blob_log(
                image, 
                min_sigma=self.min_sigma, 
                max_sigma=self.max_sigma, 
                num_sigma=self.num_sigma, 
                threshold=self.threshold, 
                overlap=self.overlap
            )
            
            # Extract coordinates (y, x)
            coordinates = blobs[:, 0:2]
            
            # Cache result
            self._cache[image_hash] = coordinates.copy()
            # Limit cache size
            if len(self._cache) > self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            
            return coordinates
            
        except Exception as e:
            logger.error(f"Error in wavelet particle detection: {str(e)}")
            raise


class LocalMaximaDetector:
    """
    Local maxima-based particle detector.
    
    Parameters
    ----------
    min_distance : int, optional
        Minimum distance between particles, by default 5
    threshold_abs : float, optional
        Absolute threshold for detection, by default 0.0
    threshold_rel : float, optional
        Relative threshold for detection, by default 0.1
    preprocess : bool, optional
        Whether to apply preprocessing, by default True
    """
    
    def __init__(self, min_distance: int = 5, threshold_abs: float = 0.0,
                 threshold_rel: float = 0.1, preprocess: bool = True) -> None:
        self.min_distance = min_distance
        self.threshold_abs = threshold_abs
        self.threshold_rel = threshold_rel
        self.preprocess = preprocess
    
    def detect(self, image: ImageArray) -> CoordinateArray:
        """
        Detect particles using local maxima.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image
            
        Returns
        -------
        numpy.ndarray
            Array of particle coordinates with shape (n_particles, 2)
        """
        try:
            # Input validation
            if not isinstance(image, np.ndarray):
                raise TypeError("Input image must be a numpy array")
            if image.ndim != 2:
                raise ValueError("Input image must be 2-dimensional")
                
            # Preprocess image if requested
            if self.preprocess:
                # Apply Gaussian filter for noise reduction
                image = ndimage.gaussian_filter(image, sigma=1.0)
            
            # Find local maxima
            coordinates = peak_local_max(
                image, 
                min_distance=self.min_distance, 
                threshold_abs=self.threshold_abs, 
                threshold_rel=self.threshold_rel
            )
            
            return coordinates
            
        except Exception as e:
            logger.error(f"Error in local maxima particle detection: {str(e)}")
            raise


def get_detector(method: str = "wavelet", **kwargs: Any) -> Union[WaveletDetector, UNetDetector, LocalMaximaDetector]:
    """
    Factory function to get appropriate detector based on method.
    
    Parameters
    ----------
    method : str, optional
        Detection method, one of 'wavelet', 'unet', 'local_maxima', by default "wavelet"
    **kwargs
        Additional parameters for the detector
    
    Returns
    -------
    Union[WaveletDetector, UNetDetector, LocalMaximaDetector]
        Initialized detector object
    
    Raises
    ------
    ValueError
        If method is unknown
    
    Examples
    --------
    >>> detector = get_detector("wavelet", min_sigma=2.0)
    >>> coordinates = detector.detect(image)
    """
    methods = {
        "wavelet": WaveletDetector,
        "unet": UNetDetector,
        "local_maxima": LocalMaximaDetector
    }
    
    if method not in methods:
        raise ValueError(f"Unknown detection method: {method}. Available methods: {list(methods.keys())}")
    
    return methods[method](**kwargs)
'''

# Write the corrected code to a file
with open('detector_corrected.py', 'w') as f:
    f.write(corrected_detector)

print("Created detector_corrected.py with all improvements implemented.")