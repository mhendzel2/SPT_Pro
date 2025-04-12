
"""
Image processing module for SPT Analysis.

This module provides functions for preprocessing microscopy images,
including denoising, background correction, and enhancement.
"""

import numpy as np
import scipy.ndimage as ndi
from skimage import filters, restoration, exposure, transform
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


def enhance_contrast(image, percentile=(2, 98), method='stretch'):
    """
    Enhance contrast in microscopy image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    percentile : tuple, optional
        Percentile range for contrast stretching, by default (2, 98)
    method : str, optional
        Method for contrast enhancement ('stretch', 'equalize', 'adaptive'), by default 'stretch'
        
    Returns
    -------
    numpy.ndarray
        Contrast-enhanced image
    """
    try:
        logger.debug(f"Enhancing contrast using method: {method}")
        
        # Apply contrast enhancement based on method
        if method == 'stretch':
            # Contrast stretching
            p_low, p_high = percentile
            low, high = np.percentile(image, (p_low, p_high))
            enhanced = exposure.rescale_intensity(image, in_range=(low, high))
        
        elif method == 'equalize':
            # Histogram equalization
            enhanced = exposure.equalize_hist(image)
        
        elif method == 'adaptive':
            # Adaptive histogram equalization
            enhanced = exposure.equalize_adapthist(image)
        
        else:
            raise ValueError(f"Unknown contrast enhancement method: {method}")
        
        return enhanced
    
    except Exception as e:
        logger.error(f"Error enhancing contrast: {str(e)}")
        raise


def denoise_image(image, method='gaussian', **kwargs):
    """
    Denoise microscopy image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    method : str, optional
        Denoising method ('gaussian', 'median', 'bilateral', 'tv', 'wavelet'), by default 'gaussian'
    **kwargs
        Additional parameters for specific methods
        
    Returns
    -------
    numpy.ndarray
        Denoised image
    """
    try:
        logger.debug(f"Denoising image using method: {method}")
        
        # Apply denoising based on method
        if method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            denoised = filters.gaussian(image, sigma=sigma)
        
        elif method == 'median':
            size = kwargs.get('size', 3)
            denoised = filters.median(image, footprint=np.ones((size, size)))
        
        elif method == 'bilateral':
            sigma_color = kwargs.get('sigma_color', 0.1)
            sigma_spatial = kwargs.get('sigma_spatial', 2)
            denoised = restoration.denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
        
        elif method == 'tv':
            weight = kwargs.get('weight', 0.1)
            denoised = restoration.denoise_tv_chambolle(image, weight=weight)
        
        elif method == 'wavelet':
            wavelet = kwargs.get('wavelet', 'db1')
            denoised = restoration.denoise_wavelet(image, wavelet=wavelet)
        
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        
        return denoised
    
    except Exception as e:
        logger.error(f"Error denoising image: {str(e)}")
        raise


def correct_background(image, method='rolling_ball', **kwargs):
    """
    Correct background in microscopy image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    method : str, optional
        Background correction method ('rolling_ball', 'tophat', 'subtract'), by default 'rolling_ball'
    **kwargs
        Additional parameters for specific methods
        
    Returns
    -------
    numpy.ndarray
        Background-corrected image
    """
    try:
        logger.debug(f"Correcting background using method: {method}")
        
        # Apply background correction based on method
        if method == 'rolling_ball':
            radius = kwargs.get('radius', 50)
            
            # Create structuring element
            from skimage.morphology import disk
            selem = disk(radius)
            
            # Estimate background using grayscale opening
            from skimage.morphology import opening
            background = opening(image, selem)
            
            # Subtract background
            corrected = image - background
            
            # Clip negative values
            corrected = np.clip(corrected, 0, None)
        
        elif method == 'tophat':
            radius = kwargs.get('radius', 50)
            
            # Create structuring element
            from skimage.morphology import disk
            selem = disk(radius)
            
            # Apply white tophat filter
            from skimage.morphology import white_tophat
            corrected = white_tophat(image, selem)
        
        elif method == 'subtract':
            # Subtract a constant or an image
            background = kwargs.get('background', image.min())
            
            if isinstance(background, (int, float)):
                corrected = image - background
            else:
                # Ensure same shape
                if background.shape != image.shape:
                    raise ValueError(f"Background shape {background.shape} does not match image shape {image.shape}")
                corrected = image - background
            
            # Clip negative values
            corrected = np.clip(corrected, 0, None)
        
        else:
            raise ValueError(f"Unknown background correction method: {method}")
        
        return corrected
    
    except Exception as e:
        logger.error(f"Error correcting background: {str(e)}")
        raise


def normalize_image(image, method='minmax', **kwargs):
    """
    Normalize intensity in microscopy image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    method : str, optional
        Normalization method ('minmax', 'zscore', 'percentile'), by default 'minmax'
    **kwargs
        Additional parameters for specific methods
        
    Returns
    -------
    numpy.ndarray
        Normalized image
    """
    try:
        logger.debug(f"Normalizing image using method: {method}")
        
        # Apply normalization based on method
        if method == 'minmax':
            # Min-max normalization
            min_val = kwargs.get('min_val', image.min())
            max_val = kwargs.get('max_val', image.max())
            
            if min_val == max_val:
                normalized = np.zeros_like(image)
            else:
                normalized = (image - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            # Z-score normalization
            mean = kwargs.get('mean', image.mean())
            std = kwargs.get('std', image.std())
            
            if std == 0:
                normalized = np.zeros_like(image)
            else:
                normalized = (image - mean) / std
        
        elif method == 'percentile':
            # Percentile normalization
            p_low = kwargs.get('p_low', 1)
            p_high = kwargs.get('p_high', 99)
            
            low, high = np.percentile(image, (p_low, p_high))
            
            if low == high:
                normalized = np.zeros_like(image)
            else:
                normalized = np.clip((image - low) / (high - low), 0, 1)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    except Exception as e:
        logger.error(f"Error normalizing image: {str(e)}")
        raise


def register_image_stack(image_stack, method='translation', **kwargs):
    """
    Register images in a stack to correct for drift.
    
    Parameters
    ----------
    image_stack : numpy.ndarray
        Input image stack with shape (n_frames, height, width)
    method : str, optional
        Registration method ('translation', 'rigid', 'affine'), by default 'translation'
    **kwargs
        Additional parameters for specific methods
        
    Returns
    -------
    numpy.ndarray
        Registered image stack
    dict
        Registration parameters for each frame
    """
    try:
        from skimage.registration import phase_cross_correlation
        from scipy.ndimage import shift, rotate, affine_transform
        
        logger.debug(f"Registering image stack using method: {method}")
        
        # Get number of frames
        n_frames = image_stack.shape[0]
        
        # Initialize registered stack
        registered_stack = np.zeros_like(image_stack)
        registered_stack[0] = image_stack[0]  # First frame as reference
        
        # Initialize registration parameters
        reg_params = {0: {'tx': 0, 'ty': 0, 'rotation': 0, 'scale': 1.0}}
        
        # Reference frame
        reference = image_stack[0]
        
        # Register each frame
        for i in range(1, n_frames):
            frame = image_stack[i]
            
            if method == 'translation':
                # Estimate translation
                shift_y, shift_x = phase_cross_correlation(reference, frame)[0]
                
                # Apply translation
                registered_stack[i] = shift(frame, (shift_y, shift_x), order=3, mode='constant')
                
                # Store parameters
                reg_params[i] = {'tx': shift_x, 'ty': shift_y, 'rotation': 0, 'scale': 1.0}
            
            elif method == 'rigid':
                # Get parameters
                max_rotation = kwargs.get('max_rotation', 10)
                rotation_steps = kwargs.get('rotation_steps', 20)
                
                # Try different rotations
                best_correlation = -1
                best_shift = (0, 0)
                best_rotation = 0
                
                for angle in np.linspace(-max_rotation, max_rotation, rotation_steps):
                    # Rotate frame
                    rotated = rotate(frame, angle, reshape=False, order=3, mode='constant')
                    
                    # Estimate translation
                    shift_y, shift_x = phase_cross_correlation(reference, rotated)[0]
                    
                    # Apply translation
                    aligned = shift(rotated, (shift_y, shift_x), order=3, mode='constant')
                    
                    # Compute correlation
                    correlation = np.corrcoef(reference.flatten(), aligned.flatten())[0, 1]
                    
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_shift = (shift_y, shift_x)
                        best_rotation = angle
                
                # Apply best transformation
                rotated = rotate(frame, best_rotation, reshape=False, order=3, mode='constant')
                registered_stack[i] = shift(rotated, best_shift, order=3, mode='constant')
                
                # Store parameters
                reg_params[i] = {'tx': best_shift[1], 'ty': best_shift[0], 
                                'rotation': best_rotation, 'scale': 1.0}
            
            elif method == 'affine':
                try:
                    from skimage.transform import AffineTransform, estimate_transform, warp
                    
                    # Feature-based registration
                    from skimage.feature import ORB, match_descriptors
                    
                    # Detect keypoints and compute descriptors
                    detector_extractor = ORB(n_keypoints=100)
                    
                    detector_extractor.detect_and_extract(reference)
                    keypoints1 = detector_extractor.keypoints
                    descriptors1 = detector_extractor.descriptors
                    
                    detector_extractor.detect_and_extract(frame)
                    keypoints2 = detector_extractor.keypoints
                    descriptors2 = detector_extractor.descriptors
                    
                    # Match descriptors
                    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)
                    
                    # Estimate affine transform
                    src = keypoints2[matches[:, 1]]
                    dst = keypoints1[matches[:, 0]]
                    
                    if len(src) >= 3:  # Need at least 3 point pairs
                        tform = estimate_transform('affine', src, dst)
                        
                        # Apply transform
                        registered_stack[i] = warp(frame, tform.inverse, order=3)
                        
                        # Store parameters
                        matrix = tform.params
                        reg_params[i] = {
                            'tx': matrix[0, 2],
                            'ty': matrix[1, 2],
                            'rotation': np.arctan2(matrix[1, 0], matrix[0, 0]),
                            'scale': np.sqrt(matrix[0, 0]**2 + matrix[1, 0]**2)
                        }
                    else:
                        # Fall back to translation if not enough matches
                        shift_y, shift_x = phase_cross_correlation(reference, frame)[0]
                        registered_stack[i] = shift(frame, (shift_y, shift_x), order=3, mode='constant')
                        reg_params[i] = {'tx': shift_x, 'ty': shift_y, 'rotation': 0, 'scale': 1.0}
                
                except:
                    # Fall back to translation if affine registration fails
                    logger.warning("Affine registration failed, falling back to translation")
                    shift_y, shift_x = phase_cross_correlation(reference, frame)[0]
                    registered_stack[i] = shift(frame, (shift_y, shift_x), order=3, mode='constant')
                    reg_params[i] = {'tx': shift_x, 'ty': shift_y, 'rotation': 0, 'scale': 1.0}
            
            else:
                raise ValueError(f"Unknown registration method: {method}")
            
            # Optionally update reference frame
            if kwargs.get('update_reference', False):
                reference = registered_stack[i]
        
        return registered_stack, reg_params
    
    except Exception as e:
        logger.error(f"Error registering image stack: {str(e)}")
        raise


def enhance_particles(image, enhancement_method='dog', **kwargs):
    """
    Enhance particles in microscopy image.
    
    ParametersCopy
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    enhancement_method : str, optional
        Enhancement method ('dog', 'log', 'doh', 'wavelet'), by default 'dog'
    **kwargs
        Additional parameters for specific methods
        
    Returns
    -------
    numpy.ndarray
        Enhanced image
    """
    try:
        logger.debug(f"Enhancing particles using method: {enhancement_method}")
        
        # Apply enhancement based on method
        if enhancement_method == 'dog':
            # Difference of Gaussians
            sigma_low = kwargs.get('sigma_low', 1.0)
            sigma_high = kwargs.get('sigma_high', 2.0)
            
            gaussian_low = filters.gaussian(image, sigma=sigma_low)
            gaussian_high = filters.gaussian(image, sigma=sigma_high)
            
            enhanced = gaussian_low - gaussian_high
        
        elif enhancement_method == 'log':
            # Laplacian of Gaussian
            sigma = kwargs.get('sigma', 2.0)
            
            enhanced = filters.gaussian(image, sigma=sigma)
            enhanced = filters.laplace(enhanced)
            enhanced = -enhanced  # Invert to make particles bright
        
        elif enhancement_method == 'doh':
            # Determinant of Hessian
            sigma = kwargs.get('sigma', 2.0)
            
            from skimage.feature import hessian_matrix, hessian_matrix_eigvals
            
            # Compute Hessian matrix
            hessian = hessian_matrix(image, sigma=sigma)
            
            # Compute eigenvalues
            eig_vals = hessian_matrix_eigvals(hessian)
            
            # Compute determinant
            enhanced = eig_vals[0] * eig_vals[1]
        
        elif enhancement_method == 'wavelet':
            # Wavelet transform
            import pywt
            
            wavelet = kwargs.get('wavelet', 'haar')
            level = kwargs.get('level', 2)
            
            # Apply wavelet transform
            coeffs = pywt.wavedec2(image, wavelet, level=level)
            
            # Extract detail coefficients
            enhanced = coeffs[1][0]  # Horizontal detail at level 1
            
            # Scale to original size
            from skimage.transform import resize
            enhanced = resize(enhanced, image.shape, anti_aliasing=True)
        
        else:
            raise ValueError(f"Unknown particle enhancement method: {enhancement_method}")
        
        # Normalize to [0, 1]
        enhanced = normalize_image(enhanced, method='minmax')
        
        return enhanced
    
    except Exception as e:
        logger.error(f"Error enhancing particles: {str(e)}")
        raise


def preprocess_stack(image_stack, steps=None, **kwargs):
    """
    Apply a sequence of preprocessing steps to an image stack.
    
    Parameters
    ----------
    image_stack : numpy.ndarray
        Input image stack with shape (n_frames, height, width)
    steps : list, optional
        List of preprocessing steps to apply, by default None
    **kwargs
        Additional parameters for specific steps
        
    Returns
    -------
    numpy.ndarray
        Preprocessed image stack
    """
    try:
        # Default preprocessing pipeline
        default_steps = [
            {'name': 'denoise', 'method': 'gaussian', 'sigma': 1.0},
            {'name': 'background', 'method': 'rolling_ball', 'radius': 50},
            {'name': 'enhance', 'method': 'dog', 'sigma_low': 1.0, 'sigma_high': 2.0},
            {'name': 'normalize', 'method': 'minmax'}
        ]
        
        # Use default steps if none provided
        steps = steps or default_steps
        
        logger.info(f"Preprocessing stack with {len(steps)} steps")
        
        # Get number of frames
        n_frames = image_stack.shape[0]
        
        # Initialize preprocessed stack
        preprocessed_stack = image_stack.copy()
        
        # Apply each step to all frames
        for step in steps:
            step_name = step.get('name', '')
            method = step.get('method', '')
            
            logger.debug(f"Applying {step_name} with method {method}")
            
            # Update step parameters with kwargs
            step_params = step.copy()
            step_params.update(kwargs.get(step_name, {}))
            
            # Remove name and method from parameters
            step_params.pop('name', None)
            method = step_params.pop('method', '')
            
            # Process all frames
            for i in range(n_frames):
                frame = preprocessed_stack[i]
                
                # Apply preprocessing step
                if step_name == 'denoise':
                    preprocessed_stack[i] = denoise_image(frame, method=method, **step_params)
                
                elif step_name == 'background':
                    preprocessed_stack[i] = correct_background(frame, method=method, **step_params)
                
                elif step_name == 'enhance':
                    preprocessed_stack[i] = enhance_particles(frame, enhancement_method=method, **step_params)
                
                elif step_name == 'normalize':
                    preprocessed_stack[i] = normalize_image(frame, method=method, **step_params)
                
                elif step_name == 'contrast':
                    preprocessed_stack[i] = enhance_contrast(frame, method=method, **step_params)
                
                else:
                    logger.warning(f"Unknown preprocessing step: {step_name}")
        
        # Apply registration if requested
        if kwargs.get('register', False):
            register_method = kwargs.get('register_method', 'translation')
            preprocessed_stack, _ = register_image_stack(preprocessed_stack, method=register_method, **kwargs)
        
        return preprocessed_stack
    
    except Exception as e:
        logger.error(f"Error preprocessing stack: {str(e)}")
        raise


def crop_image(image, roi=None, center=None, size=None):
    """
    Crop image to region of interest.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    roi : tuple, optional
        Region of interest (x_min, y_min, x_max, y_max), by default None
    center : tuple, optional
        Center coordinates (x, y), by default None
    size : tuple, optional
        Crop size (width, height), by default None
        
    Returns
    -------
    numpy.ndarray
        Cropped image
    """
    try:
        # Get image dimensions
        height, width = image.shape[-2:]
        
        # Determine crop region
        if roi is not None:
            # Use explicit ROI
            x_min, y_min, x_max, y_max = roi
        elif center is not None and size is not None:
            # Use center and size
            x, y = center
            w, h = size
            
            x_min = max(0, int(x - w / 2))
            y_min = max(0, int(y - h / 2))
            x_max = min(width, int(x + w / 2))
            y_max = min(height, int(y + h / 2))
        else:
            raise ValueError("Either roi or both center and size must be provided")
        
        # Ensure valid bounds
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(width, int(x_max))
        y_max = min(height, int(y_max))
        
        # Check dimensions
        if x_min >= x_max or y_min >= y_max:
            raise ValueError(f"Invalid crop region: ({x_min}, {y_min}, {x_max}, {y_max})")
        
        # Crop image
        if image.ndim == 2:
            # Single image
            cropped = image[y_min:y_max, x_min:x_max]
        elif image.ndim == 3:
            # Image stack
            cropped = image[:, y_min:y_max, x_min:x_max]
        else:
            raise ValueError(f"Unsupported image dimensions: {image.ndim}")
        
        return cropped, (x_min, y_min, x_max, y_max)
    
    except Exception as e:
        logger.error(f"Error cropping image: {str(e)}")
        raise


def mask_image(image, mask):
    """
    Apply binary mask to image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    mask : numpy.ndarray
        Binary mask
        
    Returns
    -------
    numpy.ndarray
        Masked image
    """
    try:
        # Ensure mask has same shape as image
        if image.shape != mask.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape}")
        
        # Apply mask
        masked = image.copy()
        masked[~mask] = 0
        
        return masked
    
    except Exception as e:
        logger.error(f"Error applying mask: {str(e)}")
        raise


def downsample_stack(image_stack, factor=2):
    """
    Downsample image stack.
    
    Parameters
    ----------
    image_stack : numpy.ndarray
        Input image stack
    factor : int, optional
        Downsampling factor, by default 2
        
    Returns
    -------
    numpy.ndarray
        Downsampled image stack
    """
    try:
        # Get original shape
        orig_shape = image_stack.shape
        
        # Calculate new shape
        new_shape = list(orig_shape)
        new_shape[1] = int(new_shape[1] / factor)
        new_shape[2] = int(new_shape[2] / factor)
        
        # Downsample
        downsampled = np.zeros(new_shape, dtype=image_stack.dtype)
        
        for i in range(orig_shape[0]):
            downsampled[i] = transform.resize(image_stack[i], new_shape[1:], 
                                             anti_aliasing=True, preserve_range=True).astype(image_stack.dtype)
        
        return downsampled
    
    except Exception as e:
        logger.error(f"Error downsampling stack: {str(e)}")
        raise

    ----------