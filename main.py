
"""
Command-line interface for SPT Analysis.

This module provides a command-line interface for the SPT Analysis package,
allowing users to perform tracking and analysis directly from the terminal.
"""

import argparse
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import yaml

# Import package modules
from spt_analysis import __version__
from spt_analysis.tracking import tracker
from spt_analysis.analysis import diffusion, motion, clustering
from spt_analysis.utils import io, processing, visualization

# Set up logging
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='SPT Analysis: Single Particle Tracking Analysis Tool'
    )
    
    parser.add_argument('--version', action='version', version=f'SPT Analysis {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Track command
    track_parser = subparsers.add_parser('track', help='Track particles in image stack')
    track_parser.add_argument('input', help='Input image stack file (TIFF)')
    track_parser.add_argument('output', help='Output tracks file (CSV, JSON, HDF5)')
    track_parser.add_argument('--detector', choices=['wavelet', 'unet', 'local_maxima'],
                            default='wavelet', help='Detection method')
    track_parser.add_argument('--linker', choices=['graph', 'mht', 'imm'],
                            default='graph', help='Linking method')
    track_parser.add_argument('--preprocess', action='store_true',
                            help='Apply preprocessing to image stack')
    track_parser.add_argument('--min-track-length', type=int, default=3,
                            help='Minimum track length to keep')
    track_parser.add_argument('--config', help='Configuration file (YAML, JSON)')
    track_parser.add_argument('--plot', action='store_true',
                            help='Generate and save plots')
    track_parser.add_argument('--plot-output', help='Output directory for plots')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze tracking data')
    analyze_parser.add_argument('input', help='Input tracks file (CSV, JSON, HDF5)')
    analyze_parser.add_argument('output', help='Output results file (JSON, HDF5)')
    analyze_parser.add_argument('--analysis', choices=['diffusion', 'motion', 'clustering', 'all'],
                              default='all', help='Analysis type')
    analyze_parser.add_argument('--pixel-size', type=float, default=0.1,
                              help='Pixel size in μm')
    analyze_parser.add_argument('--frame-interval', type=float, default=0.1,
                              help='Frame interval in seconds')
    analyze_parser.add_argument('--min-track-length', type=int, default=10,
                              help='Minimum track length for analysis')
    analyze_parser.add_argument('--config', help='Configuration file (YAML, JSON)')
    analyze_parser.add_argument('--plot', action='store_true',
                              help='Generate and save plots')
    analyze_parser.add_argument('--plot-output', help='Output directory for plots')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize tracking data')
    viz_parser.add_argument('input', help='Input tracks file (CSV, JSON, HDF5)')
    viz_parser.add_argument('--image', help='Optional background image file (TIFF)')
    viz_parser.add_argument('--output', help='Output file (PDF, HTML)')
    viz_parser.add_argument('--type', choices=['trajectories', 'summary', 'animation', 'interactive'],
                          default='summary', help='Visualization type')
    viz_parser.add_argument('--pixel-size', type=float, default=0.1,
                          help='Pixel size in μm')
    viz_parser.add_argument('--frame-interval', type=float, default=0.1,
                          help='Frame interval in seconds')
    viz_parser.add_argument('--config', help='Configuration file (YAML, JSON)')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert tracking data between formats')
    convert_parser.add_argument('input', help='Input tracks file')
    convert_parser.add_argument('output', help='Output tracks file')
    convert_parser.add_argument('--input-format', choices=['csv', 'excel', 'hdf5', 'json', 'trackmate', 'fiji'],
                              help='Input format (defaults to file extension)')
    convert_parser.add_argument('--output-format', choices=['csv', 'excel', 'hdf5', 'json'],
                              help='Output format (defaults to file extension)')
    
    return parser.parse_args()


def get_configuration(config_file, default_config=None):
    """
    Load configuration from file.
    
    Parameters
    ----------
    config_file : str
        Path to configuration file
    default_config : dict, optional
        Default configuration, by default None
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    # Start with default config if provided
    config = default_config.copy() if default_config else {}
    
    # Load from file if provided
    if config_file and os.path.exists(config_file):
        try:
            loaded_config = io.load_config(config_file)
            config.update(loaded_config)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    return config


def run_tracking(args):
    """
    Run particle tracking.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    logger.info("Starting particle tracking")
    
    # Load configuration
    default_config = {
        'detector': {
            'method': args.detector,
            'params': {}
        },
        'linker': {
            'method': args.linker,
            'params': {}
        },
        'preprocessing': {
            'apply': args.preprocess,
            'steps': [
                {'name': 'denoise', 'method': 'gaussian', 'sigma': 1.0},
                {'name': 'background', 'method': 'rolling_ball', 'radius': 50},
                {'name': 'enhance', 'method': 'dog', 'sigma_low': 1.0, 'sigma_high': 2.0},
                {'name': 'normalize', 'method': 'minmax'}
            ]
        },
        'tracking': {
            'min_track_length': args.min_track_length
        }
    }
    
    config = get_configuration(args.config, default_config)
    
    # Load image stack
    try:
        image_stack = io.load_image_stack(args.input)
        logger.info(f"Loaded image stack with shape {image_stack.shape}")
    except Exception as e:
        logger.error(f"Error loading image stack: {str(e)}")
        sys.exit(1)
    
    # Preprocess image stack if requested
    if config['preprocessing']['apply']:
        logger.info("Preprocessing image stack")
        try:
            image_stack = processing.preprocess_stack(
                image_stack, 
                steps=config['preprocessing']['steps']
            )
        except Exception as e:
            logger.error(f"Error preprocessing image stack: {str(e)}")
            sys.exit(1)
    
    # Initialize tracker
    try:
        particle_tracker = tracker.ParticleTracker(
            detector_method=config['detector']['method'],
            detector_params=config['detector']['params'],
            linker_method=config['linker']['method'],
            linker_params=config['linker']['params']
        )
    except Exception as e:
        logger.error(f"Error initializing tracker: {str(e)}")
        sys.exit(1)
    
    # Perform tracking
    try:
        logger.info("Performing tracking")
        track