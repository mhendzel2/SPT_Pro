
SPT Analysis: A comprehensive package for Single Particle Tracking Analysis with 
Bayesian Diffusion State Analysis and Microenvironment Analysis.

This package provides tools for:
- Particle detection and tracking in noisy, high-density environments
- Track preprocessing and filtering
- Mean squared displacement (MSD) analysis
- Diffusion coefficient calculation
- Bayesian diffusion state analysis
- Microenvironment-specific analysis
- Nuclear compartment analysis
- Visualization and reporting
"""

__version__ = '1.0.0'

# Import key components for easier access
from spt_analysis.core.analysis import TrackAnalyzer
from spt_analysis.core.enhanced_analysis import EnhancedTrackAnalyzer
from spt_analysis.tracking.tracker import Tracker
from spt_analysis.preprocessing.filters import preprocess_tracks
from spt_analysis.microenvironment.compartment_analyzer import CompartmentAnalyzer

# Set up package-wide logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
