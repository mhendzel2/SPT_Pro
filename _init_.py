
"""
SPT Analysis package for single-particle tracking data analysis.

This package provides tools for tracking, analyzing, and visualizing
single-particle tracking data from microscopy experiments.
"""

__version__ = '0.1.0'

import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import key modules for convenience
from . import tracking
from . import analysis
from . import visualization
from . import utils
from . import project    # Add project module
try:
    from . import gui    # Try to import GUI module
except ImportError:
    pass                # GUI dependencies might not be installed
