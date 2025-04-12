"""
Boundary crossing analysis module for SPT Analysis.

This module provides tools for analyzing particle behavior at boundaries
between compartments, including crossing dynamics and angular distributions.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)


class BoundaryCrossingAnalyzer:
    """
    Analyzer for boundary crossing and angular distributions.
    
    Parameters
    ----------
    dt : float, optional
        Time between frames in seconds, by default 0.014
    """
    
    def __init__(self, dt=0.014):
        self.dt = dt
        
        # Results storage
        self.crossing_events = {}
        self.boundary_params = {}
        self.angular_distributions = {}
    
    def analyze_boundary_crossings(self, tracks_df, compartment_masks):
        """
        Analyze boundary crossing events between compartments.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        compartment_masks : dict
            Dictionary mapping compartment names to binary masks
            
        Returns
        -------
        dict
            Dictionary with boundary crossing results
        """
        try:
            if not compartment_masks or len(compartment_masks) < 2:
                return {'status': 'Need at least two compartments for boundary analysis'}
            
            # Create labeled map from compartment masks
            first_mask = next(iter(compartment_masks.values()))
            labeled_map = np.zeros_like(first_mask, dtype=np.int32)
            
            compartment_names = list(compartment_masks.keys())
            for i, name in enumerate(compartment_names, start=1):
                labeled_map[compartment_masks[name] > 0] = i
            
            # Find boundary pixels using dilation
            from scipy import ndimage
            boundaries = {}
            for i, name1 in enumerate(compartment_names, start=1):
                for j, name2 in enumerate(compartment_names[i:], start=i+1):
                    # Create binary masks for each compartment
                    mask1 = labeled_map == i
                    mask2 = labeled_map == j
                    # Dilate both masks to identify pixels near the boundary
                    mask1_dilated = ndimage.binary_dilation(mask1)
                    mask2_dilated = ndimage.binary_dilation(mask2)
                    # The boundary is defined as the intersection of the two dilated masks
                    boundary = mask1_dilated & mask2_dilated
                    boundaries[f"{name1}-{name2}"] = boundary
            
            # Analyze crossings from the tracking data
            crossing_events = []
            for track_id, track_df in tracks_df.groupby('track_id'):
                track_df = track_df.sort_values('frame')
                positions = track_df[['x', 'y']].values.astype(int)
                frames = track_df['frame'].values
                
                # For each pair of consecutive positions, determine if there is a crossing
                for i in range(len(positions) - 1):
                    x1, y1 = positions[i]
                    x2, y2 = positions[i+1]
                    
                    # Determine the compartments at current and next positions
                    if 0 <= y1 < labeled_map.shape[0] and 0 <= x1 < labeled_map.shape[1]:
                        curr_label = labeled_map[y1, x1]
                        curr_comp = compartment_names[curr_label - 1] if curr_label > 0 else None
                    else:
                        curr
