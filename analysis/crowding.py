"""
Crowding analysis module for SPT Analysis.

This module provides tools for analyzing the effects of molecular crowding
on particle trajectories using MSD scaling, non-Gaussian parameters,
and other viscoelastic indicators.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

logger = logging.getLogger(__name__)


class CrowdingAnalyzer:
    """
    Analyzer for molecular crowding effects.
    
    Parameters
    ----------
    dt : float, optional
        Time between frames in seconds, by default 0.014
    particle_radius : float, optional
        Particle radius in nm, by default 5.0
    temperature : float, optional
        Temperature in K, by default 298
    """
    
    def __init__(self, dt=0.014, particle_radius=5.0, temperature=298):
        self.dt = dt
        self.particle_radius = particle_radius * 1e-9  # Convert to meters
        self.temperature = temperature
        self.kB = 1.38e-23  # Boltzmann constant [J/K]
        
        # Results storage
        self.crowding_parameters = {}
        self.scaling_analysis = {}
        self.non_gaussian_parameters = {}
    
    def analyze_crowding_effects(self, tracks_df, compartment_masks=None):
        """
        Analyze crowding effects on particle motion.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        compartment_masks : dict, optional
            Dictionary mapping compartment names to binary masks, by default None
            
        Returns
        -------
        dict
            Dictionary with crowding analysis results
        """
        try:
            # Calculate MSD for all tracks
            msd_results = self._calculate_msd_all_tracks(tracks_df)
            
            # Calculate MSD for each compartment if masks provided
            compartment_msds = {}
            
            if compartment_masks:
                for comp_name, mask in compartment_masks.items():
                    # Filter tracks by compartment
                    comp_tracks = self._filter_tracks_by_compartment(tracks_df, mask)
                    
                    if len(comp_tracks) > 0:
                        compartment_msds[comp_name] = self._calculate_msd_all_tracks(comp_tracks)
            else:
                compartment_msds = None
            
            # Extract crowding parameters
            crowding_params = {}
            scaling_results = {}
            non_gaussian_params = {}
            
            # All tracks
            crowding_params['all'] = self._extract_crowding_parameters(msd_results)
            scaling_results['all'] = self._analyze_msd_scaling(msd_results)
            non_gaussian_params['all'] = self._calculate_non_gaussian_parameter(tracks_df)
            
            # Compartment-specific analysis
            if compartment_msds:
                for comp, comp_msd in compartment_msds.items():
                    crowding_params[comp] = self._extract_crowding_parameters(comp_msd)
                    scaling_results[comp] = self._analyze_msd_scaling(comp_msd)
                    
                    # Filter tracks by compartment for NGP
                    comp_tracks = self._filter_tracks_by_compartment(
                        tracks_df, 
                        compartment_masks[comp]
                    )
                    non_gaussian_params[comp] = self._calculate_non_gaussian_parameter(comp_tracks)
            
            # Store results
            self.crowding_parameters = crowding_params
            self.scaling_analysis = scaling_results
            self.non_gaussian_parameters = non_gaussian_params
            
            return {
                'crowding_parameters': crowding_params,
                'scaling_analysis': scaling_results,
                '
