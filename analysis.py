
"""
Core analysis module for SPT Analysis.

This module provides base functionality for analyzing single particle tracking data,
including mean squared displacement calculation, diffusion coefficient estimation,
and rheological property calculation.
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.special import gamma
from sklearn.mixture import GaussianMixture
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from numba import jit

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrackAnalyzer:
    """
    Analyzer for single particle tracking data.
    
    This class provides methods for calculating and analyzing various metrics
    from particle tracking data, such as mean squared displacement, diffusion
    coefficients, and rheological properties.
    
    Parameters
    ----------
    dt : float, optional
        Time interval between frames in seconds, by default 14e-3
    k_B : float, optional
        Boltzmann constant, by default 1.38e-23
    T : float, optional
        Temperature in Kelvin, by default 298
    a : float, optional
        Particle radius in meters, by default 10e-9
    config : dict, optional
        Additional configuration parameters, by default None
    """
    
    def __init__(self, dt=14e-3, k_B=1.38e-23, T=298, a=10e-9, config=None):
        """Initialize the track analyzer with physical parameters."""
        self.dt = dt
        self.k_B = k_B
        self.T = T
        self.a = a
        self.frequencies = np.logspace(0, 3, 50)
        
        # Default configuration
        self.config = {
            'max_workers': min(os.cpu_count(), 4),  # Number of parallel workers
            'bootstrap_samples': 100,               # Number of bootstrap samples for error estimation
            'msd_max_lag_factor': 0.25,             # Maximum lag time as fraction of track length
            'use_numba': True,                      # Whether to use Numba for acceleration
        }
        
        # Update with user configuration
        if config is not None:
            self.config.update(config)
    
    @staticmethod
    def power_law(tau, A, alpha):
        """
        MSD power-law function.
        
        Parameters
        ----------
        tau : numpy.ndarray
            Lag times
        A : float
            Amplitude
        alpha : float
            Power law exponent
            
        Returns
        -------
        numpy.ndarray
            MSD values according to power law
        """
        return A * tau**alpha
    
    def analyze_diffusion_modes(self, msd_values, tau):
        """
        Analyze different diffusion modes based on MSD scaling.
        
        Parameters
        ----------
        msd_values : numpy.ndarray
            MSD values
        tau : numpy.ndarray
            Lag times
            
        Returns
        -------
        str
            Diffusion mode description
        float
            Mean slope
        """
        try:
            # Calculate local slopes in log-log space
            slopes = np.diff(np.log(msd_values)) / np.diff(np.log(tau))
            mean_slope = np.mean(slopes)
            
            # Classify diffusion mode based on slope
            if mean_slope > 1.8:
                return "Directed motion", mean_slope
            elif mean_slope < 0.9:
                return "Confined diffusion", mean_slope
            else:
                return "Normal diffusion", mean_slope
        
        except Exception as e:
            logger.error(f"Error in diffusion mode analysis: {str(e)}")
            return "Analysis error", np.nan
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _fast_msd_calculation(positions, max_lag):
        """
        Numba-accelerated MSD calculation.
        
        Parameters
        ----------
        positions : numpy.ndarray
            Array of positions with shape (n_points, 2)
        max_lag : int
            Maximum lag time
            
        Returns
        -------
        numpy.ndarray
            MSD values
        """
        n_points = len(positions)
        msd_values = np.zeros(max_lag)
        
        for lag in range(1, max_lag + 1):
            # Calculate displacements for this lag
            disp = positions[lag:] - positions[:-lag]
            sq_disp = disp[:, 0]**2 + disp[:, 1]**2  # Faster than np.sum(disp**2, axis=1)
            msd_values[lag-1] = np.mean(sq_disp)
        
        return msd_values
    
    def calculate_msd(self, positions, max_lag=None):
        """
        Calculate mean squared displacement with confidence intervals using bootstrapping.
        
        Parameters
        ----------
        positions : numpy.ndarray
            Array of positions with shape (n_points, 2)
        max_lag : int, optional
            Maximum lag time, by default None (will use fraction of track length)
            
        Returns
        -------
        numpy.ndarray
            Lag times
        numpy.ndarray
            MSD values
        numpy.ndarray
            MSD standard errors
        """
        try:
            n_points = len(positions)
            
            # Determine maximum lag time
            if max_lag is None:
                max_lag = int(n_points * self.config['msd_max_lag_factor'])
            
            max_lag = min(max_lag, n_points - 1)  # Ensure max_lag is valid
            
            # Fast MSD calculation with Numba if enabled
            if self.config['use_numba']:
                msd_values = self._fast_msd_calculation(positions, max_lag)
            else:
                msd_values = np.zeros(max_lag)
                for lag in range(1, max_lag + 1):
                    disp = positions[lag:] - positions[:-lag]
                    sq_disp = np.sum(disp**2, axis=1)
                    msd_values[lag-1] = np.mean(sq_disp)
            
            # Bootstrap for error estimation
            msd_stderr = np.zeros(max_lag)
            for lag in range(1, max_lag + 1):
                disp = positions[lag:] - positions[:-lag]
                sq_disp = np.sum(disp**2, axis=1)
                
                bootstrap_msds = []
                for _ in range(self.config['bootstrap_samples']):
                    boot_sample = np.random.choice(sq_disp, size=len(sq_disp), replace=True)
                    bootstrap_msds.append(np.mean(boot_sample))
                
                msd_stderr[lag-1] = np.std(bootstrap_msds)
            
            # Create lag time array
            tau = np.arange(1, max_lag + 1) * self.dt
            
            return tau, msd_values, msd_stderr
        
        except Exception as e:
            logger.error(f"Error in MSD calculation: {str(e)}")
            raise
    
    def analyze_track(self, track_df):
        """
        Analyze a single track.
        
        Parameters
        ----------
        track_df : pandas.DataFrame
            DataFrame containing track data with X, Y coordinates
            
        Returns
        -------
        dict
            Dictionary of analysis results
        """
        try:
            # Extract positions and other data
            if not {'X', 'Y'}.issubset(track_df.columns):
                raise ValueError("Track DataFrame must contain X and Y columns")
            
            positions = track_df[['X', 'Y']].values
            
            # Get intensity if available
            intensity = track_df['max_hoechst_intensity'].mean() if 'max_hoechst_intensity' in track_df.columns else None
            
            # Calculate jumps (step sizes)
            jumps = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            
            # Calculate MSD
            tau, msd, msd_err = self.calculate_msd(positions)
            
            # Fit power law to MSD
            try:
                popt, pcov = curve_fit(self.power_law, tau, msd, p0=[0.1, 1.0])
                A_fit, alpha_fit = popt
                alpha_err = np.sqrt(pcov[1, 1])
            except RuntimeError:
                logger.warning(f"Power law fit failed, using fallback estimate")
                log_tau = np.log(tau)
                log_msd = np.log(msd)
                slope, intercept = np.polyfit(log_tau, log_msd, 1)
                A_fit = np.exp(intercept)
                alpha_fit = slope
                alpha_err = np.nan
            
            # Analyze diffusion mode
            diffusion_mode, slope = self.analyze_diffusion_modes(msd, tau)
            
            # Calculate rheological properties
            G_star = (self.k_B * self.T * (self.frequencies**alpha_fit)) / (np.pi * self.a * A_fit * gamma(1 + alpha_fit))
            G_prime = G_star * np.cos(np.pi * alpha_fit / 2)
            G_loss = G_star * np.sin(np.pi * alpha_fit / 2)
            viscosity = np.mean(G_loss / self.frequencies)
            
            # Calculate track statistics
            track_length = len(positions)
            track_duration = (track_length - 1) * self.dt
            mean_jump = np.mean(jumps)
            std_jump = np.std(jumps)
            
            # Diffusion coefficient (D = MSD/4t for 2D)
            D = msd[0] / (4 * tau[0])
            
            # Calculate spring constant
            spring_constant = 6 * np.pi * viscosity * self.a
            
            # Combine results
            results = {
                'track_id': track_df['TrackID'].iloc[0] if 'TrackID' in track_df.columns else None,
                'track_length': track_length,
                'track_duration': track_duration,
                'jumps': jumps,
                'mean_jump': mean_jump,
                'std_jump': std_jump,
                'intensity': intensity,
                'msd': msd,
                'msd_err': msd_err,
                'tau': tau,
                'diffusion_coefficient': D,
                'alpha': alpha_fit,
                'alpha_err': alpha_err,
                'diffusion_mode': diffusion_mode,
                'G_prime': G_prime,
                'G_loss': G_loss,
                'viscosity': viscosity,
                'spring_constant': spring_constant
            }
            
            return results
        
        except Exception as e:
            logger.error(f"Error in track analysis: {str(e)}")
            raise
    
    def analyze_dataset(self, df):
        """
        Analyze entire dataset with parallel processing.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing tracking data with TrackID, X, Y coordinates
            
        Returns
        -------
        list
            List of analysis results for each track
        """
        try:
            # Check input data
            if not {'TrackID', 'X', 'Y'}.issubset(df.columns):
                raise ValueError("DataFrame must contain TrackID, X, and Y columns")
            
            track_ids = df['TrackID'].unique()
            logger.info(f"Analyzing {len(track_ids)} tracks...")
            
            # Group data by track ID
            track_dfs = [df[df['TrackID'] == track_id] for track_id in track_ids]
            
            # Process tracks in parallel
            with ProcessPoolExecutor(max_workers=self.config['max_workers']) as executor:
                track_results = list(executor.map(self.analyze_track, track_dfs))
            
            logger.info(f"Analysis complete for {len(track_results)} tracks")
            
            return track_results
        
        except Exception as e:
            logger.error(f"Error in dataset analysis: {str(e)}")
            raise
    
    def calculate_ensemble_statistics(self, track_results):
        """
        Calculate ensemble statistics across all tracks.
        
        Parameters
        ----------
        track_results : list
            List of analysis results for each track
            
        Returns
        -------
        dict
            Dictionary of ensemble statistics
        """
        try:
            # Extract key metrics from all tracks
            alphas = [result['alpha'] for result in track_results if 'alpha' in result]
            diffusion_coefficients = [result['diffusion_coefficient'] for result in track_results if 'diffusion_coefficient' in result]
            viscosities = [result['viscosity'] for result in track_results if 'viscosity' in result]
            spring_constants = [result['spring_constant'] for result in track_results if 'spring_constant' in result]
            
            # Count diffusion modes
            diffusion_modes = [result['diffusion_mode'] for result in track_results if 'diffusion_mode' in result]
            mode_counts = {}
            for mode in diffusion_modes:
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            # Calculate ensemble MSD if tracks have similar lag times
            # This is a simplified approach - more sophisticated methods may be needed
            all_tau = track_results[0]['tau'] if track_results else np.array([])
            all_msd = np.zeros((len(track_results), len(all_tau))) if track_results else np.array([])
            
            for i, result in enumerate(track_results):
                if len(result['tau']) == len(all_tau):
                    all_msd[i, :] = result['msd']
            
            ensemble_msd = np.mean(all_msd, axis=0)
            ensemble_msd_err = np.std(all_msd, axis=0) / np.sqrt(all_msd.shape[0])
            
            # Return ensemble statistics
            return {
                'n_tracks': len(track_results),
                'alpha_mean': np.mean(alphas),
                'alpha_std': np.std(alphas),
                'diffusion_coefficient_mean': np.mean(diffusion_coefficients),
                'diffusion_coefficient_std': np.std(diffusion_coefficients),
                'viscosity_mean': np.mean(viscosities),
                'viscosity_std': np.std(viscosities),
                'spring_constant_mean': np.mean(spring_constants),
                'spring_constant_std': np.std(spring_constants),
                'diffusion_mode_counts': mode_counts,
                'ensemble_tau': all_tau,
                'ensemble_msd': ensemble_msd,
                'ensemble_msd_err': ensemble_msd_err
            }
        
        except Exception as e:
            logger.error(f"Error in ensemble statistics calculation: {str(e)}")
            raise
