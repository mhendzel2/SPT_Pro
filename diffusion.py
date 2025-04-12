
"""
Diffusion analysis module for SPT Analysis.

This module provides tools for diffusion coefficient calculation, MSD analysis,
and motion classification for single-particle tracking data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class DiffusionAnalyzer:
    """
    Analyzer for diffusion behavior and motion characteristics.
    
    Parameters
    ----------
    pixel_size : float, optional
        Pixel size in μm, by default 0.1
    frame_interval : float, optional
        Time between frames in seconds, by default 0.1
    """
    
    def __init__(self, pixel_size=0.1, frame_interval=0.1):
        self.pixel_size = pixel_size
        self.frame_interval = frame_interval
    
    def compute_msd(self, tracks_df, max_lag=20, min_track_length=10):
        """
        Compute mean squared displacement for each track.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        max_lag : int, optional
            Maximum time lag to consider, by default 20
        min_track_length : int, optional
            Minimum track length to analyze, by default 10
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with MSD values for each track and lag time
        """
        try:
            tracks_df = tracks_df.copy()
            
            # Convert coordinates to μm
            tracks_df['x_um'] = tracks_df['x'] * self.pixel_size
            tracks_df['y_um'] = tracks_df['y'] * self.pixel_size
            
            # Get unique track IDs
            track_ids = tracks_df['track_id'].unique()
            
            msd_data = []
            
            for track_id in track_ids:
                # Get track data
                track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
                
                # Skip tracks that are too short
                if len(track) < min_track_length:
                    continue
                
                # Compute MSD for different lag times
                for lag in range(1, min(max_lag + 1, len(track))):
                    # Get positions separated by lag
                    positions = track[['x_um', 'y_um', 'frame']].values
                    lagged_positions = positions[lag:]
                    original_positions = positions[:-lag]
                    
                    # Compute squared displacements
                    squared_displacements = np.sum(
                        (lagged_positions[:, :2] - original_positions[:, :2])**2,
                        axis=1
                    )
                    
                    # Compute time lag in seconds
                    time_lag = lag * self.frame_interval
                    
                    # Add to MSD data
                    msd_data.extend([{
                        'track_id': track_id,
                        'lag': lag,
                        'time_lag': time_lag,
                        'msd': sd
                    } for sd in squared_displacements])
            
            msd_df = pd.DataFrame(msd_data)
            
            # Compute ensemble average MSD per lag time
            ensemble_msd = msd_df.groupby('lag')['msd'].mean().reset_index()
            ensemble_msd['time_lag'] = ensemble_msd['lag'] * self.frame_interval
            
            return msd_df, ensemble_msd
        
        except Exception as e:
            logger.error(f"Error computing MSD: {str(e)}")
            raise
    
    def fit_diffusion_models(self, ensemble_msd, max_fit_points=10):
        """
        Fit different diffusion models to MSD curve.
        
        Parameters
        ----------
        ensemble_msd : pandas.DataFrame
            DataFrame with ensemble MSD values
        max_fit_points : int, optional
            Maximum number of points to use for fitting, by default 10
            
        Returns
        -------
        dict
            Dictionary with fitted model parameters
        """
        try:
            # Limit number of points for fitting
            fit_data = ensemble_msd.head(max_fit_points)
            
            # Extract time lags and MSD values
            time_lags = fit_data['time_lag'].values
            msd_values = fit_data['msd'].values
            
            # Model 1: Simple diffusion (MSD = 4Dt)
            def simple_diffusion(t, D):
                return 4 * D * t
            
            # Model 2: Anomalous diffusion (MSD = 4Dt^α)
            def anomalous_diffusion(t, D, alpha):
                return 4 * D * t**alpha
            
            # Model 3: Confined diffusion (MSD = L²[1-exp(-12Dt/L²)])
            def confined_diffusion(t, D, L):
                return L**2 * (1 - np.exp(-12 * D * t / L**2))
            
            # Model 4: Directed diffusion (MSD = 4Dt + (vt)²)
            def directed_diffusion(t, D, v):
                return 4 * D * t + (v * t)**2
            
            # Fit models
            from scipy.optimize import curve_fit
            
            results = {}
            
            # Simple diffusion fit
            try:
                popt, pcov = curve_fit(simple_diffusion, time_lags, msd_values)
                D_simple = popt[0]
                D_simple_err = np.sqrt(np.diag(pcov))[0]
                
                # Calculate R²
                residuals = msd_values - simple_diffusion(time_lags, D_simple)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((msd_values - np.mean(msd_values))**2)
                r_squared_simple = 1 - (ss_res / ss_tot)
                
                results['simple_diffusion'] = {
                    'D': D_simple,
                    'D_err': D_simple_err,
                    'r_squared': r_squared_simple
                }
            except Exception as e:
                logger.warning(f"Error fitting simple diffusion model: {str(e)}")
            
            # Anomalous diffusion fit
            try:
                popt, pcov = curve_fit(anomalous_diffusion, time_lags, msd_values, bounds=([0, 0], [np.inf, 2]))
                D_anom, alpha = popt
                D_anom_err, alpha_err = np.sqrt(np.diag(pcov))
                
                # Calculate R²
                residuals = msd_values - anomalous_diffusion(time_lags, D_anom, alpha)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((msd_values - np.mean(msd_values))**2)
                r_squared_anom = 1 - (ss_res / ss_tot)
                
                results['anomalous_diffusion'] = {
                    'D': D_anom,
                    'D_err': D_anom_err,
                    'alpha': alpha,
                    'alpha_err': alpha_err,
                    'r_squared': r_squared_anom
                }
            except Exception as e:
                logger.warning(f"Error fitting anomalous diffusion model: {str(e)}")
            
            # Confined diffusion fit
            try:
                popt, pcov = curve_fit(confined_diffusion, time_lags, msd_values, bounds=([0, 0], [np.inf, np.inf]))
                D_conf, L = popt
                D_conf_err, L_err = np.sqrt(np.diag(pcov))
                
                # Calculate R²
                residuals = msd_values - confined_diffusion(time_lags, D_conf, L)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((msd_values - np.mean(msd_values))**2)
                r_squared_conf = 1 - (ss_res / ss_tot)
                
                results['confined_diffusion'] = {
                    'D': D_conf,
                    'D_err': D_conf_err,
                    'L': L,
                    'L_err': L_err,
                    'r_squared': r_squared_conf
                }
            except Exception as e:
                logger.warning(f"Error fitting confined diffusion model: {str(e)}")
            
            # Directed diffusion fit
            try:
                popt, pcov = curve_fit(directed_diffusion, time_lags, msd_values)
                D_dir, v = popt
                D_dir_err, v_err = np.sqrt(np.diag(pcov))
                
                # Calculate R²
                residuals = msd_values - directed_diffusion(time_lags, D_dir, v)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((msd_values - np.mean(msd_values))**2)
                r_squared_dir = 1 - (ss_res / ss_tot)
                
                results['directed_diffusion'] = {
                    'D': D_dir,
                    'D_err': D_dir_err,
                    'v': v,
                    'v_err': v_err,
                    'r_squared': r_squared_dir
                }
            except Exception as e:
                logger.warning(f"Error fitting directed diffusion model: {str(e)}")
            
            # Find best model based on R²
            if results:
                best_model = max(results.items(), key=lambda x: x[1]['r_squared'])[0]
                results['best_model'] = best_model
            
            return results
        
        except Exception as e:
            logger.error(f"Error fitting diffusion models: {str(e)}")
            raise
    
    def classify_tracks(self, tracks_df, min_track_length=20, n_clusters=3):
        """
        Classify tracks based on diffusion properties.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        min_track_length : int, optional
            Minimum track length to consider, by default 20
        n_clusters : int, optional
            Number of clusters for classification, by default 3
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with track classifications
        """
        try:
            # Compute per-track diffusion parameters
            track_ids = tracks_df['track_id'].unique()
            
            track_features = []
            
            for track_id in track_ids:
                # Get track data
                track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
                
                # Skip tracks that are too short
                if len(track) < min_track_length:
                    continue
                
                # Convert coordinates to μm
                x = track['x'] * self.pixel_size
                y = track['y'] * self.pixel_size
                
                # Compute track length
                positions = np.vstack([x, y]).T
                displacements = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                track_length = np.sum(displacements)
                
                # Compute features for classification
                
                # 1. Instantaneous diffusion coefficient
                # Use first 4 points of MSD curve for linear fit
                msd_values = []
                
                for lag in range(1, min(5, len(track) - 1)):
                    dt = lag * self.frame_interval
                    dx = x.iloc[lag:].values - x.iloc[:-lag].values
                    dy = y.iloc[lag:].values - y.iloc[:-lag].values
                    msd = np.mean(dx**2 + dy**2)
                    msd_values.append((dt, msd))
                
                dt_values, msd_values = zip(*msd_values)
                slope, _, _, _, _ = stats.linregress(dt_values, msd_values)
                D_inst = slope / 4.0
                
                # 2. Asymmetry (ratio of eigenvalues of gyration tensor)
                if len(track) >= 4:
                    positions = np.vstack([x, y]).T
                    positions_centered = positions - np.mean(positions, axis=0)
                    gyration_tensor = np.cov(positions_centered.T)
                    eig_vals = np.linalg.eigvals(gyration_tensor)
                    asymmetry = np.min(eig_vals) / np.max(eig_vals) if np.max(eig_vals) > 0 else 1.0
                else:
                    asymmetry = 1.0
                
                # 3. Straightness (end-to-end distance / path length)
                if track_length > 0:
                    end_to_end = np.sqrt(
                        (x.iloc[-1] - x.iloc[0])**2 + 
                        (y.iloc[-1] - y.iloc[0])**2
                    )
                    straightness = end_to_end / track_length
                else:
                    straightness = 0.0
                
                # 4. Anomalous exponent (α) from MSD curve
                if len(msd_values) >= 3:
                    log_dt = np.log(dt_values)
                    log_msd = np.log(msd_values)
                    slope, _, _, _, _ = stats.linregress(log_dt, log_msd)
                    alpha = slope
                else:
                    alpha = 1.0  # Default to normal diffusion
                
                # Add features
                track_features.append({
                    'track_id': track_id,
                    'D': D_inst,
                    'asymmetry': asymmetry,
                    'straightness': straightness,
                    'alpha': alpha,
                    'track_length': len(track)
                })
            
            # Create DataFrame with features
            features_df = pd.DataFrame(track_features)
            
            if len(features_df) < n_clusters:
                logger.warning(f"Not enough tracks ({len(features_df)}) for {n_clusters} clusters")
                return None
            
            # Normalize features for clustering
            from sklearn.preprocessing import StandardScaler
            
            feature_columns = ['D', 'asymmetry', 'straightness', 'alpha']
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_df[feature_columns])
            
            # Cluster tracks using Gaussian Mixture Model
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            cluster_labels = gmm.fit_predict(features_scaled)
            
            # Add cluster labels to features DataFrame
            features_df['cluster'] = cluster_labels
            
            # Compute cluster characteristics
            cluster_characteristics = []
            
            for cluster_id in range(n_clusters):
                cluster_tracks = features_df[features_df['cluster'] == cluster_id]
                
                # Skip empty clusters
                if len(cluster_tracks) == 0:
                    continue
                
                # Compute mean features
                mean_D = cluster_tracks['D'].mean()
                mean_asymmetry = cluster_tracks['asymmetry'].mean()
                mean_straightness = cluster_tracks['straightness'].mean()
                mean_alpha = cluster_tracks['alpha'].mean()
                
                # Classify based on features
                if mean_alpha > 1.1 and mean_straightness > 0.6:
                    motion_type = "Directed"
                elif mean_alpha < 0.9 and mean_asymmetry < 0.4:
                    motion_type = "Confined"
                elif 0.9 <= mean_alpha <= 1.1:
                    motion_type = "Brownian"
                else:
                    motion_type = "Mixed"
                
                cluster_characteristics.append({
                    'cluster': cluster_id,
                    'motion_type': motion_type,
                    'mean_D': mean_D,
                    'mean_asymmetry': mean_asymmetry,
                    'mean_straightness': mean_straightness,
                    'mean_alpha': mean_alpha,
                    'track_count': len(cluster_tracks)
                })
            
            # Create DataFrame with cluster characteristics
            cluster_df = pd.DataFrame(cluster_characteristics)
            
            # Add motion type to features DataFrame
            motion_map = {row['cluster']: row['motion_type'] for _, row in cluster_df.iterrows()}
            features_df['motion_type'] = features_df['cluster'].map(motion_map)
            
            return features_df, cluster_df
        
        except Exception as e:
            logger.error(f"Error classifying tracks: {str(e)}")
            raise
    
    def compute_diffusion_coefficient(self, tracks_df, method="individual", min_track_length=10, max_lag=4):
        """
        Compute diffusion coefficients for tracks.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        method : str, optional
            Method for computing diffusion coefficient ('individual', 'ensemble'), by default "individual"
        min_track_length : int, optional
            Minimum track length to consider, by default 10
        max_lag : int, optional
            Maximum time lag to use for fit, by default 4
            
        Returns
        -------
        dict or pandas.DataFrame
            Dictionary with ensemble diffusion coefficient or DataFrame with individual coefficients
        """
        try:
            # Convert coordinates to μm
            tracks_df = tracks_df.copy()
            tracks_df['x_um'] = tracks_df['x'] * self.pixel_size
            tracks_df['y_um'] = tracks_df['y'] * self.pixel_size
            
            if method == "ensemble":
                # Compute ensemble MSD
                _, ensemble_msd = self.compute_msd(
                    tracks_df, 
                    max_lag=max_lag, 
                    min_track_length=min_track_length
                )
                
                # Linear fit of the first few points
                time_lags = ensemble_msd['time_lag'].values[:max_lag]
                msd_values = ensemble_msd['msd'].values[:max_lag]
                
                # MSD = 4Dt
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_lags, msd_values)
                
                D = slope / 4.0
                D_err = std_err / 4.0
                
                return {
                    'D': D,Copy
                    'D_err': D_err,
                    'r_squared': r_value**2,
                    'intercept': intercept
                }
            
            elif method == "individual":
                # Compute diffusion coefficient for each track
                track_ids = tracks_df['track_id'].unique()
                
                diffusion_data = []
                
                for track_id in track_ids:
                    # Get track data
                    track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
                    
                    # Skip tracks that are too short
                    if len(track) < min_track_length:
                        continue
                    
                    # Compute MSD for different lag times
                    lag_times = []
                    msd_values = []
                    
                    for lag in range(1, min(max_lag + 1, len(track))):
                        # Get positions separated by lag
                        positions = track[['x_um', 'y_um', 'frame']].values
                        lagged_positions = positions[lag:]
                        original_positions = positions[:-lag]
                        
                        # Compute squared displacements
                        squared_displacements = np.sum(
                            (lagged_positions[:, :2] - original_positions[:, :2])**2,
                            axis=1
                        )
                        
                        # Compute time lag in seconds
                        time_lag = lag * self.frame_interval
                        
                        # Compute mean squared displacement
                        msd = np.mean(squared_displacements)
                        
                        lag_times.append(time_lag)
                        msd_values.append(msd)
                    
                    # Fit MSD curve
                    if len(lag_times) > 1:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(lag_times, msd_values)
                        
                        # MSD = 4Dt
                        D = slope / 4.0
                        D_err = std_err / 4.0
                        
                        # Add to diffusion data
                        diffusion_data.append({
                            'track_id': track_id,
                            'D': D,
                            'D_err': D_err,
                            'r_squared': r_value**2,
                            'intercept': intercept,
                            'track_length': len(track)
                        })
                
                return pd.DataFrame(diffusion_data)
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        except Exception as e:
            logger.error(f"Error computing diffusion coefficient: {str(e)}")
            raise
