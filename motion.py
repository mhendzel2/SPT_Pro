
"""
Motion analysis module for SPT Analysis.

This module provides tools for analyzing the motion characteristics of tracked particles,
including velocity, directionality, and confinement.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class MotionAnalyzer:
    """
    Analyzer for particle motion characteristics.
    
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
    
    def compute_velocities(self, tracks_df, smooth=True, window=3):
        """
        Compute instantaneous and average velocities for each track.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        smooth : bool, optional
            Whether to apply smoothing to velocity calculations, by default True
        window : int, optional
            Window size for smoothing, by default 3
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with velocity data
        """
        try:
            tracks_df = tracks_df.copy()
            
            # Convert coordinates to μm
            tracks_df['x_um'] = tracks_df['x'] * self.pixel_size
            tracks_df['y_um'] = tracks_df['y'] * self.pixel_size
            
            # Create result DataFrame
            result_data = []
            
            # Process each track
            for track_id, track in tracks_df.groupby('track_id'):
                # Sort by frame
                track = track.sort_values('frame')
                
                # Skip tracks that are too short
                if len(track) < 2:
                    continue
                
                # Get positions and frames
                positions = np.vstack([track['x_um'], track['y_um']]).T
                frames = track['frame'].values
                
                # Optionally smooth positions
                if smooth and len(positions) >= window:
                    # Apply rolling average
                    positions_smoothed = np.zeros_like(positions)
                    
                    for i in range(len(positions)):
                        start_idx = max(0, i - window // 2)
                        end_idx = min(len(positions), i + window // 2 + 1)
                        positions_smoothed[i] = np.mean(positions[start_idx:end_idx], axis=0)
                    
                    positions = positions_smoothed
                
                # Compute displacements
                displacements = np.diff(positions, axis=0)
                
                # Compute time intervals
                time_intervals = np.diff(frames) * self.frame_interval
                
                # Compute instantaneous velocities
                velocities = displacements / time_intervals[:, np.newaxis]
                
                # Compute speed (magnitude of velocity)
                speeds = np.sqrt(np.sum(velocities**2, axis=1))
                
                # Compute average velocity and speed
                avg_velocity = np.mean(velocities, axis=0)
                avg_speed = np.mean(speeds)
                
                # Compute direction changes
                directions = np.arctan2(velocities[:, 1], velocities[:, 0])
                direction_changes = np.diff(directions)
                
                # Wrap direction changes to [-π, π]
                direction_changes = np.mod(direction_changes + np.pi, 2 * np.pi) - np.pi
                
                # Compute mean and std of direction changes
                mean_direction_change = np.mean(np.abs(direction_changes)) if len(direction_changes) > 0 else 0
                std_direction_change = np.std(np.abs(direction_changes)) if len(direction_changes) > 0 else 0
                
                # Add data to result
                result_data.append({
                    'track_id': track_id,
                    'n_points': len(track),
                    'total_distance': np.sum(np.sqrt(np.sum(displacements**2, axis=1))),
                    'net_displacement': np.sqrt(np.sum((positions[-1] - positions[0])**2)),
                    'avg_speed': avg_speed,
                    'avg_velocity_x': avg_velocity[0],
                    'avg_velocity_y': avg_velocity[1],
                    'avg_velocity_magnitude': np.sqrt(np.sum(avg_velocity**2)),
                    'max_speed': np.max(speeds),
                    'mean_direction_change': mean_direction_change,
                    'std_direction_change': std_direction_change
                })
            
            return pd.DataFrame(result_data)
        
        except Exception as e:
            logger.error(f"Error computing velocities: {str(e)}")
            raise
    
    def compute_confinement_ratio(self, tracks_df, window_size=5):
        """
        Compute confinement ratio for each track.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        window_size : int, optional
            Window size for computing confinement ratio, by default 5
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with confinement ratio data
        """
        try:
            tracks_df = tracks_df.copy()
            
            # Convert coordinates to μm
            tracks_df['x_um'] = tracks_df['x'] * self.pixel_size
            tracks_df['y_um'] = tracks_df['y'] * self.pixel_size
            
            # Create result DataFrame
            result_data = []
            
            # Process each track
            for track_id, track in tracks_df.groupby('track_id'):
                # Sort by frame
                track = track.sort_values('frame')
                
                # Skip tracks that are too short
                if len(track) < window_size + 1:
                    continue
                
                # Get positions
                positions = np.vstack([track['x_um'], track['y_um']]).T
                frames = track['frame'].values
                
                # Compute confinement ratio for each window
                confinement_data = []
                
                for i in range(len(positions) - window_size + 1):
                    window_positions = positions[i:i+window_size]
                    
                    # Compute net displacement
                    net_displacement = np.sqrt(np.sum((window_positions[-1] - window_positions[0])**2))
                    
                    # Compute total path length
                    path_length = np.sum(
                        np.sqrt(np.sum(np.diff(window_positions, axis=0)**2, axis=1))
                    )
                    
                    # Compute confinement ratio
                    conf_ratio = net_displacement / path_length if path_length > 0 else 1.0
                    
                    confinement_data.append({
                        'track_id': track_id,
                        'frame': frames[i + window_size - 1],
                        'window_start_frame': frames[i],
                        'window_end_frame': frames[i + window_size - 1],
                        'confinement_ratio': conf_ratio
                    })
                
                # Add to result data
                result_data.extend(confinement_data)
            
            return pd.DataFrame(result_data)
        
        except Exception as e:
            logger.error(f"Error computing confinement ratio: {str(e)}")
            raise
    
    def compute_turning_angles(self, tracks_df, min_displacement=0.1):
        """
        Compute turning angles between consecutive displacements.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        min_displacement : float, optional
            Minimum displacement to consider, by default 0.1
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with turning angle data
        """
        try:
            tracks_df = tracks_df.copy()
            
            # Convert coordinates to μm
            tracks_df['x_um'] = tracks_df['x'] * self.pixel_size
            tracks_df['y_um'] = tracks_df['y'] * self.pixel_size
            
            # Create result DataFrame
            result_data = []
            
            # Process each track
            for track_id, track in tracks_df.groupby('track_id'):
                # Sort by frame
                track = track.sort_values('frame')
                
                # Skip tracks that are too short
                if len(track) < 3:
                    continue
                
                # Get positions and frames
                positions = np.vstack([track['x_um'], track['y_um']]).T
                frames = track['frame'].values
                
                # Compute displacements between consecutive positions
                displacements = np.diff(positions, axis=0)
                displacement_magnitudes = np.sqrt(np.sum(displacements**2, axis=1))
                
                # Skip if displacements are too small
                if np.any(displacement_magnitudes < min_displacement):
                    valid_indices = displacement_magnitudes >= min_displacement
                    displacements = displacements[valid_indices]
                    displacement_magnitudes = displacement_magnitudes[valid_indices]
                    
                    if len(displacements) < 2:
                        continue
                
                # Compute turning angles
                turning_angles = []
                
                for i in range(len(displacements) - 1):
                    # Normalize displacements
                    v1 = displacements[i] / displacement_magnitudes[i]
                    v2 = displacements[i+1] / displacement_magnitudes[i+1]
                    
                    # Compute dot product and angle
                    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    
                    # Determine sign of angle using cross product
                    cross_product = np.cross(v1, v2)
                    angle = angle if cross_product >= 0 else -angle
                    
                    turning_angles.append(angle)
                
                # Add to result data
                for i, angle in enumerate(turning_angles):
                    result_data.append({
                        'track_id': track_id,
                        'frame': frames[i+2],  # Frame of the third point
                        'turning_angle': angle,
                        'turning_angle_degrees': angle * 180 / np.pi
                    })
            
            return pd.DataFrame(result_data)
        
        except Exception as e:
            logger.error(f"Error computing turning angles: {str(e)}")
            raise
    
    def analyze_track_shape(self, tracks_df, min_track_length=10):
        """
        Analyze the shape of tracks using principal component analysis.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        min_track_length : int, optional
            Minimum track length to consider, by default 10
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with track shape analysis
        """
        try:
            tracks_df = tracks_df.copy()
            
            # Convert coordinates to μm
            tracks_df['x_um'] = tracks_df['x'] * self.pixel_size
            tracks_df['y_um'] = tracks_df['y'] * self.pixel_size
            
            # Create result DataFrame
            result_data = []
            
            # Process each track
            for track_id, track in tracks_df.groupby('track_id'):
                # Skip tracks that are too short
                if len(track) < min_track_length:
                    continue
                
                # Get positions
                positions = np.vstack([track['x_um'], track['y_um']]).T
                
                # Center positions
                positions_centered = positions - np.mean(positions, axis=0)
                
                # Compute covariance matrix (gyration tensor)
                cov_matrix = np.cov(positions_centered.T)
                
                # Compute eigenvalues and eigenvectors
                eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
                
                # Sort eigenvalues and eigenvectors
                idx = eig_vals.argsort()[::-1]
                eig_vals = eig_vals[idx]
                eig_vecs = eig_vecs[:, idx]
                
                # Compute shape metrics
                
                # Radius of gyration
                rg = np.sqrt(np.sum(eig_vals))
                
                # Asphericity (0 for circle, 1 for line)
                asphericity = np.abs(eig_vals[0] - eig_vals[1]) / (eig_vals[0] + eig_vals[1])
                
                # Major and minor axes lengths
                major_axis = 2 * np.sqrt(3 * eig_vals[0])
                minor_axis = 2 * np.sqrt(3 * eig_vals[1])
                
                # Orientation angle
                orientation = np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0])
                
                # Add to result data
                result_data.append({
                    'track_id': track_id,
                    'n_points': len(track),
                    'rg': rg,
                    'asphericity': asphericity,
                    'major_axis': major_axis,
                    'minor_axis': minor_axis,
                    'axis_ratio': minor_axis / major_axis if major_axis > 0 else 1.0,
                    'orientation': orientation,
                    'orientation_degrees': orientation * 180 / np.pi
                })
            
            return pd.DataFrame(result_data)
        
        except Exception as e:
            logger.error(f"Error analyzing track shape: {str(e)}")
            raise
