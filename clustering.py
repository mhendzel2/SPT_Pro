
"""
Clustering analysis module for SPT Analysis.

This module provides tools for identifying particle clusters, hotspots,
and other spatial patterns in single-particle tracking data.
"""

import numpy as np
import pandas as pd
from scipy import spatial, stats
from sklearn.cluster import DBSCAN, KMeans
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """
    Analyzer for spatial clustering of particle trajectories.
    
    Parameters
    ----------
    pixel_size : float, optional
        Pixel size in μm, by default 0.1
    """
    
    def __init__(self, pixel_size=0.1):
        self.pixel_size = pixel_size
    
    def dbscan_clustering(self, positions, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering on positions.
        
        Parameters
        ----------
        positions : numpy.ndarray
            Array of positions with shape (n_points, 2)
        eps : float, optional
            DBSCAN epsilon parameter, by default 0.5
        min_samples : int, optional
            DBSCAN min_samples parameter, by default 5
            
        Returns
        -------
        numpy.ndarray
            Array of cluster labels
        """
        try:
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(positions)
            
            return cluster_labels
        
        except Exception as e:
            logger.error(f"Error in DBSCAN clustering: {str(e)}")
            raise
    
    def analyze_clusters(self, tracks_df, eps=0.5, min_samples=5, time_window=None):
        """
        Identify and analyze clusters in track data.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        eps : float, optional
            DBSCAN epsilon parameter in μm, by default 0.5
        min_samples : int, optional
            DBSCAN min_samples parameter, by default 5
        time_window : tuple, optional
            Time window (start_frame, end_frame) to analyze, by default None
            
        Returns
        -------
        tuple
            Tuple containing cluster DataFrame and clustered positions
        """
        try:
            # Convert coordinates to μm
            tracks_df = tracks_df.copy()
            tracks_df['x_um'] = tracks_df['x'] * self.pixel_size
            tracks_df['y_um'] = tracks_df['y'] * self.pixel_size
            
            # Filter by time window if provided
            if time_window is not None:
                start_frame, end_frame = time_window
                tracks_df = tracks_df[(tracks_df['frame'] >= start_frame) & 
                                      (tracks_df['frame'] <= end_frame)]
            
            # Get positions
            positions = tracks_df[['x_um', 'y_um']].values
            
            # Perform clustering
            cluster_labels = self.dbscan_clustering(positions, eps=eps, min_samples=min_samples)
            
            # Add cluster labels to DataFrame
            clustered_df = tracks_df.copy()
            clustered_df['cluster'] = cluster_labels
            
            # Analyze clusters
            cluster_stats = []
            
            for cluster_id in np.unique(cluster_labels):
                # Skip noise points (cluster_id = -1)
                if cluster_id == -1:
                    continue
                
                # Get points in this cluster
                cluster_points = positions[cluster_labels == cluster_id]
                
                # Compute cluster properties
                centroid = np.mean(cluster_points, axis=0)
                radius = np.sqrt(np.max(np.sum((cluster_points - centroid)**2, axis=1)))
                density = len(cluster_points) / (np.pi * radius**2) if radius > 0 else 0
                
                # Compute convex hull
                from scipy.spatial import ConvexHull
                
                if len(cluster_points) >= 3:
                    hull = ConvexHull(cluster_points)
                    hull_area = hull.volume  # In 2D, volume is actually area
                    hull_perimeter = np.sum(hull.simplex_volume)
                else:
                    hull_area = 0
                    hull_perimeter = 0
                
                # Add to cluster statistics
                cluster_stats.append({
                    'cluster_id': cluster_id,
                    'n_points': len(cluster_points),
                    'centroid_x': centroid[0],
                    'centroid_y': centroid[1],
                    'radius': radius,
                    'density': density,
                    'area': hull_area,
                    'perimeter': hull_perimeter
                })
            
            # Create cluster stats DataFrame
            cluster_df = pd.DataFrame(cluster_stats)
            
            return cluster_df, clustered_df
        
        except Exception as e:
            logger.error(f"Error analyzing clusters: {str(e)}")
            raise
    
    def compute_ripley_k(self, tracks_df, r_max=10.0, n_points=20, time_window=None):
        """
        Compute Ripley's K function for spatial point pattern analysis.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        r_max : float, optional
            Maximum radius to evaluate K function, by default 10.0
        n_points : int, optional
            Number of evaluation points, by default 20
        time_window : tuple, optional
            Time window (start_frame, end_frame) to analyze, by default None
            
        Returns
        -------
        dict
            Dictionary with Ripley's K function results
        """
        try:
            # Convert coordinates to μm
            tracks_df = tracks_df.copy()
            tracks_df['x_um'] = tracks_df['x'] * self.pixel_size
            tracks_df['y_um'] = tracks_df['y'] * self.pixel_size
            
            # Filter by time window if provided
            if time_window is not None:
                start_frame, end_frame = time_window
                tracks_df = tracks_df[(tracks_df['frame'] >= start_frame) & 
                                      (tracks_df['frame'] <= end_frame)]
            
            # Get positions
            positions = tracks_df[['x_um', 'y_um']].values
            
            # Generate evaluation radii
            radii = np.linspace(0.1, r_max, n_points)
            
            # Compute area of analysis region
            x_min, y_min = positions.min(axis=0)
            x_max, y_max = positions.max(axis=0)
            area = (x_max - x_min) * (y_max - y_min)
            
            # Compute pairwise distances
            distances = spatial.distance.pdist(positions)
            
            # Compute Ripley's K function
            k_values = []
            l_values = []  # Ripley's L function: L(r) = sqrt(K(r)/π)
            
            for r in radii:
                # Count pairs with distance < r
                count = np.sum(distances < r)
                
                # Compute K(r)
                k_r = area * count / (positions.shape[0] * (positions.shape[0] - 1))
                k_values.append(k_r)
                
                # Compute L(r)
                l_r = np.sqrt(k_r / np.pi) - r
                l_values.append(l_r)
            
            return {
                'r': radii,
                'K': np.array(k_values),
                'L': np.array(l_values)
            }
        
        except Exception as e:
            logger.error(f"Error computing Ripley's K function: {str(e)}")
            raise
    
    def identify_hotspots(self, tracks_df, bandwidth=1.0, threshold=0.7, grid_size=100):
        """
        Identify hotspots using kernel density estimation.
        
        Parameters
        ----------
        tracks_df : pandas.DataFrame
            DataFrame with track data
        bandwidth : float, optional
            Bandwidth for kernel density estimation, by default 1.0
        threshold : float, optional
            Threshold for hotspot identification, by default 0.7
        grid_size : int, optional
            Grid size for density evaluation, by default 100
            
        Returns
        -------
        tuple
            Tuple containing hotspot DataFrame and density grid
        """
        try:
            from scipy.stats import gaussian_kde
            
            # Convert coordinates to μm
            tracks_df = tracks_df.copy()
            tracks_df['x_um'] = tracks_df['x'] * self.pixel_size
            tracks_df['y_um'] = tracks_df['y'] * self.pixel_size
            
            # Get positions
            positions = tracks_df[['x_um', 'y_um']].values
            
            # Create grid for density evaluation
            x_min, y_min = positions.min(axis=0)
            x_max, y_max = positions.max(axis=0)
            
            # Add margin
            margin = 0.1 * max(x_max - x_min, y_max - y_min)
            x_min -= margin
            x_max += margin
            y_min -= margin
            y_max += margin
            
            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)
            X, Y = np.meshgrid(x_grid, y_grid)
            grid_positions = np.vstack([X.ravel(), Y.ravel()]).T
            
            # Compute density
            if len(positions) > 1:
                kde = gaussian_kde(positions.T, bw_method=bandwidth)
                density = kde(grid_positions.T).reshape(grid_size, grid_size)
                
                # Normalize density
                density = density / density.max()
            else:
                density = np.zeros((grid_size, grid_size))
            
            # Identify hotspots
            hotspot_mask = density > threshold
            
            # Label hotspots
            from scipy.ndimage import label
            
            labeled_hotspots, num_hotspots = label(hotspot_mask)
            
            # Analyze hotspots
            hotspot_stats = []
            
            for hotspot_id in range(1, num_hotspots + 1):
                # Get grid positions for this hotspot
                hotspot_indices = np.where(labeled_hotspots.ravel() == hotspot_id)[0]
                hotspot_positions = grid_positions[hotspot_indices]
                
                # Compute hotspot properties
                centroid = np.mean(hotspot_positions, axis=0)
                
                # Find tracks in this hotspot
                track_in_hotspot = []
                
                for _, row in tracks_df.iterrows():
                    pos = np.array([row['x_um'], row['y_um']])
                    
                    # Find closest grid point
                    dist = np.sqrt(np.sum((grid_positions - pos)**2, axis=1))
                    closest_idx = np.argmin(dist)
                    
                    if labeled_hotspots.ravel()[closest_idx] == hotspot_id:
                        track_in_hotspot.append(row['track_id'])
                
                # Compute hotspot area
                pixel_area = (x_max - x_min) * (y_max - y_min) / (grid_size * grid_size)
                area = len(hotspot_indices) * pixel_area
                
                # Average density
                avg_density = np.mean(density.ravel()[hotspot_indices])
                
                # Add to hotspot statistics
                hotspot_stats.append({
                    'hotspot_id': hotspot_id,
                    'centroid_x': centroid[0],
                    'centroid_y': centroid[1],
                    'area': area,
                    'avg_density': avg_density,
                    'n_tracks': len(set(track_in_hotspot))
                })
            
            # Create hotspot stats DataFrame
            hotspot_df = pd.DataFrame(hotspot_stats)
            
            # Create density grid dataframe for visualization
            grid_df = pd.DataFrame({
                'x': X.ravel(),
                'y': Y.ravel(),
                'density': density.ravel(),
                'hotspot': labeled_hotspots.ravel()
            })
            
            return hotspot_df, grid_df
        
        except Exception as e:
            logger.error(f"Error identifying hotspots: {str(e)}")
            raise
