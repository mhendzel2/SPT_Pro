
"""
Clustering visualization module for SPT Analysis.

This module provides functions for visualizing spatial clustering results,
including Ripley's K function, density-based clusters, and hotspots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


def plot_spatial_clusters(tracks_df, clustered_df, cluster_df, background=None,
                         cmap='tab10', ax=None, figsize=(10, 8), 
                         pixel_size=1.0, title=None):
    """
    Plot spatial clusters of track positions.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    clustered_df : pandas.DataFrame
        DataFrame with cluster assignments
    cluster_df : pandas.DataFrame
        DataFrame with cluster statistics
    background : numpy.ndarray, optional
        Background image, by default None
    cmap : str, optional
        Colormap to use, by default 'tab10'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 8)
    pixel_size : float, optional
        Pixel size for proper scaling, by default 1.0
    title : str, optional
        Plot title, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Plot background image if provided
        if background is not None:
            ax.imshow(background, cmap='gray', alpha=0.5)
        
        # Get cluster IDs
        cluster_ids = cluster_df['cluster_id'].unique()
        
        # Create colormap
        cmap_obj = plt.get_cmap(cmap, len(cluster_ids))
        
        # Plot noise points
        noise_points = clustered_df[clustered_df['cluster'] == -1]
        if not noise_points.empty:
            noise_x = noise_points['x'] * pixel_size
            noise_y = noise_points['y'] * pixel_size
            ax.scatter(noise_x, noise_y, c='gray', s=10, alpha=0.5, label='Noise')
        
        # Plot clusters
        for i, cluster_id in enumerate(cluster_ids):
            # Get cluster data
            cluster_points = clustered_df[clustered_df['cluster'] == cluster_id]
            
            # Get cluster statistics
            cluster_stats = cluster_df[cluster_df['cluster_id'] == cluster_id].iloc[0]
            
            # Get cluster coordinates
            x = cluster_points['x'] * pixel_size
            y = cluster_points['y'] * pixel_size
            
            # Get color
            color = cmap_obj(i % len(cluster_ids))
            
            # Plot cluster points
            ax.scatter(x, y, c=[color], s=20, alpha=0.7, label=f'Cluster {cluster_id}')
            
            # Draw cluster circle
            centroid_x = cluster_stats['centroid_x']
            centroid_y = cluster_stats['centroid_y']
            radius = cluster_stats['radius']
            
            circle = plt.Circle((centroid_x, centroid_y), radius, fill=False, 
                               edgecolor=color, linestyle='--', linewidth=1.5)
            ax.add_patch(circle)
            
            # Add cluster ID label
            ax.text(centroid_x, centroid_y, str(cluster_id), 
                   ha='center', va='center', fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Add legend with limited items
        if len(cluster_ids) <= 10:
            ax.legend(loc='upper right')
        
        # Set labels and title
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        if title:
            ax.set_title(title)
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting spatial clusters: {str(e)}")
        raise


def plot_cluster_stats(cluster_df, ax=None, figsize=(10, 6), title=None):
    """
    Plot statistics for spatial clusters.
    
    Parameters
    ----------
    cluster_df : pandas.DataFrame
        DataFrame with cluster statistics
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 6)
    title : str, optional
        Plot title, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Get cluster IDs
        cluster_ids = cluster_df['cluster_id'].values
        
        # Plot cluster properties
        width = 0.3
        x = np.arange(len(cluster_ids))
        
        # Plot number of points
        ax.bar(x - width, cluster_df['n_points'], width, label='Points')
        
        # Plot density (scaled for visibility)
        density_scaled = cluster_df['density'] * 10
        ax.bar(x, density_scaled, width, label='Density (×10)')
        
        # Plot radius
        ax.bar(x + width, cluster_df['radius'], width, label='Radius (μm)')
        
        # Set x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels([f'Cluster {cid}' for cid in cluster_ids])
        ax.set_xlabel('Cluster')
        
        # Add legend
        ax.legend()
        
        # Set title
        if title:
            ax.set_title(title)
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting cluster statistics: {str(e)}")
        raise


def plot_ripley_k(ripley_k_result, reference_line=True, ax=None, figsize=(10, 6), title=None):
    """
    Plot Ripley's K and L functions.
    
    Parameters
    ----------
    ripley_k_result : dict
        Dictionary with Ripley's K function results
    reference_line : bool, optional
        Whether to plot CSR reference line, by default True
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 6)
    title : str, optional
        Plot title, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Get data
        r = ripley_k_result['r']
        K = ripley_k_result['K']
        L = ripley_k_result['L']
        
        # Plot Ripley's K function
        ax.plot(r, K, 'b-', linewidth=2, label="K(r)")
        
        # Plot reference line for complete spatial randomness (CSR)
        if reference_line:
            # For CSR, K(r) = πr²
            K_csr = np.pi * r**2
            ax.plot(r, K_csr, 'k--', linewidth=1, label="CSR")
        
        # Add second axis for L function
        ax2 = ax.twinx()
        ax2.plot(r, L, 'r-', linewidth=2, label="L(r)")
        
        # Plot reference line for L function
        if reference_line:
            # For CSR, L(r) = 0
            ax2.plot(r, np.zeros_like(r), 'k:', linewidth=1)
        
        # Set labels
        ax.set_xlabel('r (μm)')
        ax.set_ylabel("Ripley's K(r)")
        ax2.set_ylabel("Ripley's L(r)")
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Set title
        if title:
            ax.set_title(title)
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting Ripley's K function: {str(e)}")
        raise


def plot_density_heatmap(grid_df, tracks_df=None, hotspot_df=None, 
                        ax=None, figsize=(10, 8), cmap='hot', 
                        alpha=0.7, title=None):
    """
    Plot density heatmap with optional hotspots and tracks.
    
    Parameters
    ----------
    grid_df : pandas.DataFrame
        DataFrame with density grid data
    tracks_df : pandas.DataFrame, optional
        DataFrame with track data, by default None
    hotspot_df : pandas.DataFrame, optional
        DataFrame with hotspot data, by default None
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 8)
    cmap : str, optional
        Colormap to use, by default 'hot'
    alpha : float, optional
        Heatmap opacity, by default 0.7
    title : str, optional
        Plot title, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Get grid data
        x = grid_df['x'].values
        y = grid_df['y'].values
        density = grid_df['density'].values
        
        # Determine grid size
        from math import sqrt
        grid_size = int(sqrt(len(x)))
        
        # Reshape data
        X = x.reshape(grid_size, grid_size)
        Y = y.reshape(grid_size, grid_size)
        Z = density.reshape(grid_size, grid_size)
        
        # Plot density heatmap
        im = ax.pcolormesh(X, Y, Z, cmap=cmap, alpha=alpha, shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density')
        
        # Plot tracks if provided
        if tracks_df is not None:
            for track_id, track in tracks_df.groupby('track_id'):
                # Sort track by frame
                track = track.sort_values('frame')
                
                # Get coordinates
                x = track['x_um'] if 'x_um' in track.columns else track['x']
                y = track['y_um'] if 'y_um' in track.columns else track['y']
                
                # Plot track
                ax.plot(x, y, 'w-', linewidth=0.5, alpha=0.5)
                ax.plot(x.iloc[0], y.iloc[0], 'wo', markersize=3, alpha=0.7)
                ax.plot(x.iloc[-1], y.iloc[-1], 'ws', markersize=3, alpha=0.7)
        
        # Plot hotspots if provided
        if hotspot_df is not None:
            # Create colormap for hotspots
            hotspot_cmap = plt.get_cmap('tab10', len(hotspot_df))
            
            # Plot each hotspot
            for i, (_, hotspot) in enumerate(hotspot_df.iterrows()):
                # Get hotspot properties
                centroid_x = hotspot['centroid_x']
                centroid_y = hotspot['centroid_y']
                
                # Create a marker for the hotspot
                circle = plt.Circle((centroid_x, centroid_y), hotspot['area']**0.5/2, 
                                  fill=False, edgecolor=hotspot_cmap(i), 
                                  linestyle='-', linewidth=2)
                ax.add_patch(circle)
                
                # Add hotspot ID label
                ax.text(centroid_x, centroid_y, str(hotspot['hotspot_id']), 
                       ha='center', va='center', fontsize=10, 
                       color='white', fontweight='bold')
        
        # Set labels and title
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        if title:
            ax.set_title(title)
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting density heatmap: {str(e)}")
        raise


def plot_cluster_time_evolution(tracks_df, frame_groups=10, eps=0.5, min_samples=5,
                              ax=None, figsize=(12, 10), cmap='tab10', 
                              pixel_size=1.0, title=None):
    """
    Plot time evolution of spatial clusters.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    frame_groups : int, optional
        Number of time groups to analyze, by default 10
    eps : float, optional
        DBSCAN epsilon parameter, by default 0.5
    min_samples : int, optional
        DBSCAN min_samples parameter, by default 5
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (12, 10)
    cmap : str, optional
        Colormap to use, by default 'tab10'
    pixel_size : float, optional
        Pixel size for proper scaling, by default 1.0
    title : str, optional
        Plot title, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        from sklearn.cluster import DBSCAN
        
        # Convert coordinates to μm
        tracks_df = tracks_df.copy()
        tracks_df['x_um'] = tracks_df['x'] * pixel_size
        tracks_df['y_um'] = tracks_df['y'] * pixel_size
        
        # Create figure if needed
        if ax is None:
            # Create a grid of subplots
            n_cols = min(3, frame_groups)
            n_rows = (frame_groups + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten()
        else:
            axes = [ax]
            frame_groups = 1
        
        # Get frame range
        min_frame = tracks_df['frame'].min()
        max_frame = tracks_df['frame'].max()
        frame_range = max_frame - min_frame
        
        # Divide into time groups
        group_size = frame_range / frame_groups
        
        # Process each time group
        for group_idx in range(frame_groups):
            # Skip if we don't have enough axes
            if group_idx >= len(axes):
                break
            
            # Get frame interval
            start_frame = min_frame + group_idx * group_size
            end_frame = min_frame + (group_idx + 1) * group_size
            
            # Filter tracks for this time group
            group_tracks = tracks_df[(tracks_df['frame'] >= start_frame) & 
                                    (tracks_df['frame'] < end_frame)]
            
            # Skip if not enough points
            if len(group_tracks) < min_samples:
                continue
            
            # Get positions
            positions = group_tracks[['x_um', 'y_um']].values
            
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(positions)
            
            # Add cluster labels to track data
            group_tracks = group_tracks.copy()
            group_tracks['cluster'] = cluster_labels
            
            # Get unique clusters
            clusters = sorted(set(cluster_labels))
            clusters.remove(-1) if -1 in clusters else clusters
            
            # Create colormap
            cmap_obj = plt.get_cmap(cmap, len(clusters))
            
            # Get current axis
            curr_ax = axes[group_idx]
            
            # Plot noise points
            noise_points = group_tracks[group_tracks['cluster'] == -1]
            if not noise_points.empty:
                curr_ax.scatter(noise_points['x_um'], noise_points['y_um'], 
                              c='gray', s=10, alpha=0.5)
            
            # Plot clusters
            for i, cluster_id in enumerate(clusters):
                # Get cluster points
                cluster_points = group_tracks[group_tracks['cluster'] == cluster_id]
                
                # Get coordinates
                x = cluster_points['x_um']
                y = cluster_points['y_um']
                
                # Get color
                color = cmap_obj(i % len(clusters))
                
                # Plot cluster points
                curr_ax.scatter(x, y, c=[color], s=20, alpha=0.7, label=f'Cluster {cluster_id}')
                
                # Compute cluster centroid
                centroid_x = x.mean()
                centroid_y = y.mean()
                
                # Add cluster ID label
                curr_ax.text(centroid_x, centroid_y, str(cluster_id), 
                           ha='center', va='center', fontsize=10, 
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Set axis title
            curr_ax.set_title(f'Frames {int(start_frame)}-{int(end_frame)}')
            
            # Set labels for outer axes
            if group_idx % n_cols == 0:
                curr_ax.set_ylabel('Y (μm)')
            if group_idx >= (n_rows - 1) * n_cols:
                curr_ax.set_xlabel('X (μm)')
            
            # Set aspect ratio to equal
            curr_ax.set
Copy
            # Set aspect ratio to equal
            curr_ax.set_aspect('equal')
        
        # Hide empty axes
        for i in range(frame_groups, len(axes)):
            axes[i].axis('off')
        
        # Add main title
        if title:
            fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.9)
        
        return axes
    
    except Exception as e:
        logger.error(f"Error plotting cluster time evolution: {str(e)}")
        raise


def plot_spatial_distribution(tracks_df, background=None, binsize=10, 
                             smooth=True, sigma=1.0, ax=None, figsize=(10, 8),
                             cmap='hot_r', pixel_size=1.0, title=None):
    """
    Plot spatial distribution of tracks using 2D histogram.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    background : numpy.ndarray, optional
        Background image, by default None
    binsize : int, optional
        Bin size for histogram, by default 10
    smooth : bool, optional
        Whether to apply Gaussian smoothing, by default True
    sigma : float, optional
        Sigma for Gaussian smoothing, by default 1.0
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 8)
    cmap : str, optional
        Colormap to use, by default 'hot_r'
    pixel_size : float, optional
        Pixel size for proper scaling, by default 1.0
    title : str, optional
        Plot title, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        from scipy.ndimage import gaussian_filter
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Plot background image if provided
        if background is not None:
            ax.imshow(background, cmap='gray', alpha=0.5)
        
        # Convert coordinates to μm
        tracks_df = tracks_df.copy()
        tracks_df['x_um'] = tracks_df['x'] * pixel_size
        tracks_df['y_um'] = tracks_df['y'] * pixel_size
        
        # Get positions
        x = tracks_df['x_um'].values
        y = tracks_df['y_um'].values
        
        # Create 2D histogram
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Add margin
        margin_x = 0.05 * (x_max - x_min)
        margin_y = 0.05 * (y_max - y_min)
        
        x_bins = np.arange(x_min - margin_x, x_max + margin_x + binsize, binsize)
        y_bins = np.arange(y_min - margin_y, y_max + margin_y + binsize, binsize)
        
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
        
        # Apply Gaussian smoothing if requested
        if smooth:
            hist = gaussian_filter(hist, sigma=sigma)
        
        # Plot 2D histogram
        im = ax.pcolormesh(x_edges, y_edges, hist.T, cmap=cmap, alpha=0.7, shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count')
        
        # Set labels and title
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        if title:
            ax.set_title(title)
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting spatial distribution: {str(e)}")
        raise
