
"""
Track visualization module for SPT Analysis.

This module provides functions for visualizing track data, including
trajectory overlays, heatmaps, and various track statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


def plot_tracks(tracks_df, background=None, colorby='track_id', cmap='viridis', 
                alpha=0.7, linewidth=1, markersize=3, ax=None, figsize=(10, 8),
                max_tracks=None, pixel_size=1.0, title=None):
    """
    Plot particle tracks.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    background : numpy.ndarray, optional
        Background image, by default None
    colorby : str, optional
        Column to use for track coloring, by default 'track_id'
    cmap : str, optional
        Colormap to use, by default 'viridis'
    alpha : float, optional
        Track opacity, by default 0.7
    linewidth : float, optional
        Track line width, by default 1
    markersize : float, optional
        Track marker size, by default 3
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 8)
    max_tracks : int, optional
        Maximum number of tracks to plot, by default None
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
Copy
        # Get unique track IDs
        track_ids = tracks_df['track_id'].unique()
        
        # Limit number of tracks if requested
        if max_tracks is not None and len(track_ids) > max_tracks:
            track_ids = np.random.choice(track_ids, max_tracks, replace=False)
        
        # Create colormap
        cmap_obj = plt.get_cmap(cmap)
        
        # Determine color values based on colorby parameter
        if colorby == 'track_id':
            # Use track_id for coloring
            color_values = track_ids
            norm = mcolors.Normalize(vmin=min(track_ids), vmax=max(track_ids))
        else:
            # Try to use specified column
            if colorby in tracks_df.columns:
                # Get unique values for each track
                color_values = []
                
                for track_id in track_ids:
                    track_data = tracks_df[tracks_df['track_id'] == track_id]
                    color_values.append(track_data[colorby].mean())
                
                norm = mcolors.Normalize(vmin=min(color_values), vmax=max(color_values))
            else:
                # Fall back to track_id
                logger.warning(f"Column {colorby} not found, using track_id for coloring")
                color_values = track_ids
                norm = mcolors.Normalize(vmin=min(track_ids), vmax=max(track_ids))
        
        # Plot each track
        for i, track_id in enumerate(track_ids):
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            # Get track coordinates, scale with pixel size
            x = track_data['x'] * pixel_size
            y = track_data['y'] * pixel_size
            
            # Get color
            color = cmap_obj(norm(color_values[i]))
            
            # Plot track
            ax.plot(x, y, '-', linewidth=linewidth, color=color, alpha=alpha)
            ax.plot(x, y, 'o', markersize=markersize, color=color, alpha=alpha)
            
            # Mark start and end points
            ax.plot(x.iloc[0], y.iloc[0], 'o', markersize=markersize+2, color=color)
            ax.plot(x.iloc[-1], y.iloc[-1], 's', markersize=markersize+2, color=color)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(colorby)
        
        # Set labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        if title:
            ax.set_title(title)
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting tracks: {str(e)}")
        raise


def plot_track_heatmap(tracks_df, bin_size=5, smoothing=1.0, cmap='hot', 
                      ax=None, figsize=(10, 8), pixel_size=1.0, title=None):
    """
    Plot a heatmap of track positions.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    bin_size : int, optional
        Bin size for histogram, by default 5
    smoothing : float, optional
        Smoothing sigma for Gaussian filter, by default 1.0
    cmap : str, optional
        Colormap to use, by default 'hot'
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
        from scipy.ndimage import gaussian_filter
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Get track coordinates, scale with pixel size
        x = tracks_df['x'] * pixel_size
        y = tracks_df['y'] * pixel_size
        
        # Determine histogram edges
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Add margin
        margin_x = 0.05 * (x_max - x_min)
        margin_y = 0.05 * (y_max - y_min)
        
        x_min -= margin_x
        x_max += margin_x
        y_min -= margin_y
        y_max += margin_y
        
        # Create bins
        x_bins = np.arange(x_min, x_max + bin_size, bin_size)
        y_bins = np.arange(y_min, y_max + bin_size, bin_size)
        
        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
        
        # Apply Gaussian smoothing
        if smoothing > 0:
            hist = gaussian_filter(hist, sigma=smoothing)
        
        # Plot heatmap
        im = ax.imshow(
            hist.T,
            origin='lower',
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap=cmap,
            aspect='auto'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Counts')
        
        # Set labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        if title:
            ax.set_title(title)
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting track heatmap: {str(e)}")
        raise


def plot_track_stats(tracks_df, stats_df, stat_column, cmap='viridis',
                    alpha=0.7, linewidth=1, markersize=3, ax=None, 
                    figsize=(10, 8), pixel_size=1.0, title=None):
    """
    Plot tracks colored by a statistic from a separate DataFrame.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    stats_df : pandas.DataFrame
        DataFrame with track statistics
    stat_column : str
        Column in stats_df to use for coloring
    cmap : str, optional
        Colormap to use, by default 'viridis'
    alpha : float, optional
        Track opacity, by default 0.7
    linewidth : float, optional
        Track line width, by default 1
    markersize : float, optional
        Track marker size, by default 3
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
        
        # Get unique track IDs
        track_ids = tracks_df['track_id'].unique()
        
        # Create stats lookup table
        stats_lookup = {}
        
        for _, row in stats_df.iterrows():
            stats_lookup[row['track_id']] = row[stat_column]
        
        # Filter tracks with available statistics
        valid_track_ids = [track_id for track_id in track_ids if track_id in stats_lookup]
        
        # Create colormap
        cmap_obj = plt.get_cmap(cmap)
        
        # Get color values
        color_values = [stats_lookup[track_id] for track_id in valid_track_ids]
        norm = mcolors.Normalize(vmin=min(color_values), vmax=max(color_values))
        
        # Plot each track
        for i, track_id in enumerate(valid_track_ids):
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            # Get track coordinates, scale with pixel size
            x = track_data['x'] * pixel_size
            y = track_data['y'] * pixel_size
            
            # Get color
            color = cmap_obj(norm(color_values[i]))
            
            # Plot track
            ax.plot(x, y, '-', linewidth=linewidth, color=color, alpha=alpha)
            ax.plot(x, y, 'o', markersize=markersize, color=color, alpha=alpha)
            
            # Mark start and end points
            ax.plot(x.iloc[0], y.iloc[0], 'o', markersize=markersize+2, color=color)
            ax.plot(x.iloc[-1], y.iloc[-1], 's', markersize=markersize+2, color=color)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(stat_column)
        
        # Set labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        if title:
            ax.set_title(title)
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting track statistics: {str(e)}")
        raise


def plot_track_clusters(tracks_df, cluster_column='cluster', background=None,
                        cmap='tab10', alpha=0.7, linewidth=1, markersize=3,
                        ax=None, figsize=(10, 8), pixel_size=1.0, title=None):
    """
    Plot tracks colored by cluster assignment.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data including cluster assignments
    cluster_column : str, optional
        Column with cluster assignments, by default 'cluster'
    background : numpy.ndarray, optional
        Background image, by default None
    cmap : str, optional
        Colormap to use, by default 'tab10'
    alpha : float, optional
        Track opacity, by default 0.7
    linewidth : float, optional
        Track line width, by default 1
    markersize : float, optional
        Track marker size, by default 3
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
        
        # Get unique track IDs
        track_ids = tracks_df['track_id'].unique()
        
        # Ensure cluster column exists
        if cluster_column not in tracks_df.columns:
            raise ValueError(f"Column {cluster_column} not found in tracks_df")
        
        # Create cluster lookup table
        cluster_lookup = {}
        
        for track_id in track_ids:
            track_data = tracks_df[tracks_df['track_id'] == track_id]
            # Use most frequent cluster value
            cluster_lookup[track_id] = track_data[cluster_column].value_counts().index[0]
        
        # Get unique clusters
        clusters = sorted(set(cluster_lookup.values()))
        
        # Create colormap
        cmap_obj = plt.get_cmap(cmap)
        
        # Plot each track
        for track_id in track_ids:
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            # Get track coordinates, scale with pixel size
            x = track_data['x'] * pixel_size
            y = track_data['y'] * pixel_size
            
            # Get color
            cluster = cluster_lookup[track_id]
            
            # Skip noise points (cluster = -1)
            if cluster == -1:
                color = 'gray'
                label = 'Noise' if track_id == track_ids[0] else None
            else:
                color_idx = clusters.index(cluster) % len(cmap_obj.colors)
                color = cmap_obj(color_idx)
                label = f'Cluster {cluster}' if track_id == track_ids[0] else None
            
            # Plot track
            ax.plot(x, y, '-', linewidth=linewidth, color=color, alpha=alpha, label=label)
            ax.plot(x, y, 'o', markersize=markersize, color=color, alpha=alpha)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Set labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        if title:
            ax.set_title(title)
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting track clusters: {str(e)}")
        raise


def plot_track_density(tracks_df, bandwidth=1.0, grid_size=100, cmap='hot',
                      ax=None, figsize=(10, 8), pixel_size=1.0, title=None):
    """
    Plot kernel density estimation of track positions.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    bandwidth : float, optional
        Bandwidth for kernel density estimation, by default 1.0
    grid_size : int, optional
        Grid size for density evaluation, by default 100
    cmap : str, optional
        Colormap to use, by default 'hot'
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
        from scipy.stats import gaussian_kde
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Get track coordinates, scale with pixel size
        x = tracks_df['x'] * pixel_size
        y = tracks_df['y'] * pixel_size
        
        # Create grid for density evaluation
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Add margin
        margin_x = 0.05 * (x_max - x_min)
        margin_y = 0.05 * (y_max - y_min)
        
        x_min -= margin_x
        x_max += margin_x
        y_min -= margin_y
        y_max += margin_y
        
        # Create grid
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        # Compute density
        if len(x) > 1:
            kernel = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
            Z = kernel(positions).reshape(grid_size, grid_size)
        else:
            Z = np.zeros((grid_size, grid_size))
        
        # Plot density
        im = ax.imshow(
            Z,
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            cmap=cmap,
            aspect='auto'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density')
        
        # Set labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        if title:
            ax.set_title(title)
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting track density: {str(e)}")
        raise


def plot_msd_curves(msd_df, ensemble_df=None, max_lag=None, track_ids=None,
                   ax=None, figsize=(10, 6), title=None):
    """
    Plot mean squared displacement curves.
    
    Parameters
    ----------
    msd_df : pandas.DataFrame
        DataFrame with MSD data for individual tracks
    ensemble_df : pandas.DataFrame, optional
        DataFrame with ensemble MSD data, by default None
    max_lag : int, optional
        Maximum lag to plot, by default None
    track_ids : list, optional
        List of track IDs to include, by default None
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
        
        # Filter by max lag if provided
        if max_lag is not None:
            msd_df = msd_df[msd_df['lag'] <= max_lag].copy()
            if ensemble_df is not None:
                ensemble_df = ensemble_df[ensemble_df['lag'] <= max_lag].copy()
        
        # Get unique track IDs
        if track_ids is None:
            track_ids = msd_df['track_id'].unique()
            
            # Limit to 10 random tracks if too many
            if len(track_ids) > 10:
                track_ids = np.random.choice(track_ids, 10, replace=False)
        
        # Plot MSD curves for individual tracks
        for track_id in track_ids:
            track_data = msd_df[msd_df['track_id'] == track_id]
            track_data = track_data.groupby('lag')['msd'].mean().reset_index()
            ax.plot(track_data['lag'], track_data['msd'], 'o-', alpha=0.5, linewidth=1, label=f'Track {track_id}')
        
        # Plot ensemble MSD if provided
        if ensemble_df is not None:
            ax.plot(ensemble_df['lag'], ensemble_df['msd'], 'k-', linewidth=2, label='Ensemble')
        
        # Set labels and title
        ax.set_xlabel('Lag (frames)')
        ax.set_ylabel('MSD')
        if title:
            ax.set_title(title)
        
        # Add legend
        if len(track_ids) <= 10:
            ax.legend()
        
        # Log-log scale often helps visualization
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting MSD curves: {str(e)}")
        raise
