
"""
Utility functions for visualization in SPT Analysis.

This module provides common utility functions for visualization, including
colormap generation, figure management, and video creation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import colorsys
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import logging

logger = logging.getLogger(__name__)


def create_colormap(n_colors, cmap='viridis', start=0.0, stop=1.0, alpha=1.0):
    """
    Create a colormap with specified number of colors.
    
    Parameters
    ----------
    n_colors : int
        Number of colors in the colormap
    cmap : str, optional
        Base colormap name, by default 'viridis'
    start : float, optional
        Start point in the colormap, by default 0.0
    stop : float, optional
        End point in the colormap, by default 1.0
    alpha : float, optional
        Alpha value for colors, by default 1.0
        
    Returns
    -------
    list
        List of RGBA colors
    """
    try:
        base_cmap = plt.get_cmap(cmap)
        color_list = []
        
        for i in range(n_colors):
            # Get color from the colormap
            color_val = start + (stop - start) * i / max(1, n_colors - 1)
            rgba = list(base_cmap(color_val))
            
            # Set alpha
            rgba[3] = alpha
            
            color_list.append(rgba)
        
        return color_list
    
    except Exception as e:
        logger.error(f"Error creating colormap: {str(e)}")
        raise


def create_distinct_colors(n_colors, alpha=1.0):
    """
    Create a list of distinct colors using HSV color space.
    
    Parameters
    ----------
    n_colors : int
        Number of colors
    alpha : float, optional
        Alpha value for colors, by default 1.0
        
    Returns
    -------
    list
        List of RGBA colors
    """
    try:
        colors = []
        for i in range(n_colors):
            # Use golden ratio to get well-distributed hues
            h = i * 0.618033988749895 % 1
            # Use fixed saturation and value for visibility
            s = 0.8
            v = 0.9
            
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            
            # Add alpha
            colors.append((r, g, b, alpha))
        
        return colors
    
    except Exception as e:
        logger.error(f"Error creating distinct colors: {str(e)}")
        raise


def save_figure(fig, filename, dpi=300, transparent=False, bbox_inches='tight'):
    """
    Save figure to file with proper settings.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename
    dpi : int, optional
        Resolution in dots per inch, by default 300
    transparent : bool, optional
        Whether to use transparent background, by default False
    bbox_inches : str, optional
        Bounding box inches, by default 'tight'
        
    Returns
    -------
    str
        Path to saved file
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save figure
        fig.savefig(filename, dpi=dpi, transparent=transparent, bbox_inches=bbox_inches)
        logger.info(f"Figure saved to {filename}")
        
        return filename
    
    except Exception as e:
        logger.error(f"Error saving figure: {str(e)}")
        raise


def create_track_animation(tracks_df, output_file, fps=10, dpi=100, 
                         background=None, trail_length=10, marker_size=5,
                         figsize=(10, 8), cmap='viridis', pixel_size=1.0,
                         title=None):
    """
    Create an animation of particle tracks.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    output_file : str
        Output filename
    fps : int, optional
        Frames per second, by default 10
    dpi : int, optional
        Resolution in dots per inch, by default 100
    background : numpy.ndarray, optional
        Background image, by default None
    trail_length : int, optional
        Length of particle trails, by default 10
    marker_size : int, optional
        Size of particle markers, by default 5
    figsize : tuple, optional
        Figure size, by default (10, 8)
    cmap : str, optional
        Colormap for tracks, by default 'viridis'
    pixel_size : float, optional
        Pixel size for proper scaling, by default 1.0
    title : str, optional
        Title for the animation, by default None
        
    Returns
    -------
    str
        Path to saved animation file
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up plot
        if background is not None:
            ax.imshow(background, cmap='gray', alpha=0.5)
        
        # Set labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        if title:
            ax.set_title(title)
        
        # Get unique track IDs and frames
        track_ids = tracks_df['track_id'].unique()
        frames = sorted(tracks_df['frame'].unique())
        
        # Create colormap
        cmap_obj = plt.get_cmap(cmap)
        colors = create_colormap(len(track_ids), cmap=cmap)
        
        # Create a scatter plot for current positions
        scatter = ax.scatter([], [], s=marker_size, c=[], cmap=cmap_obj)
        
        # Create a line collection for trails
        line_collection = LineCollection([], linewidths=1, alpha=0.5)
        ax.add_collection(line_collection)
        
        # Set axis limits
        x_min, x_max = tracks_df['x'].min(), tracks_df['x'].max()
        y_min, y_max = tracks_df['y'].min(), tracks_df['y'].max()
        
        # Add margin
        margin_x = 0.05 * (x_max - x_min)
        margin_y = 0.05 * (y_max - y_min)
        
        ax.set_xlim(x_min - margin_x, x_max + margin_x)
        ax.set_ylim(y_max + margin_y, y_min - margin_y)  # Inverted for image coordinates
        
        # Set aspect ratio to equal
        ax.set_aspect('equal')
        
        # Function to initialize animation
        def init():
            scatter.set_offsets(np.empty((0, 2)))
            line_collection.set_segments([])
            return scatter, line_collection
        
        # Function to update animation for each frame
        def update(frame_idx):
            # Get current frame
            frame = frames[frame_idx]
            
            # Get points for current frame
            current_points = tracks_df[tracks_df['frame'] == frame]
            
            # Update scatter plot
            if not current_points.empty:
                positions = current_points[['x', 'y']].values * pixel_size
                colors = [track_ids.tolist().index(tid) for tid in current_points['track_id']]
                
                scatter.set_offsets(positions)
                scatter.set_array(np.array(colors))
            else:
                scatter.set_offsets(np.empty((0, 2)))
                scatter.set_array(np.array([]))
            
            # Update trails
            segments = []
            segment_colors = []
            
            for i, track_id in enumerate(track_ids):
                # Get track data up to current frame
                track = tracks_df[(tracks_df['track_id'] == track_id) & 
                                  (tracks_df['frame'] <= frame)]
                
                # Skip if track hasn't started yet
                if len(track) == 0:
                    continue
                
                # Sort by frame
                track = track.sort_values('frame')
                
                # Get trail points (limited by trail_length)
                if len(track) > 1:
                    trail_points = track.iloc[-min(len(track), trail_length+1):][['x', 'y']].values * pixel_size
                    
                    # Create line segments
                    for j in range(len(trail_points) - 1):
                        segments.append([trail_points[j], trail_points[j+1]])
                        segment_colors.append(colors[i])
            
            line_collection.set_segments(segments)
            line_collection.set_color(segment_colors)
            
            return scatter, line_collection
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                      init_func=init, blit=True)
        
        # Save animation
        if output_file.endswith('.mp4'):
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(output_file, writer=writer, dpi=dpi)
        else:
            anim.save(output_file, fps=fps, dpi=dpi)
        
        # Close figure
        plt.close(fig)
        
        logger.info(f"Animation saved to {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error creating track animation: {str(e)}")
        raise


def create_interactive_plot(tracks_df, diffusion_df=None, cluster_df=None,
                           figsize=(10, 8), cmap='viridis', pixel_size=1.0,
                           title=None):
    """
    Create an interactive plot of tracks with hover information.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    diffusion_df : pandas.DataFrame, optional
        DataFrame with diffusion coefficients, by default None
    cluster_df : pandas.DataFrame, optional
        DataFrame with cluster assignments, by default None
    figsize : tuple, optional
        Figure size, by default (10, 8)
    cmap : str, optional
        Colormap for tracks, by default 'viridis'
    pixel_size : float, optional
        Pixel size for proper scaling, by default 1.0
    title : str, optional
        Title for the plot, by default None
        
    Returns
    -------
    matplotlib.figure.Figure
        Interactive figure
    """
    try:
        # Try to import interactive plotting libraries
        try:
            import mpld3
            from mpld3 import plugins
        except ImportError:
            logger.warning("mpld3 not installed. Using static plot instead.")
            # Fall back to static plot
            fig, ax = plt.subplots(figsize=figsize)
            
            for track_id, track in tracks_df.groupby('track_id'):
                track = track.sort_values('frame')
                x = track['x'] * pixel_size
                y = track['y'] * pixel_size
                ax.plot(x, y, '-o', linewidth=1, markersize=3)
            
            if title:
                ax.set_title(title)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.invert_yaxis()
            
            return fig
        
        # Create figure for interactive plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create lookup tables for additional data
        diff_lookup = {}
        cluster_lookup = {}
        
        if diffusion_df is not None:
            for _, row in diffusion_df.iterrows():
                diff_lookup[row['track_id']] = row['D']
        
        if cluster_df is not None:
            for track_id, track in tracks_df.groupby('track_id'):
                # Get most common cluster assignment
                if 'cluster' in track.columns:
                    cluster_lookup[track_id] = track['cluster'].mode()[0]
        
        # Plot each track
        points = []
        track_lines = []
        
        for track_id, track in tracks_df.groupby('track_id'):
            # Sort track by frame
            track = track.sort_values('frame')
            
            # Get coordinates
            x = track['x'] * pixel_size
            y = track['y'] * pixel_size
            
            # Plot track
            line, = ax.plot(x, y, '-', linewidth=1, alpha=0.7)
            track_lines.append(line)
            
            # Plot points
            scatter = ax.scatter(x, y, s=20, alpha=0.7)
            points.append(scatter)
            
            # Create tooltip text
            tooltip_text = [f"Track ID: {track_id}<br>"]
            
            if track_id in diff_lookup:
                tooltip_text.append(f"Diffusion coefficient: {diff_lookup[track_id]:.3f} μm²/s<br>")
            
            if track_id in cluster_lookup:
                tooltip_text.append(f"Cluster: {cluster_lookup[track_id]}<br>")
            
            tooltip_text.append(f"Length: {len(track)} points<br>")
            
            # Add tooltip
            plugins.connect(fig, plugins.PointHTMLTooltip(scatter, tooltip_text))
            
        # Set up plot
        if title:
            ax.set_title(title)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.invert_yaxis()
        
        # Add zoom and pan controls
        plugins.connect(fig, plugins.Zoom())
        plugins.connect(fig, plugins.BoxZoom())
        plugins.connect(fig, plugins.Reset())
        
        # Convert to interactive plot
        interactive_plot = mpld3.fig_to_html(fig)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating interactive plot: {str(e)}")
        raise


def plot_overlay_image(image, tracks_df, alpha=0.5, figsize=(10, 8), 
                     cmap_image='gray', cmap_tracks='viridis', pixel_size=1.0, 
                     title=None):
    """
    Plot tracks overlaid on an image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Image array
    tracks_df : pandas.DataFrame
        DataFrame with track data
    alpha : float, optional
        Opacity of tracks, by default 0.5
    figsize : tuple, optional
        Figure size, by default (10, 8)
    cmap_image : str, optional
        Colormap for image, by default 'gray'
    cmap_tracks : str, optional
        Colormap for tracks, by default 'viridis'
    pixel_size : float, optional
        Pixel size for proper scaling, by default 1.0
    title : str, optional
        Title for the plot, by default None
        
    Returns
    -------
    matplotlib.axes.Axes
        Plot axes
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot image
        ax.imshow(image, cmap=cmap_image)
        
        # Get unique track IDs
        track_ids = tracks_df['track_id'].unique()
        
        # Create colormap
        cmap_obj = plt.get_cmap(cmap_tracks)
        norm = plt.Normalize(vmin=0, vmax=len(track_ids)-1)
        
        # Plot tracks
        for i, track_id in enumerate(track_ids):
            track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            # Get coordinates
            x = track['x'] * pixel_size
            y = track['y'] * pixel_size
            
            # Get color
            color = cmap_obj(norm(i))
            
            # Plot track
            ax.plot(x, y, '-', color=color, linewidth=1, alpha=alpha)
            ax.plot(x.iloc[0], y.iloc[0], 'o', color=color, markersize=5, alpha=alpha)
            ax.plot(x.iloc[-1], y.iloc[-1], 's', color=color, markersize=5, alpha=alpha)
        
        # Set labels and title
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        if title:
            ax.set_title(title)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting overlay image: {str(e)}")
        raise
