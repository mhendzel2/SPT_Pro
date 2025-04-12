
"""
Visualization utilities for SPT Analysis.

This module provides helper functions for visualizing tracking data,
preprocessing steps, and analysis results in various formats.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import io
import base64
import logging

logger = logging.getLogger(__name__)


def plot_preprocessing_steps(image, preproc_steps, titles=None, figsize=(15, 5)):
    """
    Plot original image and preprocessing steps.
    
    Parameters
    ----------
    image : numpy.ndarray
        Original image
    preproc_steps : list
        List of preprocessed images
    titles : list, optional
        List of subplot titles, by default None
    figsize : tuple, optional
        Figure size, by default (15, 5)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    try:
        # Create figure
        n_plots = len(preproc_steps) + 1  # Original + preprocessing steps
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        
        # Ensure axes is an array
        if n_plots == 1:
            axes = [axes]
        
        # Create default titles if not provided
        if titles is None:
            titles = ['Original'] + [f'Step {i+1}' for i in range(len(preproc_steps))]
        
        # Plot original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(titles[0])
        axes[0].axis('off')
        
        # Plot preprocessing steps
        for i, preproc_img in enumerate(preproc_steps):
            axes[i+1].imshow(preproc_img, cmap='gray')
            axes[i+1].set_title(titles[i+1])
            axes[i+1].axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting preprocessing steps: {str(e)}")
        raise


def plot_detection_overlay(image, detections, marker_size=10, color='red', figsize=(10, 8)):
    """
    Plot detections overlaid on image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Image
    detections : numpy.ndarray
        Array of detection coordinates with shape (n_detections, 2)
    marker_size : int, optional
        Marker size, by default 10
    color : str, optional
        Marker color, by default 'red'
    figsize : tuple, optional
        Figure size, by default (10, 8)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot image
        ax.imshow(image, cmap='gray')
        
        # Plot detections
        if len(detections) > 0:
            # Extract coordinates (y, x)
            y, x = detections[:, 0], detections[:, 1]
            
            # Plot detections
            ax.scatter(x, y, s=marker_size, c=color, marker='o', alpha=0.7)
        
        # Set title
        ax.set_title(f'Detected Particles: {len(detections)}')
        
        # Remove axes
        ax.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting detection overlay: {str(e)}")
        raise


def plot_tracking_summary(tracks_df, figsize=(15, 10)):
    """
    Plot tracking summary with multiple visualizations.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with tracking data
    figsize : tuple, optional
        Figure size, by default (15, 10)
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    try:
        # Create figure with grid layout
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # 1. Track trajectories
        ax1 = fig.add_subplot(gs[0, 0])
        plot_track_trajectories(ax1, tracks_df)
        
        # 2. Track length histogram
        ax2 = fig.add_subplot(gs[0, 1])
        plot_track_length_histogram(ax2, tracks_df)
        
        # 3. Particles per frame
        ax3 = fig.add_subplot(gs[0, 2])
        plot_particles_per_frame(ax3, tracks_df)
        
        # 4. Track displacement histogram
        ax4 = fig.add_subplot(gs[1, 0])
        plot_displacement_histogram(ax4, tracks_df)
        
        # 5. Track start positions
        ax5 = fig.add_subplot(gs[1, 1])
        plot_track_positions(ax5, tracks_df, position_type='start')
        
        # 6. Track end positions
        ax6 = fig.add_subplot(gs[1, 2])
        plot_track_positions(ax6, tracks_df, position_type='end')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting tracking summary: {str(e)}")
        raise


def plot_track_trajectories(ax, tracks_df, max_tracks=None, cmap='viridis', alpha=0.7):
    """
    Plot track trajectories.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    tracks_df : pandas.DataFrame
        DataFrame with tracking data
    max_tracks : int, optional
        Maximum number of tracks to plot, by default None
    cmap : str, optional
        Colormap name, by default 'viridis'
    alpha : float, optional
        Opacity, by default 0.7
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes object
    """
    try:
        # Get unique track IDs
        track_ids = tracks_df['track_id'].unique()
        
        # Limit number of tracks
        if max_tracks is not None and len(track_ids) > max_tracks:
            np.random.seed(42)  # For reproducibility
            track_ids = np.random.choice(track_ids, max_tracks, replace=False)
        
        # Create colormap
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=0, vmax=len(track_ids) - 1)
        
        # Plot each track
        for i, track_id in enumerate(track_ids):
            # Get track data
            track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            # Skip if track is too short
            if len(track) < 2:
                continue
            
            # Get coordinates
            x = track['x']
            y = track['y']
            
            # Get color
            color = cmap_obj(norm(i))
            
            # Plot trajectory
            ax.plot(x, y, '-', color=color, alpha=alpha, linewidth=1)
            
            # Mark start and end points
            ax.plot(x.iloc[0], y.iloc[0], 'o', color=color, markersize=5, alpha=alpha)
            ax.plot(x.iloc[-1], y.iloc[-1], 's', color=color, markersize=5, alpha=alpha)
        
        # Set title and labels
        ax.set_title(f'Track Trajectories (n={len(track_ids)})')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        # Set aspect ratio to equal
        ax.set_aspect('equal', adjustable='box')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting track trajectories: {str(e)}")
        raise


def plot_track_length_histogram(ax, tracks_df, bins=20, color='steelblue'):
    """
    Plot histogram of track lengths.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    tracks_df : pandas.DataFrame
        DataFrame with tracking data
    bins : int, optional
        Number of histogram bins, by default 20
    color : str, optional
        Bar color, by default 'steelblue'
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes object
    """
    try:
        # Compute track lengths
        track_lengths = tracks_df.groupby('track_id').size()
        
        # Plot histogram
        ax.hist(track_lengths, bins=bins, color=color, alpha=0.7, edgecolor='black')
        
        # Set title and labels
        ax.set_title('Track Length Distribution')
        ax.set_xlabel('Track Length (frames)')
        ax.set_ylabel('Count')
        
        # Add statistics
        mean_length = track_lengths.mean()
        median_length = track_lengths.median()
        
        ax.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.1f}')
        ax.axvline(median_length, color='green', linestyle=':', label=f'Median: {median_length:.1f}')
        
        # Add legend
        ax.legend()
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting track length histogram: {str(e)}")
        raise


def plot_particles_per_frame(ax, tracks_df, color='steelblue'):
    """
    Plot number of particles per frame.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    tracks_df : pandas.DataFrame
        DataFrame with tracking data
    color : str, optional
        Line color, by default 'steelblue'
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes object
    """
    try:
        # Count particles per frame
        particles_per_frame = tracks_df.groupby('frame').size()
        
        # Plot counts
        ax.plot(particles_per_frame.index, particles_per_frame.values, 
               '-o', color=color, alpha=0.7, markersize=4)
        
        # Set title and labels
        ax.set_title('Particles per Frame')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Particle Count')
        
        # Add statistics
        mean_count = particles_per_frame.mean()
        ax.axhline(mean_count, color='red', linestyle='--', 
                  label=f'Mean: {mean_count:.1f}')
        
        # Add legend
        ax.legend()
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting particles per frame: {str(e)}")
        raise


def plot_displacement_histogram(ax, tracks_df, bins=20, color='steelblue'):
    """
    Plot histogram of frame-to-frame displacements.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    tracks_df : pandas.DataFrame
        DataFrame with tracking data
    bins : int, optional
        Number of histogram bins, by default 20
    color : str, optional
        Bar color, by default 'steelblue'
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes object
    """
    try:
        # Compute displacements
        displacements = []
        
        for track_id, track in tracks_df.groupby('track_id'):
            # Sort by frame
            track = track.sort_values('frame')
            
            # Skip if track is too short
            if len(track) < 2:
                continue
            
            # Compute displacements
            dx = np.diff(track['x'])
            dy = np.diff(track['y'])
            d = np.sqrt(dx**2 + dy**2)
            
            displacements.extend(d)
        
        # Plot histogram
        ax.hist(displacements, bins=bins, color=color, alpha=0.7, edgecolor='black')
        
        # Set title and labels
        ax.set_title('Displacement Distribution')
        ax.set_xlabel('Displacement (pixels/frame)')
        ax.set_ylabel('Count')
        
        # Add statistics
        mean_disp = np.mean(displacements)
        median_disp = np.median(displacements)
        
        ax.axvline(mean_disp, color='red', linestyle='--', label=f'Mean: {mean_disp:.2f}')
        ax.axvline(median_disp, color='green', linestyle=':', label=f'Median: {median_disp:.2f}')
        
        # Add legend
        ax.legend()
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting displacement histogram: {str(e)}")
        raise


def plot_track_positions(ax, tracks_df, position_type='start', marker_size=20, 
                       cmap='viridis', alpha=0.7):
    """
    Plot start or end positions of tracks.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    tracks_df : pandas.DataFrame
        DataFrame with tracking data
    position_type : str, optional
        Position type ('start' or 'end'), by default 'start'
    marker_size : int, optional
        Marker size, by default 20
    cmap : str, optional
        Colormap name, by default 'viridis'
    alpha : float, optional
        Opacity, by default 0.7
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes object
    """
    try:
        # Get track positions
        positions = []
        
        for track_id, track in tracks_df.groupby('track_id'):
            # Sort by frame
            track = track.sort_values('frame')
            
            # Skip if track is too short
            if len(track) < 2:
                continue
            
            # Get position
            if position_type == 'start':
                positions.append((track.iloc[0]['x'], track.iloc[0]['y'], track.iloc[0]['frame']))
            elif position_type == 'end':
                positions.append((track.iloc[-1]['x'], track.iloc[-1]['y'], track.iloc[-1]['frame']))
            else:
                raise ValueError(f"Invalid position type: {position_type}")
        
        # Convert to array
        positions = np.array(positions)
        
        # Plot positions
        scatter = ax.scatter(positions[:, 0], positions[:, 1], s=marker_size, 
                           c=positions[:, 2], cmap=cmap, alpha=alpha)
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(scatter, cax=cax, label='Frame')
        
        # Set title and labels
        ax.set_title(f'Track {position_type.capitalize()} Positions')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        # Set aspect ratio to equal
        ax.set_aspect('equal', adjustable='box')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting track positions: {str(e)}")
        raise


def create_tracking_animation(image_stack, tracks_df, output_path, fps=10, dpi=100,
                            marker_size=10, trail_length=10, cmap='viridis', 
                            show_frame_number=True):
    """
    Create animation of particle tracking.
    
    Parameters
    ----------
    image_stack : numpy.ndarray
        Image stack with shape (n_frames, height, width)
    tracks_df : pandas.DataFrame
        DataFrame with tracking data
    output_path : str
        Output file path
    fps : int, optional
        Frames per second, by default 10
    dpi : int, optional
        Resolution in dots per inch, by default 100
    marker_size : int, optional
        Marker size, by default 10
    trail_length : int, optional
        Length of particle trails, by default 10
    cmap : str, optional
        Colormap name for particles, by default 'viridis'
    show_frame_number : bool, optional
        Whether to show frame number, by default True
        
    Returns
    -------
    str
        Path to saved animation
    """
    try:
        logger.info("Creating tracking animation")
        
        # Get number of frames
        n_frames = image_stack.shape[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique track IDs
        track_ids = tracks_df['track_id'].unique()
        
        # Create colormap
        cmap_obj = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=0, vmax=len(track_ids) - 1)
        
        # Function to update each frame
        def update(frame):
            # Clear axis
            ax.clear()
            
            # Plot current image
            ax.imshow(image_stack[frame], cmap='gray')
            
            # Plot tracks
            for i, track_id in enumerate(track_ids):
                # Get track data
                track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
                
                # Skip if track isn't present in this frame range
                if track['frame'].max() < frame or track['frame'].min() > frame:
                    continue
                
                # Get all points up to current frame
                track_up_to_frame = track[track['frame'] <= frame]
                
                # Skip if no points
                if len(track_up_to_frame) == 0:
                    continue
                
                # Get current position
                current_pos = track_up_to_frame[track_up_to_frame['frame'] == 
                                             track_up_to_frame['frame'].max()]
                
                # Skip if not in current frame
                if len(current_pos) == 0 or current_pos['frame'].iloc[0] < frame - trail_length:
                    continue
                
                # Get trail
                start_frame = max(0, current_pos['frame'].iloc[0] - trail_length)
                trail = track[(track['frame'] >= start_frame) & 
                             (track['frame'] <= current_pos['frame'].iloc[0])]
                
                # Get color
                color = cmap_obj(norm(i))
                
                # Plot trail
                if len(trail) > 1:
                    ax.plot(trail['x'], trail['y'], '-', color=color, alpha=0.5, linewidth=1)
                
                # Plot current position
                ax.plot(current_pos['x'], current_pos['y'], 'o', color=color, 
                       markersize=marker_size/2, alpha=0.8)
            
            # Show frame number
            if show_frame_number:
                ax.text(0.02, 0.98, f'Frame: {frame}', transform=ax.transAxes, 
                       color='white', fontsize=12, verticalalignment='top',
                       bbox=dict(facecolor='black', alpha=0.5))
            
            # Remove axes
            ax.axis('off')
            
            return [ax]
        
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)
        
        # Save animation
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(output_path, writer=writer, dpi=dpi)
        
        # Close figure
        plt.close(fig)
        
        logger.info(f"Animation saved to {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating tracking animation: {str(e)}")
        raise


def create_trajectory_plot_html(tracks_df, width=800, height=600):
    """
    Create interactive HTML plot of trajectories.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with tracking data
    width : int, optional
        Plot width in pixels, by default 800
    height : int
Copy
    height : int, optional
        Plot height in pixels, by default 600
        
    Returns
    -------
    str
        HTML content with interactive plot
    """
    try:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.error("Plotly not installed. Please install with 'pip install plotly'")
            return "<p>Error: Plotly not installed</p>"
        
        # Create figure
        fig = make_subplots(rows=1, cols=1)
        
        # Get unique track IDs
        track_ids = tracks_df['track_id'].unique()
        
        # Plot each track
        for track_id in track_ids:
            # Get track data
            track = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            # Skip if track is too short
            if len(track) < 2:
                continue
            
            # Get coordinates
            x = track['x']
            y = track['y']
            frames = track['frame']
            
            # Create hover text
            hover_text = [f"Track ID: {track_id}<br>Frame: {frame}<br>X: {x:.2f}<br>Y: {y:.2f}" 
                         for x, y, frame in zip(x, y, frames)]
            
            # Plot trajectory
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,
                    mode='lines+markers',
                    name=f'Track {track_id}',
                    hoverinfo='text',
                    hovertext=hover_text,
                    line=dict(width=1),
                    marker=dict(size=5)
                )
            )
        
        # Update layout
        fig.update_layout(
            title='Particle Trajectories',
            xaxis_title='X (pixels)',
            yaxis_title='Y (pixels)',
            width=width,
            height=height,
            hovermode='closest',
            template='plotly_white',
            yaxis=dict(scaleanchor="x", scaleratio=1, autorange='reversed')
        )
        
        # Convert to HTML
        html = fig.to_html(include_plotlyjs='cdn', full_html=False)
        
        return html
    
    except Exception as e:
        logger.error(f"Error creating interactive trajectory plot: {str(e)}")
        return f"<p>Error creating plot: {str(e)}</p>"


def create_summary_report(tracks_df, analysis_results=None, image=None, width=900, height=700):
    """
    Create HTML summary report with plots and statistics.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with tracking data
    analysis_results : dict, optional
        Dictionary with analysis results, by default None
    image : numpy.ndarray, optional
        Representative image for background, by default None
    width : int, optional
        Plot width in pixels, by default 900
    height : int, optional
        Plot height in pixels, by default 700
        
    Returns
    -------
    str
        HTML content with summary report
    """
    try:
        # Create summary figures
        
        # Trajectory plot
        fig_traj = plt.figure(figsize=(8, 6))
        ax_traj = fig_traj.add_subplot(111)
        plot_track_trajectories(ax_traj, tracks_df)
        
        # Track length histogram
        fig_lengths = plt.figure(figsize=(8, 6))
        ax_lengths = fig_lengths.add_subplot(111)
        plot_track_length_histogram(ax_lengths, tracks_df)
        
        # Particles per frame
        fig_counts = plt.figure(figsize=(8, 6))
        ax_counts = fig_counts.add_subplot(111)
        plot_particles_per_frame(ax_counts, tracks_df)
        
        # Displacement histogram
        fig_displacements = plt.figure(figsize=(8, 6))
        ax_displacements = fig_displacements.add_subplot(111)
        plot_displacement_histogram(ax_displacements, tracks_df)
        
        # Convert figures to base64 for embedding
        def fig_to_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return img_str
        
        traj_b64 = fig_to_base64(fig_traj)
        lengths_b64 = fig_to_base64(fig_lengths)
        counts_b64 = fig_to_base64(fig_counts)
        displacements_b64 = fig_to_base64(fig_displacements)
        
        # Add image if provided
        image_b64 = None
        if image is not None:
            fig_img = plt.figure(figsize=(8, 6))
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.title('Representative Image')
            image_b64 = fig_to_base64(fig_img)
        
        # Calculate statistics
        track_count = tracks_df['track_id'].nunique()
        point_count = len(tracks_df)
        
        track_lengths = tracks_df.groupby('track_id').size()
        mean_length = track_lengths.mean()
        max_length = track_lengths.max()
        min_length = track_lengths.min()
        
        frame_counts = tracks_df.groupby('frame').size()
        mean_count = frame_counts.mean()
        max_count = frame_counts.max()
        
        # Create HTML report
        html = f"""
        <html>
        <head>
            <title>SPT Analysis Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .container {{ max-width: {width}px; margin: 0 auto; }}
                .stats {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); grid-gap: 10px; }}
                .stat-item {{ background-color: #ffffff; padding: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                .stat-value {{ font-size: 20px; font-weight: bold; color: #3498db; }}
                .stat-label {{ font-size: 14px; color: #7f8c8d; }}
                .plots {{ display: grid; grid-template-columns: 1fr 1fr; grid-gap: 20px; }}
                .plot {{ background-color: #ffffff; padding: 10px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .plot img {{ max-width: 100%; height: auto; }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #7f8c8d; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Single Particle Tracking Analysis Report</h1>
                
                <div class="stats">
                    <h2>Summary Statistics</h2>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value">{track_count}</div>
                            <div class="stat-label">Total Tracks</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{point_count}</div>
                            <div class="stat-label">Total Points</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{mean_length:.1f}</div>
                            <div class="stat-label">Mean Track Length</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{min_length}</div>
                            <div class="stat-label">Minimum Track Length</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{max_length}</div>
                            <div class="stat-label">Maximum Track Length</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value">{mean_count:.1f}</div>
                            <div class="stat-label">Mean Particles per Frame</div>
                        </div>
                    </div>
                </div>
        """
        
        # Add representative image if available
        if image_b64:
            html += f"""
                <div class="plot">
                    <h2>Representative Image</h2>
                    <img src="data:image/png;base64,{image_b64}" alt="Representative Image">
                </div>
            """
        
        # Add plots
        html += f"""
                <h2>Visualization</h2>
                <div class="plots">
                    <div class="plot">
                        <h3>Trajectories</h3>
                        <img src="data:image/png;base64,{traj_b64}" alt="Trajectories">
                    </div>
                    <div class="plot">
                        <h3>Track Lengths</h3>
                        <img src="data:image/png;base64,{lengths_b64}" alt="Track Lengths">
                    </div>
                    <div class="plot">
                        <h3>Particles per Frame</h3>
                        <img src="data:image/png;base64,{counts_b64}" alt="Particles per Frame">
                    </div>
                    <div class="plot">
                        <h3>Displacements</h3>
                        <img src="data:image/png;base64,{displacements_b64}" alt="Displacements">
                    </div>
                </div>
        """
        
        # Add analysis results if provided
        if analysis_results:
            html += """
                <h2>Analysis Results</h2>
            """
            
            # Check for diffusion analysis
            if 'diffusion' in analysis_results:
                diffusion = analysis_results['diffusion']
                html += f"""
                <div class="plot">
                    <h3>Diffusion Analysis</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value">{diffusion.get('D', 0):.4f}</div>
                            <div class="stat-label">Diffusion Coefficient (μm²/s)</div>
                        </div>
                """
                
                if 'alpha' in diffusion:
                    html += f"""
                        <div class="stat-item">
                            <div class="stat-value">{diffusion.get('alpha', 1):.2f}</div>
                            <div class="stat-label">Anomalous Exponent</div>
                        </div>
                    """
                
                html += """
                    </div>
                </div>
                """
            
            # Check for clustering analysis
            if 'clustering' in analysis_results:
                clustering = analysis_results['clustering']
                html += f"""
                <div class="plot">
                    <h3>Clustering Analysis</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div class="stat-value">{clustering.get('n_clusters', 0)}</div>
                            <div class="stat-label">Number of Clusters</div>
                        </div>
                    </div>
                </div>
                """
        
        # Close HTML
        html += f"""
                <div class="footer">
                    <p>Generated with SPT Analysis | {len(tracks_df)} points in {tracks_df['track_id'].nunique()} tracks</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    except Exception as e:
        logger.error(f"Error creating summary report: {str(e)}")
        return f"<p>Error creating report: {str(e)}</p>"


def plot_msd_results(msd_df, pixel_size=0.1, frame_interval=0.1, ax=None, figsize=(10, 8),
                   ensemble=True, fit_curve=True, fit_range=None, individual_tracks=False):
    """
    Plot MSD results.
    
    Parameters
    ----------
    msd_df : pandas.DataFrame
        DataFrame with MSD results
    pixel_size : float, optional
        Pixel size in μm, by default 0.1
    frame_interval : float, optional
        Frame interval in seconds, by default 0.1
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 8)
    ensemble : bool, optional
        Whether to plot ensemble average, by default True
    fit_curve : bool, optional
        Whether to plot fit curve, by default True
    fit_range : tuple, optional
        Range for fitting (min_lag, max_lag), by default None
    individual_tracks : bool, optional
        Whether to plot individual tracks, by default False
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes object
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Plot individual tracks if requested
        if individual_tracks:
            track_ids = msd_df['track_id'].unique()
            
            # Limit to 20 random tracks if too many
            if len(track_ids) > 20:
                np.random.seed(42)  # For reproducibility
                track_ids = np.random.choice(track_ids, 20, replace=False)
            
            # Plot each track
            for track_id in track_ids:
                track_msd = msd_df[msd_df['track_id'] == track_id]
                
                # Group by lag
                grouped = track_msd.groupby('lag')
                mean_msd = grouped['msd'].mean()
                
                # Convert to physical units
                lag_time = grouped['lag'].first() * frame_interval
                msd_um = mean_msd * pixel_size**2
                
                # Plot track MSD
                ax.plot(lag_time, msd_um, 'o-', alpha=0.3, linewidth=1, markersize=3)
        
        # Plot ensemble average
        if ensemble:
            # Group by lag
            grouped = msd_df.groupby('lag')
            mean_msd = grouped['msd'].mean()
            std_msd = grouped['msd'].std()
            
            # Convert to physical units
            lag_time = grouped['lag'].first() * frame_interval
            msd_um = mean_msd * pixel_size**2
            std_um = std_msd * pixel_size**2
            
            # Plot ensemble average with errorbars
            ax.errorbar(lag_time, msd_um, yerr=std_um, fmt='o-', color='black', 
                       linewidth=2, markersize=6, label='Ensemble Average')
            
            # Fit curve
            if fit_curve:
                # Get fit range
                if fit_range is None:
                    min_lag = 1
                    max_lag = min(10, len(lag_time))
                else:
                    min_lag, max_lag = fit_range
                
                # Ensure valid range
                max_lag = min(max_lag, len(lag_time))
                min_lag = max(min_lag, 1)
                
                # Get data for fitting
                fit_time = lag_time.iloc[min_lag-1:max_lag]
                fit_msd = msd_um.iloc[min_lag-1:max_lag]
                
                # Linear fit: MSD = 4Dt
                from scipy.stats import linregress
                
                slope, intercept, r_value, p_value, std_err = linregress(fit_time, fit_msd)
                
                # Calculate diffusion coefficient
                D = slope / 4.0
                D_err = std_err / 4.0
                
                # Plot fit line
                fit_line = intercept + slope * fit_time
                ax.plot(fit_time, fit_line, '--', color='red', linewidth=2, 
                       label=f'Fit: D = {D:.4f} ± {D_err:.4f} μm²/s')
                
                # Add shaded area for fit range
                ax.axvspan(fit_time.iloc[0], fit_time.iloc[-1], alpha=0.1, color='red')
        
        # Set labels and title
        ax.set_xlabel('Lag Time (s)')
        ax.set_ylabel('MSD (μm²)')
        ax.set_title('Mean Squared Displacement')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        if ensemble or fit_curve:
            ax.legend()
        
        # Log-log scale often helps visualization
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting MSD results: {str(e)}")
        raise


def plot_diffusion_coefficient_distribution(diffusion_df, ax=None, figsize=(10, 6), 
                                          bins=30, log_scale=True, kde=True):
    """
    Plot distribution of diffusion coefficients.
    
    Parameters
    ----------
    diffusion_df : pandas.DataFrame
        DataFrame with diffusion coefficients
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 6)
    bins : int, optional
        Number of histogram bins, by default 30
    log_scale : bool, optional
        Whether to use log scale for x-axis, by default True
    kde : bool, optional
        Whether to plot kernel density estimate, by default True
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes object
    """
    try:
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Get diffusion coefficients
        D_values = diffusion_df['D'].values
        
        # Apply log transformation if requested
        if log_scale:
            # Filter out non-positive values
            D_values = D_values[D_values > 0]
            D_values = np.log10(D_values)
            xlabel = 'log₁₀ D (μm²/s)'
        else:
            xlabel = 'D (μm²/s)'
        
        # Plot histogram with KDE
        sns.histplot(D_values, bins=bins, kde=kde, ax=ax)
        
        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.set_title('Diffusion Coefficient Distribution')
        
        # Add statistics
        mean_D = np.mean(D_values)
        median_D = np.median(D_values)
        
        ax.axvline(mean_D, color='red', linestyle='--', label=f'Mean: {mean_D:.2f}')
        ax.axvline(median_D, color='green', linestyle=':', label=f'Median: {median_D:.2f}')
        
        # Add legend
        ax.legend()
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting diffusion coefficient distribution: {str(e)}")
        raise


def plot_correlation_matrix(data_df, columns=None, ax=None, figsize=(10, 8), cmap='coolwarm'):
    """
    Plot correlation matrix of selected columns.
    
    Parameters
    ----------
    data_df : pandas.DataFrame
        DataFrame with data
    columns : list, optional
        Columns to include in correlation matrix, by default None
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
    figsize : tuple, optional
        Figure size, by default (10, 8)
    cmap : str, optional
        Colormap name, by default 'coolwarm'
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes object
    """
    try:
        # Select columns
        if columns is not None:
            data = data_df[columns]
        else:
            data = data_df.select_dtypes(include=[np.number])
        
        # Compute correlation matrix
        corr = data.corr()
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        plt.colorbar(im, cax=cax)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)
        
        # Add correlation values
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(i, j, f'{corr.iloc[j, i]:.2f}', ha='center', va='center', 
                       color='white' if abs(corr.iloc[j, i]) > 0.5 else 'black')
        
        # Set title
        ax.set_title('Correlation Matrix')
        
        return ax
    
    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {str(e)}")
        raise
