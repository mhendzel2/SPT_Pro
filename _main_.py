
"""
Command-line interface for SPT Analysis.

This module provides a command-line interface for the SPT Analysis package,
allowing users to perform tracking and analysis directly from the terminal.
"""

import argparse
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import yaml

# Import package modules
from spt_analysis import __version__
from spt_analysis.tracking import tracker
from spt_analysis.analysis import diffusion, motion, clustering
from spt_analysis.utils import io, processing, visualization

# Set up logging
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='SPT Analysis: Single Particle Tracking Analysis Tool'
    )
    
    parser.add_argument('--version', action='version', version=f'SPT Analysis {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Track command
    track_parser = subparsers.add_parser('track', help='Track particles in image stack')
    track_parser.add_argument('input', help='Input image stack file (TIFF)')
    track_parser.add_argument('output', help='Output tracks file (CSV, JSON, HDF5)')
    track_parser.add_argument('--detector', choices=['wavelet', 'unet', 'local_maxima'],
                            default='wavelet', help='Detection method')
    track_parser.add_argument('--linker', choices=['graph', 'mht', 'imm'],
                            default='graph', help='Linking method')
    track_parser.add_argument('--preprocess', action='store_true',
                            help='Apply preprocessing to image stack')
    track_parser.add_argument('--min-track-length', type=int, default=3,
                            help='Minimum track length to keep')
    track_parser.add_argument('--config', help='Configuration file (YAML, JSON)')
    track_parser.add_argument('--plot', action='store_true',
                            help='Generate and save plots')
    track_parser.add_argument('--plot-output', help='Output directory for plots')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze tracking data')
    analyze_parser.add_argument('input', help='Input tracks file (CSV, JSON, HDF5)')
    analyze_parser.add_argument('output', help='Output results file (JSON, HDF5)')
    analyze_parser.add_argument('--analysis', choices=['diffusion', 'motion', 'clustering', 'all'],
                              default='all', help='Analysis type')
    analyze_parser.add_argument('--pixel-size', type=float, default=0.1,
                              help='Pixel size in μm')
    analyze_parser.add_argument('--frame-interval', type=float, default=0.1,
                              help='Frame interval in seconds')
    analyze_parser.add_argument('--min-track-length', type=int, default=10,
                              help='Minimum track length for analysis')
    analyze_parser.add_argument('--config', help='Configuration file (YAML, JSON)')
    analyze_parser.add_argument('--plot', action='store_true',
                              help='Generate and save plots')
    analyze_parser.add_argument('--plot-output', help='Output directory for plots')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize tracking data')
    viz_parser.add_argument('input', help='Input tracks file (CSV, JSON, HDF5)')
    viz_parser.add_argument('--image', help='Optional background image file (TIFF)')
    viz_parser.add_argument('--output', help='Output file (PDF, HTML)')
    viz_parser.add_argument('--type', choices=['trajectories', 'summary', 'animation', 'interactive'],
                          default='summary', help='Visualization type')
    viz_parser.add_argument('--pixel-size', type=float, default=0.1,
                          help='Pixel size in μm')
    viz_parser.add_argument('--frame-interval', type=float, default=0.1,
                          help='Frame interval in seconds')
    viz_parser.add_argument('--config', help='Configuration file (YAML, JSON)')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert tracking data between formats')
    convert_parser.add_argument('input', help='Input tracks file')
    convert_parser.add_argument('output', help='Output tracks file')
    convert_parser.add_argument('--input-format', choices=['csv', 'excel', 'hdf5', 'json', 'trackmate', 'fiji'],
                              help='Input format (defaults to file extension)')
    convert_parser.add_argument('--output-format', choices=['csv', 'excel', 'hdf5', 'json'],
                              help='Output format (defaults to file extension)')
    
    # GUI command (new)
    gui_parser = subparsers.add_parser('gui', help='Launch graphical user interface')
    
    return parser.parse_args()


def get_configuration(config_file, default_config=None):
    """
    Load configuration from file.
    
    Parameters
    ----------
    config_file : str
        Path to configuration file
    default_config : dict, optional
        Default configuration, by default None
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    # Start with default config if provided
    config = default_config.copy() if default_config else {}
    
    # Load from file if provided
    if config_file and os.path.exists(config_file):
        try:
            loaded_config = io.load_config(config_file)
            config.update(loaded_config)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    return config


def run_tracking(args):
    """
    Run particle tracking.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    logger.info("Starting particle tracking")
    
    # Load configuration
    default_config = {
        'detector': {
            'method': args.detector,
            'params': {}
        },
        'linker': {
            'method': args.linker,
            'params': {}
        },
        'preprocessing': {
            'apply': args.preprocess,
            'steps': [
                {'name': 'denoise', 'method': 'gaussian', 'sigma': 1.0},
                {'name': 'background', 'method': 'rolling_ball', 'radius': 50},
                {'name': 'enhance', 'method': 'dog', 'sigma_low': 1.0, 'sigma_high': 2.0},
                {'name': 'normalize', 'method': 'minmax'}
            ]
        },
        'tracking': {
            'min_track_length': args.min_track_length
        }
    }
    
    config = get_configuration(args.config, default_config)
    
    # Load image stack
    try:
        image_stack = io.load_image_stack(args.input)
        logger.info(f"Loaded image stack with shape {image_stack.shape}")
    except Exception as e:
        logger.error(f"Error loading image stack: {str(e)}")
        sys.exit(1)
    
    # Preprocess image stack if requested
    if config['preprocessing']['apply']:
        logger.info("Preprocessing image stack")
        try:
            image_stack = processing.preprocess_stack(
                image_stack, 
                steps=config['preprocessing']['steps']
            )
        except Exception as e:
            logger.error(f"Error preprocessing image stack: {str(e)}")
            sys.exit(1)
    
    # Initialize tracker
    try:
        particle_tracker = tracker.ParticleTracker(
            detector_method=config['detector']['method'],
            detector_params=config['detector']['params'],
            linker_method=config['linker']['method'],
            linker_params=config['linker']['params']
        )
    except Exception as e:
        logger.error(f"Error initializing tracker: {str(e)}")
        sys.exit(1)
    
    # Perform tracking
    try:
        logger.info("Performing tracking")
        tracks_df = particle_tracker.track(image_stack)
        logger.info(f"Tracking completed: {len(tracks_df)} points in {tracks_df['track_id'].nunique()} tracks")
    except Exception as e:
        logger.error(f"Error during tracking: {str(e)}")
        sys.exit(1)
    
    # Filter short tracks
    min_length = config['tracking']['min_track_length']
    if min_length > 1:
        try:
            filtered_df = particle_tracker.filter_tracks(min_length=min_length)
            logger.info(f"Filtered tracks: {len(filtered_df)} points in {filtered_df['track_id'].nunique()} tracks")
            tracks_df = filtered_df
        except Exception as e:
            logger.error(f"Error filtering tracks: {str(e)}")
    
    # Save tracks
    try:
        io.save_tracks(tracks_df, args.output)
        logger.info(f"Saved tracks to {args.output}")
    except Exception as e:
        logger.error(f"Error saving tracks: {str(e)}")
        sys.exit(1)
    
    # Generate and save plots if requested
    if args.plot:
        plot_dir = args.plot_output or os.path.dirname(args.output)
        os.makedirs(plot_dir, exist_ok=True)
        
        try:
            # Create summary figure
            logger.info("Generating summary plots")
            fig = visualization.plot_tracking_summary(tracks_df)
            plot_path = os.path.join(plot_dir, 'tracking_summary.png')
            visualization.save_figure(fig, plot_path)
            plt.close(fig)
            logger.info(f"Saved summary plot to {plot_path}")
            
            # Create detection overlay figure (first and last frames)
            for frame_idx in [0, image_stack.shape[0] - 1]:
                frame = image_stack[frame_idx]
                frame_tracks = tracks_df[tracks_df['frame'] == frame_idx]
                
                if len(frame_tracks) > 0:
                    detections = frame_tracks[['y', 'x']].values
                    fig = visualization.plot_detection_overlay(frame, detections)
                    plot_path = os.path.join(plot_dir, f'detections_frame_{frame_idx}.png')
                    visualization.save_figure(fig, plot_path)
                    plt.close(fig)
            
            # Track trajectories plot
            fig, ax = plt.subplots(figsize=(10, 8))
            visualization.plot_track_trajectories(ax, tracks_df)
            plot_path = os.path.join(plot_dir, 'trajectories.png')
            visualization.save_figure(fig, plot_path)
            plt.close(fig)
            
            # Create diagnostics plots
            try:
                diagnostics = particle_tracker.track_diagnostics()
                with open(os.path.join(plot_dir, 'track_diagnostics.json'), 'w') as f:
                    json.dump(diagnostics, f, indent=2)
            except Exception as e:
                logger.error(f"Error generating diagnostics: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
    
    logger.info("Tracking completed successfully")


def run_analysis(args):
    """
    Run tracking data analysis.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    logger.info("Starting tracking data analysis")
    
    # Load configuration
    default_config = {
        'diffusion': {
            'method': 'ensemble',
            'max_lag': 20,
            'min_track_length': args.min_track_length
        },
        'motion': {
            'min_track_length': args.min_track_length,
            'window_size': 5
        },
        'clustering': {
            'eps': 0.5,
            'min_samples': 5,
            'method': 'dbscan'
        },
        'general': {
            'pixel_size': args.pixel_size,
            'frame_interval': args.frame_interval
        }
    }
    
    config = get_configuration(args.config, default_config)
    
    # Load tracks
    try:
        tracks_df = io.load_tracks(args.input)
        logger.info(f"Loaded {len(tracks_df)} track points in {tracks_df['track_id'].nunique()} tracks")
    except Exception as e:
        logger.error(f"Error loading tracks: {str(e)}")
        sys.exit(1)
    
    # Initialize results dictionary
    results = {}
    
    # Perform analyses
    if args.analysis in ['diffusion', 'all']:
        logger.info("Performing diffusion analysis")
        pixel_size = config['general']['pixel_size']
        frame_interval = config['general']['frame_interval']
        
        try:
            diffusion_analyzer = diffusion.DiffusionAnalyzer(
                pixel_size=pixel_size, 
                frame_interval=frame_interval
            )
            
            # Compute MSD
            msd_df, ensemble_df = diffusion_analyzer.compute_msd(
                tracks_df, 
                max_lag=config['diffusion']['max_lag'], 
                min_track_length=config['diffusion']['min_track_length']
            )
            
            # Fit diffusion models
            fit_results = diffusion_analyzer.fit_diffusion_models(ensemble_df)
            
            # Compute diffusion coefficients
            diffusion_df = diffusion_analyzer.compute_diffusion_coefficient(
                tracks_df, 
                method=config['diffusion']['method'], 
                min_track_length=config['diffusion']['min_track_length']
            )
            
            # Store results
            results['diffusion'] = {
                'msd_df': msd_df,
                'ensemble_df': ensemble_df,
                'fit_results': fit_results,
                'diffusion_df': diffusion_df
            }
            
            logger.info("Diffusion analysis completed")
            
            if args.plot:
                plot_dir = args.plot_output or os.path.dirname(args.output)
                os.makedirs(plot_dir, exist_ok=True)
                
                # MSD plot
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_msd_results(
                    msd_df, 
                    pixel_size=pixel_size, 
                    frame_interval=frame_interval, 
                    ax=ax, 
                    ensemble=True, 
                    fit_curve=True
                )
                plot_path = os.path.join(plot_dir, 'msd_analysis.png')
                visualization.save_figure(fig, plot_path)
                plt.close(fig)
                
                # Diffusion coefficient distribution
                if isinstance(diffusion_df, pd.DataFrame):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    visualization.plot_diffusion_coefficient_distribution(
                        diffusion_df, 
                        ax=ax, 
                        log_scale=True
                    )
                    plot_path = os.path.join(plot_dir, 'diffusion_coefficient_distribution.png')
                    visualization.save_figure(fig, plot_path)
                    plt.close(fig)
        
        except Exception as e:
            logger.error(f"Error in diffusion analysis: {str(e)}")
    
    if args.analysis in ['motion', 'all']:
        logger.info("Performing motion analysis")
        pixel_size = config['general']['pixel_size']
        frame_interval = config['general']['frame_interval']
        
        try:
            motion_analyzer = motion.MotionAnalyzer(
                pixel_size=pixel_size, 
                frame_interval=frame_interval
            )
            
            # Compute velocities
            velocity_df = motion_analyzer.compute_velocities(
                tracks_df, 
                smooth=True, 
                window=3
            )
            
            # Compute confinement ratio
            confinement_df = motion_analyzer.compute_confinement_ratio(
                tracks_df, 
                window_size=config['motion']['window_size']
            )
            
            # Compute turning angles
            turning_df = motion_analyzer.compute_turning_angles(
                tracks_df, 
                min_displacement=0.1
            )
            
            # Analyze track shapes
            shape_df = motion_analyzer.analyze_track_shape(
                tracks_df, 
                min_track_length=config['motion']['min_track_length']
            )
            
            # Store results
            results['motion'] = {
                'velocity_df': velocity_df,
                'confinement_df': confinement_df,
                'turning_df': turning_df,
                'shape_df': shape_df
            }
            
            logger.info("Motion analysis completed")
            
            if args.plot:
                plot_dir = args.plot_output or os.path.dirname(args.output)
                os.makedirs(plot_dir, exist_ok=True)
                
                # Velocity distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(velocity_df['avg_speed'], bins=30, alpha=0.7, color='steelblue')
                ax.set_xlabel('Average Speed (μm/s)')
                ax.set_ylabel('Count')
                ax.set_title('Speed Distribution')
                plot_path = os.path.join(plot_dir, 'speed_distribution.png')
                visualization.save_figure(fig, plot_path)
                plt.close(fig)
                
                # Turning angle distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(turning_df['turning_angle_degrees'], bins=36, alpha=0.7, color='steelblue')
                ax.set_xlabel('Turning Angle (degrees)')
                ax.set_ylabel('Count')
                ax.set_title('Turning Angle Distribution')
                plot_path = os.path.join(plot_dir, 'turning_angle_distribution.png')
                visualization.save_figure(fig, plot_path)
                plt.close(fig)
        
        except Exception as e:
            logger.error(f"Error in motion analysis: {str(e)}")
    
    if args.analysis in ['clustering', 'all']:
        logger.info("Performing clustering analysis")
        pixel_size = config['general']['pixel_size']
        
        try:
            cluster_analyzer = clustering.ClusterAnalyzer(pixel_size=pixel_size)
            
            # Perform clustering analysis
            cluster_df, clustered_df = cluster_analyzer.analyze_clusters(
                tracks_df, 
                eps=config['clustering']['eps'], 
                min_samples=config['clustering']['min_samples']
            )
            
            # Compute Ripley's K function
            ripley_k = cluster_analyzer.compute_ripley_k(
                tracks_df, 
                r_max=10.0, 
                n_points=20
            )
            
            # Identify hotspots
            hotspot_df, density_grid = cluster_analyzer.identify_hotspots(
                tracks_df, 
                bandwidth=1.0, 
                threshold=0.7
            )
            
            # Store results
            results['clustering'] = {
                'cluster_df': cluster_df,
                'clustered_df': clustered_df,
                'ripley_k': ripley_k,
                'hotspot_df': hotspot_df,
                'density_grid': density_grid
            }
            
            logger.info("Clustering analysis completed")
            
            if args.plot:
                plot_dir = args.plot_output or os.path.dirname(args.output)
                os.makedirs(plot_dir, exist_ok=True)
                
                # Spatial clusters
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_spatial_clusters(
                    tracks_df, 
                    clustered_df, 
                    cluster_df, 
                    ax=ax, 
                    pixel_size=pixel_size
                )
                plot_path = os.path.join(plot_dir, 'spatial_clusters.png')
                visualization.save_figure(fig, plot_path)
                plt.close(fig)
                
                # Ripley's K function
                fig, ax = plt.subplots(figsize=(10, 6))
                visualization.plot_ripley_k(ripley_k, ax=ax)
                plot_path = os.path.join(plot_dir, 'ripley_k.png')
                visualization.save_figure(fig, plot_path)
                plt.close(fig)
                
                # Density heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_density_heatmap(
                    density_grid, 
                    tracks_df=tracks_df, 
                    hotspot_df=hotspot_df, 
                    ax=ax
                )
                plot_path = os.path.join(plot_dir, 'density_heatmap.png')
                visualization.save_figure(fig, plot_path)
                plt.close(fig)
        
        except Exception as e:
            logger.error(f"Error in clustering analysis: {str(e)}")
    
    # Save results
    try:
        io.save_analysis_results(results, args.output)
        logger.info(f"Saved analysis results to {args.output}")
    except Exception as e:
        logger.error(f"Error saving analysis results: {str(e)}")
        sys.exit(1)
    
    logger.info("Analysis completed successfully")


def run_visualization(args):
    """
    Run tracking data visualization.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    logger.info("Starting tracking data visualization")
    
    # Load configuration
    default_config = {
        'visualization': {
            'type': args.type,
            'cmap': 'viridis',
            'max_tracks': 100,
            'trail_length': 10,
            'figure_dpi': 300
        },
        'general': {
            'pixel_size': args.pixel_size,
            'frame_interval': args.frame_interval
        }
    }
    
    config = get_configuration(args.config, default_config)
    
    # Load tracks
    try:
        tracks_df = io.load_tracks(args.input)
        logger.info(f"Loaded {len(tracks_df)} track points in {tracks_df['track_id'].nunique()} tracks")
    except Exception as e:
        logger.error(f"Error loading tracks: {str(e)}")
        sys.exit(1)
    
    # Load image if provided
    image = None
    if args.image:
        try:
            image_stack = io.load_image_stack(args.image)
            image = image_stack[0]  # Use first frame as background
            logger.info(f"Loaded background image with shape {image.shape}")
        except Exception as e:
            logger.warning(f"Error loading background image: {str(e)}")
    
    # Determine output path
    output_path = args.output
    if not output_path:
        base_dir = os.path.dirname(args.input)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        
        if args.type == 'trajectories':
            output_path = os.path.join(base_dir, f"{base_name}_trajectories.pdf")
        elif args.type == 'summary':
            output_path = os.path.join(base_dir, f"{base_name}_summary.pdf")
        elif args.type == 'animation':
            output_path = os.path.join(base_dir, f"{base_name}_animation.mp4")
        elif args.type == 'interactive':
            output_path = os.path.join(base_dir, f"{base_name}_interactive.html")
    
    # Create visualization
    if args.type == 'trajectories':
        logger.info("Creating trajectory plot")
        try:
            # Create PDF with trajectory plot
            with PdfPages(output_path) as pdf:
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_track_trajectories(
                    ax, 
                    tracks_df, 
                    max_tracks=config['visualization']['max_tracks'], 
                    cmap=config['visualization']['cmap']
                )
                
                if image is not None:
                    ax.imshow(image, cmap='gray', alpha=0.3)
                
                pdf.savefig(fig)
                plt.close(fig)
            
            logger.info(f"Saved trajectory plot to {output_path}")
        
        except Exception as e:
            logger.error(f"Error creating trajectory plot: {str(e)}")
            sys.exit(1)
    
    elif args.type == 'summary':
        logger.info("Creating summary report")
        try:
            # Create PDF with multiple plots
            with PdfPages(output_path) as pdf:
                # Trajectory plot
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_track_trajectories(
                    ax, 
                    tracks_df, 
                    max_tracks=config['visualization']['max_tracks'], 
                    cmap=config['visualization']['cmap']
                )
                pdf.savefig(fig)
                plt.close(fig)
                
                # Track length histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                visualization.plot_track_length_histogram(ax, tracks_df)
                pdf.savefig(fig)
                plt.close(fig)
                
                # Particles per frame
                fig, ax = plt.subplots(figsize=(10, 6))
                visualization.plot_particles_per_frame(ax, tracks_df)
                pdf.savefig(fig)
                plt.close(fig)
                
                # Displacement histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                visualization.plot_displacement_histogram(ax, tracks_df)
                pdf.savefig(fig)
                plt.close(fig)
                
                # Track positions
                fig, ax = plt.subplots(figsize=(10, 8))
                visualization.plot_track_positions(
                    ax, 
                    tracks_df, 
                    position_type='start', 
                    cmap=config['visualization']['cmap']
                )
                pdf.savefig(fig)
                plt.close(fig)
            
            logger.info(f"Saved summary report to {output_path}")
        
        except Exception as e:
            logger.error(f"Error creating summary report: {str(e)}")
            sys.exit(1)
    
    elif args.type == 'animation':
        logger.info("Creating tracking animation")
        try:
            if args.image:
                # Load full image stack
                image_stack = io.load_image_stack(args.image)
                
                # Create animation
                visualization.create_tracking_animation(
                    image_stack, 
                    tracks_df, 
                    output_path, 
                    fps=10, 
                    dpi=100, 
                    marker_size=10, 
                    trail_length=config['visualization']['trail_length'], 
                    cmap=config['visualization']['cmap']
                )
            else:
                logger.error("Image stack required for animation")
                sys.exit(1)
            
            logger.info(f"Saved tracking animation to {output_path}")
        
        except Exception as e:
            logger.error(f"Error creating tracking animation: {str(e)}")
            sys.exit(1)
    
    elif args.type == 'interactive':
        logger.info("Creating interactive visualization")
        try:
            # Create HTML file with interactive plot
            html = visualization.create_trajectory_plot_html(
                tracks_df, 
                width=800, 
                height=600
            )
            
            with open(output_path, 'w') as f:
                f.write(html)
            
            logger.info(f"Saved interactive visualization to {output_path}")
        
        except Exception as e:
            logger.error(f"Error creating interactive visualization: {str(e)}")
            sys.exit(1)
    
    logger.info("Visualization completed successfully")


def run_conversion(args):
    """
    Run track data conversion.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    logger.info("Starting track data conversion")
    
    try:
        # Determine input format
        input_format = args.input_format
        if not input_format:
            ext = os.path.splitext(args.input)[1].lower()
            if ext == '.csv':
                input_format = 'csv'
            elif ext in ['.xls', '.xlsx']:
                input_format = 'excel'
            elif ext == '.h5':
                input_format = 'hdf5'
            elif ext == '.json':
                input_format = 'json'
            elif ext == '.xml':
                input_format = 'trackmate'
            else:
                logger.warning(f"Could not determine input format from extension {ext}, assuming CSV")
                input_format = 'csv'
        
        # Load tracks based on format
        if input_format == 'trackmate':
            tracks_df = io.import_trackmate_xml(args.input)
        elif input_format == 'fiji':
            tracks_df = io.import_from_fiji_results(args.input)
        else:
            tracks_df = io.load_tracks(args.input, format=input_format)
        
        logger.info(f"Loaded {len(tracks_df)} track points in {tracks_df['track_id'].nunique()} tracks")
        
        # Determine output format
        output_format = args.output_format
        if not output_format:
            ext = os.path.splitext(args.output)[1].lower()
            if ext == '.csv':
                output_format = 'csv'
            elif ext in ['.xls', '.xlsx']:
                output_format = 'excel'
            elif ext == '.h5':
                output_format = 'hdf5'
            elif ext == '.json':
                output_format = 'json'
            else:
                logger.warning(f"Could not determine output format from extension {ext}, assuming CSV")
                output_format = 'csv'
        
        # Save tracks in new format
        io.save_tracks(tracks_df, args.output, format=output_format)
        
        logger.info(f"Converted tracks from {input_format} to {output_format}")
        logger.info(f"Saved converted tracks to {args.output}")
    
    except Exception as e:
        logger.error(f"Error converting tracks: {str(e)}")
        sys.exit(1)


def launch_gui():
    """Launch the graphical user interface."""
    try:
        from PyQt5.QtWidgets import
Copy
def launch_gui():
    """Launch the graphical user interface."""
    try:
        from PyQt5.QtWidgets import QApplication
        from spt_analysis.gui.main_window import EnhancedTrackingAnalysisGUI
    except ImportError:
        logger.error("PyQt5 is required for the GUI. Please install with 'pip install pyqt5'")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    window = EnhancedTrackingAnalysisGUI()
    window.show()
    sys.exit(app.exec_())


def main():
    """Main function to handle command-line interface."""
    args = parse_arguments()
    
    if args.command == 'track':
        run_tracking(args)
    elif args.command == 'analyze':
        run_analysis(args)
    elif args.command == 'visualize':
        run_visualization(args)
    elif args.command == 'convert':
        run_conversion(args)
    elif args.command == 'gui':
        launch_gui()
    else:
        print("Please specify a command. Use -h or --help for help.")
        sys.exit(1)


if __name__ == "__main__":
    main()
