
"""
Input/output module for SPT Analysis.

This module provides functions for loading and saving tracking data,
supporting various file formats including CSV, Excel, HDF5, and TIF stacks.
"""

import numpy as np
import pandas as pd
import os
import json
import yaml
import tifffile
import h5py
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


def load_image_stack(file_path, scale=1.0, subset=None):
    """
    Load microscopy image stack from file.
    
    Parameters
    ----------
    file_path : str
        Path to image file (TIF or similar)
    scale : float, optional
        Scaling factor for image intensity, by default 1.0
    subset : tuple, optional
        Frame subset to load (start, end), by default None
        
    Returns
    -------
    numpy.ndarray
        Image stack with shape (n_frames, height, width)
    """
    try:
        logger.info(f"Loading image stack from {file_path}")
        
        # Check file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Load image stack using tifffile
        with tifffile.TiffFile(file_path) as tif:
            if subset is not None:
                start, end = subset
                image_stack = tif.asarray(key=range(start, end))
            else:
                image_stack = tif.asarray()
        
        # Apply scaling
        if scale != 1.0:
            image_stack = image_stack * scale
        
        # Check if it's a proper stack
        if image_stack.ndim == 2:
            # Single image, add frame dimension
            image_stack = image_stack[np.newaxis, :, :]
        
        logger.info(f"Loaded image stack with shape {image_stack.shape}")
        
        return image_stack
    
    except Exception as e:
        logger.error(f"Error loading image stack: {str(e)}")
        raise


def save_image_stack(image_stack, file_path, compress=True):
    """
    Save image stack to file.
    
    Parameters
    ----------
    image_stack : numpy.ndarray
        Image stack with shape (n_frames, height, width)
    file_path : str
        Output file path
    compress : bool, optional
        Whether to apply compression, by default True
        
    Returns
    -------
    str
        Path to saved file
    """
    try:
        logger.info(f"Saving image stack to {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save image stack
        if compress:
            tifffile.imwrite(file_path, image_stack, compress=6)
        else:
            tifffile.imwrite(file_path, image_stack)
        
        logger.info(f"Saved image stack with shape {image_stack.shape}")
        
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving image stack: {str(e)}")
        raise


def load_tracks(file_path, format=None):
    """
    Load track data from file.
    
    Parameters
    ----------
    file_path : str
        Path to track data file
    format : str, optional
        File format ('csv', 'excel', 'hdf5', 'json'), by default None
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with track data
    """
    try:
        logger.info(f"Loading tracks from {file_path}")
        
        # Check file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Track file not found: {file_path}")
        
        # Determine format if not provided
        if format is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                format = 'csv'
            elif ext in ['.xls', '.xlsx']:
                format = 'excel'
            elif ext == '.h5':
                format = 'hdf5'
            elif ext == '.json':
                format = 'json'
            else:
                raise ValueError(f"Unknown file extension: {ext}")
        
        # Load data based on format
        if format == 'csv':
            tracks_df = pd.read_csv(file_path)
        elif format == 'excel':
            tracks_df = pd.read_excel(file_path)
        elif format == 'hdf5':
            with h5py.File(file_path, 'r') as f:
                # Check if there's a 'tracks' dataset
                if 'tracks' in f:
                    # HDF5 dataset to DataFrame
                    data = f['tracks'][:]
                    columns = f['tracks'].attrs.get('columns', [])
                    
                    if not columns:
                        # Try to infer columns
                        if data.dtype.names:
                            columns = list(data.dtype.names)
                        else:
                            columns = [f'col_{i}' for i in range(data.shape[1])]
                    
                    tracks_df = pd.DataFrame(data, columns=columns)
                else:
                    # Try pandas HDF5 format
                    tracks_df = pd.read_hdf(file_path, key='tracks')
        elif format == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # List of tracks
                tracks = []
                
                for track_id, track in enumerate(data):
                    for point in track:
                        point['track_id'] = point.get('track_id', track_id)
                        tracks.append(point)
                
                tracks_df = pd.DataFrame(tracks)
            else:
                # Direct DataFrame representation
                tracks_df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Ensure required columns exist
        required_columns = ['track_id', 'frame', 'x', 'y']
        missing_columns = [col for col in required_columns if col not in tracks_df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            
            # Try to map standard column names
            column_mapping = {}
            
            for req_col in missing_columns:
                # Common alternatives
                alternatives = {
                    'track_id': ['trajectory', 'particle', 'id', 'track'],
                    'frame': ['t', 'time', 'frame_number'],
                    'x': ['position_x', 'x_position', 'X'],
                    'y': ['position_y', 'y_position', 'Y']
                }
                
                # Try to find alternative
                for alt in alternatives.get(req_col, []):
                    if alt in tracks_df.columns:
                        column_mapping[alt] = req_col
                        break
            
            # Apply mapping
            if column_mapping:
                tracks_df = tracks_df.rename(columns=column_mapping)
                logger.info(f"Mapped columns: {column_mapping}")
                
                # Check again
                missing_columns = [col for col in required_columns if col not in tracks_df.columns]
        
        # If still missing columns, raise error
        if missing_columns:
            raise ValueError(f"Missing required columns after mapping: {missing_columns}")
        
        logger.info(f"Loaded {len(tracks_df)} track points, {tracks_df['track_id'].nunique()} tracks")
        
        return tracks_df
    
    except Exception as e:
        logger.error(f"Error loading tracks: {str(e)}")
        raise


def save_tracks(tracks_df, file_path, format=None):
    """
    Save track data to file.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    file_path : str
        Output file path
    format : str, optional
        File format ('csv', 'excel', 'hdf5', 'json'), by default None
        
    Returns
    -------
    str
        Path to saved file
    """
    try:
        logger.info(f"Saving tracks to {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Determine format if not provided
        if format is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                format = 'csv'
            elif ext in ['.xls', '.xlsx']:
                format = 'excel'
            elif ext == '.h5':
                format = 'hdf5'
            elif ext == '.json':
                format = 'json'
            else:
                raise ValueError(f"Unknown file extension: {ext}")
        
        # Save data based on format
        if format == 'csv':
            tracks_df.to_csv(file_path, index=False)
        elif format == 'excel':
            tracks_df.to_excel(file_path, index=False)
        elif format == 'hdf5':
            tracks_df.to_hdf(file_path, key='tracks', mode='w')
        elif format == 'json':
            # Group by track for better organization
            tracks_json = []
            
            for track_id, group in tracks_df.groupby('track_id'):
                track = group.drop('track_id', axis=1).to_dict('records')
                tracks_json.append({
                    'track_id': track_id,
                    'points': track
                })
            
            with open(file_path, 'w') as f:
                json.dump(tracks_json, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(tracks_df)} track points, {tracks_df['track_id'].nunique()} tracks")
        
        return file_path
Copy
    except Exception as e:
        logger.error(f"Error saving tracks: {str(e)}")
        raise


def load_config(file_path):
    """
    Load configuration from YAML or JSON file.
    
    Parameters
    ----------
    file_path : str
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    try:
        logger.info(f"Loading configuration from {file_path}")
        
        # Check file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Determine format
        ext = os.path.splitext(file_path)[1].lower()
        
        # Load configuration
        if ext in ['.yaml', '.yml']:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
        elif ext == '.json':
            with open(file_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {ext}")
        
        logger.info(f"Loaded configuration with {len(config)} top-level entries")
        
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise


def save_config(config, file_path):
    """
    Save configuration to YAML or JSON file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    file_path : str
        Output file path
        
    Returns
    -------
    str
        Path to saved file
    """
    try:
        logger.info(f"Saving configuration to {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Determine format
        ext = os.path.splitext(file_path)[1].lower()
        
        # Save configuration
        if ext in ['.yaml', '.yml']:
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif ext == '.json':
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration format: {ext}")
        
        logger.info(f"Saved configuration with {len(config)} top-level entries")
        
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        raise


def load_analysis_results(file_path):
    """
    Load analysis results from file.
    
    Parameters
    ----------
    file_path : str
        Path to results file
        
    Returns
    -------
    dict
        Dictionary of analysis results
    """
    try:
        logger.info(f"Loading analysis results from {file_path}")
        
        # Check file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Results file not found: {file_path}")
        
        # Determine format
        ext = os.path.splitext(file_path)[1].lower()
        
        # Load results
        if ext == '.h5':
            results = {}
            
            with h5py.File(file_path, 'r') as f:
                # Get list of datasets
                def get_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        # Convert dataset to appropriate type
                        if obj.attrs.get('type') == 'DataFrame':
                            # Convert to DataFrame
                            data = obj[:]
                            columns = obj.attrs.get('columns', [])
                            
                            if not columns:
                                if data.dtype.names:
                                    columns = list(data.dtype.names)
                                else:
                                    columns = [f'col_{i}' for i in range(data.shape[1])]
                            
                            results[name] = pd.DataFrame(data, columns=columns)
                        else:
                            # Regular array
                            results[name] = obj[:]
                
                # Traverse all datasets
                f.visititems(get_datasets)
        elif ext == '.json':
            with open(file_path, 'r') as f:
                results = json.load(f)
                
                # Convert DataFrame-like structures
                for key, value in results.items():
                    if isinstance(value, dict) and 'columns' in value and 'data' in value:
                        results[key] = pd.DataFrame(value['data'], columns=value['columns'])
        else:
            raise ValueError(f"Unsupported results format: {ext}")
        
        logger.info(f"Loaded analysis results with {len(results)} entries")
        
        return results
    
    except Exception as e:
        logger.error(f"Error loading analysis results: {str(e)}")
        raise


def save_analysis_results(results, file_path):
    """
    Save analysis results to file.
    
    Parameters
    ----------
    results : dict
        Dictionary of analysis results
    file_path : str
        Output file path
        
    Returns
    -------
    str
        Path to saved file
    """
    try:
        logger.info(f"Saving analysis results to {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Determine format
        ext = os.path.splitext(file_path)[1].lower()
        
        # Save results
        if ext == '.h5':
            with h5py.File(file_path, 'w') as f:
                for key, value in results.items():
                    if isinstance(value, pd.DataFrame):
                        # Save DataFrame
                        dataset = f.create_dataset(key, data=value.values)
                        dataset.attrs['type'] = 'DataFrame'
                        dataset.attrs['columns'] = list(value.columns)
                    elif isinstance(value, np.ndarray):
                        # Save array
                        f.create_dataset(key, data=value)
                    elif isinstance(value, (dict, list)):
                        # Save JSON serializable
                        f.create_dataset(key, data=np.array(str(json.dumps(value))))
                        f.attrs['type'] = 'json'
        elif ext == '.json':
            # Convert results to JSON serializable
            json_results = {}
            
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    # Convert DataFrame
                    json_results[key] = {
                        'columns': list(value.columns),
                        'data': value.to_dict('records')
                    }
                elif isinstance(value, np.ndarray):
                    # Convert array
                    json_results[key] = value.tolist()
                else:
                    # Use as is if JSON serializable
                    json_results[key] = value
            
            with open(file_path, 'w') as f:
                json.dump(json_results, f, indent=2)
        else:
            raise ValueError(f"Unsupported results format: {ext}")
        
        logger.info(f"Saved analysis results with {len(results)} entries")
        
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving analysis results: {str(e)}")
        raise


def import_trackmate_xml(file_path, scale_factor=1.0):
    """
    Import tracks from TrackMate XML file.
    
    Parameters
    ----------
    file_path : str
        Path to TrackMate XML file
    scale_factor : float, optional
        Scaling factor for coordinates, by default 1.0
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with track data
    """
    try:
        import xml.etree.ElementTree as ET
        
        logger.info(f"Importing TrackMate tracks from {file_path}")
        
        # Check file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"TrackMate file not found: {file_path}")
        
        # Parse XML
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Get particle data
        particle_data = []
        
        # Find all spots
        for spots in root.findall('.//AllSpots'):
            for frame in spots.findall('SpotsInFrame'):
                frame_id = int(frame.get('frame'))
                
                for spot in frame.findall('Spot'):
                    # Extract position and properties
                    spot_id = int(spot.get('ID'))
                    x = float(spot.get('POSITION_X')) * scale_factor
                    y = float(spot.get('POSITION_Y')) * scale_factor
                    z = float(spot.get('POSITION_Z')) * scale_factor if 'POSITION_Z' in spot.attrib else 0.0
                    
                    # Add to data
                    particle_data.append({
                        'spot_id': spot_id,
                        'frame': frame_id,
                        'x': x,
                        'y': y,
                        'z': z
                    })
        
        # Create DataFrame
        spots_df = pd.DataFrame(particle_data)
        
        # Create track mapping
        track_mapping = {}
        
        # Find all tracks
        for tracks in root.findall('.//Model/AllTracks'):
            for track in tracks.findall('Track'):
                track_id = int(track.get('TRACK_ID'))
                
                for edge in track.findall('Edge'):
                    source_id = int(edge.get('SPOT_SOURCE_ID'))
                    target_id = int(edge.get('SPOT_TARGET_ID'))
                    
                    track_mapping[source_id] = track_id
                    track_mapping[target_id] = track_id
        
        # Map tracks to spots
        spots_df['track_id'] = spots_df['spot_id'].map(track_mapping)
        
        # Remove spots without track
        tracks_df = spots_df.dropna(subset=['track_id']).copy()
        
        # Ensure track_id is integer
        tracks_df['track_id'] = tracks_df['track_id'].astype(int)
        
        logger.info(f"Imported {len(tracks_df)} track points, {tracks_df['track_id'].nunique()} tracks")
        
        return tracks_df
    
    except Exception as e:
        logger.error(f"Error importing TrackMate tracks: {str(e)}")
        raise


def import_from_fiji_results(file_path):
    """
    Import track data from Fiji Results table.
    
    Parameters
    ----------
    file_path : str
        Path to Fiji Results table (CSV or similar)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with track data
    """
    try:
        logger.info(f"Importing tracks from Fiji Results: {file_path}")
        
        # Check file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Fiji Results file not found: {file_path}")
        
        # Read file
        table_df = pd.read_csv(file_path, sep='\t')
        
        # Check expected columns
        expected_columns = ['Track', 'Slice', 'X', 'Y']
        missing_columns = [col for col in expected_columns if col not in table_df.columns]
        
        if missing_columns:
            # Try alternative columns
            alternatives = {
                'Track': ['Track_ID', 'TrackID', 'Trajectory', 'Particle'],
                'Slice': ['Frame', 'Time', 'T'],
                'X': ['x', 'xpos', 'X_position'],
                'Y': ['y', 'ypos', 'Y_position']
            }
            
            column_mapping = {}
            
            for missing_col in missing_columns:
                for alt in alternatives.get(missing_col, []):
                    if alt in table_df.columns:
                        column_mapping[alt] = missing_col
                        break
            
            # Apply mapping
            if column_mapping:
                table_df = table_df.rename(columns=column_mapping)
                logger.info(f"Mapped columns: {column_mapping}")
            
            # Check again
            missing_columns = [col for col in expected_columns if col not in table_df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns after mapping: {missing_columns}")
        
        # Create result DataFrame
        tracks_df = pd.DataFrame({
            'track_id': table_df['Track'],
            'frame': table_df['Slice'] - 1,  # Convert 1-based to 0-based indexing
            'x': table_df['X'],
            'y': table_df['Y']
        })
        
        # Add any additional columns
        additional_columns = [col for col in table_df.columns if col not in expected_columns]
        for col in additional_columns:
            tracks_df[col.lower()] = table_df[col]
        
        logger.info(f"Imported {len(tracks_df)} track points, {tracks_df['track_id'].nunique()} tracks")
        
        return tracks_df
    
    except Exception as e:
        logger.error(f"Error importing from Fiji Results: {str(e)}")
        raise


def export_to_csv(tracks_df, file_path, include_metadata=True):
    """
    Export tracks to CSV file with optional metadata.
    
    Parameters
    ----------
    tracks_df : pandas.DataFrame
        DataFrame with track data
    file_path : str
        Output file path
    include_metadata : bool, optional
        Whether to include metadata, by default True
        
    Returns
    -------
    str
        Path to saved file
    """
    try:
        logger.info(f"Exporting tracks to CSV: {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Create a copy of the DataFrame
        export_df = tracks_df.copy()
        
        # Add metadata if requested
        if include_metadata:
            # Add export timestamp
            from datetime import datetime
            
            export_df['export_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Add software info
            export_df['software'] = 'SPT Analysis'
            
            # Add version info
            try:
                from .. import __version__
                export_df['version'] = __version__
            except:
                export_df['version'] = 'unknown'
        
        # Save to CSV
        export_df.to_csv(file_path, index=False)
        
        logger.info(f"Exported {len(export_df)} track points, {export_df['track_id'].nunique()} tracks")
        
        return file_path
    
    except Exception as e:
        logger.error(f"Error exporting tracks to CSV: {str(e)}")
        raise


def save_plot(fig, file_path, dpi=300, format=None):
    """
    Save plot to file.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    file_path : str
        Output file path
    dpi : int, optional
        Resolution in dots per inch, by default 300
    format : str, optional
        Output format ('png', 'pdf', 'svg', etc.), by default None
        
    Returns
    -------
    str
        Path to saved file
    """
    try:
        logger.info(f"Saving plot to {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Determine format if not provided
        if format is None:
            format = os.path.splitext(file_path)[1].lower().lstrip('.')
            if not format:
                format = 'png'
                file_path += '.png'
        
        # Save figure
        fig.savefig(file_path, dpi=dpi, format=format, bbox_inches='tight')
        
        logger.info(f"Saved plot to {file_path}")
        
        return file_path
    
    except Exception as e:
        logger.error(f"Error saving plot: {str(e)}")
        raise
