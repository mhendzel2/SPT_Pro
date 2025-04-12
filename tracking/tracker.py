
"""
Tracker module for SPT Analysis.

This module integrates detection and linking to create a complete tracking pipeline
for single-particle tracking, with capabilities for handling complex tracking scenarios.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from . import detector, linker

logger = logging.getLogger(__name__)


class ParticleTracker:
    """
    Main tracker class that integrates detection and linking.
    
    Parameters
    ----------
    detector_method : str, optional
        Detection method to use, by default "wavelet"
    detector_params : dict, optional
        Parameters for the detector, by default None
    linker_method : str, optional
        Linking method to use, by default "graph"
    linker_params : dict, optional
        Parameters for the linker, by default None
    """
    
    def __init__(self, detector_method="wavelet", detector_params=None,
                 linker_method="graph", linker_params=None):
        self.detector_method = detector_method
        self.detector_params = detector_params or {}
        self.linker_method = linker_method
        self.linker_params = linker_params or {}
        
        # Initialize detector and linker
        self.detector = detector.get_detector(detector_method, **self.detector_params)
        self.linker = linker.get_linker(linker_method, **self.linker_params)
        
        # Storage for tracks
        self._tracks = None
        self._track_df = None
    
    def track(self, frames, batch_size=None):
        """
        Perform tracking on a sequence of frames.
        
        Parameters
        ----------
        frames : list or numpy.ndarray
            List or array of image frames
        batch_size : int, optional
            Batch size for processing frames, by default None
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with track data
        """
        try:
            n_frames = len(frames)
            frame_indices = list(range(n_frames))
            
            logger.info(f"Starting tracking with {n_frames} frames")
            
            # Detect particles in all frames
            detections = []
            
            if batch_size is None:
                # Process all frames at once
                for i, frame in enumerate(frames):
                    logger.debug(f"Detecting particles in frame {i}/{n_frames}")
                    frame_detections = self.detector.detect(frame)
                    detections.append(frame_detections)
            else:
                # Process frames in batches
                for batch_start in range(0, n_frames, batch_size):
                    batch_end = min(batch_start + batch_size, n_frames)
                    batch_frames = frames[batch_start:batch_end]
                    batch_indices = frame_indices[batch_start:batch_end]
                    
                    logger.debug(f"Processing batch {batch_start}-{batch_end}")
                    
                    for i, frame in zip(batch_indices, batch_frames):
                        logger.debug(f"Detecting particles in frame {i}/{n_frames}")
                        frame_detections = self.detector.detect(frame)
                        detections.append(frame_detections)
            
            logger.info(f"Detected particles in {n_frames} frames")
            
            # Link detections into tracks
            logger.info("Linking particles into tracks")
            tracks = self.linker.link(detections, frame_indices)
            
            logger.info(f"Found {len(tracks)} tracks")
            
            # Convert tracks to DataFrame
            track_data = []
            
            for track_id, track in enumerate(tracks):
                for frame, frame_idx, det_idx in track:
                    position = detections[frame_idx][det_idx]
                    track_data.append({
                        'track_id': track_id,
                        'frame': frame,
                        'y': position[0],
                        'x': position[1]
                    })
            
            self._tracks = tracks
            self._track_df = pd.DataFrame(track_data)
            
            return self._track_df
        
        except Exception as e:
            logger.error(f"Error in tracking: {str(e)}")
            raise
    
    def get_tracks(self):
        """
        Get the tracks detected by the tracker.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with track data
        """
        if self._track_df is None:
            raise ValueError("No tracks available. Run track() first.")
        
        return self._track_df
    
    def track_diagnostics(self):
        """
        Get diagnostics for the tracked particles.
        
        Returns
        -------
        dict
            Dictionary with track diagnostics
        """
        if self._track_df is None:
            raise ValueError("No tracks available. Run track() first.")
        
        # Calculate track statistics
        track_lengths = self._track_df.groupby('track_id').size()
        total_tracks = len(track_lengths)
        mean_track_length = track_lengths.mean()
        median_track_length = track_lengths.median()
        max_track_length = track_lengths.max()
        
        # Calculate frame statistics
        detections_per_frame = self._track_df.groupby('frame').size()
        mean_detections = detections_per_frame.mean()
        
        # Calculate displacement statistics
        tracks_grouped = self._track_df.sort_values(['track_id', 'frame']).groupby('track_id')
        
        displacements = []
        
        for _, track in tracks_grouped:
            if len(track) >= 2:
                # Calculate displacements between consecutive frames
                for i in range(len(track) - 1):
                    x1, y1 = track.iloc[i][['x', 'y']]
                    x2, y2 = track.iloc[i+1][['x', 'y']]
                    
                    displacement = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    displacements.append(displacement)
        
        mean_displacement = np.mean(displacements) if displacements else 0
        median_displacement = np.median(displacements) if displacements else 0
        max_displacement = np.max(displacements) if displacements else 0
        
        return {
            'total_tracks': total_tracks,
            'mean_track_length': mean_track_length,
            'median_track_length': median_track_length,
            'max_track_length': max_track_length,
            'mean_detections_per_frame': mean_detections,
            'mean_displacement': mean_displacement,
            'median_displacement': median_displacement,
            'max_displacement': max_displacement
        }
    
    def filter_tracks(self, min_length=None, max_gap=None, roi=None):
        """
        Filter tracks based on criteria.
        
        Parameters
        ----------
        min_length : int, optional
            Minimum track length, by default None
        max_gap : int, optional
            Maximum allowed gap in frames, by default None
        roi : tuple, optional
            Region of interest (x_min, y_min, x_max, y_max), by default None
            
        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame with track data
        """
        if self._track_df is None:
            raise ValueError("No tracks available. Run track() first.")
        
        filtered_df = self._track_df.copy()
        
        # Filter by track length
        if min_length is not None:
            track_lengths = filtered_df.groupby('track_id').size()
            valid_tracks = track_lengths[track_lengths >= min_length].index
            filtered_df = filtered_df[filtered_df['track_id'].isin(valid_tracks)]
        
        # Filter by maximum gap
        if max_gap is not None:
            # Function to check gaps in track
            def has_valid_gaps(track, max_gap):
                frames = sorted(track['frame'].unique())
                
                for i in range(len(frames) - 1):
                    if frames[i+1] - frames[i] > max_gap + 1:
                        return False
                
                return True
            
            # Check each track
            valid_tracks = []
            
            for track_id, track in filtered_df.groupby('track_id'):
                if has_valid_gaps(track, max_gap):
                    valid_tracks.append(track_id)
            
            filtered_df = filtered_df[filtered_df['track_id'].isin(valid_tracks)]
        
        # Filter by region of interest
        if roi is not None:
            x_min, y_min, x_max, y_max = roi
            
            # Keep only points inside ROI
            filtered_df = filtered_df[
                (filtered_df['x'] >= x_min) & 
                (filtered_df['x'] <= x_max) & 
                (filtered_df['y'] >= y_min) & 
                (filtered_df['y'] <= y_max)
            ]
            
            # Remove tracks that are too short after ROI filtering
            if min_length is not None:
                track_lengths = filtered_df.groupby('track_id').size()
                valid_tracks = track_lengths[track_lengths >= min_length].index
                filtered_df = filtered_df[filtered_df['track_id'].isin(valid_tracks)]
        
        return filtered_df
