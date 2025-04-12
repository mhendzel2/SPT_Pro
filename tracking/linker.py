
"""
Track linking module for SPT Analysis.

This module provides algorithms for linking particle detections into tracks,
with a focus on handling challenging conditions such as high particle density,
temporary disappearances, and crossing trajectories.
"""

import numpy as np
import scipy.optimize
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class GraphBasedLinker:
    """
    Graph-based global optimization for track linking.
    
    This linker formulates track linking as a global optimization problem,
    which is particularly effective for high-density scenarios.
    
    Parameters
    ----------
    max_distance : float, optional
        Maximum distance for linking between frames, by default 15.0
    max_gap : int, optional
        Maximum number of frames to allow gap closing, by default 2
    cost_factor : float, optional
        Cost scaling factor for distance, by default 1.5
    """
    
    def __init__(self, max_distance=15.0, max_gap=2, cost_factor=1.5):
        self.max_distance = max_distance
        self.max_gap = max_gap
        self.cost_factor = cost_factor
    
    def link(self, detections, frames):
        """
        Link detections into tracks.
        
        Parameters
        ----------
        detections : list
            List of detection arrays, one per frame
        frames : list
            List of frame numbers
            
        Returns
        -------
        list
            List of tracks, each containing a list of detection indices
        """
        try:
            n_frames = len(frames)
            
            if n_frames < 2:
                logger.warning("Need at least 2 frames for linking")
                return []
            
            # Build graph
            G = nx.DiGraph()
            
            # Add source and sink nodes
            G.add_node('source')
            G.add_node('sink')
            
            # Add detection nodes
            for frame_idx, frame_dets in enumerate(detections):
                for det_idx, detection in enumerate(frame_dets):
                    node_id = f"f{frame_idx}_d{det_idx}"
                    G.add_node(node_id, pos=detection, frame=frames[frame_idx])
                    
                    # Connect source to all detections in first frame
                    if frame_idx == 0:
                        G.add_edge('source', node_id, weight=0)
                    
                    # Connect all detections in last frame to sink
                    if frame_idx == n_frames - 1:
                        G.add_edge(node_id, 'sink', weight=0)
            
            # Add edges between detections in consecutive frames
            for curr_frame_idx in range(n_frames - 1):
                for next_frame_idx in range(curr_frame_idx + 1, min(curr_frame_idx + self.max_gap + 1, n_frames)):
                    gap = next_frame_idx - curr_frame_idx
                    
                    # Skip if no detections in either frame
                    if len(detections[curr_frame_idx]) == 0 or len(detections[next_frame_idx]) == 0:
                        continue
                    
                    # Calculate distance matrix
                    curr_dets = detections[curr_frame_idx]
                    next_dets = detections[next_frame_idx]
                    
                    for curr_idx, curr_det in enumerate(curr_dets):
                        curr_node = f"f{curr_frame_idx}_d{curr_idx}"
                        
                        for next_idx, next_det in enumerate(next_dets):
                            next_node = f"f{next_frame_idx}_d{next_idx}"
                            
                            # Calculate distance
                            distance = np.sqrt(np.sum((curr_det - next_det)**2))
                            
                            # Skip if distance is too large
                            if distance > self.max_distance * (gap**0.5):
                                continue
                            
                            # Cost increases with distance and gap
                            cost = distance * self.cost_factor**gap
                            
                            # Add edge
                            G.add_edge(curr_node, next_node, weight=cost)
            
            # Find minimum cost paths using a greedy approach
            paths = []
            
            # Find source-connected nodes
            source_nodes = list(G.successors('source'))
            
            # Greedy path finding
            while source_nodes:
                best_path = None
                best_cost = float('inf')
                
                for start_node in source_nodes:
                    # Find a path from this node to sink
                    try:
                        path = nx.shortest_path(G, start_node, 'sink', weight='weight')
                        cost = sum(G[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_path = path
                    except nx.NetworkXNoPath:
                        continue
                
                if best_path:
                    # Add path to results
                    paths.append(best_path[:-1])  # Exclude sink node
                    
                    # Remove nodes in this path from graph
                    G.remove_nodes_from(best_path[:-1])
                    
                    # Update source nodes
                    source_nodes = list(G.successors('source'))
                else:
                    break
            
            # Convert paths to tracks
            tracks = []
            
            for path in paths:
                track = []
                
                for node in path:
                    if node != 'source' and node != 'sink':
                        frame_idx, det_idx = map(int, node.replace('f', '').replace('d', '').split('_'))
                        track.append((frames[frame_idx], frame_idx, det_idx))
                
                tracks.append(sorted(track))
            
            return tracks
        
        except Exception as e:
            logger.error(f"Error in graph-based linking: {str(e)}")
            raise


class MultipleHypothesisTracker:
    """
    Multiple Hypothesis Tracking (MHT) for ambiguous track linking.
    
    This tracker maintains multiple track hypotheses when associations are ambiguous,
    resolving them when more information becomes available.
    
    Parameters
    ----------
    max_gap_closing : int, optional
        Maximum number of frames to allow gap closing, by default 3
    max_distance : float, optional
        Maximum distance for linking between frames, by default 15.0
    max_hypotheses : int, optional
        Maximum number of hypotheses to maintain, by default 10
    """
    
    def __init__(self, max_gap_closing=3, max_distance=15.0, max_hypotheses=10):
        self.max_gap_closing = max_gap_closing
        self.max_distance = max_distance
        self.max_hypotheses = max_hypotheses
    
    def link(self, detections, frames):
        """
        Link detections into tracks using multiple hypothesis tracking.
        
        Parameters
        ----------
        detections : list
            List of detection arrays, one per frame
        frames : list
            List of frame numbers
            
        Returns
        -------
        list
            List of tracks, each containing a list of detection indices
        """
        try:
            n_frames = len(frames)
            
            if n_frames < 2:
                logger.warning("Need at least 2 frames for linking")
                return []
            
            # Initialize active tracks
            active_tracks = []
            
            # Process first frame
            for i in range(len(detections[0])):
                track = [(frames[0], 0, i)]
                active_tracks.append(track)
            
            # Process subsequent frames
            for frame_idx in range(1, n_frames):
                frame_dets = detections[frame_idx]
                frame_num = frames[frame_idx]
                
                if len(frame_dets) == 0:
                    continue
                
                # For each active track, generate hypotheses
                all_hypotheses = []
                
                for track_idx, track in enumerate(active_tracks):
                    last_frame, last_frame_idx, last_det_idx = track[-1]
                    
                    # Skip if track hasn't been updated recently
                    if last_frame_idx < frame_idx - self.max_gap_closing:
                        continue
                    
                    # Get last detection position
                    last_pos = detections[last_frame_idx][last_det_idx]
                    
                    # Generate hypotheses for this track
                    track_hypotheses = []
                    
                    # Option 1: Track ends
                    track_hypotheses.append((track_idx, None, 0.0))
                    
                    # Option 2: Track continues with one of the detections
                    for det_idx, detection in enumerate(frame_dets):
                        distance = np.sqrt(np.sum((last_pos - detection)**2))
                        
                        if distance <= self.max_distance:
                            cost = distance
                            track_hypotheses.append((track_idx, det_idx, cost))
                    
                    all_hypotheses.extend(track_hypotheses)
                
                # Sort hypotheses by cost
                all_hypotheses.sort(key=lambda x: x[2])
                
                # Greedy hypothesis selection
                used_tracks = set()
                used_detections = set()
                selected_hypotheses = []
                
                for track_idx, det_idx, cost in all_hypotheses:
                    if track_idx in used_tracks:
                        continue
                    
                    if det_idx is not None and det_idx in used_detections:
                        continue
                    
                    selected_hypotheses.append((track_idx, det_idx, cost))
                    
                    if det_idx is not None:
                        used_tracks.add(track_idx)
                        used_detections.add(det_idx)
                    
                    if len(selected_hypotheses) >= min(len(active_tracks), len(frame_dets) + len(active_tracks) - len(used_tracks)):
                        break
                
                # Update tracks based on selected hypotheses
                updated_tracks = []
                
                for track_idx, det_idx, _ in selected_hypotheses:
                    if det_idx is not None:
                        # Track continues
                        new_track = active_tracks[track_idx].copy()
                        new_track.append((frame_num, frame_idx, det_idx))
                        updated_tracks.append(new_track)
                    else:
                        # Track ends
                        updated_tracks.append(active_tracks[track_idx])
                
                # Create new tracks for unmatched detections
                for det_idx in range(len(frame_dets)):
                    if det_idx not in used_detections:
                        new_track = [(frame_num, frame_idx, det_idx)]
                        updated_tracks.append(new_track)
                
                # Update active tracks
                active_tracks = updated_tracks
            
            # Filter out very short tracks
            filtered_tracks = [track for track in active_tracks if len(track) >= 3]
            
            return filtered_tracks
        
        except Exception as e:
            logger.error(f"Error in multiple hypothesis tracking: {str(e)}")
            raise


class IMMLinker:
    """
    Interacting Multiple Model (IMM) filter for track linking.
    
    This linker adapts to different types of motion (Brownian, directed, confined)
    by switching between motion models.
    
    Parameters
    ----------
    motion_models : list, optional
        List of motion models to use, by default ['brownian', 'directed', 'confined']
    max_distance : float, optional
        Maximum distance for linking between frames, by default 15.0
    """
    
    def __init__(self, motion_models=None, max_distance=15.0):
        self.motion_models = motion_models or ['brownian', 'directed', 'confined']
        self.max_distance = max_distance
        
        # Model transition matrix
        self.model_trans_prob = np.array([
            [0.8, 0.1, 0.1],  # Brownian to (Brownian, Directed, Confined)
            [0.1, 0.8, 0.1],  # Directed to (Brownian, Directed, Confined)
            [0.1, 0.1, 0.8]   # Confined to (Brownian, Directed, Confined)
        ])
    
    def _predict_brownian(self, pos, vel, dt=1.0):
        """Predict next position using Brownian motion model"""
        # Brownian motion has zero mean displacement
        pred_pos = pos
        pred_vel = np.zeros_like(vel)
        
        # Uncertainty grows with time
        uncertainty = np.sqrt(dt) * 5.0
        
        return pred_pos, pred_vel, uncertainty
    
    def _predict_directed(self, pos, vel, dt=1.0):
        """Predict next position using directed motion model"""
        # Directed motion continues with constant velocity
        pred_pos = pos + vel * dt
        pred_vel = vel
        
        # Fixed uncertainty
        uncertainty = 3.0
        
        return pred_pos, pred_vel, uncertainty
    
    def _predict_confined(self, pos, vel, dt=1.0):
        """Predict next position using confined motion model"""
        # Confined motion has a restoring force towards origin
        center = np.zeros_like(pos)
        for track_pos in self.track_history[-min(5, len(self.track_history)):]:
            center += track_pos
        center /= min(5, len(self.track_history))
        
        # Predict position with drift towards center
        pred_pos = pos + 0.5 * (center - pos) * dt
        pred_vel = (center - pos) * 0.5
        
        # Small uncertainty
        uncertainty = 2.0
        
        return pred_pos, pred_vel, uncertainty
    
    def link(self, detections, frames):
        """
        Link detections into tracks using IMM filtering.
        
        Parameters
        ----------
        detections : list
            List of detection arrays, one per frame
        frames : list
            List of frame numbers
            
        Returns
        -------
        list
            List of tracks, each containing a list of detection indices
        """
        try:
            n_frames = len(frames)
            
            if n_frames < 2:
                logger.warning("Need at least 2 frames for linking")
                return []
            
            # Initialize active tracks
            active_tracks = []
            track_states = []
            
            # Process first frame - initialize tracks
            for i in range(len(detections[0])):
                active_tracks.append([(frames[0], 0, i)])
                
                # Initialize state with position, velocity, and model probabilities
                state = {
                    'pos': detections[0][i],
                    'vel': np.zeros(2),
                    'model_probs': np.array([0.8, 0.1, 0.1])  # Initially favor Brownian
                }
                track_states.append(state)
            
            # Process subsequent frames
            for frame_idx in range(1, n_frames):
                frame_dets = detections[frame_idx]
                frame_num = frames[frame_idx]
                
                if len(frame_dets) == 0:
                    continue
                
                # For each active track, predict next position using IMM
                predictions = []
                
                for track_idx, (track, state) in enumerate(zip(active_tracks, track_states)):
                    # Skip tracks that haven't been updated recently
                    if track[-1][1] < frame_idx - 1:
                        predictions.append(None)
                        continue
                    
                    # Get current state
                    pos = state['pos']
                    vel = state['vel']
                    model_probs = state['model_probs']
                    
                    # Predict using each model
                    model_predictions = []
                    model_uncertainties = []
                    
                    # Save track history for confined model
                    self.track_history = [detections[t[1]][t[2]] for t in track]
                    
                    # Brownian model prediction
                    pred_pos_b, pred_vel_b, uncertainty_b = self._predict_brownian(pos, vel)
                    model_predictions.append((pred_pos_b, pred_vel_b))
                    model_uncertainties.append(uncertainty_b)
                    
                    # Directed model prediction
                    pred_pos_d, pred_vel_d, uncertainty_d = self._predict_directed(pos, vel)
                    model_predictions.append((pred_pos_d, pred_vel_d))
                    model_uncertainties.append(uncertainty_d)
                    
                    # Confined model prediction
                    pred_pos_c, pred_vel_c, uncertainty_c = self._predict_confined(pos, vel)
                    model_predictions.append((pred_pos_c, pred_vel_c))
                    model_uncertainties.append(uncertainty_c)
                    
                    # Combine predictions using model probabilities
                    combined_pos = np.zeros(2)
                    for i, (pred_pos, _) in enumerate(model_predictions):
                        combined_pos += model_probs[i] * pred_pos
                    
                    # Calculate predicted position and uncertainty
                    pred_pos = combined_pos
                    uncertainty = np.sum(model_probs * np.array(model_uncertainties))
                    
                    predictions.append((pred_pos, uncertainty))
                
                # Create cost matrix for assignment
                cost_matrix = np.zeros((len(active_tracks), len(frame_dets)))
                cost_matrix.fill(float('inf'))
                
                for track_idx, prediction in enumerate(predictions):
                    if prediction is None:
                        continue
                    
                    pred_pos, uncertainty = prediction
                    
                    for det_idx, detection in enumerate(frame_dets):
                        distance = np.sqrt(np.sum((pred_pos - detection)**2))
                        
                        if distance <= self.max_distance:
                            # Cost based on distance and uncertainty
                            cost = distance / (uncertainty + 1e-6)
                            cost_matrix[track_idx, det_idx] = cost
                
                # Solve assignment problem
                track_indices, det_indices = scipy.optimize.linear_sum_assignment(cost_matrix)
                
                # Update matched tracks
                updated_tracks = []
                updated_states = []
                matched_dets = set()
                
                for track_idx, det_idx in zip(track_indices, det_indices):
                    if cost_matrix[track_idx, det_idx] < float('inf'):
                        # Update track
                        track = active_tracks[track_idx].copy()
                        track.append((frame_num, frame_idx, det_idx))
                        updated_tracks.append(track)
                        
                        # Calculate velocity
                        curr_pos = frame_dets[det_idx]
                        prev_pos = detections[track[-2][1]][track[-2][2]]
                        dt = frame_num - track[-2][0]
                        vel = (curr_pos - prev_pos) / max(dt, 1)
                        
                        # Update state
                        state = track_states[track_idx].copy()
                        state['pos'] = curr_pos
                        state['vel'] = vel
                        
                        # Update model probabilities based on prediction error
                        if predictions[track_idx] is not None:
                            pred_pos, _ = predictions[track_idx]
                            error = np.sqrt(np.sum((pred_pos - curr_pos)**2))
                            
                            # Simple update rule: favor models with lower error
                            model_errors = []
                            
                            for i, motion_model in enumerate(self.motion_models):
                                if motion_model == 'brownian':
                                    model_pred, _, _ = self._predict_brownian(state['pos'], state['vel'])
                                elif motion_model == 'directed':
                                    model_pred, _, _ = self._predict_directed(state['pos'], state['vel'])
                                elif motion_model == 'confined':
Copy                                    model_pred, _, _ = self._predict_confined(state['pos'], state['vel'])
                                
                                model_error = np.sqrt(np.sum((model_pred - curr_pos)**2))
                                model_errors.append(model_error)
                            
                            # Convert errors to probabilities (smaller error -> higher probability)
                            model_errors = np.array(model_errors) + 1e-6  # Avoid division by zero
                            model_likelihoods = 1.0 / model_errors
                            model_likelihoods /= np.sum(model_likelihoods)
                            
                            # Combine with prior using model transition matrix
                            mixed_probs = np.dot(state['model_probs'], self.model_trans_prob)
                            posterior = mixed_probs * model_likelihoods
                            posterior /= np.sum(posterior)
                            
                            state['model_probs'] = posterior
                        
                        updated_states.append(state)
                        matched_dets.add(det_idx)
                
                # Add unmatched tracks that are still recent
                for track_idx, track in enumerate(active_tracks):
                    if track_idx not in track_indices and track[-1][1] >= frame_idx - 1:
                        updated_tracks.append(track)
                        updated_states.append(track_states[track_idx])
                
                # Create new tracks for unmatched detections
                for det_idx in range(len(frame_dets)):
                    if det_idx not in matched_dets:
                        new_track = [(frame_num, frame_idx, det_idx)]
                        updated_tracks.append(new_track)
                        
                        # Initialize state for new track
                        new_state = {
                            'pos': frame_dets[det_idx],
                            'vel': np.zeros(2),
                            'model_probs': np.array([0.8, 0.1, 0.1])  # Initially favor Brownian
                        }
                        updated_states.append(new_state)
                
                # Update active tracks and states
                active_tracks = updated_tracks
                track_states = updated_states
            
            # Filter out very short tracks
            filtered_tracks = [track for track in active_tracks if len(track) >= 3]
            
            return filtered_tracks
        
        except Exception as e:
            logger.error(f"Error in IMM tracking: {str(e)}")
            raise


def get_linker(method="graph", **kwargs):
    """
    Factory function to get appropriate linker based on method.
    
    Parameters
    ----------
    method : str, optional
        Linking method, one of 'graph', 'mht', 'imm', by default "graph"
    **kwargs
        Additional parameters for the linker
    
    Returns
    -------
    object
        Linker object
    
    Raises
    ------
    ValueError
        If method is unknown
    """
    methods = {
        "graph": GraphBasedLinker,
        "mht": MultipleHypothesisTracker,
        "imm": IMMLinker
    }
    
    if method not in methods:
        raise ValueError(f"Unknown linking method: {method}. Available methods: {list(methods.keys())}")
    
    return methods[method](**kwargs)
