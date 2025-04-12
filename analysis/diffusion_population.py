# Detect significant change
if abs(msj2 - msj1) > 0.5 * max(msj1, msj2):
    segment_boundaries.append((current_segment_start, i))
    current_segment_start = i

# Add final segment
segment_boundaries.append((current_segment_start, len(squared_jumps)))

else:
    raise ValueError(f"Unknown segmentation method: {method}")

# Filter out segments that are too short
segment_boundaries = [
    seg for seg in segment_boundaries 
    if seg[1] - seg[0] >= self.min_segment_length
]

# Analyze each segment
segments = []

for start_idx, end_idx in segment_boundaries:
    # Get segment positions and frames
    segment_positions = positions[start_idx:end_idx+1]
    segment_frames = frames[start_idx:end_idx+1]

    # Skip if segment is too short
    if len(segment_positions) < self.min_segment_length:
        continue

    # Calculate MSD for different lag times
    lag_times = []
    msd_values = []

    for lag in range(1, min(11, len(segment_positions)//2)):
        disp = segment_positions[lag:] - segment_positions[:-lag]
        sq_disp = np.sum(disp**2, axis=1)
        lag_times.append(lag * self.dt)
        msd_values.append(np.mean(sq_disp))

    # Skip if MSD calculation failed
    if not msd_values:
        continue

    # Fit power law to MSD (MSD ~ t^alpha)
    try:
        log_tau = np.log(lag_times)
        log_msd = np.log(msd_values)
        slope, intercept = np.polyfit(log_tau, log_msd, 1)
        
        alpha = slope
        D = np.exp(intercept - np.log(4))
    except Exception:
        alpha = None
        D = None

    # Calculate mean jump
    segment_jumps = np.sqrt(np.sum(np.diff(segment_positions, axis=0)**2, axis=1))
    mean_jump = np.mean(segment_jumps)

    # Classify diffusion mode
    if alpha is not None:
        if alpha < 0.7:
            diffusion_mode = "Subdiffusion"
        elif alpha > 1.3:
            diffusion_mode = "Superdiffusion"
        else:
            diffusion_mode = "Normal diffusion"
    else:
        diffusion_mode = "Unknown"

    # Store segment information
    segments.append({
        'start_idx': start_idx,
        'end_idx': end_idx,
        'start_frame': segment_frames[0],
        'end_frame': segment_frames[-1],
        'n_frames': len(segment_frames),
        'positions': segment_positions,
        'alpha': alpha,
        'diffusion_coefficient': D,
        'mean_jump': mean_jump,
        'diffusion_mode': diffusion_mode
    })

# Store segments for this track
segmented_trajectories[track_id] = segments

# Store results
self.segmented_trajectories = segmented_trajectories

return segmented_trajectories
