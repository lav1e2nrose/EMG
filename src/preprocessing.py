"""
Preprocessing module for EMG signal analysis.
Includes filtering, segmentation, and sliding window operations.
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, hilbert

THREE_STATE_RATIO_MIN = 0.1
THREE_STATE_RATIO_MAX = 0.9

# Improved segmentation parameters
DEFAULT_RMS_WINDOW_MS = 50      # Window size for RMS envelope calculation (ms)
DEFAULT_MIN_SEGMENT_MS = 500    # Minimum segment duration (ms) 
DEFAULT_MERGE_GAP_MS = 300      # Merge segments closer than this (ms)
DEFAULT_SAMPLING_RATE = 2000    # Default sampling rate (Hz)

# Fallback parameters for segment detection when correlation matching fails
FALLBACK_MIN_SEGMENT_MS_OPTIONS = [200, 300, 500]  # Try increasingly strict min segment lengths
FALLBACK_THRESHOLD_MULTIPLIERS = [1.0, 0.8, 0.6, 0.5]  # Try increasingly sensitive thresholds


def load_emg_data(filepath):
    """
    Load EMG data from CSV file.
    
    Args:
        filepath: str, path to CSV file
    
    Returns:
        np.array: EMG signal data (1D array)
    """
    # Try reading without header first
    data = pd.read_csv(filepath, header=None)
    
    # Check if first row is a header by trying to convert to float
    first_val = data.iloc[0, 0]
    is_header = False
    if isinstance(first_val, str):
        try:
            float(first_val)
        except ValueError:
            # First value is a non-numeric string, so this is likely a header
            is_header = True
    
    if is_header:
        # File has a header, reload with header
        data = pd.read_csv(filepath)
        # Use second column if available (typically amplitude data), otherwise first
        if data.shape[1] > 1:
            return data.iloc[:, 1].values.astype(float).flatten()
        else:
            return data.iloc[:, 0].values.astype(float).flatten()
    
    # If multi-column, use second column (typically amplitude data)
    if data.shape[1] > 1:
        return data.iloc[:, 1].values.astype(float).flatten()
    
    return data.values.astype(float).flatten()


def bandpass_filter(data, lowcut=20, highcut=450, fs=2000, order=4):
    """
    Apply bandpass filter to EMG signal.
    
    Args:
        data: array-like, EMG signal
        lowcut: float, lower cutoff frequency in Hz
        highcut: float, upper cutoff frequency in Hz
        fs: int, sampling rate in Hz
        order: int, filter order
    
    Returns:
        np.array: filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data


def notch_filter(data, freq=50, fs=2000, quality=30):
    """
    Apply notch filter to remove power line interference.
    
    Args:
        data: array-like, EMG signal
        freq: float, frequency to remove (e.g., 50Hz or 60Hz)
        fs: int, sampling rate in Hz
        quality: float, quality factor
    
    Returns:
        np.array: filtered signal
    """
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    
    b, a = iirnotch(w0, quality)
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data


def preprocess_signal(data, fs=2000):
    """
    Apply full preprocessing pipeline to EMG signal.
    
    Args:
        data: array-like, raw EMG signal
        fs: int, sampling rate in Hz
    
    Returns:
        np.array: preprocessed signal
    """
    # Apply bandpass filter
    data_bp = bandpass_filter(data, lowcut=20, highcut=450, fs=fs)
    
    # Apply notch filter to remove 50Hz power line interference
    data_filtered = notch_filter(data_bp, freq=50, fs=fs)
    
    return data_filtered


def create_sliding_windows(data, window_size, step_size):
    """
    Create sliding windows from signal data.
    
    Args:
        data: array-like, signal data
        window_size: int, size of each window in samples
        step_size: int, step size between windows in samples
    
    Returns:
        np.array: array of windows, shape (n_windows, window_size)
        np.array: array of window start indices
    """
    windows = []
    start_indices = []
    
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        windows.append(data[start:end])
        start_indices.append(start)
    
    return np.array(windows), np.array(start_indices)


def compute_rms(window):
    """
    Compute Root Mean Square (RMS) of a signal window.
    
    Args:
        window: array-like, signal window
    
    Returns:
        float: RMS value
    """
    return np.sqrt(np.mean(window ** 2))


def compute_rms_envelope(signal, window_size, fs=2000):
    """
    Compute RMS envelope of the signal using a sliding window.
    
    Args:
        signal: array-like, EMG signal
        window_size: int, window size in samples
        fs: int, sampling rate in Hz
    
    Returns:
        np.array: RMS envelope
    """
    if window_size <= 0:
        window_size = int(DEFAULT_RMS_WINDOW_MS * fs / 1000)
    
    # Use convolution for efficient RMS computation
    squared = signal ** 2
    kernel = np.ones(window_size) / window_size
    mean_squared = np.convolve(squared, kernel, mode='same')
    rms_envelope = np.sqrt(mean_squared)
    
    return rms_envelope


def compute_mav_envelope(signal, window_size):
    """
    Compute Mean Absolute Value (MAV) envelope of the signal.
    
    Args:
        signal: array-like, EMG signal
        window_size: int, window size in samples
    
    Returns:
        np.array: MAV envelope
    """
    kernel = np.ones(window_size) / window_size
    mav_envelope = np.convolve(np.abs(signal), kernel, mode='same')
    return mav_envelope


def compute_adaptive_threshold(envelope, method='otsu', percentile=75):
    """
    Compute adaptive threshold for activity detection.
    
    Args:
        envelope: array-like, signal envelope (RMS or MAV)
        method: str, thresholding method ('otsu', 'percentile', 'mean_std')
        percentile: int, percentile value for 'percentile' method
    
    Returns:
        float: threshold value
    """
    if method == 'otsu':
        # Otsu's method for automatic thresholding
        hist, bin_edges = np.histogram(envelope, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Compute cumulative sums and means
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        
        mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
        mean2 = (np.cumsum((hist * bin_centers)[::-1])[::-1]) / (weight2 + 1e-10)
        
        # Compute between-class variance
        variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
        
        # Find optimal threshold
        idx = np.argmax(variance)
        threshold = bin_centers[idx]
        
    elif method == 'percentile':
        threshold = np.percentile(envelope, percentile)
        
    elif method == 'mean_std':
        # Mean + standard deviation based threshold
        mean_val = np.mean(envelope)
        std_val = np.std(envelope)
        threshold = mean_val + 0.5 * std_val
        
    else:
        # Default: use median
        threshold = np.median(envelope)
    
    return threshold


def detect_activity_regions(signal, fs=2000, rms_window_ms=None, 
                            min_segment_ms=None, merge_gap_ms=None,
                            threshold_method='otsu', threshold_multiplier=1.0):
    """
    Detect activity regions in EMG signal using RMS envelope and adaptive thresholding.
    This is the improved segmentation algorithm that uses:
    1. RMS envelope for smooth activity detection
    2. Adaptive thresholding (Otsu's method by default)
    3. Minimum segment length filtering
    4. Segment merging for close segments
    
    Args:
        signal: array-like, preprocessed EMG signal
        fs: int, sampling rate in Hz
        rms_window_ms: int, RMS window size in ms (default: 50)
        min_segment_ms: int, minimum segment duration in ms (default: 500)
        merge_gap_ms: int, merge segments closer than this in ms (default: 300)
        threshold_method: str, method for threshold computation ('otsu', 'percentile', 'mean_std')
        threshold_multiplier: float, multiplier for threshold adjustment
    
    Returns:
        list: list of tuples (start, end) for each detected segment
    """
    # Use default values if not provided
    if rms_window_ms is None:
        rms_window_ms = DEFAULT_RMS_WINDOW_MS
    if min_segment_ms is None:
        min_segment_ms = DEFAULT_MIN_SEGMENT_MS
    if merge_gap_ms is None:
        merge_gap_ms = DEFAULT_MERGE_GAP_MS
    
    # Convert ms to samples
    rms_window = int(rms_window_ms * fs / 1000)
    min_segment_samples = int(min_segment_ms * fs / 1000)
    merge_gap_samples = int(merge_gap_ms * fs / 1000)
    
    # Compute RMS envelope
    rms_env = compute_rms_envelope(signal, rms_window, fs)
    
    # Compute adaptive threshold
    threshold = compute_adaptive_threshold(rms_env, method=threshold_method)
    threshold *= threshold_multiplier
    
    # Find regions above threshold
    above_threshold = rms_env > threshold
    
    # Find segment boundaries
    segments = []
    in_segment = False
    segment_start = 0
    
    for i, is_active in enumerate(above_threshold):
        if is_active and not in_segment:
            in_segment = True
            segment_start = i
        elif not is_active and in_segment:
            in_segment = False
            segment_end = i
            segments.append((segment_start, segment_end))
    
    # Handle case where signal ends while in segment
    if in_segment:
        segments.append((segment_start, len(signal)))
    
    # Filter by minimum length
    filtered_segments = []
    for start, end in segments:
        if end - start >= min_segment_samples:
            filtered_segments.append((start, end))
    segments = filtered_segments
    
    # Merge close segments
    if len(segments) > 1:
        merged = [segments[0]]
        for cur_start, cur_end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if cur_start - prev_end <= merge_gap_samples:
                # Merge with previous
                merged[-1] = (prev_start, cur_end)
            else:
                merged.append((cur_start, cur_end))
        segments = merged
    
    return segments


def segment_signal_improved(signal, fs=2000, rms_window_ms=50, 
                           min_segment_ms=500, merge_gap_ms=300,
                           use_dual_threshold=True):
    """
    Improved signal segmentation using multi-feature approach.
    
    This function combines:
    1. RMS envelope for amplitude detection
    2. MAV envelope for activity confirmation
    3. Dual-threshold hysteresis for stable boundaries
    4. Post-processing for segment refinement
    
    Args:
        signal: array-like, preprocessed EMG signal
        fs: int, sampling rate in Hz
        rms_window_ms: int, RMS window size in ms
        min_segment_ms: int, minimum segment duration in ms
        merge_gap_ms: int, merge segments closer than this in ms
        use_dual_threshold: bool, use hysteresis thresholding
    
    Returns:
        list: list of tuples (start, end) for each detected segment
    """
    # Convert ms to samples
    rms_window = max(1, int(rms_window_ms * fs / 1000))
    min_segment_samples = int(min_segment_ms * fs / 1000)
    merge_gap_samples = int(merge_gap_ms * fs / 1000)
    
    # Compute RMS envelope
    rms_env = compute_rms_envelope(signal, rms_window, fs)
    
    # Compute MAV envelope for confirmation
    mav_env = compute_mav_envelope(signal, rms_window)
    
    # Normalize envelopes
    rms_norm = rms_env / (np.max(rms_env) + 1e-10)
    mav_norm = mav_env / (np.max(mav_env) + 1e-10)
    
    # Combined envelope (weighted average)
    combined_env = 0.7 * rms_norm + 0.3 * mav_norm
    
    if use_dual_threshold:
        # Dual threshold (hysteresis) for stable boundaries
        # High threshold for segment start, low threshold for segment end
        high_thresh = compute_adaptive_threshold(combined_env, method='otsu')
        low_thresh = high_thresh * 0.6  # 60% of high threshold
        
        # Apply hysteresis thresholding
        segments = []
        in_segment = False
        segment_start = 0
        
        for i, val in enumerate(combined_env):
            if not in_segment and val > high_thresh:
                in_segment = True
                segment_start = i
            elif in_segment and val < low_thresh:
                in_segment = False
                segments.append((segment_start, i))
        
        if in_segment:
            segments.append((segment_start, len(signal)))
    else:
        # Simple thresholding
        threshold = compute_adaptive_threshold(combined_env, method='otsu')
        above_threshold = combined_env > threshold
        
        segments = []
        in_segment = False
        segment_start = 0
        
        for i, is_active in enumerate(above_threshold):
            if is_active and not in_segment:
                in_segment = True
                segment_start = i
            elif not is_active and in_segment:
                in_segment = False
                segments.append((segment_start, i))
        
        if in_segment:
            segments.append((segment_start, len(signal)))
    
    # Filter by minimum length
    filtered_segments = []
    for start, end in segments:
        if end - start >= min_segment_samples:
            filtered_segments.append((start, end))
    segments = filtered_segments
    
    # Merge close segments
    if len(segments) > 1:
        merged = [segments[0]]
        for cur_start, cur_end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if cur_start - prev_end <= merge_gap_samples:
                merged[-1] = (prev_start, cur_end)
            else:
                merged.append((cur_start, cur_end))
        segments = merged
    
    # Refine segment boundaries by extending to local minima
    refined_segments = []
    for start, end in segments:
        # Extend start backward to local minimum (within limit)
        search_start = max(0, start - min_segment_samples // 4)
        if search_start < start:
            local_min_start = search_start + np.argmin(combined_env[search_start:start])
            # Only extend if it's actually a minimum
            if combined_env[local_min_start] < combined_env[start] * 0.8:
                start = local_min_start
        
        # Extend end forward to local minimum (within limit)
        search_end = min(len(signal), end + min_segment_samples // 4)
        if end < search_end and end > 0:
            local_min_end = end + np.argmin(combined_env[end:search_end])
            if local_min_end < len(combined_env) and combined_env[local_min_end] < combined_env[end-1] * 0.8:
                end = local_min_end
        
        refined_segments.append((start, end))
    
    return refined_segments


def label_windows_from_segments(signal_length, segment_ranges, window_size, step_size,
                                overlap_threshold=0.1):
    """
    Create binary labels for sliding windows based on manual segmentation.
    Label is 1 if window overlaps with any segment, 0 otherwise.
    
    Args:
        signal_length: int, total length of signal
        segment_ranges: list of tuples (start, end) for active segments
        window_size: int, size of each window in samples
        step_size: int, step size between windows in samples
        overlap_threshold: float, minimum fraction of window that must overlap
                          with a segment to be considered active
    
    Returns:
        np.array: binary labels for each window (1=active, 0=inactive)
        np.array: array of window start indices
    """
    labels = []
    start_indices = []
    
    effective_threshold = max(overlap_threshold, 0.001)
    min_overlap = max(1, int(np.ceil(window_size * effective_threshold)))
    
    for start in range(0, signal_length - window_size + 1, step_size):
        end = start + window_size
        start_indices.append(start)
        
        # Check if window overlaps with any segment
        is_active = False
        
        for seg_start, seg_end in segment_ranges:
            overlap = min(end, seg_end) - max(start, seg_start)
            
            # Consider window active if sufficient overlap with any segment
            if overlap >= min_overlap:
                is_active = True
                break
        
        labels.append(1 if is_active else 0)
    
    return np.array(labels), np.array(start_indices)


def label_signal_three_state(signal_length, segment_ranges, concentric_ratio=0.6):
    """
    Create per-sample three-state labels.

    States:
        0: Background (rest)
        1: Concentric / primary activation (large packet)
        2: Eccentric / secondary activation (small packet)

    Each manual segment is split into two phases based on concentric_ratio.
    """
    labels = np.zeros(signal_length, dtype=int)
    ratio = np.clip(concentric_ratio, THREE_STATE_RATIO_MIN, THREE_STATE_RATIO_MAX)

    for seg_start, seg_end in segment_ranges:
        seg_start = max(0, int(seg_start))
        seg_end = min(signal_length, int(seg_end))
        if seg_end <= seg_start:
            continue

        seg_len = seg_end - seg_start
        split = seg_start + int(seg_len * ratio)
        labels[seg_start:split] = 1
        labels[split:seg_end] = 2

    return labels


def create_sequence_windows_for_segmentation(signal, labels, sequence_length, step_size):
    """
    Create paired signal/label sequences for CRNN training or inference.
    """
    windows = []
    label_windows = []

    for start in range(0, len(signal) - sequence_length + 1, step_size):
        end = start + sequence_length
        windows.append(signal[start:end])
        label_windows.append(labels[start:end])

    return np.array(windows), np.array(label_windows)


def decode_three_state_predictions(predictions, min_length=200, merge_gap=100):
    """
    Convert per-sample three-state predictions into (start, end) segments.
    """
    segments = []
    in_segment = False
    start_idx = 0

    for idx, label in enumerate(predictions):
        if label != 0 and not in_segment:
            in_segment = True
            start_idx = idx
        elif label == 0 and in_segment:
            in_segment = False
            end_idx = idx
            if end_idx - start_idx >= min_length:
                segments.append((start_idx, end_idx))

    if in_segment:
        end_idx = len(predictions)
        if end_idx - start_idx >= min_length:
            segments.append((start_idx, end_idx))

    # Merge close segments to handle brief pauses
    if len(segments) > 1:
        merged = [segments[0]]
        for cur_start, cur_end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if cur_start - prev_end <= merge_gap:
                merged[-1] = (prev_start, cur_end)
            else:
                merged.append((cur_start, cur_end))
        segments = merged

    return segments


def get_segment_ranges_from_files(raw_filepath, segment_dir):
    """
    Get segment ranges by comparing raw file length with segment files.
    This estimates where segments came from in the original signal.
    
    Args:
        raw_filepath: str, path to raw CSV file
        segment_dir: str, path to directory containing segment files
    
    Returns:
        list: list of tuples (start, end) for each segment
    """
    import os
    
    # Load raw signal to get total length
    raw_signal = load_emg_data(raw_filepath)
    raw_length = len(raw_signal)
    
    # Load all segments
    segment_files = sorted([f for f in os.listdir(segment_dir) if f.endswith('.csv')])
    segments = []
    
    current_pos = 0
    for seg_file in segment_files:
        seg_path = os.path.join(segment_dir, seg_file)
        seg_data = load_emg_data(seg_path)
        seg_length = len(seg_data)
        
        # Find best match position in raw signal using correlation
        # For efficiency, we'll use a simplified approach
        # Assuming segments appear sequentially in the raw file
        segments.append((current_pos, current_pos + seg_length))
        current_pos += seg_length
        
        # Add gap between segments (estimated)
        if current_pos < raw_length:
            current_pos += int(0.5 * 2000)  # 0.5 second gap
    
    return segments


def find_segment_in_raw_signal(raw_signal, segment_signal, search_window=10000,
                               min_correlation=0.7):
    """
    Find where a segment appears in the raw signal using correlation.
    
    Args:
        raw_signal: array-like, full raw signal
        segment_signal: array-like, segment to find
        search_window: int, maximum search window size
        min_correlation: float, minimum correlation to accept a match
    
    Returns:
        int: start index of best match, or None if not found
    """
    seg_len = len(segment_signal)
    
    if seg_len > len(raw_signal):
        return None
    
    # Use correlation to find best match
    best_correlation = -np.inf
    best_start = 0
    
    # Normalize signals for correlation
    seg_norm = (segment_signal - np.mean(segment_signal)) / (np.std(segment_signal) + 1e-10)
    
    for start in range(0, len(raw_signal) - seg_len + 1, 100):  # Step by 100 for efficiency
        window = raw_signal[start:start + seg_len]
        window_norm = (window - np.mean(window)) / (np.std(window) + 1e-10)
        
        if np.std(window_norm) < 1e-10 or np.std(seg_norm) < 1e-10:
            continue
        
        correlation = np.corrcoef(seg_norm, window_norm)[0, 1]
        if np.isnan(correlation):
            continue
        
        if correlation > best_correlation:
            best_correlation = correlation
            best_start = start
    
    # Return if correlation is high enough
    if best_correlation >= min_correlation:
        return best_start
    
    return None


def improved_get_segment_ranges(raw_filepath, segment_dir):
    """
    Get segment ranges by finding actual matches between raw and segment signals.
    Falls back to automatic RMS-based detection if correlation matching fails.
    
    Args:
        raw_filepath: str, path to raw CSV file
        segment_dir: str, path to directory containing segment files
    
    Returns:
        list: list of tuples (start, end) for each segment
    """
    import os
    
    # Load raw signal
    raw_signal = load_emg_data(raw_filepath)
    raw_filtered = preprocess_signal(raw_signal)
    
    # Load all segments and their metadata
    segment_files = sorted([f for f in os.listdir(segment_dir) if f.endswith('.csv')])
    num_manual_segments = len(segment_files)
    
    # Track total length for fallback decision
    total_segment_length = 0
    for seg_file in segment_files:
        seg_path = os.path.join(segment_dir, seg_file)
        seg_data = load_emg_data(seg_path)
        total_segment_length += len(seg_data)
    
    # Try correlation-based matching first
    segment_ranges = []
    for seg_file in segment_files:
        seg_path = os.path.join(segment_dir, seg_file)
        seg_data = load_emg_data(seg_path)
        seg_filtered = preprocess_signal(seg_data)
        
        # Find segment in raw signal
        start_idx = find_segment_in_raw_signal(raw_filtered, seg_filtered)
        
        if start_idx is not None:
            end_idx = start_idx + len(seg_data)
            segment_ranges.append((start_idx, end_idx))
    
    # If correlation matching found segments, return them
    if len(segment_ranges) > 0:
        return segment_ranges
    
    # Fallback: Use automatic RMS-based detection with multiple parameter attempts
    # Try different parameter combinations to find segments
    if num_manual_segments > 0 and total_segment_length > 0:
        # Try with different min_segment_ms and threshold combinations
        # Start with more permissive parameters (smaller min_segment_ms)
        for min_seg_ms in FALLBACK_MIN_SEGMENT_MS_OPTIONS:
            for threshold_mult in FALLBACK_THRESHOLD_MULTIPLIERS:
                auto_segments = detect_activity_regions(
                    raw_filtered, 
                    fs=2000,
                    min_segment_ms=min_seg_ms,
                    merge_gap_ms=200,  # Use smaller merge gap for better segment separation
                    threshold_multiplier=threshold_mult
                )
                
                # If we found at least some segments, use them
                if len(auto_segments) > 0:
                    return auto_segments
    
    # Last resort: return empty list (no segments found)
    return segment_ranges


def detect_activity_segments(predictions, start_indices, min_length=1000, merge_gap=500):
    """
    Convert binary predictions to segment ranges.
    
    Args:
        predictions: array-like, binary predictions for windows (1=active, 0=inactive)
        start_indices: array-like, start index of each window
        min_length: int, minimum segment length in samples
        merge_gap: int, merge segments separated by less than this many samples
    
    Returns:
        list: list of tuples (start, end) for each detected segment
    """
    segments = []
    in_segment = False
    segment_start = None
    
    for i, (pred, start_idx) in enumerate(zip(predictions, start_indices)):
        if pred == 1 and not in_segment:
            # Start of new segment
            in_segment = True
            segment_start = start_idx
        elif pred == 0 and in_segment:
            # End of segment
            in_segment = False
            segment_end = start_idx
            
            # Only add if segment is long enough
            if segment_end - segment_start >= min_length:
                segments.append((segment_start, segment_end))
    
    # Handle case where last window is active
    if in_segment and segment_start is not None:
        segment_end = start_indices[-1] + (start_indices[1] - start_indices[0])  # Estimate
        if segment_end - segment_start >= min_length:
            segments.append((segment_start, segment_end))
    
    # Merge close segments
    if len(segments) > 1:
        merged_segments = [segments[0]]
        for current_start, current_end in segments[1:]:
            last_start, last_end = merged_segments[-1]
            
            if current_start - last_end <= merge_gap:
                # Merge with previous segment
                merged_segments[-1] = (last_start, current_end)
            else:
                merged_segments.append((current_start, current_end))
        
        segments = merged_segments
    
    return segments
