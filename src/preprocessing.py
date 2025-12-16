"""
Preprocessing module for EMG signal analysis.
Includes filtering, segmentation, and sliding window operations.
"""
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch


def load_emg_data(filepath):
    """
    Load EMG data from CSV file.
    
    Args:
        filepath: str, path to CSV file
    
    Returns:
        np.array: EMG signal data (1D array)
    """
    data = pd.read_csv(filepath, header=None)
    return data.values.flatten()


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
                               min_correlation=0.9):
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
    
    # Load all segments
    segment_files = sorted([f for f in os.listdir(segment_dir) if f.endswith('.csv')])
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
