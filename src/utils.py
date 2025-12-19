"""
Utility functions for EMG signal analysis.
Includes filename parsing and visualization utilities.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def parse_filename(filename):
    """
    Parse EMG data filename to extract metadata.
    
    Format: [Amplitude]_[Fatigue]_[SubjectID]_[ActionID].csv
    - Amplitudes: full, half, invalid
    - Fatigues: free, light, medium, heavy
    - Constraints:
        * half/invalid imply free fatigue
        * light/medium/heavy imply full amplitude
    
    Args:
        filename: str, name of the file (with or without .csv extension)
    
    Returns:
        dict: {'amplitude': str, 'fatigue': str, 'subject_id': str, 'action_id': str}
        or None if parsing fails
    """
    # Remove .csv extension and any path
    basename = os.path.basename(filename)
    basename = basename.replace('.csv', '')
    
    # Try to match the pattern
    pattern = r'^(full|half|invalid)_(free|light|medium|heavy)_(\d+)_(\d+)$'
    match = re.match(pattern, basename)
    
    if match:
        amplitude, fatigue, subject_id, action_id = match.groups()
        return {
            'amplitude': amplitude,
            'fatigue': fatigue,
            'subject_id': subject_id,
            'action_id': action_id,
            'filename': basename
        }
    
    # Try without segment suffix
    pattern_seg = r'^(full|half|invalid)_(free|light|medium|heavy)_(\d+)_(\d+)_seg\d+$'
    match = re.match(pattern_seg, basename)
    
    if match:
        amplitude, fatigue, subject_id, action_id = match.groups()
        return {
            'amplitude': amplitude,
            'fatigue': fatigue,
            'subject_id': subject_id,
            'action_id': action_id,
            'filename': basename
        }
    
    return None


def validate_labels(metadata):
    """
    Validate that amplitude-fatigue combinations follow constraints.
    
    Args:
        metadata: dict with 'amplitude' and 'fatigue' keys
    
    Returns:
        bool: True if valid, False otherwise
    """
    amplitude = metadata['amplitude']
    fatigue = metadata['fatigue']
    
    # half/invalid imply free fatigue
    if amplitude in ['half', 'invalid'] and fatigue != 'free':
        return False
    
    # light/medium/heavy imply full amplitude
    if fatigue in ['light', 'medium', 'heavy'] and amplitude != 'full':
        return False
    
    return True


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', 
                         save_path=None, figsize=(8, 6)):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: array-like, true labels
        y_pred: array-like, predicted labels
        labels: list, label names for display
        title: str, plot title
        save_path: str, path to save the plot (optional)
        figsize: tuple, figure size
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_signal_with_segments(signal, segment_indices, sampling_rate=2000, 
                              title='EMG Signal with Segments', save_path=None):
    """
    Plot EMG signal with detected segment boundaries.
    
    Args:
        signal: array-like, EMG signal
        segment_indices: list of tuples (start, end) for each segment
        sampling_rate: int, sampling rate in Hz
        title: str, plot title
        save_path: str, path to save the plot (optional)
    """
    time = np.arange(len(signal)) / sampling_rate
    
    plt.figure(figsize=(15, 5))
    plt.plot(time, signal, 'b-', linewidth=0.5, alpha=0.7, label='EMG Signal')
    
    # Plot segment boundaries
    for i, (start, end) in enumerate(segment_indices):
        plt.axvline(x=start/sampling_rate, color='r', linestyle='--', linewidth=1, alpha=0.6)
        plt.axvline(x=end/sampling_rate, color='g', linestyle='--', linewidth=1, alpha=0.6)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Signal plot saved to {save_path}")
    
    plt.close()


def print_classification_report(y_true, y_pred, labels, target_names=None):
    """
    Print classification report with precision, recall, f1-score.
    
    Args:
        y_true: array-like, true labels
        y_pred: array-like, predicted labels
        labels: list, unique label values
        target_names: list, label names for display (optional)
    """
    if target_names is None:
        target_names = labels
    
    report = classification_report(y_true, y_pred, labels=labels, 
                                   target_names=target_names)
    print(report)


def get_train_files(train_dir):
    """
    Get all training files from the train directory.
    
    Args:
        train_dir: str, path to train directory
    
    Returns:
        dict: {'raw_files': list of raw csv files, 
               'segment_dirs': dict mapping raw file to segment directory}
    """
    raw_files = []
    segment_dirs = {}
    
    for item in os.listdir(train_dir):
        item_path = os.path.join(train_dir, item)
        
        if item.endswith('.csv'):
            raw_files.append(item_path)
            
            # Check for corresponding segment directory
            segment_dir_name = item.replace('.csv', '_segments')
            segment_dir_path = os.path.join(train_dir, segment_dir_name)
            
            if os.path.isdir(segment_dir_path):
                segment_dirs[item_path] = segment_dir_path
    
    return {'raw_files': sorted(raw_files), 'segment_dirs': segment_dirs}


def load_segments_metadata(segment_dir):
    """
    Load metadata for all segments in a directory.
    
    Args:
        segment_dir: str, path to segment directory
    
    Returns:
        list of dicts with segment metadata
    """
    segments = []
    
    for seg_file in sorted(os.listdir(segment_dir)):
        if seg_file.endswith('.csv'):
            seg_path = os.path.join(segment_dir, seg_file)
            metadata = parse_filename(seg_file)
            
            if metadata:
                metadata['path'] = seg_path
                segments.append(metadata)
    
    return segments
