"""
Test set prediction script with comprehensive visualization.
Segments test signals, predicts amplitude/fatigue for each segment, and generates visualizations.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import (
    load_emg_data, preprocess_signal, create_sliding_windows,
    detect_activity_segments, create_sequence_windows_for_segmentation,
    decode_three_state_predictions
)
from features import extract_features_from_windows, extract_segment_features
from models import ActivityDetector, AmplitudeClassifier, FatigueClassifier, CRNNActivitySegmenter
from utils import parse_filename


def segment_signal_with_crnn(signal, segmenter):
    """
    Predict three-state labels and convert to segments using CRNN segmenter.
    
    Args:
        signal: array-like, preprocessed EMG signal
        segmenter: CRNNActivitySegmenter, trained segmenter
    
    Returns:
        list: list of tuples (start, end) for detected segments
    """
    seq_len = segmenter.sequence_length
    step = segmenter.step_size

    # Pad the tail so that the final window is complete
    signal_length = len(signal)
    if signal_length < seq_len:
        pad_length = seq_len - signal_length
    else:
        remainder = (signal_length - seq_len) % step
        pad_length = 0 if remainder == 0 else step - remainder

    if pad_length > 0:
        signal = np.pad(signal, (0, pad_length), mode="edge")

    dummy_labels = np.zeros(len(signal), dtype=int)
    windows, _ = create_sequence_windows_for_segmentation(signal, dummy_labels, seq_len, step)
    preds = segmenter.predict(windows)
    # Reconstruct per-sample predictions by overlap-averaging (simple majority)
    per_sample = np.zeros(len(signal), dtype=int)
    counts = np.zeros(len(signal), dtype=int)
    for i, start in enumerate(range(0, len(signal) - seq_len + 1, step)):
        per_sample[start:start + seq_len] += preds[i]
        counts[start:start + seq_len] += 1
    # Prevent division by zero for samples at signal boundaries not covered by windows
    counts[counts == 0] = 1
    per_sample = (per_sample / counts).round().astype(int)
    per_sample = np.clip(per_sample, 0, 2)
    return decode_three_state_predictions(per_sample)


def segment_signal_with_detector(signal, detector, window_size=200, step_size=100):
    """
    Segment signal using ActivityDetector (sliding window classifier).
    
    Args:
        signal: array-like, preprocessed EMG signal
        detector: ActivityDetector, trained detector
        window_size: int, window size in samples
        step_size: int, step size in samples
    
    Returns:
        list: list of tuples (start, end) for detected segments
    """
    # Create sliding windows
    windows, start_indices = create_sliding_windows(signal, window_size, step_size)
    
    # Extract features
    features, _ = extract_features_from_windows(windows)
    
    # Predict activity
    predictions = detector.predict(features)
    
    # Convert predictions to segments with lower min_length for better detection
    # min_length=200 means 0.1s minimum segment, merge_gap=500 means 0.25s gap allowed
    segments = detect_activity_segments(predictions, start_indices, min_length=200, merge_gap=500)
    
    return segments


def plot_signal_with_predictions(signal, segments, predictions, sampling_rate=2000,
                                 title='EMG Signal Segmentation and Prediction',
                                 save_path=None):
    """
    Plot EMG signal with detected segments and their predictions.
    
    Args:
        signal: array-like, EMG signal
        segments: list of tuples (start, end) for each segment
        predictions: list of dicts with 'amplitude' and 'fatigue' for each segment
        sampling_rate: int, sampling rate in Hz
        title: str, plot title
        save_path: str, path to save the plot (optional)
    """
    time = np.arange(len(signal)) / sampling_rate
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    
    # Plot 1: Full signal with segments
    ax1.plot(time, signal, 'b-', linewidth=0.5, alpha=0.7, label='EMG Signal')
    
    # Color map for amplitudes
    amp_colors = {'full': 'green', 'half': 'orange', 'invalid': 'red'}
    
    for i, ((start, end), pred) in enumerate(zip(segments, predictions)):
        color = amp_colors.get(pred['amplitude'], 'gray')
        ax1.axvspan(start/sampling_rate, end/sampling_rate, alpha=0.3, color=color,
                   label=f"Seg {i+1}: {pred['amplitude']}/{pred['fatigue']}" if i < 5 else "")
        ax1.axvline(x=start/sampling_rate, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axvline(x=end/sampling_rate, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if len(segments) <= 5:
        ax1.legend(loc='upper right', fontsize=10)
    
    # Plot 2: Segment annotations
    ax2.set_ylim(0, 3)
    ax2.set_yticks([0.5, 1.5, 2.5])
    ax2.set_yticklabels(['Fatigue', 'Amplitude', 'Active'])
    
    # Plot active segments
    for i, (start, end) in enumerate(segments):
        ax2.barh(2.5, (end-start)/sampling_rate, left=start/sampling_rate, 
                height=0.8, color='lightblue', alpha=0.7, edgecolor='black')
        # Add segment number
        mid_time = (start + end) / 2 / sampling_rate
        ax2.text(mid_time, 2.5, f'{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Plot amplitude predictions
    for i, ((start, end), pred) in enumerate(zip(segments, predictions)):
        color = amp_colors.get(pred['amplitude'], 'gray')
        ax2.barh(1.5, (end-start)/sampling_rate, left=start/sampling_rate,
                height=0.8, color=color, alpha=0.7, edgecolor='black')
        ax2.text((start + end) / 2 / sampling_rate, 1.5, pred['amplitude'][:4],
                ha='center', va='center', fontsize=9)
    
    # Plot fatigue predictions
    fat_colors = {'free': 'lightgreen', 'light': 'yellow', 'medium': 'orange', 'heavy': 'red'}
    for i, ((start, end), pred) in enumerate(zip(segments, predictions)):
        color = fat_colors.get(pred['fatigue'], 'gray')
        ax2.barh(0.5, (end-start)/sampling_rate, left=start/sampling_rate,
                height=0.8, color=color, alpha=0.7, edgecolor='black')
        ax2.text((start + end) / 2 / sampling_rate, 0.5, pred['fatigue'][:4],
                ha='center', va='center', fontsize=9)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_title('Segmentation and Predictions Timeline', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.close()


def plot_segment_features_heatmap(segments_features, predictions, save_path=None):
    """
    Plot heatmap of features for all segments.
    
    Args:
        segments_features: list of dicts, features for each segment
        predictions: list of dicts with 'amplitude' and 'fatigue' for each segment
        save_path: str, path to save the plot (optional)
    """
    if len(segments_features) == 0:
        print("No segments to plot features for.")
        return
    
    # Select key features to display
    key_features = ['rms', 'mav', 'zc', 'ssc', 'wl', 'var', 'mean_freq', 'median_freq']
    
    # Create feature matrix
    feature_matrix = []
    segment_labels = []
    
    for i, (features, pred) in enumerate(zip(segments_features, predictions)):
        feature_values = [features.get(f, 0) for f in key_features]
        feature_matrix.append(feature_values)
        segment_labels.append(f"S{i+1}\n{pred['amplitude'][:4]}/{pred['fatigue'][:4]}")
    
    feature_matrix = np.array(feature_matrix)
    
    # Normalize each feature column for better visualization
    for j in range(feature_matrix.shape[1]):
        col = feature_matrix[:, j]
        if col.max() - col.min() > 0:
            feature_matrix[:, j] = (col - col.min()) / (col.max() - col.min())
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(6, len(segments_features) * 0.3)))
    
    sns.heatmap(feature_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=key_features, yticklabels=segment_labels,
                cbar_kws={'label': 'Normalized Feature Value'}, ax=ax)
    
    ax.set_title('Segment Features Heatmap (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Segments', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature heatmap saved to {save_path}")
    
    plt.close()


def predict_test_file(signal_path, detector, amp_clf, fat_clf,
                      segmenter=None, use_crnn=False):
    """
    Predict amplitude and fatigue for a single test file.
    
    Args:
        signal_path: str, path to test signal CSV
        detector: ActivityDetector, trained activity detector (used if use_crnn=False)
        amp_clf: AmplitudeClassifier, trained amplitude classifier
        fat_clf: FatigueClassifier, trained fatigue classifier
        segmenter: CRNNActivitySegmenter, trained CRNN segmenter (used if use_crnn=True)
        use_crnn: bool, if True use CRNN segmenter, else use ActivityDetector
    
    Returns:
        tuple: (segments, predictions, segments_features, filtered_signal)
    """
    print(f"\nProcessing: {os.path.basename(signal_path)}")
    
    # Load and preprocess signal
    raw_signal = load_emg_data(signal_path)
    filtered_signal = preprocess_signal(raw_signal)
    print(f"Signal length: {len(filtered_signal)} samples ({len(filtered_signal)/2000:.2f} seconds)")
    
    # Detect segments using selected method
    if use_crnn and segmenter is not None:
        segments = segment_signal_with_crnn(filtered_signal, segmenter)
    else:
        segments = segment_signal_with_detector(filtered_signal, detector)
    print(f"Detected segments: {len(segments)}")
    
    # Classify each segment
    segment_predictions = []
    segments_features = []
    
    for i, (start, end) in enumerate(segments):
        # Extract segment
        segment_signal = filtered_signal[start:end]
        
        # Extract features
        seg_features = extract_segment_features(segment_signal)
        segments_features.append(seg_features)
        
        feature_array = np.array([list(seg_features.values())])
        
        # Predict amplitude
        amplitude = amp_clf.predict(feature_array)[0]
        amp_proba = amp_clf.predict_proba(feature_array)[0]
        
        # Predict fatigue (only for full amplitude)
        if amplitude == 'full':
            fatigue = fat_clf.predict(feature_array)[0]
            fat_proba = fat_clf.predict_proba(feature_array)[0]
        else:
            fatigue = 'free'  # By constraint
            fat_proba = None
        
        segment_predictions.append({
            'segment_id': i + 1,
            'start': start,
            'end': end,
            'duration_s': (end - start) / 2000,
            'amplitude': amplitude,
            'amplitude_confidence': np.max(amp_proba),
            'fatigue': fatigue,
            'fatigue_confidence': np.max(fat_proba) if fat_proba is not None else 1.0
        })
        
        print(f"  Segment {i+1}: [{start:6d}-{end:6d}] ({(end-start)/2000:.2f}s) -> "
              f"{amplitude:8s} (conf: {np.max(amp_proba):.3f}), "
              f"{fatigue:6s} (conf: {np.max(fat_proba) if fat_proba is not None else 1.0:.3f})")
    
    return segments, segment_predictions, segments_features, filtered_signal


def main():
    """Main test prediction execution."""
    print("=" * 80)
    print("EMG Test Set Prediction with Visualization")
    print("=" * 80)
    
    # Check if models exist - prefer ActivityDetector, fall back to CRNN
    use_crnn = False
    if os.path.exists('models/activity_detector.pkl'):
        print("\nUsing ActivityDetector for segmentation...")
    elif os.path.exists('models/crnn_activity_segmenter.pt'):
        print("\nUsing CRNN Segmenter for segmentation...")
        use_crnn = True
    else:
        print("Error: Models not found. Please run 'python main.py' first to train models.")
        return
    
    # Load models
    print("\nLoading trained models...")
    
    detector = None
    segmenter = None
    
    if use_crnn:
        segmenter = CRNNActivitySegmenter()
        segmenter.load('models/crnn_activity_segmenter.pt')
    else:
        detector = ActivityDetector()
        detector.load('models/activity_detector.pkl')
    
    amp_clf = AmplitudeClassifier()
    amp_clf.load('models/amplitude_classifier.pkl')
    
    fat_clf = FatigueClassifier()
    fat_clf.load('models/fatigue_classifier.pkl')
    
    # Find test files
    test_dir = 'test'
    if not os.path.exists(test_dir):
        print(f"Error: Test directory '{test_dir}' not found!")
        print("Using sample files from training directory instead...")
        test_dir = 'train'
    
    # Get test files (CSV files in test directory)
    import glob
    test_files = glob.glob(os.path.join(test_dir, '*.csv'))
    
    if not test_files:
        print(f"Error: No CSV files found in '{test_dir}' directory!")
        return
    
    print(f"\nFound {len(test_files)} test files")
    
    # Create output directory
    os.makedirs('predictions', exist_ok=True)
    
    # Process each test file
    all_results = []
    
    for test_file in test_files[:10]:  # Limit to first 10 files for demo
        basename = os.path.basename(test_file).replace('.csv', '')
        
        try:
            # Predict
            segments, predictions, segments_features, filtered_signal = predict_test_file(
                test_file, detector, amp_clf, fat_clf,
                segmenter=segmenter, use_crnn=use_crnn
            )
            
            # Store results
            all_results.append({
                'file': basename,
                'segments': segments,
                'predictions': predictions,
                'segments_features': segments_features
            })
            
            # Generate visualizations
            if len(segments) > 0:
                # Full signal segmentation plot
                plot_signal_with_predictions(
                    filtered_signal,
                    segments,
                    predictions,
                    title=f'EMG Signal Segmentation: {basename}',
                    save_path=f'predictions/{basename}_segmentation.png'
                )
                
                # Segment features heatmap
                plot_segment_features_heatmap(
                    segments_features,
                    predictions,
                    save_path=f'predictions/{basename}_features.png'
                )
        
        except Exception as e:
            print(f"Error processing {test_file}: {str(e)}")
            continue
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    
    total_segments = 0
    amp_counts = {}
    fat_counts = {}
    
    for result in all_results:
        total_segments += len(result['segments'])
        for pred in result['predictions']:
            amp = pred['amplitude']
            fat = pred['fatigue']
            amp_counts[amp] = amp_counts.get(amp, 0) + 1
            fat_counts[fat] = fat_counts.get(fat, 0) + 1
    
    print(f"\nTotal files processed: {len(all_results)}")
    print(f"Total segments detected: {total_segments}")
    
    print("\nPredicted Amplitude Distribution:")
    for amp in sorted(amp_counts.keys()):
        print(f"  {amp}: {amp_counts[amp]} segments")
    
    print("\nPredicted Fatigue Distribution:")
    for fat in sorted(fat_counts.keys()):
        print(f"  {fat}: {fat_counts[fat]} segments")
    
    print(f"\nVisualization files saved to 'predictions/' directory")
    print("=" * 80)


if __name__ == '__main__':
    main()
