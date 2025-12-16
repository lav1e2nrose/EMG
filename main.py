"""
Main entry point for EMG signal analysis pipeline.
Orchestrates data loading, preprocessing, feature extraction, training, and evaluation.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import (
    load_emg_data, preprocess_signal, create_sliding_windows,
    label_windows_from_segments, improved_get_segment_ranges,
    detect_activity_segments, label_signal_three_state,
    create_sequence_windows_for_segmentation, decode_three_state_predictions
)
from features import extract_features_from_windows, extract_segment_features
from models import ActivityDetector, AmplitudeClassifier, FatigueClassifier, CRNNActivitySegmenter
from utils import (
    parse_filename, get_train_files, load_segments_metadata,
    plot_confusion_matrix, print_classification_report
)


def train_activity_detector(train_dir, window_size=200, step_size=100, 
                            model_type='random_forest'):
    """
    Train activity detector using raw files and manual segments.
    
    Args:
        train_dir: str, path to training directory
        window_size: int, window size in samples (200 samples = 0.1s at 2000Hz)
        step_size: int, step size in samples
        model_type: str, 'random_forest' or 'xgboost'
    
    Returns:
        ActivityDetector: trained model
    """
    print("\n=== Training Activity Detector ===")
    
    # Get training files
    file_data = get_train_files(train_dir)
    
    X_all = []
    y_all = []
    
    # Process files with manual segments
    for raw_file, segment_dir in file_data['segment_dirs'].items():
        print(f"Processing {os.path.basename(raw_file)}...")
        
        # Load and preprocess raw signal
        raw_signal = load_emg_data(raw_file)
        filtered_signal = preprocess_signal(raw_signal)
        
        # Get segment ranges
        segment_ranges = improved_get_segment_ranges(raw_file, segment_dir)
        
        if len(segment_ranges) == 0:
            print(f"  Warning: No segments found for {raw_file}")
            continue
        
        # Create sliding windows
        windows, start_indices = create_sliding_windows(
            filtered_signal, window_size, step_size
        )
        
        # Label windows
        labels, _ = label_windows_from_segments(
            len(filtered_signal), segment_ranges, window_size, step_size
        )
        
        # Extract features
        features, feature_names = extract_features_from_windows(windows)
        
        print(f"  Windows: {len(windows)}, Active: {np.sum(labels)}, "
              f"Inactive: {len(labels) - np.sum(labels)}")
        
        X_all.append(features)
        y_all.append(labels)
    
    # Combine all data
    X = np.vstack(X_all)
    y = np.hstack(y_all)
    
    print(f"\nTotal samples: {len(y)}")
    print(f"Active: {np.sum(y)} ({100*np.sum(y)/len(y):.1f}%)")
    print(f"Inactive: {len(y) - np.sum(y)} ({100*(len(y)-np.sum(y))/len(y):.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print(f"\nTraining {model_type} model...")
    detector = ActivityDetector(model_type=model_type, decision_threshold=0.6)
    detector.fit(X_train, y_train)
    
    # Evaluate
    y_pred = detector.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nActivity Detector Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Inactive', 'Active']))
    
    return detector


def train_crnn_activity_segmenter(train_dir, sequence_length=400, step_size=200, epochs=3):
    """
    Train CRNN-based three-state activity segmenter.
    """
    print("\n=== Training CRNN Activity Segmenter ===")

    file_data = get_train_files(train_dir)
    window_buffer = []
    label_buffer = []

    for raw_file, segment_dir in file_data['segment_dirs'].items():
        print(f"Processing {os.path.basename(raw_file)} for CRNN...")
        raw_signal = load_emg_data(raw_file)
        filtered_signal = preprocess_signal(raw_signal)
        segment_ranges = improved_get_segment_ranges(raw_file, segment_dir)

        if len(segment_ranges) == 0:
            print(f"  Warning: No segments found for {raw_file}")
            continue

        per_sample_labels = label_signal_three_state(len(filtered_signal), segment_ranges)
        windows, label_windows = create_sequence_windows_for_segmentation(
            filtered_signal, per_sample_labels, sequence_length, step_size
        )

        window_buffer.append(windows)
        label_buffer.append(label_windows)

    if not window_buffer:
        raise RuntimeError("No data available for CRNN training.")

    X = np.vstack(window_buffer)
    y = np.vstack(label_buffer)

    print(f"CRNN training samples: {len(X)}, sequence length: {sequence_length}")

    segmenter = CRNNActivitySegmenter(sequence_length=sequence_length, step_size=step_size)
    segmenter.fit(X, y, epochs=epochs)
    return segmenter


def segment_signal_with_crnn(signal, segmenter):
    """
    Predict three-state labels and convert to segments.
    """
    seq_len = segmenter.sequence_length
    step = segmenter.step_size

    # Pad tail if needed
    pad_length = (-(len(signal) - seq_len)) % step
    if pad_length:
        signal = np.pad(signal, (0, pad_length), mode='edge')

    dummy_labels = np.zeros(len(signal), dtype=int)
    windows, _ = create_sequence_windows_for_segmentation(signal, dummy_labels, seq_len, step)
    preds = segmenter.predict(windows)
    # Reconstruct per-sample predictions by overlap-averaging (simple majority)
    per_sample = np.zeros(len(signal), dtype=int)
    counts = np.zeros(len(signal), dtype=int)
    for i, start in enumerate(range(0, len(signal) - seq_len + 1, step)):
        per_sample[start:start + seq_len] += preds[i]
        counts[start:start + seq_len] += 1
    counts[counts == 0] = 1
    per_sample = (per_sample / counts).round().astype(int)
    return decode_three_state_predictions(per_sample)


def segment_signal_with_detector(signal, detector, window_size=200, step_size=100):
    """
    Segment a signal using trained activity detector.
    
    Args:
        signal: array-like, preprocessed signal
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
    
    # Convert predictions to segments
    segments = detect_activity_segments(predictions, start_indices)
    
    return segments


def train_classifiers(train_dir, model_type='random_forest', 
                     filter_amplitude_by_fatigue=False, filter_fatigue_by_amplitude=False):
    """
    Train amplitude and fatigue classifiers using segmented data.
    
    Args:
        train_dir: str, path to training directory
        model_type: str, 'random_forest' or 'xgboost'
        filter_amplitude_by_fatigue: bool, if True, use only 'free' fatigue for amplitude training
        filter_fatigue_by_amplitude: bool, if True, use only 'full' amplitude for fatigue training
    
    Returns:
        tuple: (AmplitudeClassifier, FatigueClassifier)
    """
    print("\n=== Training Classifiers ===")
    
    # Collect all segment data
    segments_data = []
    
    file_data = get_train_files(train_dir)
    
    print("\n--- Loading Segment Data ---")
    for raw_file, segment_dir in file_data['segment_dirs'].items():
        print(f"Loading segments from {os.path.basename(segment_dir)}...")
        
        segments = load_segments_metadata(segment_dir)
        
        for seg_meta in segments:
            # Load segment signal
            seg_signal = load_emg_data(seg_meta['path'])
            seg_filtered = preprocess_signal(seg_signal)
            
            # Extract features
            features = extract_segment_features(seg_filtered)
            
            # Store data
            seg_meta['features'] = features
            segments_data.append(seg_meta)
    
    print(f"\nTotal segments loaded: {len(segments_data)}")
    
    # Print detailed statistics
    print("\n--- Training Data Statistics ---")
    print(f"Total training actions (CSV files): {len(file_data['segment_dirs'])}")
    
    # Count by amplitude
    amplitude_counts = {}
    fatigue_counts = {}
    subject_counts = {}
    
    for seg in segments_data:
        amp = seg['amplitude']
        fat = seg['fatigue']
        subj = seg['subject_id']
        
        amplitude_counts[amp] = amplitude_counts.get(amp, 0) + 1
        fatigue_counts[fat] = fatigue_counts.get(fat, 0) + 1
        subject_counts[subj] = subject_counts.get(subj, 0) + 1
    
    print("\nAmplitude Distribution:")
    for amp in sorted(amplitude_counts.keys()):
        print(f"  {amp}: {amplitude_counts[amp]} samples")
    
    print("\nFatigue Distribution:")
    for fat in sorted(fatigue_counts.keys()):
        print(f"  {fat}: {fatigue_counts[fat]} samples")
    
    print("\nSubject Distribution:")
    for subj in sorted(subject_counts.keys()):
        print(f"  Subject {subj}: {subject_counts[subj]} samples")
    
    # Prepare data for amplitude classification
    X_amp = []
    y_amp = []
    subj_amp = []
    
    for seg in segments_data:
        if seg['features'] is not None:
            # Apply filtering if requested
            if filter_amplitude_by_fatigue and seg['fatigue'] != 'free':
                continue
            
            feature_values = list(seg['features'].values())
            X_amp.append(feature_values)
            y_amp.append(seg['amplitude'])
            subj_amp.append(seg['subject_id'])
    
    X_amp = np.array(X_amp)
    y_amp = np.array(y_amp)
    subj_amp = np.array(subj_amp)
    
    # Train amplitude classifier
    print("\n--- Training Amplitude Classifier ---")
    if filter_amplitude_by_fatigue:
        print("Using filtered data: only 'free' fatigue samples")
    print(f"Total samples: {len(y_amp)}")
    unique_classes, class_counts = np.unique(y_amp, return_counts=True)
    print(f"Samples per amplitude class:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  {cls}: {count} samples")
    
    X_amp_train, X_amp_test, y_amp_train, y_amp_test = train_test_split(
        X_amp, y_amp, test_size=0.2, random_state=42, stratify=y_amp
    )
    
    amp_classifier = AmplitudeClassifier(model_type=model_type)
    amp_classifier.fit(X_amp_train, y_amp_train)
    
    y_amp_pred = amp_classifier.predict(X_amp_test)
    amp_accuracy = accuracy_score(y_amp_test, y_amp_pred)
    
    print(f"\nAmplitude Classifier Accuracy: {amp_accuracy:.4f}")
    print("\nClassification Report:")
    print_classification_report(
        y_amp_test, y_amp_pred,
        labels=amp_classifier.get_classes(),
        target_names=amp_classifier.get_classes()
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_amp_test, y_amp_pred,
        labels=amp_classifier.get_classes(),
        title='Amplitude Classification Confusion Matrix',
        save_path='amplitude_confusion_matrix.png'
    )
    
    # Prepare data for fatigue classification (only 'full' amplitude)
    X_fat = []
    y_fat = []
    subj_fat = []
    
    for seg in segments_data:
        if seg['amplitude'] == 'full' and seg['features'] is not None:
            # Apply filtering if requested (already filtered by amplitude='full')
            if filter_fatigue_by_amplitude and seg['amplitude'] != 'full':
                continue
            
            feature_values = list(seg['features'].values())
            X_fat.append(feature_values)
            y_fat.append(seg['fatigue'])
            subj_fat.append(seg['subject_id'])
    
    X_fat = np.array(X_fat)
    y_fat = np.array(y_fat)
    subj_fat = np.array(subj_fat)
    
    # Train fatigue classifier
    print("\n--- Training Fatigue Classifier ---")
    print("Using only 'full' amplitude samples (by constraint)")
    print(f"Total samples: {len(y_fat)}")
    unique_classes, class_counts = np.unique(y_fat, return_counts=True)
    print(f"Samples per fatigue class:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  {cls}: {count} samples")
    
    X_fat_train, X_fat_test, y_fat_train, y_fat_test = train_test_split(
        X_fat, y_fat, test_size=0.2, random_state=42, stratify=y_fat
    )
    
    fat_classifier = FatigueClassifier(model_type=model_type)
    fat_classifier.fit(X_fat_train, y_fat_train)
    
    y_fat_pred = fat_classifier.predict(X_fat_test)
    fat_accuracy = accuracy_score(y_fat_test, y_fat_pred)
    
    print(f"\nFatigue Classifier Accuracy: {fat_accuracy:.4f}")
    print("\nClassification Report:")
    print_classification_report(
        y_fat_test, y_fat_pred,
        labels=fat_classifier.get_classes(),
        target_names=fat_classifier.get_classes()
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_fat_test, y_fat_pred,
        labels=fat_classifier.get_classes(),
        title='Fatigue Classification Confusion Matrix',
        save_path='fatigue_confusion_matrix.png'
    )
    
    return amp_classifier, fat_classifier


def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("EMG Signal Analysis Pipeline")
    print("=" * 60)
    
    # Configuration
    TRAIN_DIR = 'train'
    WINDOW_SIZE = 200  # 0.1 seconds at 2000 Hz
    STEP_SIZE = 100    # 0.05 seconds at 2000 Hz (50% overlap)
    MODEL_TYPE = 'random_forest'  # or 'xgboost'
    
    # Filtering options for better accuracy
    FILTER_AMPLITUDE_BY_FATIGUE = False  # Set to True to use only 'free' fatigue for amplitude
    FILTER_FATIGUE_BY_AMPLITUDE = True   # Already enforced by using only 'full' amplitude
    
    # Check if train directory exists
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory '{TRAIN_DIR}' not found!")
        return
    
    # Step 1: Train CRNN activity segmenter
    print("\nStep 1: Training CRNN Activity Segmenter...")
    os.makedirs('models', exist_ok=True)
    crnn_segmenter = train_crnn_activity_segmenter(
        TRAIN_DIR,
        sequence_length=400,
        step_size=200,
        epochs=3
    )
    crnn_segmenter.save('models/crnn_activity_segmenter.pt')
    
    # Step 2: Train amplitude and fatigue classifiers
    print("\nStep 2: Training Amplitude and Fatigue Classifiers...")
    amp_classifier, fat_classifier = train_classifiers(
        TRAIN_DIR,
        model_type=MODEL_TYPE,
        filter_amplitude_by_fatigue=FILTER_AMPLITUDE_BY_FATIGUE,
        filter_fatigue_by_amplitude=FILTER_FATIGUE_BY_AMPLITUDE
    )
    
    # Save classifiers
    amp_classifier.save('models/amplitude_classifier.pkl')
    fat_classifier.save('models/fatigue_classifier.pkl')
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print("\nModels saved to 'models/' directory:")
    print("  - activity_detector.pkl")
    print("  - amplitude_classifier.pkl")
    print("  - fatigue_classifier.pkl")
    print("\nConfusion matrices saved:")
    print("  - amplitude_confusion_matrix.png")
    print("  - fatigue_confusion_matrix.png")


if __name__ == '__main__':
    main()
