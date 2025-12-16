"""
Example script demonstrating how to use trained models for inference.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import load_emg_data, preprocess_signal, create_sliding_windows
from features import extract_features_from_windows, extract_segment_features
from models import ActivityDetector, AmplitudeClassifier, FatigueClassifier
from utils import plot_signal_with_segments


def segment_and_classify_signal(signal_path, detector_path, amp_clf_path, fat_clf_path,
                                window_size=200, step_size=100):
    """
    Complete pipeline: segment signal and classify each segment.
    
    Args:
        signal_path: str, path to raw EMG signal CSV
        detector_path: str, path to trained activity detector
        amp_clf_path: str, path to trained amplitude classifier
        fat_clf_path: str, path to trained fatigue classifier
        window_size: int, window size in samples
        step_size: int, step size in samples
    
    Returns:
        list: list of dicts with segment info and predictions
    """
    print(f"Processing: {signal_path}")
    
    # Load models
    print("Loading models...")
    detector = ActivityDetector()
    detector.load(detector_path)
    
    amp_clf = AmplitudeClassifier()
    amp_clf.load(amp_clf_path)
    
    fat_clf = FatigueClassifier()
    fat_clf.load(fat_clf_path)
    
    # Load and preprocess signal
    print("Loading and preprocessing signal...")
    raw_signal = load_emg_data(signal_path)
    filtered_signal = preprocess_signal(raw_signal)
    print(f"Signal length: {len(filtered_signal)} samples ({len(filtered_signal)/2000:.2f} seconds)")
    
    # Create sliding windows for activity detection
    print("Detecting active segments...")
    windows, start_indices = create_sliding_windows(filtered_signal, window_size, step_size)
    
    # Extract features
    features, _ = extract_features_from_windows(windows)
    
    # Predict activity
    predictions = detector.predict(features)
    active_windows = np.sum(predictions)
    print(f"Active windows: {active_windows}/{len(predictions)} ({100*active_windows/len(predictions):.1f}%)")
    
    # Convert predictions to segments
    from preprocessing import detect_activity_segments
    segments = detect_activity_segments(predictions, start_indices)
    print(f"Detected segments: {len(segments)}")
    
    # Classify each segment
    results = []
    for i, (start, end) in enumerate(segments):
        print(f"\nSegment {i+1}: [{start}-{end}] ({(end-start)/2000:.2f}s)")
        
        # Extract segment
        segment_signal = filtered_signal[start:end]
        
        # Extract features
        seg_features = extract_segment_features(segment_signal)
        feature_array = np.array([list(seg_features.values())])
        
        # Predict amplitude
        amplitude = amp_clf.predict(feature_array)[0]
        amp_proba = amp_clf.predict_proba(feature_array)[0]
        print(f"  Amplitude: {amplitude} (confidence: {np.max(amp_proba):.3f})")
        
        # Predict fatigue (only for full amplitude)
        if amplitude == 'full':
            fatigue = fat_clf.predict(feature_array)[0]
            fat_proba = fat_clf.predict_proba(feature_array)[0]
            print(f"  Fatigue: {fatigue} (confidence: {np.max(fat_proba):.3f})")
        else:
            fatigue = 'free'  # By constraint
            print(f"  Fatigue: {fatigue} (by constraint)")
        
        results.append({
            'segment_id': i + 1,
            'start': start,
            'end': end,
            'duration_s': (end - start) / 2000,
            'amplitude': amplitude,
            'fatigue': fatigue
        })
    
    # Visualize
    if len(segments) > 0:
        plot_signal_with_segments(
            filtered_signal,
            segments,
            title=f'EMG Signal Segmentation: {os.path.basename(signal_path)}',
            save_path=f'segmentation_{os.path.basename(signal_path).replace(".csv", ".png")}'
        )
    
    return results


def main():
    """Example usage."""
    # Check if models exist
    if not os.path.exists('models/activity_detector.pkl'):
        print("Error: Models not found. Please run 'python main.py' first to train models.")
        return
    
    # Find a test file
    test_files = []
    if os.path.exists('train'):
        import glob
        test_files = glob.glob('train/full_free_*.csv')[:1]  # Take first file as example
    
    if not test_files:
        print("Error: No training files found for testing.")
        return
    
    # Process example file
    test_file = test_files[0]
    results = segment_and_classify_signal(
        test_file,
        'models/activity_detector.pkl',
        'models/amplitude_classifier.pkl',
        'models/fatigue_classifier.pkl'
    )
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in results:
        print(f"Segment {result['segment_id']}: {result['amplitude']}/{result['fatigue']} "
              f"({result['duration_s']:.2f}s)")


if __name__ == '__main__':
    main()
