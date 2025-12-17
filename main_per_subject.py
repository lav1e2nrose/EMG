"""
Main entry point for EMG signal analysis pipeline with per-subject learning.
Orchestrates data loading, preprocessing, feature extraction, training, and evaluation.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import (
    load_emg_data, preprocess_signal, create_sliding_windows,
    label_windows_from_segments, improved_get_segment_ranges,
    detect_activity_segments, segment_signal_improved
)
from features import extract_features_from_windows, extract_segment_features
from models import ActivityDetector, AmplitudeClassifier, FatigueClassifier
from per_subject_learning import PerSubjectClassifier, extract_subject_features
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


def train_per_subject_classifiers(train_dir, model_type='random_forest', 
                                  filter_amplitude_by_fatigue=True,
                                  filter_fatigue_by_amplitude=True,
                                  use_per_subject=True,
                                  n_folds=5):
    """
    Train amplitude and fatigue classifiers with per-subject learning and k-fold cross-validation.
    
    Args:
        train_dir: str, path to training directory
        model_type: str, 'random_forest' or 'xgboost'
        filter_amplitude_by_fatigue: bool, if True, use only 'free' fatigue for amplitude
                                     (Default: True - amplitude classes only apply to 'free' state)
        filter_fatigue_by_amplitude: bool, if True, use only 'full' amplitude for fatigue
                                     (Default: True - fatigue only applies to 'full' amplitude)
        use_per_subject: bool, if True, use per-subject learning
        n_folds: int, number of folds for cross-validation (default: 5)
    
    Returns:
        tuple: (classifier, classifier) for amplitude and fatigue
    """
    print("\n=== Training Classifiers with Per-Subject Learning and K-Fold CV ===")
    
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
    
    # Count by amplitude, fatigue, and subject
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
    # IMPORTANT: Amplitude classification (full/half/invalid) only applies to 'free' fatigue state
    X_amp = []
    y_amp = []
    subj_amp = []
    
    for seg in segments_data:
        if seg['features'] is not None:
            # Apply filtering: amplitude classes only apply when fatigue is 'free'
            if filter_amplitude_by_fatigue and seg['fatigue'] != 'free':
                continue
            
            feature_values = list(seg['features'].values())
            X_amp.append(feature_values)
            y_amp.append(seg['amplitude'])
            subj_amp.append(seg['subject_id'])
    
    X_amp = np.array(X_amp)
    y_amp = np.array(y_amp)
    subj_amp = np.array(subj_amp)
    
    # Train amplitude classifier with k-fold cross-validation
    print("\n--- Training Amplitude Classifier with K-Fold CV ---")
    if filter_amplitude_by_fatigue:
        print("Using filtered data: only 'free' fatigue samples (amplitude classification only applies to 'free' state)")
    print(f"Total samples: {len(y_amp)}")
    unique_classes, class_counts = np.unique(y_amp, return_counts=True)
    print(f"Samples per amplitude class:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  {cls}: {count} samples")
    print(f"Using {n_folds}-fold cross-validation")
    
    # K-fold cross-validation for amplitude classifier
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_y_amp_true = []
    all_y_amp_pred = []
    amp_fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_amp, y_amp)):
        X_train, X_test = X_amp[train_idx], X_amp[test_idx]
        y_train, y_test = y_amp[train_idx], y_amp[test_idx]
        subj_train, subj_test = subj_amp[train_idx], subj_amp[test_idx]
        
        # Train model for this fold
        if use_per_subject:
            fold_classifier = PerSubjectClassifier(model_type=model_type)
            fold_classifier.fit(X_train, y_train, subj_train)
            y_pred = fold_classifier.predict(X_test, subj_test)
        else:
            fold_classifier = AmplitudeClassifier(model_type=model_type)
            fold_classifier.fit(X_train, y_train)
            y_pred = fold_classifier.predict(X_test)
        
        # Collect predictions
        all_y_amp_true.extend(y_test)
        all_y_amp_pred.extend(y_pred)
        
        fold_acc = accuracy_score(y_test, y_pred)
        amp_fold_accuracies.append(fold_acc)
        print(f"  Fold {fold+1}: Accuracy = {fold_acc:.4f}")
    
    # Train final model on all data
    if use_per_subject:
        amp_classifier = PerSubjectClassifier(model_type=model_type)
        amp_classifier.fit(X_amp, y_amp, subj_amp)
    else:
        amp_classifier = AmplitudeClassifier(model_type=model_type)
        amp_classifier.fit(X_amp, y_amp)
    
    # Compute overall metrics from cross-validation
    all_y_amp_true = np.array(all_y_amp_true)
    all_y_amp_pred = np.array(all_y_amp_pred)
    
    amp_accuracy = accuracy_score(all_y_amp_true, all_y_amp_pred)
    print(f"\nAmplitude Classifier CV Accuracy: {amp_accuracy:.4f} (+/- {np.std(amp_fold_accuracies):.4f})")
    print("\nClassification Report (aggregated from all folds):")
    print_classification_report(
        all_y_amp_true, all_y_amp_pred,
        labels=amp_classifier.get_classes(),
        target_names=amp_classifier.get_classes()
    )
    
    # Plot confusion matrix with aggregated predictions from all folds
    suffix = '_ps' if use_per_subject else ''
    plot_confusion_matrix(
        all_y_amp_true, all_y_amp_pred,
        labels=amp_classifier.get_classes(),
        title=f'Amplitude Classification Confusion Matrix ({n_folds}-Fold CV)',
        save_path=f'amplitude_confusion_matrix{suffix}.png'
    )
    
    # Prepare data for fatigue classification
    # IMPORTANT: Fatigue classification only applies to 'full' amplitude samples
    X_fat = []
    y_fat = []
    subj_fat = []
    
    for seg in segments_data:
        if seg['features'] is not None:
            # Fatigue classification only applies to 'full' amplitude
            if filter_fatigue_by_amplitude and seg['amplitude'] != 'full':
                continue
            
            feature_values = list(seg['features'].values())
            X_fat.append(feature_values)
            y_fat.append(seg['fatigue'])
            subj_fat.append(seg['subject_id'])
    
    X_fat = np.array(X_fat)
    y_fat = np.array(y_fat)
    subj_fat = np.array(subj_fat)
    
    # Train fatigue classifier with k-fold cross-validation
    print("\n--- Training Fatigue Classifier with K-Fold CV ---")
    print("Using only 'full' amplitude samples (fatigue classification only applies to 'full' amplitude)")
    print(f"Total samples: {len(y_fat)}")
    unique_classes, class_counts = np.unique(y_fat, return_counts=True)
    print(f"Samples per fatigue class:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  {cls}: {count} samples")
    print(f"Using {n_folds}-fold cross-validation")
    
    # K-fold cross-validation for fatigue classifier
    all_y_fat_true = []
    all_y_fat_pred = []
    fat_fold_accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_fat, y_fat)):
        X_train, X_test = X_fat[train_idx], X_fat[test_idx]
        y_train, y_test = y_fat[train_idx], y_fat[test_idx]
        subj_train, subj_test = subj_fat[train_idx], subj_fat[test_idx]
        
        # Train model for this fold
        if use_per_subject:
            fold_classifier = PerSubjectClassifier(model_type=model_type)
            fold_classifier.fit(X_train, y_train, subj_train)
            y_pred = fold_classifier.predict(X_test, subj_test)
        else:
            fold_classifier = FatigueClassifier(model_type=model_type)
            fold_classifier.fit(X_train, y_train)
            y_pred = fold_classifier.predict(X_test)
        
        # Collect predictions
        all_y_fat_true.extend(y_test)
        all_y_fat_pred.extend(y_pred)
        
        fold_acc = accuracy_score(y_test, y_pred)
        fat_fold_accuracies.append(fold_acc)
        print(f"  Fold {fold+1}: Accuracy = {fold_acc:.4f}")
    
    # Train final model on all data
    if use_per_subject:
        fat_classifier = PerSubjectClassifier(model_type=model_type)
        fat_classifier.fit(X_fat, y_fat, subj_fat)
    else:
        fat_classifier = FatigueClassifier(model_type=model_type)
        fat_classifier.fit(X_fat, y_fat)
    
    # Compute overall metrics from cross-validation
    all_y_fat_true = np.array(all_y_fat_true)
    all_y_fat_pred = np.array(all_y_fat_pred)
    
    fat_accuracy = accuracy_score(all_y_fat_true, all_y_fat_pred)
    print(f"\nFatigue Classifier CV Accuracy: {fat_accuracy:.4f} (+/- {np.std(fat_fold_accuracies):.4f})")
    print("\nClassification Report (aggregated from all folds):")
    print_classification_report(
        all_y_fat_true, all_y_fat_pred,
        labels=fat_classifier.get_classes(),
        target_names=fat_classifier.get_classes()
    )
    
    # Plot confusion matrix with aggregated predictions from all folds
    plot_confusion_matrix(
        all_y_fat_true, all_y_fat_pred,
        labels=fat_classifier.get_classes(),
        title=f'Fatigue Classification Confusion Matrix ({n_folds}-Fold CV)',
        save_path=f'fatigue_confusion_matrix{suffix}.png'
    )
    
    return amp_classifier, fat_classifier


def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("EMG Signal Analysis Pipeline with Per-Subject Learning")
    print("=" * 60)
    
    # Configuration
    TRAIN_DIR = 'train'
    WINDOW_SIZE = 200  # 0.1 seconds at 2000 Hz
    STEP_SIZE = 100    # 0.05 seconds at 2000 Hz (50% overlap)
    MODEL_TYPE = 'random_forest'  # or 'xgboost'
    N_FOLDS = 5  # Number of folds for cross-validation
    
    # Per-subject learning options
    USE_PER_SUBJECT = True  # Enable per-subject learning for better accuracy
    
    # Filtering options
    # Amplitude classification only applies to 'free' fatigue state
    FILTER_AMPLITUDE_BY_FATIGUE = True   # Use only 'free' fatigue for amplitude classification
    # Fatigue classification only applies to 'full' amplitude
    FILTER_FATIGUE_BY_AMPLITUDE = True   # Use only 'full' amplitude for fatigue classification
    
    # Check if train directory exists
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory '{TRAIN_DIR}' not found!")
        return
    
    # Step 1: Train activity detector
    print("\nStep 1: Training Activity Detector...")
    activity_detector = train_activity_detector(
        TRAIN_DIR, 
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        model_type=MODEL_TYPE
    )
    
    # Save activity detector
    os.makedirs('models', exist_ok=True)
    activity_detector.save('models/activity_detector.pkl')
    
    # Step 2: Train amplitude and fatigue classifiers with per-subject learning and k-fold CV
    print("\nStep 2: Training Classifiers with Per-Subject Learning and K-Fold CV...")
    amp_classifier, fat_classifier = train_per_subject_classifiers(
        TRAIN_DIR,
        model_type=MODEL_TYPE,
        filter_amplitude_by_fatigue=FILTER_AMPLITUDE_BY_FATIGUE,
        filter_fatigue_by_amplitude=FILTER_FATIGUE_BY_AMPLITUDE,
        use_per_subject=USE_PER_SUBJECT,
        n_folds=N_FOLDS
    )
    
    # Save classifiers
    if USE_PER_SUBJECT:
        amp_classifier.save('models/amplitude_classifier_ps.pkl')
        fat_classifier.save('models/fatigue_classifier_ps.pkl')
        print("\nPer-subject models saved:")
        print("  - amplitude_classifier_ps.pkl")
        print("  - fatigue_classifier_ps.pkl")
    else:
        amp_classifier.save('models/amplitude_classifier.pkl')
        fat_classifier.save('models/fatigue_classifier.pkl')
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    print("\nModels saved to 'models/' directory:")
    print("  - activity_detector.pkl")
    if USE_PER_SUBJECT:
        print("  - amplitude_classifier_ps.pkl (per-subject)")
        print("  - fatigue_classifier_ps.pkl (per-subject)")
    else:
        print("  - amplitude_classifier.pkl")
        print("  - fatigue_classifier.pkl")
    print("\nConfusion matrices saved:")
    if USE_PER_SUBJECT:
        print("  - amplitude_confusion_matrix_ps.png")
        print("  - fatigue_confusion_matrix_ps.png")
    else:
        print("  - amplitude_confusion_matrix.png")
        print("  - fatigue_confusion_matrix.png")


if __name__ == '__main__':
    main()
