# EMG Signal Analysis - Usage Guide

This guide provides detailed instructions for using the EMG signal analysis pipeline with all its features.

## Quick Start

### 1. Standard Training Pipeline

Train models with standard approach:

```bash
python main.py
```

**Output:**
- Detailed training statistics (samples per class, subject distribution)
- Trained models saved to `models/` directory
- Confusion matrices saved as PNG files

### 2. Per-Subject Learning Pipeline

Train models with per-subject learning for better accuracy:

```bash
python main_per_subject.py
```

**Output:**
- Per-subject model statistics showing how many subject-specific models were trained
- Enhanced models that learn individual patterns per subject
- Separate confusion matrices for per-subject models

### 3. Test Set Prediction

Predict on test data with comprehensive visualization:

```bash
python predict_test_set.py
```

**Output:**
- Signal segmentation plots for each test file
- Feature heatmaps showing segment characteristics
- Prediction summary with class distributions
- All visualizations saved to `predictions/` directory

## Detailed Training Statistics

The enhanced training pipeline outputs detailed statistics to help you understand your data:

```
--- Training Data Statistics ---
Total training actions (CSV files): 31

Amplitude Distribution:
  full: 141 samples
  half: 33 samples
  invalid: 42 samples

Fatigue Distribution:
  free: 115 samples
  heavy: 30 samples
  light: 35 samples
  medium: 36 samples

Subject Distribution:
  Subject 1: 98 samples
  Subject 2: 46 samples
  Subject 3: 72 samples
```

This helps you identify:
- Class imbalances that might affect training
- Per-subject data distribution
- Whether you have sufficient samples for each class

## Per-Subject Learning Explained

### Why Use Per-Subject Learning?

Different people have different EMG signal patterns, especially for fatigue detection:
- Individual muscle activation patterns vary
- Fatigue manifests differently across subjects
- Some subjects may have stronger or weaker signals

### How It Works

The per-subject classifier:
1. Trains a separate model for each subject (if they have sufficient data)
2. Also trains a global model on all subjects combined
3. At prediction time:
   - Uses subject-specific model if available
   - Falls back to global model for unknown subjects

### Example Output

```
Training per-subject models for 3 subjects...
  Subject 1: 55 samples
  Subject 2: 19 samples
  Subject 3: 38 samples

Training global model on all 112 samples...
Per-subject training complete: 3 subject-specific models
```

## Test Set Prediction Workflow

### Input

Place your test signals in the `test/` directory:
```
test/
├── signal1.csv
├── signal2.csv
└── signal3.csv
```

Or the script will use files from `train/` directory if no `test/` directory exists.

### Process

For each test file, the script:
1. Loads and preprocesses the signal
2. Uses the activity detector to find active segments
3. Classifies each segment for amplitude and fatigue
4. Generates visualizations

### Output Visualizations

**1. Segmentation Plot** (`[filename]_segmentation.png`):
- Top panel: Full signal with colored segment regions
- Bottom panel: Timeline showing:
  - Active segments (blue bars)
  - Amplitude predictions (color-coded: green=full, orange=half, red=invalid)
  - Fatigue predictions (color-coded: lightgreen=free, yellow=light, orange=medium, red=heavy)

**2. Feature Heatmap** (`[filename]_features.png`):
- Rows: Each detected segment
- Columns: Key features (RMS, MAV, ZC, SSC, WL, VAR, mean_freq, median_freq)
- Values: Normalized feature values (0-1 scale for easy comparison)
- Row labels: Segment ID with amplitude/fatigue prediction

### Prediction Summary

At the end, you get a summary across all processed files:

```
PREDICTION SUMMARY
================================================================================

Total files processed: 5
Total segments detected: 14

Predicted Amplitude Distribution:
  full: 13 segments
  half: 1 segments

Predicted Fatigue Distribution:
  free: 11 segments
  heavy: 1 segments
  medium: 2 segments

Visualization files saved to 'predictions/' directory
```

## Configuration Options

### Main Pipeline Options

In `main.py` or `main_per_subject.py`:

```python
# Window parameters for activity detection
WINDOW_SIZE = 200    # Window size in samples (0.1s at 2000 Hz)
STEP_SIZE = 100      # Step size in samples (0.05s, 50% overlap)

# Model selection
MODEL_TYPE = 'random_forest'  # Options: 'random_forest', 'xgboost'

# Per-subject learning (main_per_subject.py only)
USE_PER_SUBJECT = True  # Enable per-subject learning

# Data filtering for improved accuracy
FILTER_AMPLITUDE_BY_FATIGUE = False  # If True, use only 'free' fatigue for amplitude training
FILTER_FATIGUE_BY_AMPLITUDE = True   # Already enforced (only 'full' amplitude for fatigue)
```

### Prediction Options

In `predict_test_set.py`:

```python
# Window parameters (should match training)
window_size = 200
step_size = 100

# Limit number of files to process (for testing)
for test_file in test_files[:10]:  # Process first 10 files
```

## Best Practices

### For Training

1. **Check Class Balance**: Review the training statistics to ensure balanced classes
2. **Consider Per-Subject Learning**: Use `main_per_subject.py` if you have multiple subjects
3. **Filter Data**: Try filtering options if one classifier performs poorly:
   - Set `FILTER_AMPLITUDE_BY_FATIGUE = True` to use only "free" fatigue for amplitude training
   - This can help if amplitude classifier is confused by fatigue variations

### For Prediction

1. **Verify Models**: Ensure models are trained before running `predict_test_set.py`
2. **Check Visualizations**: Review the generated plots to verify segmentation quality
3. **Analyze Feature Heatmaps**: Use feature heatmaps to understand what distinguishes different segments

### For Better Fatigue Accuracy

Fatigue detection is challenging. To improve:

1. **Use Per-Subject Learning**: Different people show fatigue differently
2. **Collect More Data**: More fatigue examples help the model learn patterns
3. **Check Subject Distribution**: Ensure each subject has examples of different fatigue levels
4. **Review Predictions**: Use visualizations to identify where the model struggles

## Troubleshooting

### Error: "Models not found"

Run training first:
```bash
python main.py
# or
python main_per_subject.py
```

### Error: "No CSV files found"

Ensure your data directory has CSV files:
```bash
ls train/*.csv
# or
ls test/*.csv
```

### Low Accuracy

Try these approaches:

1. **Use per-subject learning**: `python main_per_subject.py`
2. **Enable filtering**: Set `FILTER_AMPLITUDE_BY_FATIGUE = True`
3. **Collect more training data**: Especially for underrepresented classes
4. **Try different model**: Change `MODEL_TYPE` to 'xgboost'

### Chinese Font Warnings

These are harmless if your test files have Chinese names. The visualizations still work correctly, just without proper Chinese character rendering.

## API Reference

### Loading Models

```python
from src.models import ActivityDetector, AmplitudeClassifier, FatigueClassifier
from src.per_subject_learning import PerSubjectClassifier

# Standard models
detector = ActivityDetector()
detector.load('models/activity_detector.pkl')

amp_clf = AmplitudeClassifier()
amp_clf.load('models/amplitude_classifier.pkl')

# Per-subject models
amp_clf_ps = PerSubjectClassifier()
amp_clf_ps.load('models/amplitude_classifier_ps.pkl')
```

### Making Predictions

```python
from src.preprocessing import load_emg_data, preprocess_signal, create_sliding_windows
from src.features import extract_features_from_windows, extract_segment_features
import numpy as np

# Load and preprocess
signal = load_emg_data('test.csv')
filtered = preprocess_signal(signal)

# Detect segments
windows, indices = create_sliding_windows(filtered, window_size=200, step_size=100)
features, _ = extract_features_from_windows(windows)
predictions = detector.predict(features)

# Convert to segments
from src.preprocessing import detect_activity_segments
segments = detect_activity_segments(predictions, indices)

# Classify each segment
for start, end in segments:
    segment_signal = filtered[start:end]
    seg_features = extract_segment_features(segment_signal)
    feature_array = np.array([list(seg_features.values())])
    
    amplitude = amp_clf.predict(feature_array)[0]
    if amplitude == 'full':
        fatigue = fat_clf.predict(feature_array)[0]
    else:
        fatigue = 'free'  # By constraint
    
    print(f"Segment [{start}-{end}]: {amplitude}/{fatigue}")
```

### Per-Subject Predictions

```python
# If you know the subject IDs
subject_ids = ['1', '1', '2', '3', '1']  # One per segment
predictions = amp_clf_ps.predict(features, subject_ids)

# If you don't know subject IDs (uses global model)
predictions = amp_clf_ps.predict(features)
```

## Examples

See the main scripts for complete examples:
- `main.py`: Standard training pipeline
- `main_per_subject.py`: Per-subject learning pipeline
- `predict_test_set.py`: Complete prediction workflow with visualization
- `example_inference.py`: Simple inference example

## Support

For issues or questions:
1. Check this usage guide
2. Review the main README.md
3. Examine the example scripts
4. Check the code comments in `src/` modules
