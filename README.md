# EMG Signal Analysis Pipeline

A complete machine learning pipeline for EMG (Electromyography) signal analysis using Python. This project implements supervised learning for:

1. **Segmentation**: Detecting active muscle contractions in continuous EMG signals
2. **Action Recognition**: Classifying action amplitude (Full, Half, Invalid)
3. **Fatigue Detection**: Classifying muscle fatigue levels (Free, Light, Medium, Heavy)

## Features

- **Advanced Signal Processing**: Bandpass filtering (20-450 Hz) and Notch filtering (50 Hz) for 2000 Hz EMG data
- **Supervised Segmentation**: Activity detection using sliding window features and Random Forest/XGBoost classifiers
- **Advanced Feature Extraction**:
  - Time-domain: RMS, MAV, Zero Crossings (ZC), Slope Sign Changes (SSC), Waveform Length (WL)
  - Frequency-domain: Mean Frequency, Median Frequency
  - Time-Frequency: Discrete Wavelet Transform (DWT) and Wavelet Packet Decomposition (WPD) energy ratios
  - Autoregressive (AR) coefficients
- **Multi-class Classification**: Separate models for amplitude and fatigue detection
- **Per-Subject Learning**: Subject-specific model training for improved accuracy
- **Test Set Prediction**: Comprehensive prediction with detailed visualization
- **Detailed Training Statistics**: Sample counts per class, subject distribution
- **Visualization**: Confusion matrices, signal segmentation plots, feature heatmaps

## Project Structure

```
EMG/
├── train/                          # Training data
│   ├── *.csv                       # Raw EMG signals
│   └── *_segments/                 # Manual segmentation examples
├── test/                           # Test data (optional)
│   └── *.csv                       # Test EMG signals
├── src/
│   ├── preprocessing.py            # Signal filtering and segmentation
│   ├── features.py                 # Advanced feature extraction
│   ├── models.py                   # ML model wrappers
│   ├── per_subject_learning.py    # Per-subject learning module
│   └── utils.py                    # Utilities and visualization
├── main.py                         # Main pipeline entry point
├── main_per_subject.py             # Pipeline with per-subject learning
├── predict_test_set.py             # Test set prediction with visualization
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Data Format

**Filename Convention**: `[Amplitude]_[Fatigue]_[SubjectID]_[ActionID].csv`

- **Amplitudes**: `full`, `half`, `invalid`
- **Fatigue Levels**: `free`, `light`, `medium`, `heavy`
- **Sampling Rate**: 2000 Hz

**Constraints**:
- `half`/`invalid` amplitudes always have `free` fatigue
- `light`/`medium`/`heavy` fatigue levels only occur with `full` amplitude

## Installation

```bash
# Install required packages
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- PyWavelets
- xgboost

## Usage

### Run the Complete Pipeline (Standard)

```bash
python main.py
```

This will:
1. Train an activity detector using sliding window features
2. Train amplitude and fatigue classifiers using manual segments
3. Output detailed training statistics (samples per class, subject distribution)
4. Save trained models to `models/` directory
5. Generate confusion matrices as PNG files

### Run the Pipeline with Per-Subject Learning

```bash
python main_per_subject.py
```

This enhanced version:
1. Trains separate models for each subject to learn individual patterns
2. Combines per-subject and global models for better accuracy
3. Especially improves fatigue classification (different people have different fatigue patterns)
4. Outputs per-subject model statistics

### Test Set Prediction with Visualization

```bash
python predict_test_set.py
```

This will:
1. Load trained models from `models/` directory
2. Segment and classify signals from `test/` directory (or uses training data if no test directory)
3. Generate comprehensive visualizations for each signal:
   - Full signal with segmentation boundaries
   - Amplitude and fatigue predictions per segment
   - Feature heatmap showing characteristics of each segment
4. Save all visualizations to `predictions/` directory
5. Output prediction summary statistics

### Run Unit Tests

```bash
python -m unittest tests/test_pipeline.py
```

### Pipeline Configuration

Edit parameters in `main.py` or `main_per_subject.py`:

```python
WINDOW_SIZE = 200    # Window size (0.1s at 2000 Hz)
STEP_SIZE = 100      # Step size (0.05s, 50% overlap)
MODEL_TYPE = 'random_forest'  # or 'xgboost'

# Per-subject learning options (main_per_subject.py only)
USE_PER_SUBJECT = True  # Enable per-subject learning

# Filtering options for better accuracy
FILTER_AMPLITUDE_BY_FATIGUE = False  # Use only 'free' fatigue for amplitude training
FILTER_FATIGUE_BY_AMPLITUDE = True   # Use only 'full' amplitude for fatigue training
```

### Output

The standard pipeline (`main.py`) generates:
- **models/**
  - `activity_detector.pkl`: Binary classifier for segment detection
  - `amplitude_classifier.pkl`: Multi-class classifier for amplitude
  - `fatigue_classifier.pkl`: Multi-class classifier for fatigue
- **Confusion Matrices**
  - `amplitude_confusion_matrix.png`
  - `fatigue_confusion_matrix.png`

The per-subject pipeline (`main_per_subject.py`) additionally generates:
- **models/**
  - `amplitude_classifier_ps.pkl`: Per-subject amplitude classifier
  - `fatigue_classifier_ps.pkl`: Per-subject fatigue classifier
- **Confusion Matrices**
  - `amplitude_confusion_matrix_ps.png`
  - `fatigue_confusion_matrix_ps.png`

The test prediction script (`predict_test_set.py`) generates:
- **predictions/**
  - `[filename]_segmentation.png`: Full signal with detected segments and predictions
  - `[filename]_features.png`: Feature heatmap for all segments

## Technical Approach

### 1. Activity Detection (Segmentation)

Instead of simple thresholding, we train a supervised binary classifier:
- Extract features from sliding windows (200 samples, 50% overlap)
- Label windows based on overlap with manual segments
- Train Random Forest classifier to detect active vs. inactive periods
- Convert predictions back to segment boundaries

### 2. Feature Engineering

**Hudgins' Time-Domain Features**:
- Root Mean Square (RMS)
- Mean Absolute Value (MAV)
- Zero Crossings (ZC)
- Slope Sign Changes (SSC)
- Waveform Length (WL)
- Variance (VAR)

**Frequency-Domain Features**:
- Mean Frequency
- Median Frequency

**Wavelet Features**:
- DWT energy ratios (Daubechies wavelet, level 4)
- WPD energy ratios (level 3, all leaf nodes)

**AR Model**:
- 4th order autoregressive coefficients

### 3. Classification

**Amplitude Classifier**: Distinguishes between Full, Half, and Invalid actions

**Fatigue Classifier**: Classifies fatigue level (Free, Light, Medium, Heavy) for Full amplitude actions only

Both use Random Forest or XGBoost with 80/20 train/test split.

## Performance

Example results from training:

- **Activity Detector**: ~82% accuracy (binary classification)
- **Amplitude Classifier**: ~91% accuracy (3-class)
- **Fatigue Classifier**: ~41% accuracy (4-class, challenging problem)

### Training Statistics

The enhanced pipeline now outputs detailed statistics during training:

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

### Per-Subject Learning Benefits

When using `main_per_subject.py`:
- Each subject gets a personalized model that learns their specific patterns
- Falls back to global model for unseen subjects
- Particularly beneficial for fatigue detection where individual patterns vary significantly
- The system trains 1 global model + N subject-specific models (where N = number of subjects with sufficient data)

Note: Fatigue detection is inherently difficult due to subtle signal changes and limited training data. Per-subject learning helps by accounting for individual variations in fatigue patterns.

## Extending the Pipeline

### Use Trained Models for Prediction

```python
from src.models import ActivityDetector, AmplitudeClassifier, FatigueClassifier
from src.preprocessing import load_emg_data, preprocess_signal
from src.features import extract_segment_features

# Load models
detector = ActivityDetector()
detector.load('models/activity_detector.pkl')

amp_clf = AmplitudeClassifier()
amp_clf.load('models/amplitude_classifier.pkl')

fat_clf = FatigueClassifier()
fat_clf.load('models/fatigue_classifier.pkl')

# Process new signal
signal = load_emg_data('new_data.csv')
filtered = preprocess_signal(signal)

# Segment and classify
# ... (see main.py or predict_test_set.py for complete example)
```

### Use Per-Subject Models

```python
from src.per_subject_learning import PerSubjectClassifier

# Load per-subject model
amp_clf_ps = PerSubjectClassifier()
amp_clf_ps.load('models/amplitude_classifier_ps.pkl')

# Predict with subject ID for personalized prediction
predictions = amp_clf_ps.predict(features, subject_ids=['1', '2', '1'])

# Or predict without subject ID (uses global model)
predictions = amp_clf_ps.predict(features)
```

### Visualize Test Results

The `predict_test_set.py` script provides comprehensive visualization:

1. **Signal Segmentation Plot**: Shows the complete signal with:
   - Detected segment boundaries
   - Color-coded amplitude predictions per segment
   - Timeline view with amplitude and fatigue predictions

2. **Feature Heatmap**: Displays normalized features for each segment:
   - Key time-domain features (RMS, MAV, ZC, SSC, WL, VAR)
   - Frequency-domain features (Mean/Median frequency)
   - Easy comparison across segments

### Add Custom Features

Extend `src/features.py` with additional feature extraction functions and update `extract_all_features()`.

### Try Different Models

Change `MODEL_TYPE` to experiment with different classifiers or modify `src/models.py` to add new model types.

## References

- Hudgins et al. "A New Strategy for Multifunction Myoelectric Control" (1993)
- Phinyomark et al. "Feature Reduction and Selection for EMG Signal Classification" (2012)
- EMG Signal Processing: [scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html)
- Wavelet Analysis: [PyWavelets](https://pywavelets.readthedocs.io/)

## License

This project is provided as-is for educational and research purposes.