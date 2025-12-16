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
- **Visualization**: Confusion matrices and classification reports

## Project Structure

```
EMG/
├── train/                          # Training data
│   ├── *.csv                       # Raw EMG signals
│   └── *_segments/                 # Manual segmentation examples
├── src/
│   ├── preprocessing.py            # Signal filtering and segmentation
│   ├── features.py                 # Advanced feature extraction
│   ├── models.py                   # ML model wrappers
│   └── utils.py                    # Utilities and visualization
├── main.py                         # Main pipeline entry point
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

### Run the Complete Pipeline

```bash
python main.py
```

### Run Unit Tests

```bash
python -m unittest tests/test_pipeline.py
```

This will:
1. Train an activity detector using sliding window features
2. Train amplitude and fatigue classifiers using manual segments
3. Save trained models to `models/` directory
4. Generate confusion matrices as PNG files

### Pipeline Configuration

Edit parameters in `main.py`:

```python
WINDOW_SIZE = 200    # Window size (0.1s at 2000 Hz)
STEP_SIZE = 100      # Step size (0.05s, 50% overlap)
MODEL_TYPE = 'random_forest'  # or 'xgboost'
```

### Output

The pipeline generates:
- **models/**
  - `activity_detector.pkl`: Binary classifier for segment detection
  - `amplitude_classifier.pkl`: Multi-class classifier for amplitude
  - `fatigue_classifier.pkl`: Multi-class classifier for fatigue
- **Confusion Matrices**
  - `amplitude_confusion_matrix.png`
  - `fatigue_confusion_matrix.png`

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

Note: Fatigue detection is inherently difficult due to subtle signal changes and limited training data. Performance may improve with more data and hyperparameter tuning.

## Extending the Pipeline

### Use Trained Models

```python
from src.models import ActivityDetector, AmplitudeClassifier, FatigueClassifier
from src.preprocessing import load_emg_data, preprocess_signal
from src.features import extract_segment_features

# Load models
detector = ActivityDetector()
detector.load('models/activity_detector.pkl')

amp_clf = AmplitudeClassifier()
amp_clf.load('models/amplitude_classifier.pkl')

# Process new signal
signal = load_emg_data('new_data.csv')
filtered = preprocess_signal(signal)

# Segment and classify
# ... (see main.py for complete example)
```

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