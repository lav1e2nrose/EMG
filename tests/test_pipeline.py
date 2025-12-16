"""
Unit tests for EMG signal analysis pipeline.
"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from preprocessing import (
    bandpass_filter, notch_filter, preprocess_signal,
    create_sliding_windows, label_windows_from_segments,
    detect_activity_segments
)
from features import (
    compute_rms, compute_mav, compute_zc, compute_ssc, compute_wl,
    extract_all_features
)
from utils import parse_filename, validate_labels


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""
    
    def test_bandpass_filter(self):
        """Test bandpass filter."""
        signal = np.random.randn(1000)
        filtered = bandpass_filter(signal)
        
        self.assertEqual(len(filtered), len(signal))
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_notch_filter(self):
        """Test notch filter."""
        signal = np.random.randn(1000)
        filtered = notch_filter(signal)
        
        self.assertEqual(len(filtered), len(signal))
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_preprocess_signal(self):
        """Test full preprocessing pipeline."""
        signal = np.random.randn(1000)
        processed = preprocess_signal(signal)
        
        self.assertEqual(len(processed), len(signal))
        self.assertIsInstance(processed, np.ndarray)
    
    def test_create_sliding_windows(self):
        """Test sliding window creation."""
        signal = np.arange(1000)
        window_size = 100
        step_size = 50
        
        windows, indices = create_sliding_windows(signal, window_size, step_size)
        
        self.assertEqual(windows.shape[1], window_size)
        self.assertEqual(len(windows), len(indices))
        self.assertGreater(len(windows), 0)
    
    def test_label_windows_from_segments(self):
        """Test window labeling."""
        signal_length = 1000
        segment_ranges = [(100, 300), (500, 700)]
        window_size = 50
        step_size = 25
        
        labels, indices = label_windows_from_segments(
            signal_length, segment_ranges, window_size, step_size
        )
        
        self.assertEqual(len(labels), len(indices))
        self.assertTrue(all(l in [0, 1] for l in labels))
    
    def test_label_windows_overlap_threshold(self):
        """Ensure partial overlap still marks window as active."""
        signal_length = 400
        # Very short segment that would be missed by center-only logic
        segment_ranges = [(95, 105)]
        window_size = 100
        step_size = 50
        
        labels, _ = label_windows_from_segments(
            signal_length, segment_ranges, window_size, step_size, overlap_threshold=0.05
        )
        
        start_positions = list(range(0, signal_length - window_size + 1, step_size))
        idx_for_50 = start_positions.index(50)
        
        # The window starting at 50 overlaps 10 samples with the segment
        self.assertEqual(labels[idx_for_50], 1)
    
    def test_detect_activity_segments(self):
        """Test segment detection from predictions."""
        predictions = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0])
        start_indices = np.arange(len(predictions)) * 100
        
        segments = detect_activity_segments(predictions, start_indices, min_length=100)
        
        self.assertIsInstance(segments, list)
        self.assertTrue(all(isinstance(s, tuple) and len(s) == 2 for s in segments))


class TestFeatures(unittest.TestCase):
    """Test feature extraction functions."""
    
    def test_compute_rms(self):
        """Test RMS computation."""
        window = np.array([1, 2, 3, 4, 5])
        rms = compute_rms(window)
        
        expected = np.sqrt(np.mean(window ** 2))
        self.assertAlmostEqual(rms, expected)
    
    def test_compute_mav(self):
        """Test MAV computation."""
        window = np.array([-1, -2, 3, 4, -5])
        mav = compute_mav(window)
        
        expected = np.mean(np.abs(window))
        self.assertAlmostEqual(mav, expected)
    
    def test_compute_zc(self):
        """Test zero crossing computation."""
        window = np.array([1, -1, 1, -1, 1])
        zc = compute_zc(window, threshold=0.01)
        
        self.assertIsInstance(zc, (int, np.integer))
        self.assertGreaterEqual(zc, 0)
    
    def test_compute_ssc(self):
        """Test slope sign change computation."""
        window = np.array([1, 2, 1, 2, 1])
        ssc = compute_ssc(window, threshold=0.01)
        
        self.assertIsInstance(ssc, (int, np.integer))
        self.assertGreaterEqual(ssc, 0)
    
    def test_compute_wl(self):
        """Test waveform length computation."""
        window = np.array([1, 3, 2, 4, 3])
        wl = compute_wl(window)
        
        expected = np.sum(np.abs(np.diff(window)))
        self.assertAlmostEqual(wl, expected)
    
    def test_extract_all_features(self):
        """Test complete feature extraction."""
        window = np.random.randn(200)
        features = extract_all_features(window)
        
        self.assertIsInstance(features, dict)
        self.assertIn('rms', features)
        self.assertIn('mav', features)
        self.assertIn('mean_freq', features)
        self.assertIn('median_freq', features)
        self.assertTrue(any('dwt' in k for k in features.keys()))
        self.assertTrue(any('wpd' in k for k in features.keys()))
        self.assertTrue(any('ar_coeff' in k for k in features.keys()))


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_parse_filename_valid(self):
        """Test filename parsing with valid names."""
        test_cases = [
            ('full_free_1_001.csv', 'full', 'free', '1', '001'),
            ('half_free_2_003', 'half', 'free', '2', '003'),
            ('invalid_free_3_002.csv', 'invalid', 'free', '3', '002'),
            ('full_light_1_001.csv', 'full', 'light', '1', '001'),
            ('full_medium_2_005.csv', 'full', 'medium', '2', '005'),
            ('full_heavy_3_010.csv', 'full', 'heavy', '3', '010'),
        ]
        
        for filename, exp_amp, exp_fat, exp_sub, exp_act in test_cases:
            result = parse_filename(filename)
            self.assertIsNotNone(result)
            self.assertEqual(result['amplitude'], exp_amp)
            self.assertEqual(result['fatigue'], exp_fat)
            self.assertEqual(result['subject_id'], exp_sub)
            self.assertEqual(result['action_id'], exp_act)
    
    def test_parse_filename_invalid(self):
        """Test filename parsing with invalid names."""
        invalid_names = [
            'invalid_filename.csv',
            'test.csv',
            'full_free.csv',
        ]
        
        for filename in invalid_names:
            result = parse_filename(filename)
            self.assertIsNone(result)
    
    def test_validate_labels_valid(self):
        """Test label validation with valid combinations."""
        valid_cases = [
            {'amplitude': 'full', 'fatigue': 'free'},
            {'amplitude': 'full', 'fatigue': 'light'},
            {'amplitude': 'full', 'fatigue': 'medium'},
            {'amplitude': 'full', 'fatigue': 'heavy'},
            {'amplitude': 'half', 'fatigue': 'free'},
            {'amplitude': 'invalid', 'fatigue': 'free'},
        ]
        
        for metadata in valid_cases:
            self.assertTrue(validate_labels(metadata))
    
    def test_validate_labels_invalid(self):
        """Test label validation with invalid combinations."""
        invalid_cases = [
            {'amplitude': 'half', 'fatigue': 'light'},
            {'amplitude': 'invalid', 'fatigue': 'medium'},
            {'amplitude': 'half', 'fatigue': 'heavy'},
        ]
        
        for metadata in invalid_cases:
            self.assertFalse(validate_labels(metadata))


if __name__ == '__main__':
    unittest.main()
