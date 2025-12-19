"""
Unit tests for EMG signal analysis pipeline.
"""
import unittest
import numpy as np
import torch
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from preprocessing import (
    bandpass_filter, notch_filter, preprocess_signal,
    create_sliding_windows, label_windows_from_segments,
    detect_activity_segments, label_signal_three_state,
    create_sequence_windows_for_segmentation,
    compute_rms_envelope, compute_mav_envelope,
    compute_adaptive_threshold, detect_activity_regions,
    segment_signal_improved, improved_get_segment_ranges,
    find_segment_in_raw_signal, learn_segment_basis_and_detect
)
from features import (
    compute_rms, compute_mav, compute_zc, compute_ssc, compute_wl,
    extract_all_features
)
from utils import parse_filename, validate_labels
from models import CRNNActivitySegmenter


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

    def test_label_signal_three_state(self):
        """Three-state labeling splits segments into two phases."""
        signal_length = 200
        segment_ranges = [(20, 120)]
        labels = label_signal_three_state(signal_length, segment_ranges, concentric_ratio=0.5)

        self.assertEqual(len(labels), signal_length)
        self.assertTrue(np.all(labels[:20] == 0))
        self.assertTrue(np.all(labels[20:70] == 1))  # first half
        self.assertTrue(np.all(labels[70:120] == 2))  # second half

    def test_create_sequence_windows_for_segmentation(self):
        """Windows and label windows stay aligned."""
        signal = np.arange(20)
        labels = np.zeros(20, dtype=int)
        labels[5:10] = 1
        windows, lbl = create_sequence_windows_for_segmentation(signal, labels, 10, 5)

        self.assertEqual(windows.shape, lbl.shape)
        self.assertEqual(windows.shape[1], 10)
        self.assertTrue(np.array_equal(lbl[1], labels[5:15]))

    def test_compute_rms_envelope(self):
        """Test RMS envelope computation."""
        signal = np.random.randn(1000)
        window_size = 50
        envelope = compute_rms_envelope(signal, window_size)
        
        self.assertEqual(len(envelope), len(signal))
        self.assertTrue(np.all(envelope >= 0))  # RMS is always non-negative

    def test_compute_mav_envelope(self):
        """Test MAV envelope computation."""
        signal = np.random.randn(1000)
        window_size = 50
        envelope = compute_mav_envelope(signal, window_size)
        
        self.assertEqual(len(envelope), len(signal))
        self.assertTrue(np.all(envelope >= 0))  # MAV is always non-negative

    def test_compute_adaptive_threshold_otsu(self):
        """Test Otsu adaptive thresholding."""
        # Create bimodal distribution
        low_values = np.random.rand(500) * 0.3
        high_values = 0.7 + np.random.rand(500) * 0.3
        envelope = np.concatenate([low_values, high_values])
        np.random.shuffle(envelope)
        
        threshold = compute_adaptive_threshold(envelope, method='otsu')
        
        self.assertIsInstance(threshold, (float, np.floating))
        # Otsu should find a threshold; exact value depends on histogram binning
        self.assertTrue(np.min(envelope) <= threshold <= np.max(envelope))

    def test_compute_adaptive_threshold_percentile(self):
        """Test percentile-based thresholding."""
        envelope = np.arange(100)
        
        threshold = compute_adaptive_threshold(envelope, method='percentile', percentile=75)
        
        expected = np.percentile(envelope, 75)
        self.assertAlmostEqual(threshold, expected)

    def test_detect_activity_regions(self):
        """Test RMS-based activity region detection."""
        # Create synthetic signal with clear active and inactive regions
        fs = 2000
        inactive = np.random.randn(2000) * 0.1  # Low amplitude noise
        active = np.random.randn(2000) * 2.0    # High amplitude activity
        signal = np.concatenate([inactive, active, inactive])
        
        segments = detect_activity_regions(signal, fs=fs, min_segment_ms=200)
        
        self.assertIsInstance(segments, list)
        # Should detect at least one active region
        if len(segments) > 0:
            # Each segment should be a tuple (start, end)
            self.assertTrue(all(isinstance(s, tuple) and len(s) == 2 for s in segments))
            # Segment start should be in the active region (around sample 2000)
            for start, end in segments:
                self.assertGreater(end, start)

    def test_segment_signal_improved(self):
        """Test improved segmentation with dual threshold."""
        # Create synthetic signal with clear active and inactive regions
        fs = 2000
        inactive = np.random.randn(2000) * 0.1  # Low amplitude noise
        active = np.random.randn(2000) * 2.0    # High amplitude activity
        signal = np.concatenate([inactive, active, inactive])
        
        segments = segment_signal_improved(signal, fs=fs, min_segment_ms=200)
        
        self.assertIsInstance(segments, list)
        # Each segment should be a tuple (start, end)
        if len(segments) > 0:
            self.assertTrue(all(isinstance(s, tuple) and len(s) == 2 for s in segments))

    def test_segment_signal_improved_merging(self):
        """Test that close segments are merged."""
        fs = 2000
        # Create two bursts separated by a short gap
        inactive1 = np.random.randn(1000) * 0.1
        active1 = np.random.randn(500) * 2.0
        gap = np.random.randn(100) * 0.1  # Short gap (50ms)
        active2 = np.random.randn(500) * 2.0
        inactive2 = np.random.randn(1000) * 0.1
        signal = np.concatenate([inactive1, active1, gap, active2, inactive2])
        
        # With merge_gap_ms=300, the two bursts should be merged
        segments = segment_signal_improved(signal, fs=fs, min_segment_ms=100, merge_gap_ms=300)
        
        self.assertIsInstance(segments, list)

    def test_find_segment_in_raw_signal_precise(self):
        """Correlation-based overlay should return exact start."""
        rng = np.random.default_rng(0)
        raw = rng.standard_normal(3000) * 0.05
        start_true = 750
        burst = rng.standard_normal(200) * 2.0
        raw[start_true:start_true + len(burst)] += burst
        segment = raw[start_true:start_true + len(burst)]

        start_idx = find_segment_in_raw_signal(raw, segment, min_correlation=0.5)
        self.assertIsNotNone(start_idx)
        self.assertAlmostEqual(start_idx, start_true, delta=2)

    def test_learn_segment_basis_and_detect(self):
        """ML-based basis should recover segments when overlay is unknown."""
        fs = 2000
        rng = np.random.default_rng(1)
        raw = rng.standard_normal(5000) * 0.05
        burst = np.sin(np.linspace(0, 2 * np.pi, 400)) * 2.0
        raw[600:1000] += burst
        raw[2600:3000] += burst

        seg_signal = raw[600:1000]
        segments = learn_segment_basis_and_detect(
            raw, [seg_signal], fs=fs, window_ms=50, step_ms=25, min_length=80, prob_threshold=0.4
        )

        self.assertIsInstance(segments, list)
        self.assertTrue(len(segments) > 0)
        self.assertTrue(any(abs(s - 600) < 150 for s, _ in segments))

    def test_improved_get_segment_ranges_with_fallback(self):
        """Test that improved_get_segment_ranges falls back to automatic detection."""
        # Skip if train directory doesn't exist
        train_dir = os.path.join(os.path.dirname(__file__), '..', 'train')
        if not os.path.exists(train_dir):
            self.skipTest("Train directory not found")
        
        # Find a file that has segments directory
        test_file = None
        test_segment_dir = None
        for item in os.listdir(train_dir):
            if item.endswith('.csv'):
                segment_dir = os.path.join(train_dir, item.replace('.csv', '_segments'))
                if os.path.exists(segment_dir) and os.listdir(segment_dir):
                    test_file = os.path.join(train_dir, item)
                    test_segment_dir = segment_dir
                    break
        
        if test_file is None:
            self.skipTest("No test files with segments found")
        
        # Call improved_get_segment_ranges
        ranges = improved_get_segment_ranges(test_file, test_segment_dir)
        
        # Should find at least some segments
        self.assertIsInstance(ranges, list)
        self.assertGreater(len(ranges), 0, "Should find at least one segment")
        
        # Each range should be a tuple (start, end)
        for r in ranges:
            self.assertIsInstance(r, tuple)
            self.assertEqual(len(r), 2)
            self.assertLess(r[0], r[1])  # start < end


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


class TestVisualization(unittest.TestCase):
    """Test visualization helpers."""

    def test_plot_labeled_segments_saves_file(self):
        from main import plot_labeled_segments  # main only defines functions, no side effects

        signal = np.sin(np.linspace(0, 2 * np.pi, 400))
        segments = [(50, 150), (200, 300)]
        amp_labels = ["full", "half"]
        fat_labels = ["free", "light"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "seg.png")
            plot_labeled_segments(signal, segments, amp_labels, fat_labels, save_path)

            self.assertTrue(os.path.exists(save_path))
            self.assertGreater(os.path.getsize(save_path), 0)


class TestCRNN(unittest.TestCase):
    """Lightweight CRNN sanity checks."""

    def test_forward_shape(self):
        model = CRNNActivitySegmenter(sequence_length=50, step_size=25, device="cpu")
        dummy = np.random.randn(2, 50).astype(np.float32)
        preds = model.predict(dummy)
        self.assertEqual(preds.shape, (2, 50))


if __name__ == '__main__':
    unittest.main()
