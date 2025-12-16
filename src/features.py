"""
Feature extraction module for EMG signal analysis.
Implements advanced features: time-domain, frequency-domain, wavelet, and AR features.
"""
import numpy as np
import pywt
from scipy import signal
from scipy.fft import fft, fftfreq


def compute_rms(window):
    """
    Compute Root Mean Square (RMS).
    
    Args:
        window: array-like, signal window
    
    Returns:
        float: RMS value
    """
    return np.sqrt(np.mean(window ** 2))


def compute_mav(window):
    """
    Compute Mean Absolute Value (MAV).
    
    Args:
        window: array-like, signal window
    
    Returns:
        float: MAV value
    """
    return np.mean(np.abs(window))


def compute_zc(window, threshold=0.01):
    """
    Compute Zero Crossings (ZC).
    
    Args:
        window: array-like, signal window
        threshold: float, threshold to avoid noise-induced crossings
    
    Returns:
        int: number of zero crossings
    """
    # Apply threshold
    thresholded = window.copy()
    thresholded[np.abs(thresholded) < threshold] = 0
    
    # Count zero crossings
    signs = np.sign(thresholded)
    sign_changes = np.diff(signs)
    zc_count = np.sum(np.abs(sign_changes) > 0)
    
    return zc_count


def compute_ssc(window, threshold=0.01):
    """
    Compute Slope Sign Changes (SSC).
    
    Args:
        window: array-like, signal window
        threshold: float, threshold to avoid noise-induced changes
    
    Returns:
        int: number of slope sign changes
    """
    if len(window) < 3:
        return 0
    
    ssc_count = 0
    for i in range(1, len(window) - 1):
        diff1 = window[i] - window[i-1]
        diff2 = window[i] - window[i+1]
        
        if (diff1 * diff2 >= threshold):
            ssc_count += 1
    
    return ssc_count


def compute_wl(window):
    """
    Compute Waveform Length (WL).
    
    Args:
        window: array-like, signal window
    
    Returns:
        float: waveform length
    """
    return np.sum(np.abs(np.diff(window)))


def compute_var(window):
    """
    Compute Variance (VAR).
    
    Args:
        window: array-like, signal window
    
    Returns:
        float: variance
    """
    return np.var(window)


def compute_mean_freq(window, fs=2000):
    """
    Compute Mean Frequency in the frequency domain.
    
    Args:
        window: array-like, signal window
        fs: int, sampling rate in Hz
    
    Returns:
        float: mean frequency
    """
    # Compute FFT
    N = len(window)
    fft_vals = fft(window)
    freqs = fftfreq(N, 1/fs)
    
    # Use only positive frequencies
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    power = np.abs(fft_vals[pos_mask]) ** 2
    
    # Compute mean frequency
    if np.sum(power) == 0:
        return 0
    
    mean_freq = np.sum(freqs * power) / np.sum(power)
    return mean_freq


def compute_median_freq(window, fs=2000):
    """
    Compute Median Frequency in the frequency domain.
    
    Args:
        window: array-like, signal window
        fs: int, sampling rate in Hz
    
    Returns:
        float: median frequency
    """
    # Compute FFT
    N = len(window)
    fft_vals = fft(window)
    freqs = fftfreq(N, 1/fs)
    
    # Use only positive frequencies
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    power = np.abs(fft_vals[pos_mask]) ** 2
    
    # Compute median frequency
    cumsum_power = np.cumsum(power)
    total_power = cumsum_power[-1]
    
    if total_power == 0:
        return 0
    
    median_idx = np.where(cumsum_power >= total_power / 2)[0]
    if len(median_idx) > 0:
        return freqs[median_idx[0]]
    
    return 0


def compute_dwt_features(window, wavelet='db4', level=4):
    """
    Compute Discrete Wavelet Transform (DWT) energy features.
    
    Args:
        window: array-like, signal window
        wavelet: str, wavelet type (default: 'db4' - Daubechies 4)
        level: int, decomposition level
    
    Returns:
        np.array: energy ratios for each level
    """
    # Perform DWT
    coeffs = pywt.wavedec(window, wavelet, level=level)
    
    # Compute energy for each level
    energies = [np.sum(c ** 2) for c in coeffs]
    total_energy = np.sum(energies)
    
    if total_energy == 0:
        return np.zeros(len(energies))
    
    # Return energy ratios
    energy_ratios = np.array(energies) / total_energy
    return energy_ratios


def compute_wpd_features(window, wavelet='db4', level=3):
    """
    Compute Wavelet Packet Decomposition (WPD) energy features.
    
    Args:
        window: array-like, signal window
        wavelet: str, wavelet type (default: 'db4')
        level: int, decomposition level
    
    Returns:
        np.array: energy ratios for leaf nodes
    """
    # Perform WPD
    wp = pywt.WaveletPacket(data=window, wavelet=wavelet, maxlevel=level)
    
    # Get all leaf nodes
    nodes = [node.path for node in wp.get_level(level, 'freq')]
    
    # Compute energy for each node
    energies = []
    for node_path in nodes:
        node = wp[node_path]
        energy = np.sum(node.data ** 2)
        energies.append(energy)
    
    total_energy = np.sum(energies)
    
    if total_energy == 0:
        return np.zeros(len(energies))
    
    # Return energy ratios
    energy_ratios = np.array(energies) / total_energy
    return energy_ratios


def compute_ar_coefficients(window, order=4):
    """
    Compute Autoregressive (AR) model coefficients using Burg method.
    
    Args:
        window: array-like, signal window
        order: int, AR model order
    
    Returns:
        np.array: AR coefficients
    """
    try:
        # Use Levinson-Durbin recursion for AR parameter estimation
        # Based on autocorrelation
        N = len(window)
        
        # Compute autocorrelation
        autocorr = np.correlate(window, window, mode='full')
        autocorr = autocorr[N-1:]  # Keep only non-negative lags
        autocorr = autocorr[:order+1]  # Keep up to order p
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Levinson-Durbin recursion
        ar_coeffs = np.zeros(order)
        ar_coeffs[0] = -autocorr[1]
        
        for k in range(1, order):
            # Compute reflection coefficient
            numerator = autocorr[k+1]
            for j in range(k):
                numerator += ar_coeffs[j] * autocorr[k-j]
            
            denominator = 1
            for j in range(k):
                denominator -= ar_coeffs[j] * autocorr[j+1]
            
            if abs(denominator) < 1e-10:
                break
            
            reflection = -numerator / denominator
            
            # Update coefficients
            new_coeffs = ar_coeffs.copy()
            for j in range(k):
                new_coeffs[j] = ar_coeffs[j] + reflection * ar_coeffs[k-1-j]
            new_coeffs[k] = reflection
            ar_coeffs = new_coeffs
        
        return ar_coeffs
    
    except:
        # Return zeros if computation fails
        return np.zeros(order)


def extract_all_features(window, fs=2000, wavelet='db4', dwt_level=4, 
                        wpd_level=3, ar_order=4):
    """
    Extract all features from a signal window.
    
    Args:
        window: array-like, signal window
        fs: int, sampling rate in Hz
        wavelet: str, wavelet type
        dwt_level: int, DWT decomposition level
        wpd_level: int, WPD decomposition level
        ar_order: int, AR model order
    
    Returns:
        dict: dictionary of all features
    """
    features = {}
    
    # Time-domain features (Hudgins' features)
    features['rms'] = compute_rms(window)
    features['mav'] = compute_mav(window)
    features['zc'] = compute_zc(window)
    features['ssc'] = compute_ssc(window)
    features['wl'] = compute_wl(window)
    features['var'] = compute_var(window)
    
    # Frequency-domain features
    features['mean_freq'] = compute_mean_freq(window, fs)
    features['median_freq'] = compute_median_freq(window, fs)
    
    # Wavelet features (DWT)
    dwt_features = compute_dwt_features(window, wavelet, dwt_level)
    for i, energy in enumerate(dwt_features):
        features[f'dwt_energy_{i}'] = energy
    
    # Wavelet Packet features (WPD)
    wpd_features = compute_wpd_features(window, wavelet, wpd_level)
    for i, energy in enumerate(wpd_features):
        features[f'wpd_energy_{i}'] = energy
    
    # Autoregressive coefficients
    ar_coeffs = compute_ar_coefficients(window, ar_order)
    for i, coeff in enumerate(ar_coeffs):
        features[f'ar_coeff_{i}'] = coeff
    
    return features


def extract_features_from_windows(windows, fs=2000, wavelet='db4', 
                                  dwt_level=4, wpd_level=3, ar_order=4):
    """
    Extract features from multiple windows.
    
    Args:
        windows: array-like, shape (n_windows, window_size)
        fs: int, sampling rate in Hz
        wavelet: str, wavelet type
        dwt_level: int, DWT decomposition level
        wpd_level: int, WPD decomposition level
        ar_order: int, AR model order
    
    Returns:
        np.array: feature matrix, shape (n_windows, n_features)
        list: feature names
    """
    feature_list = []
    
    for window in windows:
        features = extract_all_features(window, fs, wavelet, dwt_level, 
                                       wpd_level, ar_order)
        feature_list.append(features)
    
    # Convert to array
    if len(feature_list) > 0:
        feature_names = list(feature_list[0].keys())
        feature_matrix = np.array([[f[name] for name in feature_names] 
                                   for f in feature_list])
        return feature_matrix, feature_names
    
    return np.array([]), []


def extract_segment_features(segment_signal, fs=2000, wavelet='db4', 
                            dwt_level=4, wpd_level=3, ar_order=4):
    """
    Extract features from a complete segment (not windowed).
    Used for classification of already segmented signals.
    
    Args:
        segment_signal: array-like, complete segment signal
        fs: int, sampling rate in Hz
        wavelet: str, wavelet type
        dwt_level: int, DWT decomposition level
        wpd_level: int, WPD decomposition level
        ar_order: int, AR model order
    
    Returns:
        dict: dictionary of features
    """
    return extract_all_features(segment_signal, fs, wavelet, dwt_level, 
                               wpd_level, ar_order)
