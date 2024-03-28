import numpy as np
from scipy.signal import butter, sosfilt
from scipy.signal import wiener


from fonoSemillIAS.preproc.helpers import plot_filter_response

def band_pass_filter(signal: np.array, fs: int, low_cutoff_freq: int, high_cutoff_freq: int, plot=False):
    """
    Applies a band-pass filter to a given signal

    Args:
    - signal (np.array): Input audio signal
    - fs (int): Sampling frequency of the signal
    - low_cutoff_freq (int): Lower cutoff frequency of the band-pass filter
    - high_cutoff_freq (int): Higher cutoff frequency of the band-pass filter
    - plot (bool, optional): If True, plots the frequency response of the filter. Default is False

    Returns:
    - np.array: Filtered signal
    """
    sos = butter(20, [low_cutoff_freq, high_cutoff_freq], 'bandpass', fs=fs, output="sos")
    filtered_signal = sosfilt(sos, signal)
    
    window_size = 3
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(filtered_signal, window, mode='same')
    
    if plot:
        plot_filter_response(sos, fs, "Band-pass Filter")
    
    return smoothed_signal

def high_pass_filter(signal: np.array, fs: int, cutoff_freq: int, plot=False):
    """
    Applies a high-pass filter to a given signal

    Args:
    - signal (np.array): Input audio signal
    - fs (int): Sampling frequency of the signal
    - cutoff_freq (int): Cutoff frequency of the high-pass filter
    - plot (bool, optional): If True, plots the frequency response of the filter. Default is False

    Returns:
    - np.array: Filtered signal
    """
    sos = butter(20, cutoff_freq, 'highpass', fs=fs, output="sos")
    filtered_signal = sosfilt(sos, signal)
    
    if plot:
        plot_filter_response(sos, fs, "High-pass Filter")
    
    return filtered_signal

def low_pass_filter(signal: np.array, fs: int, cutoff_freq: int, plot=False):
    """
    Applies a low-pass filter to a given signal

    Args:
    - signal (np.array): Input audio signal
    - fs (int): Sampling frequency of the signal
    - cutoff_freq (int): Cutoff frequency of the low-pass filter
    - plot (bool, optional): If True, plots the frequency response of the filter. Default is False

    Returns:
    - np.array: Filtered signal
    """
    sos = butter(20, cutoff_freq, 'lowpass', fs=fs, output="sos")
    filtered_signal = sosfilt(sos, signal)

    if plot:
        plot_filter_response(sos, fs, "Low-pass Filter")

    return filtered_signal

def band_stop_filter(signal: np.array, fs: int, low_cutoff_freq: int, high_cutoff_freq: int, plot=False):
    """
    Applies a band-stop filter to a given signal

    Args:
    - signal (np.array): Input audio signal
    - fs (int): Sampling frequency of the signal
    - low_cutoff_freq (int): Lower cutoff frequency of the band-stop filter
    - high_cutoff_freq (int): Higher cutoff frequency of the band-stop filter
    - plot (bool, optional): If True, plots the frequency response of the filter. Default is False

    Returns:
    - np.array: Filtered signal
    """
    sos = butter(20, [low_cutoff_freq, high_cutoff_freq], 'bandstop', fs=fs, output="sos")
    filtered_signal = sosfilt(sos, signal)
    
    if plot:
        plot_filter_response(sos, fs, "Band-stop Filter")
    
    return filtered_signal

def adaptive_filter(signal: np.array) -> np.array:
    """
    Applies adaptive filtering to a given signal using Wiener filter.

    Args:
    - signal (np.array): Input signal

    Returns:
    - np.array: Filtered signal
    """
    filtered_signal = wiener(signal)
    return filtered_signal