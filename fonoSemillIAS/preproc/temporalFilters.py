import numpy as np
from scipy.signal import medfilt

def median_filter(signal: np.array, window_size: int) -> np.array:
    """
    Applies median filtering to a given signal.

    Args:
    - signal (np.array): Input signal
    - window_size (int): Size of the median filter window

    Returns:
    - np.array: Filtered signal
    """
    filtered_signal = medfilt(signal, kernel_size=window_size)
    return filtered_signal

def linear_least_squares_filter(signal: np.array, reference_signal: np.array) -> np.array:
    """
    Applies linear least squares filtering to a given signal using a reference signal.

    Args:
    - signal (np.array): Input signal
    - reference_signal (np.array): Reference signal (noise)

    Returns:
    - np.array: Filtered signal
    """
    # Assuming reference_signal has the same length as signal
    # Calculate filter coefficients using linear least squares
    filter_coefficients = np.linalg.lstsq(reference_signal.reshape(-1, 1), signal, rcond=None)[0]
    
    # Apply the filter to the signal
    filtered_signal = np.convolve(reference_signal, filter_coefficients, mode='same')
    
    return filtered_signal
