import numpy as np
from sklearn.decomposition import PCA
import noisereduce as nr # Filter
import pywt

from fonoSemillIAS.preproc.helpers import plot_coeffs

def pca_decomposition(signal: np.array, n_components: int) -> np.array:
    """
    Applies Principal Component Analysis (PCA) decomposition to a given signal.

    Args:
    - signal (np.array): Input signal
    - n_components (int): Number of components to keep

    Returns:
    - np.array: Transformed signal with reduced dimensions
    """
    pca = PCA(n_components=n_components)
    transformed_signal = pca.fit_transform(signal)
    return transformed_signal

def nrp_filter(signal:np.array, fs:int, torch: bool = False):
    """
    Apply a noise reduction filter on the provided audio signal
    ------------------------------------------------------------
    Args:
    - signal (np.array): The input audio signal array that needs noise reduction.
    - fs (int): Sampling frequency of the signal in Hz.

    Returns:
    - reduced_noise_torch (np.array): The audio signal with reduced noise.
    """
    reduced_noise_torch = nr.reduce_noise(y = signal , sr = fs, n_std_thresh_stationary=1.5,stationary=True,
                               use_torch=torch)

    return reduced_noise_torch

def DWT_filter(signal, wavelet='db6', level=5, threshold_value=0.1, plot = False):
    """
    Apply discrete wavelet transform (DWT) and filtering to the signal.

    Args:
    - signal (np.array): The input signal.
    - wavelet (str): The type of wavelet to use (default is 'db6').
    - level (int): The level of decomposition (default is 1).
    - threshold_value (float): The threshold value for thresholding.
    - plot (bool): Show the difference between coefficients before and after filtering.

    Returns:
    - np.array: The reconstructed signal after applying DWT and filtering.
    """
    # Apply DWT
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Filter detail coefficients
    filtered_coeffs = [pywt.threshold(detail_coef, threshold_value * max(detail_coef)) for detail_coef in coeffs[1:]]

    # Reconstruct the signal from the filtered coefficients
    print(len(filtered_coeffs), len(coeffs))
    reconstructed_signal = pywt.waverec([coeffs[0]] + filtered_coeffs, wavelet)

    if plot: plot_coeffs(original_coeffs=coeffs, filtered_coeffs=filtered_coeffs)

    return reconstructed_signal, [coeffs, filtered_coeffs]