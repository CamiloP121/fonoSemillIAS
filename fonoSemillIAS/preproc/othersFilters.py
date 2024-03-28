import numpy as np
from sklearn.decomposition import PCA
import noisereduce as nr # Filter

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