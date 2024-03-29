import numpy as np

def calculate_snr(signal:np.array, noise:np.array):
    """
    Calculate the Signal-to-Noise Ratio (SNR) of a signal.

    Args:
    - signal (np.array): The signal of interest.
    - noise (np.array): The noise signal.

    Returns:
    - float: The SNR in decibels (dB).
    """
    # Calculate the power of the signal and noise
    signal_power = np.mean(np.square(signal))
    noise_power = np.mean(np.square(noise))

    # Calculate SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db