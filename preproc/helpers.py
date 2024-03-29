from scipy.fft import rfft, rfftfreq # Fast Fourier Transform
from scipy.signal import sosfreqz
import numpy as np
import matplotlib.pyplot as plt


def fft(signal:np.array, fs:float, show:bool = False):
  """
  Function to apply Fast Fourier transform
  --------------------------------------------------
  Args:
  - signal (np.array): The audio signal to be plotted. Represents amplitude values.
  - fs (float): sampling frequency of the signal
  - show (bool): If true it will show the fourier transform graph, if false it will not show it. Default is false.
  Returns:
    yFreq (np.array): Represents amplitude values in frecuency domain
    Displays the plot (Opcional)
  """
  yFreq = rfft(signal) # Extracts the magnitudes
  xFreq = rfftfreq(n = len(signal), d = 1/fs) # n = length of the signal and d = 1/fs ; Extract the signal frequencies.
  if show:
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(xFreq, np.abs(yFreq)) # Magnitude spectrum
    ax.set_ylabel('Amplitude'); ax.set_title('FFT') ; ax.set_xlabel("Frequency (Hz)")
    plt.show()
  return yFreq

def plot_filter_response(sos, fs, title):
    w, h = sosfreqz(sos, fs=fs)
    plt.figure(figsize=(20, 3))
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.xscale('log')
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid(True)
    plt.show()

def get_fundamental_frequency(signal, fs):
    """
    Calculate the fundamental frequency of a voice signal.

    Args:
    - signal (np.array): The voice signal.
    - fs (int): The sampling frequency of the signal.

    Returns:
    - float: The estimated fundamental frequency.
    """
    # Calculate the autocorrelation of the signal
    autocorr = np.correlate(signal, signal, mode='full')

    # Find the first maximum after the first minimum (remove DC)
    first_min_index = np.argmax(autocorr)
    autocorr = autocorr[first_min_index:]

    # Find the second maximum (fundamental frequency) after the first maximum
    fundamental_index = np.argmax(autocorr[1:]) + 1

    # Calculate the fundamental frequency in Hz
    fundamental_freq = fs / fundamental_index
    return fundamental_freq