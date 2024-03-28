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