from scipy.fft import rfft, rfftfreq # Fast Fourier Transform
from scipy.signal import sosfreqz
import numpy as np
import matplotlib.pyplot as plt

from othersFilters import DWT_filter
from othersFilters import nrp_filter
from temporalFilters import median_filter


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

def plot_coeffs(original_coeffs, filtered_coeffs):
  """
  Plot original and filtered wavelet coefficients.

  Args:
  - original_coeffs (list of np.array): List of arrays containing the original wavelet coefficients.
  - filtered_coeffs (list of np.array): List of arrays containing the filtered wavelet coefficients.
  """
  plt.figure(figsize=(10, 6))
  for i in range(len(original_coeffs)):
      plt.subplot(len(original_coeffs), 1, i+1)
      plt.plot(original_coeffs[i], label='Original')
      plt.plot(filtered_coeffs[i], label='Filtered')
      plt.title(f'Wavelet Coefficients - Level {i+1}')
      plt.xlabel('Index')
      plt.ylabel('Coefficient Value')
      plt.legend()
  plt.tight_layout()
  plt.show()

def apply_preproc(signal, fs, torch):
  """
  Applies preprocessing to a signal.

  Parameters:
  - signal (array): The input signal.
  - fs (float): The sampling frequency of the signal.
  - torch (bool): Indicates whether to use PyTorch's adaptive threshold for the filter.

  Returns:
  - signal_pb (array): The processed signal after applying preprocessing.

  Raises:
  - Exception: If an error occurs during preprocessing.
  """

  try:
      # Applying discrete wavelet transform (DWT) filter to the signal
      signal_dwt, _ = DWT_filter(signal, wavelet="sym10", level=5, threshold_value=0.02)
      
      # Applying non-redundant noise reduction filter (NRP) to the DWT signal
      if torch: 
          signal_pb = nrp_filter(signal=signal_dwt, std_tresh=4.5, fs=fs, torch=True)
      else:
          signal_pb = nrp_filter(signal=signal_dwt, std_tresh=0.5, fs=fs, torch=False)
  except Exception as e:
      # Handling any exceptions that may occur during preprocessing
      print(e)
      raise Exception("Error in preprocessing")
  
  return signal_pb

if __name__ == "__main__":
   print("s")