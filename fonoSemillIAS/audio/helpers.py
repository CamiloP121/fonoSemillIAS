import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

def plot_sound(time:np.array, amplitude: np.array, name: str) -> None:
    """
    Plots the amplitude of an audio signal over time.
    -----------------------------------------------------------
    Args:
    - time (np.array): Array representing the time axis of the audio signal.
    - amplitude (np.array): Array representing the amplitude values of the audio signal.
    - name (str): Name of the audio file.

    Returns:
    - None
    """
    fig, ax = plt.subplots(figsize=(20,3))
    ax.plot(time, amplitude)
    ax.set_ylabel('Amplitude'); ax.set_title('Audio: ' + name) ; ax.set_xlabel("Time (Seconds)")
    plt.show()

def plot_sounds(signals:list, labels:list,time:np.array, tittle:str):
  """
  Function to plots a given audio signal against time.
  -------------------------------------------------------
  Args:
    - signals (list): List with audios signals to be plotted. Represents amplitude values.
    - time (np.array): The time values corresponding to the audio signal.
    - labels (str): List with labels signals which will be used in the plot's legend.
    - title (str): The title to be displayed on top of the plot.
  Returns:
    None. Displays the plot
  """
  fig, ax = plt.subplots(figsize=(20,3))
  for l, s in zip(labels, signals):
    ax.plot(time, s, label=l)

  ax.legend()
  ax.set_ylabel('Amplitude'); ax.set_title(tittle) ; ax.set_xlabel("Time (Seconds)")
  plt.show()

def play_sound(amplitude: np.array, fs: int):
    wn = Audio(data = amplitude, rate = fs)
    display(wn)


def add_noise(signal, noise_level, echo_delay, echo_decay, voice_attenuation_factor):
    """
    Add ambient noise and echo to an audio signal and attenuate voice.

    Args:
    - signal (np.array): The input audio signal.
    - noise_level (float): The level of ambient noise to be added.
    - echo_delay (int): The delay (in milliseconds) of the echo effect.
    - echo_decay (float): The decay factor of the echo effect.
    - voice_attenuation_factor (float): The attenuation factor for the voice (0 for complete silence, 1 for no attenuation).

    Returns:
    - np.array: The audio signal with noise, echo, and attenuated voice.
    """
    # Generate ambient noise
    noise = np.random.normal(0, noise_level, len(signal))

    # Generate echo
    echo = np.zeros_like(signal)
    echo[:echo_delay] = signal[:echo_delay]
    for i in range(echo_delay, len(signal)):
        echo[i] = signal[i] + echo_decay * echo[i - echo_delay]

    # Attenuate voice
    attenuated_signal = signal - (voice_attenuation_factor * signal)

    # Add noise, echo, and attenuated voice
    signal_with_noise_echo_and_attenuated_voice = attenuated_signal + noise + echo

    return signal_with_noise_echo_and_attenuated_voice