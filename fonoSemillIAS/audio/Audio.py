import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from fonoSemillIAS.audio.helpers import plot_sound, play_sound

class Audio():
  def __init__(self, file_path:str, debug: bool = False):
    """
    Class constructor
    ---------------------------------------
    Arg:
      - file_path (str): string with the address where the audio is located.
    Attributes:
      - path (str): string with the address where the audio is located.
      - name (str): string with the name of the audio (last element the path address)
      - amplitudes (array): array with the information of the signal
      - fs (float): sampling frequency of the signal.
    Methods:
      - open: internal operation to load the audio
      - get_info: Displays basic information of the loaded audio
      - plot_wave: Display the signal
      - play_sound: Play the audio
    """

    self.path = file_path
    self.name = file_path.split("/")[-1]
    self.fs, self.amplitude, self.channels, self.duration = self.open()
    self.time = np.arange(0, len(self.amplitude), 1) / self.fs
    self.signal = None
    self.debug = debug

    if debug: print(f"Audio {self.name} loaded correctly") 

  def open(self):
    """
    Open the audio
    """
    # Review the extention
    assert self.path.endswith(".wav"), "Not a valid extension"

    # Load the audio
    try:
      fs, amplitude = wavfile.read(self.path)
    except Exception as e:
      print(e)
      raise Exception("Failed to load audio")

    # Review num of channels
    channels = len(np.shape(amplitude))
    assert channels <= 2, "Has more than two channels"
    if channels == 2:
      # Average amplitude
      ## Review length of channels
      assert len(amplitude[:,0]) == len(amplitude[:,1]), "The channels do not have the same dimensions"
      amplitude = np.mean(amplitude, axis=1, dtype=int)

    duration = len(amplitude) / fs
    return fs, amplitude, channels, duration

  def get_info(self):
    if self.channels != 2:
      type_sound = 'mono audio'
    else:
      type_sound = 'stereo audio'
    # print info
    print('Sampling (Hz) : ',self.fs)
    print('Channels: ' + str(self.channels) + ' type ' + type_sound )
    print('Duration (s): ', self.duration)
    print('Matrix size: ', len(self.amplitude))

  def plot_sound_original(self):
    plot_sound(time = self.time, amplitude = self.amplitude, name = self.name)

  def plot_sound_filter(self):
    assert self.signal is not None, "The variable signal has not been defined in class"
    plot_sound(time = self.time, amplitude = self.signal, name = self.name.replace(".wav","_filter.wav"))

  def play_sound_original(self):
    play_sound(amplitude = self.amplitude, fs = self.fs)

  def play_sound_filter(self):
    assert self.signal is not None, "The variable signal has not been defined in class"
    play_sound(amplitude = self.signal, fs = self.fs)