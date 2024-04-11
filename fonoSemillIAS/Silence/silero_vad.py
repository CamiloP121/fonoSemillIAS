import torch
torch.set_num_threads(1)
from pathlib import Path

class SileroVAD():
    def __init__(self, model:str = "silero_vad", fs:int = 1600) -> None:
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model= model,
                              force_reload=True)
        self.fs = fs

        self.get_speech_timestamps = utils[0]
        self.read_audio = utils[2]

    def apply(self, file_name):
        wav_torch = self.read_audio(file_name, sampling_rate=self.fs)
        speech_timestamps =  self.get_speech_timestamps(wav_torch, self.model, sampling_rate = self.fs)
        return speech_timestamps

