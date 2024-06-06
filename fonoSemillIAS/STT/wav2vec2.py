import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from fonoSemillIAS.others import print_helpers as pp

class Wav2Vec2ASR():
    def __init__(self, model: str, fs:int = 16000, debug:bool = False) -> None:
        self.model_id = model
        self.fs = fs
        self.debug = debug

        self.model, self.processor = self.create_model()

    def create_model(self):
        
        if self.debug:
            pp.printy("Start create Whisper Model")
        try:
            model = Wav2Vec2ForCTC.from_pretrained(self.model_id)

            processor = Wav2Vec2Processor.from_pretrained(self.model_id)

            if self.debug:
                pp.printg("Complete download and create model")
            return model, processor
        except Exception as e:
            print(e)
            raise Exception("Error create Whisper model")
        
    def apply(self, file_path: str, generate_kwargs: dict):

        if self.debug: pp.printc("\t Start ASR process")

        waveform, _ = librosa.load(file_path, sr=self.fs)
        inputs = self.processor(waveform, sampling_rate = self.fs, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)

        text = transcription[0]
        if self.debug: 
            print("\tFinish ASR")
            print("- All transcript: ", text)

        return text