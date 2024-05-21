import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from fonoSemillIAS.others import print_helpers as pp

class WhisperASR():
    def __init__(self, model: str, debug:bool = False) -> None:
        self.model_id = model

        self.debug = debug

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model, self.processor = self.create_model()

        self.pipe = None

    def create_model(self):
        
        if self.debug:
            pp.printy("Start create Whisper Model")
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype= self.torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
            )
            model.to(self.device)

            processor = AutoProcessor.from_pretrained(self.model_id)

            if self.debug:
                pp.printg("Complete download and create model")
            return model, processor
        except Exception as e:
            print(e)
            raise Exception("Error create Whisper model")
        
    def create_pipeline(self,  max_new_tokens: int=128,
                        chunk_length_s: int =30,
                        batch_size: int = 16,
                        return_timestamps:bool =True):
        
        if self.debug:
            pp.printy("Start create Whisper Pipeline")

        try: 
            pipe = pipeline(
                        "automatic-speech-recognition",
                        model= self.model,
                        tokenizer= self.processor.tokenizer,
                        feature_extractor= self.processor.feature_extractor,
                        max_new_tokens=max_new_tokens,
                        chunk_length_s=chunk_length_s,
                        batch_size=batch_size,
                        return_timestamps=return_timestamps,
                        torch_dtype= self.torch_dtype,
                        device= self.device,
                    )
            
            return pipe
        except Exception as e:
            print(e)
            raise Exception("Error create pipeline")
        
    def apply(self, file_path: str, generate_kwargs: dict):

        assert self.pipe is None, "Empty pipeline"

        result = self.pipe(file_path, generate_kwargs= generate_kwargs)
