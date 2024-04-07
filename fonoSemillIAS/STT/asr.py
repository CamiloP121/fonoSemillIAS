import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf, open_dict

from utils import print_helpers as pp

class NeMo_ASR():
    def __init__(self, name_model:str, debug: bool = False) -> None:

        if debug: 
            pp.printy("\t Star create ASR model")
            print("Model:", name_model)

        self.name_model = name_model
        self.model = self.create_model()
        self.debug = debug

        if debug: 
            pp.printy("-----------------------")


    def create_model(self):
        try:
            asr_model = nemo_asr.models.ASRModel.from_pretrained(self.name_model)
        except Exception as e:
            print(e)
            raise Exception("Error in create ASR model")

        try:
            decoding_cfg = asr_model.cfg.decoding
            with open_dict(decoding_cfg):
                decoding_cfg.preserve_alignments = True
                decoding_cfg.compute_timestamps = True
                asr_model.change_decoding_strategy(decoding_cfg)
        except Exception as e:
            print(e)
            raise Exception("Error in modify ASR model")
        
        return asr_model

    def apply(self, file:str):
        if self.debug: pp.printc("\t Start ASR process")
        # specify flag `return_hypotheses=True``
        try:
            hypotheses = self.model.transcribe([ file ], return_hypotheses=True)
        except Exception as e:
            print(e)
            raise Exception("Error apply ASR and Hypotheses timestamps")

        # if hypotheses form a tuple (from RNNT), extract just "best" hypotheses
        try:
            if type(hypotheses) == tuple and len(hypotheses) == 2:
                hypotheses = hypotheses[0]
            timestamp_dict = hypotheses[0].timestep # extract timesteps from hypothesis of first (and only) audio file

            # For a FastConformer model, you can display the word timestamps as follows:
            # 40ms is duration of a timestep at output of the Conformer
            time_stride = 4 * self.model.cfg.preprocessor.window_stride

            word_timestamps = timestamp_dict['word']

            text = ""
            vec_timestamps = []
            for stamp in word_timestamps:
                stamp['start_offset'] = stamp['start_offset'] * time_stride
                stamp['end_offset'] = stamp['end_offset'] * time_stride
                word = stamp['char'] if 'char' in stamp else stamp['word']

                # print(f"Time : {stamp['start_offset']:0.2f} - {stamp['end_offset']:0.2f} - {word}")
                text += " " + word
                vec_timestamps.append(stamp)

        except Exception as e:
            print(e)
            raise Exception("Error transform transcript")
        

        if self.debug:
            print("\tFinish ASR")
            print("- All transcript: ", text)
            print("- Num of timestamps:", len(vec_timestamps))
            pp.printc("----------------------------")

        return text, vec_timestamps