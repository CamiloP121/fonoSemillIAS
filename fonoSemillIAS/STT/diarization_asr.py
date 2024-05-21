from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
import os
import wget
from omegaconf import OmegaConf
import shutil
import json

from fonoSemillIAS.others import print_helpers as pp

class DiarizationAsr():
    def __init__(self, work_path: str, conf_infence:str, 
                 asr_model:str, speaker_model:str , vad_model:str ,
                 debug: bool = False) -> None:
        
        assert conf_infence in ["general","meeting","telefonic"], "conf_infence not permited"
        
        if debug: 
            pp.printy("\t Star create Diarization with ASR model")
            print("Confg Inference file:", conf_infence)
            print("ASR Model:", asr_model)
            print("Speaker Model:", asr_model)
            print("Vad Model:", asr_model)
            print("Work path:", work_path)

        self.conf_file =  f"diar_infer_{conf_infence}.yaml"
        self.ASR_model = asr_model
        self.speaker_model = speaker_model
        self.vad_model = vad_model
        self.work_path = work_path
        self.filepath = "tmp_wav"
        self.debug = debug
        self.create_meta()
        self.confg = self.open_config()

        if debug: 
            pp.printy("-----------------------")

    def create_model(self):
        if self.debug: pp.printc("\t** Create model**")
        try:
            decoder = ASRDecoderTimeStamps(self.confg.diarizer)
            model = decoder.set_asr_model()
            return decoder, model
        except Exception as e:
            print(e)
            raise Exception("Error create model")

    def open_config(self):

        if self.debug: 
            pp.printc("\t** Open config**")

        try:
            if not os.path.exists(os.path.join(self.work_path,self.conf_file)):
                CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{self.conf_file}"
                CONFIG = wget.download(CONFIG_URL, self.work_path)
            else:
                CONFIG = os.path.join(self.work_path,self.conf_file)

            config = OmegaConf.load(CONFIG)
            if self.debug: 
                pp.printy("** Modify config ....")
            # Modifying arguments and  in inferece
            ## Workaround for multiprocessing hanging with ipython issue
            config.num_workers = 1
            # config.sample_rate = wav.fs
            ## Output folder: directory to store intermediate files and prediction outputs
            config.diarizer.out_dir = os.path.join(self.work_path,"output")
            ## Config inferece
            config.diarizer.manifest_filepath = os.path.join(self.work_path,'input_manifest.json')
            ## vad config
            #Here we use our inhouse pretrained NeMo VAD
            config.diarizer.vad.model_path = self.vad_model
            config.diarizer.vad.parameters.onset = 0.8
            config.diarizer.vad.parameters.offset = 0.6
            config.diarizer.vad.parameters.pad_offset = -0.05
            config.diarizer.oracle_vad = False # compute VAD provided with model_path to vad config
            ## Speaker emebbeding
            config.diarizer.speaker_embeddings.model_path = self.speaker_model
            ## ASR config
            config.diarizer.asr.model_path = self.ASR_model
            ## Neural diarizer
            # config.diarizer.msdd_model.model_path = pretrained_msdd
            # config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0] # Evaluate with T=0.7 and T=1.0
            ## Clustering
            config.diarizer.clustering.parameters.oracle_num_speakers=False # Unknown number of speakers
            config.diarizer.asr.parameters.asr_based_vad = False

            return config
        except Exception as e:
            print(e)
            raise Exception("Error in load config file")
        
    def create_meta(self):
        if self.debug: 
            pp.printc("\t** Meta file **")

        meta = {
            # 'audio_filepath': files_paths['95 MHG']["C"], # Name file (Original)
            'audio_filepath': self.filepath, # Name file (Filter)
            'offset': 0,
            'duration':None,
            'label': 'infer',
            'text': '-',
            'num_speakers': 2,
            'rttm_filepath': None,
            'uem_filepath' : None
        }

        path_inferece = os.path.join(self.work_path,'input_manifest.json')

        with open(path_inferece,'w') as fp:
            json.dump(meta,fp)
            fp.write('\n')
        
    def set_file(self, filepath:str):
        path_inferece = os.path.join(self.work_path,'input_manifest.json')
        if self.debug:
            pp.printy(f"\t Set {filepath} in input_manifest.json ")
        try:
            with open(path_inferece, 'r') as fp:
                meta = json.load(fp)

            meta['audio_filepath'] = filepath

            with open(path_inferece, 'w') as fp:
                json.dump(meta, fp) 
                fp.write('\n')

            if self.debug:
                pp.printg(f"** Complete")
            
        except Exception as e:
            print(e)
            raise Exception("Error in set new filepath")
        
    def apply(self, file ):
        if self.debug: pp.printc("\t Start ASR process")
        try:
            self.set_file(filepath = file)
            
            decoder_ts, model = self.create_model()

            word_hyp, word_ts_hyp = decoder_ts.run_ASR(model)

            asr_diar_offline = OfflineDiarWithASR(self.confg.diarizer)
            asr_diar_offline.word_ts_anchor_offset = decoder_ts.word_ts_anchor_offset

            diar_hyp, diar_score = asr_diar_offline.run_diarization(self.confg, word_ts_hyp)

            trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)

            return word_hyp, word_ts_hyp, diar_hyp, diar_score
        except Exception as e:
            print(e)
            raise Exception("Error apply Diariztion and ASR")
    