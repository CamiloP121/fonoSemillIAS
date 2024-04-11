from tqdm import tqdm
import json
import pandas as pd

from fonoSemillIAS.others.process_result import *

def apply_dasr_silences(DASR, filename, wav):
    try:
        # Initialize progress bar
        progress_bar = tqdm(total=3, desc='Progress', position=0)

        # Step 1: Apply DASR
        tqdm.write("Start step 1: Apply DASR")
        DASR.apply(file = filename)
        progress_bar.update(1)

        # Load result Json
        tqdm.write("Start step 2: Load JSON result")
        name = filename.split("/")[-1].split(".")[0]
        transcription_path_to_file = f"{DASR.work_path}/output/pred_rttms/{name}.json"
        with open(transcription_path_to_file, "r") as archivo:
            datos_json = json.load(archivo)
        progress_bar.update(1)

        # Transform results
        tqdm.write("Start step 3: Transform data")
        st = {
            "word": [],
            "start_time": [],
            "end_time": [],
            }

        for stamps in datos_json["words"]:
            st["word"].append(stamps["word"])

            t1 = round(stamps["start_time"], 3)
            t2 = round(stamps["end_time"], 3)

            st["start_time"].append( t1 )
            st["end_time"].append( t2 )

            df = pd.DataFrame(st)

        st = []

        for i in range(len(df)-1):
            start = df.iloc[i]["start_time"]
            end = df.iloc[i]["end_time"]

            start_b = df.iloc[i+1]["start_time"]
            end_b = df.iloc[i+1]["end_time"]

            if i == 0:
                st.append([0.000, start])
                st.append([end, start_b])
            elif i == len(df) - 2:
                st.append([end, start_b])
                st.append([end_b, round(wav.duration, 3)])
            else:
                st.append([end, start_b])

        data_silence = pd.DataFrame(st, columns = ["start_time","end_time"])


        arr_time_start, arr_time_end, diff = time2sample(array_time_start = data_silence["start_time"].values, 
                                                         array_time_end = data_silence["end_time"].values, 
                                                         fs = wav.fs)

        data_silence["start_sample"] = arr_time_start
        data_silence["end_sample"] = arr_time_end
        data_silence["diff_time"] = diff
        data_silence = data_silence[data_silence["diff_time"] >= 0.2]
        data_silence = data_silence.reset_index(drop = True)
        data_silence["new_silences"] = [True] * len(data_silence)

        pulse_detec = create_pulse(signal_ref = wav.amplitude, 
                                   array_sample_start = data_silence["start_sample"].values, 
                                   array_sample_end = data_silence["end_sample"].values)

        progress_bar.update(1)
        # Close progress bar
        progress_bar.close()

        return data_silence, pulse_detec

    except Exception as e:
        print(e)
        raise Exception("Error in silence detection")
        
