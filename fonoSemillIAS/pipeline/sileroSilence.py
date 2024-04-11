import pandas as pd
from tqdm import tqdm

from fonoSemillIAS.others.process_result import *

def apply_silero_silences(SVAD, filename, wav):
    try:
        # Initialize progress bar
        progress_bar = tqdm(total=2, desc='Progress', position=0)

        tqdm.write("Start step 1: Apply SVAD")
        speech_timestamps = SVAD.apply(file_name = filename)
        progress_bar.update(1)

        # Transform results
        tqdm.write("Start step 2: Transform data")
        st = []
        for value in speech_timestamps:
            start = int( value["start"] * (wav.fs / SVAD.fs))
            end = int( value["end"] * (wav.fs / SVAD.fs))

            st.append([start, end])

        df = pd.DataFrame(st, columns = ["start_sample","end_sample"])

        st = []

        for i in range(len(df)-1):
            start = int(df.iloc[i]["start_sample"])
            end = int(df.iloc[i]["end_sample"])

            start_b = int(df.iloc[i+1]["start_sample"])
            end_b = int(df.iloc[i+1]["end_sample"])

            if i == 0:
                st.append([0, start])
                st.append([end, start_b])
            elif i == len(df) - 2:
                st.append([end, start_b])
                st.append([end_b, len(wav.amplitude)])
            else:
                st.append([end, start_b])

        data_silence = pd.DataFrame(st, columns = ["start_sample","end_sample"])
        arr_time_start, arr_time_end, diff = sample2time(array_sample_start = data_silence["start_sample"].values, 
                                                         array_sample_end = data_silence["end_sample"].values, 
                                                         fs = wav.fs)

        data_silence["start_time"] = arr_time_start
        data_silence["end_time"] = arr_time_end
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