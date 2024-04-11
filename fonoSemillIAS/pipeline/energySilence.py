from tqdm import tqdm
import pandas as pd
from fonoSemillIAS.Silence.activateEnergy import *
from fonoSemillIAS.others.process_result import *

def apply_energy_silences(signal, fs, window_ms=5,):
    """
    Applies energy-based silences detection algorithm to a given signal.

    Parameters:
    - signal (array): The input signal.
    - fs (float): The sampling frequency of the signal.
    - window_ms (int): Size of the moving window for average energy calculation in milliseconds (default is 5 ms).
    - output (str): Specifies the output format. Can be "intervals" to return only the intervals of silence,
                    or "all" to return additional information including the pulse, envelogram, and all intervals.

    Returns:
    - result (dict): Dictionary containing the output based on the specified format   
    """

    try:
        # Initialize progress bar
        progress_bar = tqdm(total=6, desc='Progress', position=0)

        # Step 1: Energy signal
        tqdm.write("Start step 1: Energy signal")
        energy_wave = shannon_energy(signal = signal)
        # Update progress bar
        progress_bar.update(1)

        # Step 2: Average energy
        tqdm.write("Start step 2: Average energy")
        window = window_ms * 0.001
        average_energy = avr_energy(energy_sequence = energy_wave, window_move = int(window * fs))
        # Update progress bar
        progress_bar.update(1)

        # Step 3: Envelogram
        tqdm.write("Start step 3: Envelogram")
        envelogram_wave = envelogram(average_energy_sequence = average_energy)
        # Update progress bar
        progress_bar.update(1)

        # Step 4: Detection of silences
        ## Sub-step 4.1: Zero crossing
        tqdm.write("Start step 4.1: Zero crossing")
        zero_crossing = find_zero_crossings(envelogram = envelogram_wave)
        # Update progress bar
        progress_bar.update(1)

        ## Sub-step 4.2: Identify lobes
        tqdm.write("Start step 4.2: Identify lobes")
        lobe_indices = identify_lobes(envelogram = envelogram_wave, zero_crossings = zero_crossing)
        # Update progress bar
        progress_bar.update(1)

        ## Sub-step 4.3: Transform silence lobes into pulse
        tqdm.write("Start step 4.3: Transform silence lobes into pulse")
        intervals = identify_silence(envelogram = envelogram_wave, lobe_indices = lobe_indices)
        
        data_silence = pd.DataFrame(intervals, columns = ["start_sample","end_sample"])

        arr_time_start, arr_time_end, diff = sample2time(array_sample_start = data_silence["start_sample"].values, 
                                                         array_sample_end = data_silence["end_sample"].values, 
                                                         fs = fs)

        data_silence["start_time"] = arr_time_start
        data_silence["end_time"] = arr_time_end
        data_silence["diff_time"] = diff
        data_silence = data_silence[data_silence["diff_time"] >= 0.2]
        data_silence = data_silence.reset_index(drop = True)
        data_silence["new_silences"] = [True] * len(data_silence)
        
        pulse_detec = create_pulse(signal_ref = signal, 
                                   array_sample_start = data_silence["start_sample"].values, 
                                   array_sample_end = data_silence["end_sample"].values)
        # Update progress bar
        progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

        return data_silence, pulse_detec

    except Exception as e:
        print(e)
        raise Exception("Error in silence detection")

    

