from tqdm import tqdm
from fonoSemillIAS.Silence.activateEnergy import *

def apply_energy_silences(signal, fs, window_ms = 5, output = "intervals"):

    assert output in ["intervals", "all"], "Parameter 'output' is only intervals or all"

    try: 

        progress_bar = tqdm(total=6, desc='Progress', position=0)

        # 1. Energy signal 
        tqdm.write("Start step 1: Energy signal")
        energy_wave = shannon_energy(signal=signal)
        # Update progress bar
        progress_bar.update(1)

        # 2. Average energy
        tqdm.write("Start step 2: Average energy")
        window = window_ms * 0.001
        avr_energy = avr_energy(energy_sequence=energy_wave, window_move=int(window * fs))
        # Update progress bar
        progress_bar.update(1)

        # 3. envelogram
        tqdm.write("Start step 3: Envelogram")
        envelogram = envelogram(average_energy_sequence=avr_energy)
        # Update progress bar
        progress_bar.update(1)

        # 4. Detection of silences
        ## 4.1 zero_crossing
        tqdm.write("Start step 4.1: Zero crossing")
        zero_crossing = find_zero_crossings(envelogram=envelogram)
        # Update progress bar
        progress_bar.update(1)

        ## 4.2 identify lobes
        tqdm.write("Start step 4.2: Identify lobes")
        lobe_indices = identify_lobes(envelogram=envelogram, zero_crossings=zero_crossing)
        # Update progress bar
        progress_bar.update(1)

        ## 4.3 Transform silence lobes in pulse
        tqdm.write("Start step 4.3: Transform silence lobes in pulse")
        intervals, pulse = silence_pulse(envelogram=envelogram, lobe_indices=lobe_indices)
        # Update progress bar
        progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

        if output == "intervals":
            return {
                "intervals_silence": intervals
            }
        elif output == "all":
            return {
                "intervals_silence": intervals,
                "pulse": pulse,
                "envelogram": envelogram,
                "intervals_all": lobe_indices
            }
    except Exception as e:
        print(e)
        raise Exception("Error in detec silences")

    

