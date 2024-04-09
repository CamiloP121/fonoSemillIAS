from tqdm import tqdm
from fonoSemillIAS.Silence.activateEnergy import *

def apply_energy_silences(signal, fs, window_ms=5, output="intervals"):
    """
    Applies energy-based silences detection algorithm to a given signal.

    Parameters:
    - signal (array): The input signal.
    - fs (float): The sampling frequency of the signal.
    - window_ms (int): Size of the moving window for average energy calculation in milliseconds (default is 5 ms).
    - output (str): Specifies the output format. Can be "intervals" to return only the intervals of silence,
                    or "all" to return additional information including the pulse, envelogram, and all intervals.

    Returns:
    - result (dict): Dictionary containing the output based on the specified format:
                     - If output="intervals", returns {"intervals_silence": intervals}.
                     - If output="all", returns {"intervals_silence": intervals, "pulse": pulse,
                                                  "envelogram": envelogram, "intervals_all": lobe_indices}.
    """
    assert output in ["intervals", "all"], "Parameter 'output' must be 'intervals' or 'all'"

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
        envelogram = envelogram(average_energy_sequence = average_energy)
        # Update progress bar
        progress_bar.update(1)

        # Step 4: Detection of silences
        ## Sub-step 4.1: Zero crossing
        tqdm.write("Start step 4.1: Zero crossing")
        zero_crossing = find_zero_crossings(envelogram = envelogram)
        # Update progress bar
        progress_bar.update(1)

        ## Sub-step 4.2: Identify lobes
        tqdm.write("Start step 4.2: Identify lobes")
        lobe_indices = identify_lobes(envelogram = envelogram, zero_crossings = zero_crossing)
        # Update progress bar
        progress_bar.update(1)

        ## Sub-step 4.3: Transform silence lobes into pulse
        tqdm.write("Start step 4.3: Transform silence lobes into pulse")
        intervals, pulse = silence_pulse(envelogram = envelogram, lobe_indices = lobe_indices)
        # Update progress bar
        progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

        # Return output based on specified format
        if output == "intervals":
            return {"intervals_silence": intervals}
        elif output == "all":
            return {"intervals_silence": intervals,
                    "pulse": pulse,
                    "envelogram": envelogram,
                    "intervals_all": lobe_indices}

    except Exception as e:
        print(e)
        raise Exception("Error in silence detection")

    

