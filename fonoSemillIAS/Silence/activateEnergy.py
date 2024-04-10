import numpy as np

def shannon_energy(signal):
    """
    Calculates the Shannon energy of an audio signal.

    Parameters:
    - signal (array): The audio signal.

    Returns:
    - energy (float): The Shannon energy of the signal.
    """
    energy = -(signal**2) * np.log10(signal**2)
    return np.nan_to_num(energy)

def avr_energy(energy_sequence, window_move):
    """
    Calculates the continuous average energy for every 50 ms along the original signal.

    Parameters:
    - energy_sequence (array): The energy sequence of the original signal.
    - window_move (int): Size of the moving window in ms.
    - overlap (float): Overlap percentage between consecutive windows.

    Returns:
    - average_energy (array): The calculated continuous average energy.
    """
    # Calculate the number of samples that overlap between consecutive windows
    window_move = window_move // 2
    # Initialize the index to start the window
    start_index = 0
    average_energy, t = [], []
    # Loop through the signal with the specified overlap
    while start_index < len(energy_sequence):
        # Take the energy over the window
        window_energy = energy_sequence[start_index - window_move : start_index + window_move]
        # Calculate the average energy over the window
        average_energy.append(np.mean(window_energy))
        # Move the window with the overlap
        start_index += 1
    return np.nan_to_num(np.array(average_energy))

def envelogram(average_energy_sequence):
    """
    Calculates the normalized envelogram.

    Parameters:
    - average_energy_sequence (array): The sequence of continuous average energy.

    Returns:
    - envelogram (array): The calculated normalized envelogram.
    """
    mean = np.mean(average_energy_sequence)
    standard_deviation = np.std(average_energy_sequence)
    return (average_energy_sequence - mean) / standard_deviation

def find_zero_crossings(envelogram):
    """
    Find the indices of zero crossings in the envelogram.

    Parameters:
    - envelogram (array): The envelogram of the signal.

    Returns:
    - zero_crossings (array): Indices of zero crossings.
    """
    zero_crossings = np.where(np.diff(np.sign(envelogram)))[0]
    return zero_crossings

def identify_lobes(envelogram, zero_crossings):
    """
    Identify the lobes in the envelogram using zero crossings.

    Parameters:
    - envelogram (array): The envelogram of the signal.
    - zero_crossings (array): Indices of zero crossings.

    Returns:
    - lobe_indices (list of tuples): Indices of the lobes.
    """
    lobe_indices = []
    start_idx = 0
    for zero_crossing in zero_crossings:
        lobe_indices.append((start_idx, zero_crossing))
        start_idx = zero_crossing
    # Add the last lobe
    lobe_indices.append((start_idx, len(envelogram) - 1))
    return lobe_indices

def identify_silence(envelogram, lobe_indices):
    """
    Identify and mark regions of silence in an envelogram.

    Parameters:
    - envelogram (array): Envelogram of the signal.
    - lobe_indices (list of tuples): Indices of the lobes in the envelogram.

    Returns:
    - intervals (list of tuples): List of tuples (i, j) where the mean value of the lobe is negative.
    - pulse (array): Array marking the regions of silence with 1.
    """
    intervals = []

    for start_index, final_index in lobe_indices:
        arr = envelogram[start_index:final_index]
        if np.mean(arr) < 0:
            intervals.append((start_index, final_index))

    return intervals