import numpy as np

def sample2time(array_sample_start, array_sample_end, fs):
    """
    Convert array of sample indices to array of corresponding time points.
    
    Args:
    array_sample_start (list): List containing starting sample indices.
    array_sample_end (list): List containing ending sample indices.
    fs (int or float): Sampling frequency.
    
    Returns:
    tuple: A tuple containing three lists: 
        - array_time_start: List of corresponding starting time points.
        - array_time_end: List of corresponding ending time points.
        - diff: List of differences between ending and starting time points.
    """
    array_time_start, array_time_end, diff = [], [], []
    for start, end in zip(array_sample_start, array_sample_end):
        
        t1 = round(start / fs, 3)
        t2 = round(end / fs, 3)

        array_time_start.append( t1 )
        array_time_end.append( t2 )
        diff.append( t2 - t1 )

    return array_time_start, array_time_end, diff


def time2sample(array_time_start, array_time_end, fs):
    """
    Convert array of time points to array of corresponding sample indices.
    
    Args:
    array_time_start (list): List containing starting time points.
    array_time_end (list): List containing ending time points.
    fs (int or float): Sampling frequency.
    
    Returns:
    tuple: A tuple containing three lists: 
        - array_sample_start: List of corresponding starting sample indices.
        - array_sample_end: List of corresponding ending sample indices.
        - diff: List of differences between ending and starting sample indices.
    """
    array_sample_start, array_sample_end, diff = [], [], []
    for start, end in zip(array_time_start, array_time_end):
        
        s1 = int(start * fs)
        s2 = int(end * fs)

        array_sample_start.append( s1 )
        array_sample_end.append( s2 )
        diff.append( round(end - start,3) )

    return array_sample_start, array_sample_end, diff

def create_pulse(signal_ref, array_sample_start, array_sample_end):
    """
    Create a pulse signal based on reference signal and specified sample ranges.
    
    Args:
    signal_ref (numpy.ndarray): Reference signal array.
    array_sample_start (list): List of starting sample indices.
    array_sample_end (list): List of ending sample indices.
    
    Returns:
    numpy.ndarray: Pulse signal with 1s in specified sample ranges and 0s elsewhere.
    """
    pulse = np.zeros_like(signal_ref)
    for start, end in zip(array_sample_start, array_sample_end):
        pulse[start:end] = 1

    return pulse