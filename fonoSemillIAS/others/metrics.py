
import numpy as np

def dice_coefficient(y_true, y_pred):
    """
    Compute the Dice coefficient between two binary arrays.
    
    Args:
    y_true (numpy.ndarray): Ground truth binary array.
    y_pred (numpy.ndarray): Predicted binary array.
    
    Returns:
    float: Dice coefficient value.
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def deltas_lobes(df_tag_silences, data_detec_silence):
    delta_start = []
    for i in range(len(df_tag_silences)):
        tg = df_tag_silences.iloc[i]

        for j in range(len(data_detec_silence)):
            dect = data_detec_silence.iloc[j]
            if tg.start_sample <= dect.start_sample and dect.start_sample <= tg.end_sample:
                delta_start.append(  dect.start_time - tg.start_time )
            elif dect.start_sample <= tg.start_sample and tg.start_sample <= dect.end_sample:
                delta_start.append(  tg.start_time - dect.start_time )
            elif dect.start_sample > tg.end_sample: 
                break

        if len(delta_start) < i + 1: 
            delta_start.append("inf")

    delta_end = []
    for i in range(len(df_tag_silences)):
        tg = df_tag_silences.iloc[i]

        for j in range(len(data_detec_silence)):
            dect = data_detec_silence.iloc[j]
            if tg.start_sample <= dect.end_sample and dect.end_sample <= tg.end_sample:
                delta_end.append(  tg.end_time - dect.end_time )
            elif dect.start_sample <= tg.end_sample and tg.end_sample <= dect.end_sample:
                delta_end.append(  dect.end_time - tg.end_time  )
            elif dect.start_sample > tg.end_sample: 
                break

        if len(delta_end) < i + 1: 
            delta_end.append("inf")

    return delta_start, delta_end

    