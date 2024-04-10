
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