import numpy as np
from tensorflow import keras
from keras.models import save_model, load_model
from keras import layers, models


def Apply_Pruning_Filter(weights, std_threshold):
    """
    This function perform prune CNN filters and return the remaining filters CNN.

    * Parameters:
        weights (keras weight matrix): CNN filters BEFORE pruning.
        std_threshold (integer): threshold to prune filters.
        
    * Returns:
        filtered_weights (keras weight matrix): CNN filters AFTER pruning.
    """
    
    # Calculate sum of absolute kernel
    weights_abs = np.abs(weights)
    list_weight_abs = np.sum(weights_abs, axis=(0, 1, 2))

    # Calculate pruning threshold
    mean_value = np.mean(list_weight_abs)
    std_value = np.std(list_weight_abs)
    lower_threshold = mean_value - std_threshold * std_value
    upper_threshold = mean_value + std_threshold * std_value
    
    mask = np.logical_and(list_weight_abs >= lower_threshold, list_weight_abs <= upper_threshold)
    filtered_weights = weights[:, :, :, mask]

    return filtered_weights