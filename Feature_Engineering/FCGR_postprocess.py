#%% Documentation
"""
This script takes the npy arrays created by "CGR2FCGR.py" and performs postprocessing on them (such as normalization, etc.).
"""
#%% Imports
import numpy as np
#%% Functions
def FCGR_normalization(FCGR_array, normalization_method = 'max'):
    # FCGR_normalization: "by_sum" or "by_max" or "by_sum_and_max"
    # by_sum: divide by the sum of all bins to get the frequency (eliminating the effect the length of the sequence)
    # by_max: divide by the maximum value of the FCGR array, stretching the values between 0 and 1
    if "sum" in normalization_method:
        FCGR_array_normed = FCGR_array / FCGR_array.sum()
        if "max" in normalization_method:
            FCGR_array_normed = FCGR_array_normed / np.max(FCGR_array_normed)
    elif "max" in normalization_method:
        FCGR_array_normed = FCGR_array / np.max(FCGR_array)
    elif normalization_method == "l2_norm":
        FCGR_array_normed = np.linalg.norm(FCGR_array, order = None, axis = None)
    else:
        FCGR_array_normed = FCGR_array

    return FCGR_array_normed