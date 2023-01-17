import numpy as np
import qetpy as qp


class FeatureExtractors:
    """
    A class that contains user added features. The algoriothm
    name (function name) should not exist already 
    (in process/_features.py). 

    """

    @staticmethod
    def minmax(trace,
               window_min_index=None, window_max_index=None,
               feature_base_name='minmax',
               **kwargs):
        """
        Feature extraction for the minmax pulse value.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.


        window_min_index : int, optional
            The minium index of the window used to find  minimum
            Default: 0

        window_max_index : int, optional
            The maximum index of the window used to find minimum
            Default: end of trace

        feature_base_name : str, optional
            output feature base name 


        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """
        if window_min_index is None:
            window_min_index = 0
            
        if window_max_index is None:
            window_max_index = trace.shape[-1] - 1

            
        minmax_trace = (
            np.amax(trace[window_min_index:window_max_index])
            -np.amin(trace[window_min_index:window_max_index])
        )

        retdict = {
            feature_base_name: minmax_trace,
        }

        return retdict
