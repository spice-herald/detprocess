import numpy as np

import pytesdaq.io.hdf5 as h5io


__all__ = [
    'load_traces',
]


def load_traces(filename, channels=None, nevents=0):
    """
    Function for loading the traces from a `pytesdaq` file.

    Parameters
    ----------
    filename : str
        The full filename (including path) to the data to be loaded.
    channels : list of str, list of int, NoneType
        The channels to read from the file.

    Returns
    -------
    traces : ndarray
        A numpy array of traces of shape (number of traces, number of channels, length of individual trace.
    info_dict : dict
        A dictionary containing information that comes directly from the loaded file.

    """

    h5 = h5io.H5Reader()
    traces, info_dict = h5.read_many_events(
        filepath=filename,
        output_format=2,
        detector_chans=channels,
        nevents=nevents,
        include_metadata=True,
        adctovolt=True,
    )

    # convert traces to amps
    detector_settings = h5.get_detector_config(file_name=filename)
    close_loop_norm = [detector_settings[chan]['close_loop_norm'] for chan in channels]
    close_loop_norm_arr = np.asarray(close_loop_norm)
    traces = np.divide(traces, close_loop_norm_arr[:, np.newaxis])
    
    return traces, info_dict
