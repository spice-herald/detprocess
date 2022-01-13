import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

import pytesdaq.io.hdf5 as h5io


__all__ = [
    'load_traces',
    'load_features',
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
        A numpy array of traces of shape (number of traces, number of
        channels, length of individual trace.
    info_dict : dict
        A dictionary containing information that comes directly from
        the loaded file.

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


def load_features(file_or_folder):
    """
    Function for loading the features from an HDF5 that was created by
    detprocess.

    Parameters
    ----------
    file_or_folder : str
        The full path and filename for the detprocess HDF5 file to
        open, or the full path to the folder containing a number of
        detprocess HDF5 files to open.

    Returns
    -------
    df : Pandas.DataFrame
        A DataFrame that contains all of the extracted features for the
        specified file.

    """
    
    if Path(file_or_folder).is_dir():
        df = pd.concat(
            [
                pd.read_hdf(
                    f,
                    'detprocess_df',
                ) for f in sorted(glob(f"{file_or_folder}/*.hdf5"))
            ],
        )
    else:
        df = pd.read_hdf(
            file_or_folder,
            'detprocess_df',
        )

    return df

