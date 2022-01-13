import pandas as pd


__all__ = [
    'save_features',
]


def save_features(df, filename):
    """
    Function for saving features to an HDF5 file.

    Parameters
    ----------
    df : Pandas.DataFrame
        A DataFrame that contains all of the extracted features for a
        file.
    filename : str
        The full path and file name to save the DataFrame as.

    Returns
    -------
    None

    """

    df.to_hdf(
        filename,
        key='detprocess_df',
        mode='w'
    )
