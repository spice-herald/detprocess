import os
import pandas as pd
import numpy as np
from pprint import pprint
import pytesdaq.io as h5io
from glob import glob
from pathlib import Path
import re

__all__ = [
    'RawData'
]

class RawData:
    """
    Class to manage rawdata path
    and metadata
    """

    def __init__(self, raw_path, verbose=True):
        """
        Initialize
        """

        self._raw_path = raw_path
        self._verbose = verbose

        # file types
        self._file_types = ['cont', 'iv', 'didv',
                            'exttrig', 'calib']
        
        # initialize
        self._group_name = None
        self._base_path =  None
        self._facility = None
        self._raw_files = {'rand': {},
                           'cont': {'open': {},
                                    'restricted': {}},
                           'calib': {},
                           'iv': {},
                           'didv': {}}
        
        self._raw_metadata = {'rand': {},
                              'cont': {'open': {},
                                       'restricted': {}},
                              'calib': {},
                              'iv': {},
                              'didv': {}}


        # build map
        self._build_file_map()

        
    @property
    def verbose(self):
        return self._verbose
        
    @property
    def raw_path(self):
        return self._raw_path


    def describe(self):
        """
        Display raw data
        """
        print(f'Raw data group: {self._group_name}')
        print(f'Base path: {self._base_path}')
        print('Number of series:')
        for ftype in self._raw_files:
            if ftype != 'cont':
                if not self._raw_files[ftype]:
                    continue
                print(f' - {ftype} data: '
                      f'{len(self._raw_files[ftype])} series')
            else:
                if self._raw_files[ftype]['restricted']:
                    print(f' - restricted continuous data: '
                          f'{len(self._raw_files[ftype]["restricted"])} series')
                if self._raw_files[ftype]['open']:
                    print(f' - open continuous data: '
                          f'{len(self._raw_files[ftype]["open"])} series')
                    
    def get_group_name(self):
        """
        Get data group name
        """
        return self._group_name
    
    def get_base_path(self):
        """
        Get data group name
        """
        return self._base_path
     
    def get_facility(self):
        """
        Get data group name
        """
        return self._facility
    
    
    def get_continuous_data_map(self, series=None, restricted=False):
        """
        Get continuous data dictionary
        """
        output_map = {}
        data_map = None
        if restricted:
            data_map = self._raw_files['cont']['restricted'].copy()
        else:
            data_map = self._raw_files['cont']['open'].copy()
            
            
        if series is not None:

            if not isinstance(series, list):
                series = [series]
            for series_name in series:
                if series_name not in data_map:
                    raise ValueError(f'No series {series_name} found '
                                     f'in raw data! Check arguments')
                output_map[series_name] = data_map[series_name]
                
        else:
            output_map = data_map

        return output_map


    def _build_file_map(self):
        """
        Build file map, separate between files types
        calib, randoms, iv, continuous restricted or open

        Parameters
        ----------

        None

        Return
        -------
     
        None

        """

        # only directory for now
        if not os.path.isdir(self._raw_path):
            raise ValueError('ERROR: Expecting a raw data directory')

        # instantiate raw data reader
        h5reader = h5io.H5Reader()
        
        # base path, group_name
        self._base_path = str(Path(self._raw_path).parent)
        self._group_name = str(Path(self._raw_path).name)
        if self._verbose:
            print(f'INFO: Building file map for raw data group '
                  f'{self._group_name}')

        self._facility = None
        match = re.search(r'_I(\d+)', self._group_name)
        if match:
            self._facility = int(match.group(1))
        else:
            raise ValueError('ERROR: No facility found from group name!')
            
        # get all files
        file_list =  glob(self._raw_path + '/*.hdf5')
        if not file_list:
            raise ValueError(f'ERROR: No HDF5 files found in {self._raw_path}')
            
        # make unique and sort
        file_list = list(set(file_list))
        file_list.sort()
        
        # initialize
        separated_file_list = {}
        
        # loop 
        file_counter = 0
        file_list_copy =  file_list.copy()
        for file_name in file_list:

            # loop file type
            for ftype in self._file_types:
                ftype_ = f'{ftype}_'

                # initialize
                if ftype not in  separated_file_list:
                    separated_file_list[ftype] = []

                if ftype_ in file_name:
                    separated_file_list[ftype].append(file_name)
                    file_list_copy.remove(file_name)
                    
        # check if any files remaining
        if file_list_copy:
            raise ValueError(f'ERROR: file type unknown! '
                             f'{file_list_copy}')

        # split in function of series
        for ftype, data_list in separated_file_list.items():

            # check if empty
            if not data_list:
                continue

            # loop files
            series_name = None
            for file_name in data_list:

                # get metadata if needed
                if (series_name is None
                    or series_name not in file_name):
                    metadata = h5reader.get_metadata(file_name)
                    series_name = h5io.extract_series_name(metadata['series_num'])
                    
                # double check
                if series_name not in file_name:
                    raise ValueError(f'ERROR: Unrecognized file name '
                                     f'{file_name}!')


                # check if restricted
                is_restricted = False
                if (ftype == 'cont'
                    and 'restricted' in file_name):
                    is_restricted = True
                
                # append file
                if ftype != 'cont':
                
                    if series_name not in self._raw_files[ftype]:
                        self._raw_files[ftype][series_name] = []
                        
                    self._raw_files[ftype][series_name].append(file_name)

                elif 'restricted' in file_name:
                    if series_name not in self._raw_files['cont']['restricted']:
                        self._raw_files['cont']['restricted'][series_name] = []
                    self._raw_files['cont']['restricted'][series_name].append(
                        file_name
                    )
                else:
                    if series_name not in self._raw_files['cont']['open']:
                        self._raw_files['cont']['open'][series_name] = []
                    self._raw_files['cont']['open'][series_name].append(
                        file_name
                    )

        

        
