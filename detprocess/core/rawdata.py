import os
import pandas as pd
import numpy as np
from pprint import pprint
import pytesdaq.io as h5io
from glob import glob
from pathlib import Path
import re
import copy
import h5py
import sys

__all__ = [
    'RawData'
]

class RawData:
    """
    Class to manage rawdata path
    and metadata
    """

    def __init__(self, raw_path, series=None, data_type=None,
                 restricted=False, verbose=True):
        """
        Initialize
        """

        self._raw_path = raw_path
        self._series = series
        self._restricted = restricted
        self._verbose = verbose


        # file types
        self._available_data_types = ['cont', 'iv', 'didv',
                                      'exttrig', 'calib']
        if (data_type is not None
            and data_type not in self._available_data_types):
            raise ValueError(f'ERROR: data type {data_type} not '
                             f'recognized! Available types are: '
                             f'{self._available_data_types}')

        self._data_type = data_type
        
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

        # load medatadata
        self._load_metadata()
     
        
    @property
    def verbose(self):
        return self._verbose
        
    @property
    def raw_path(self):
        return self._raw_path

    @property
    def restricted(self):
        return self._restricted
    
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
    
    
    def get_data_files(self, data_type=None, series=None):
        """
        Get data dictionary
        """
        
        # get dictionary
        data_dict = self._get_dictionary(data_type=data_type,
                                         series=series)
        return data_dict
    

    def get_available_channels(self, data_type=None, series=None):
        """
        Get available channels in raw data
        """
        
        available_channels = []
        
        # get config
        metadata = self.get_data_config(data_type=data_type,
                                        series=series)
        # get channels
        for it, it_config in metadata.items():
            available_channels = it_config['channel_list']
            break;

        # return
        return available_channels

    
    def get_sample_rate(self, data_type=None, series=None):
        """
        Get available channels in raw data
        """
        
        sample_rate = np.nan
        
        # get config
        metadata = self.get_data_config(data_type=data_type,
                                        series=series)
        # get channels
        for it, it_config in metadata.items():
            sample_rate = it_config['overall']['sample_rate']
            break
        
        # return
        return sample_rate

    

    def get_data_config(self, data_type=None, series=None):
        """
        Get metadata
        """

        # get dictionary
        config_dict = self._get_dictionary(metadata=True,
                                         data_type=data_type,
                                         series=series)
        
        return config_dict
    
        
    def get_traces(self, series_nums, event_nums,
                   channels=None,
                   adctoamp=True,
                   include_metadata=False):

        """
        Get traces
        """

        # raw files
        raw_files = copy.deepcopy(self._raw_files)
             
        # instanciate
        h5 = h5io.H5Reader()

        # get files
        if not isinstance(series_nums, (list, tuple, np.ndarray)):
            series_nums = [series_nums]

        if not isinstance(event_nums, (list, tuple, np.ndarray)):
            event_nums = [event_nums]


        file_list = []
        for series in series_nums:

            series_name = h5io.extract_series_name(series)

            for data_type, data_series in raw_files.items():

                if (self._data_type is not None
                    and self._data_type != data_type):
                    continue

                if not data_series:
                    continue
                
                data = data_series
                if data_type == 'cont':

                    if not self._restricted:
                        data = data_series['open']
                    else:
                        data = data_series['restricted']

                for itseries, series_file_list in data.items():

                    if not series_file_list:
                        continue
                    
                    if itseries == series_name:
                        file_list.extend(series_file_list)

        # set file
        h5.set_files(file_list)


        # get traces
        traces, admins = h5.read_many_events(
            output_format=2,
            detector_chans=channels,
            event_nums=event_nums,
            series_nums=series_nums,
            adctoamp=adctoamp,
            include_metadata=True)


        # return
        if include_metadata:
            return traces, admins
        else:
            return traces

        

    
    def get_duration(self, series=None,
                     data_type=None,
                     include_nb_events=False):
        """
        Get number of events and duration
        """

        # check arguement
        if data_type is None and self._data_type is None:
            raise ValueError(f'ERROR: data_type argument '
                             f'required!')
        elif data_type is None:
            data_type = self._data_type

        # get dictionary
        data_dict = self._get_dictionary(data_type=data_type,
                                         series=series)
    
        # loop series and find number of events
        nb_events = 0
        nb_samples = None
        for series, file_list in data_dict.items():
            
            for afile in file_list:
    
                with h5py.File(afile, 'r') as f:
                
                    # Access the "adc1" group
                    adc1_group = f['adc1']
                
                    try:
                        nb_events += adc1_group.attrs['nb_events']
                    except KeyError:
                        nb_ds = 0
                        for name, item in adc1_group.items():
                            if isinstance(item, h5py.Dataset):
                                nb_ds += 1
                        nb_events += nb_ds
                
                    if nb_samples is None:
                        nb_samples = adc1_group.attrs['nb_samples']
                        sample_rate = adc1_group.attrs['sample_rate']

        trace_length = nb_samples/sample_rate
        duration = trace_length * nb_events
  
        if include_nb_events:
            return duration, nb_events
        else:
            return duration
    
    
    def _get_dictionary(self, metadata=False, data_type=None, series=None):
        """
        Get data dictionary (file names or metadata)
        """

        # copy data dict
        data_dict = None
        if metadata:
            data_dict = copy.deepcopy(self._raw_metadata)
        else:
            data_dict = copy.deepcopy(self._raw_files)

        # case return everything
        if (series is None and data_type is None
            and self._data_type is None):
            return data_dict
       
        if data_type is None:
            data_type = self._data_type
            
        
        # check data_type
        if data_type is not None:

            if data_type not in self._available_data_types:
                raise ValueError(f'ERROR: data type {data_type} not '
                                 f'recognized! Available types are: '
                                 f'{self._available_data_types}')
            
            data_dict = data_dict[data_type]

            if data_type == 'cont':
                if self._restricted:
                    data_dict = data_dict['restricted']
                else:
                    data_dict = data_dict['open']

        # check series
        data_dict_series = {}
        if series is None:
            
            if data_type is None:
                raise ValueError('ERROR: "data_type" required if '
                                 '"series" is not None')
            
            data_dict_series = data_dict
            
        else:
            
            if not isinstance(series, list):
                series = [series]
                
            for it_series in series:

                series_file_list = []
                
                if data_type is not None:
                    if it_series in data_dict:
                        series_file_list = data_dict[it_series]
                else:
                
                    for ftype in data_dict:
                        
                        if ftype == 'cont':
                            for atype in ['open', 'restricted']:
                                if it_series in data_dict['cont'][atype]:
                                    series_file_list =  (
                                        data_dict['cont'][atype][it_series]
                                    )                                
                        elif it_series in data_dict[ftype]:
                            series_file_list =  (
                                data_dict[ftype][it_series]
                            )

                if series_file_list:
                    data_dict_series[it_series] = series_file_list.copy()
                else:
                    raise ValueError(f'ERROR: series {it_series} not part '
                                     f'of raw data. Check data!')           
                        
                                
        # return
        return data_dict_series
    

        
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


        # initialize
        file_list = []
        
        # build wild card
        file_wildcard_type =  '*'
        if self._data_type is not None:
            file_wildcard_type= f'*{self._data_type}_*'
            
        if self._series is not None:
            
            series_list = self._series
            if not isinstance(self._series, list):
                series_list = [self._series]
            for it_series in series_list:
                file_wildcard = f'{file_wildcard_type}{it_series}*.hdf5'
                file_list.extend(glob(f'{self._raw_path}/{file_wildcard}'))
        else:
            file_list =  glob(f'{self._raw_path}/{file_wildcard_type}.hdf5')

        if not file_list:

            if self._data_type is not None:
                raise ValueError(f'ERROR: No HDF5 files found in {self._raw_path} '
                                 f'with data type {self._data_type}')
            else:
                raise ValueError(f'ERROR: No HDF5 files found in {self._raw_path}')
            
        # make unique and sort
        file_list = list(set(file_list))
        file_list.sort()
        
        # initialize
        separated_file_list = {}

        # data types
        data_types = self._available_data_types
        if self._data_type is not None:
            data_types = [self._data_type]
            
        # double check type and separate based on data type
        file_counter = 0
        file_list_copy =  file_list.copy()
    
        for file_name in file_list:

            base_name = os.path.basename(file_name)
                    
            # loop file type
            for ftype in data_types:
                
                ftype_ = f'{ftype}_'
              
                # initialize
                if ftype not in  separated_file_list:
                    separated_file_list[ftype] = []

                if ftype_ in base_name:
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
                
                # append file
                if ftype != 'cont':
                    if series_name not in self._raw_files[ftype]:
                        self._raw_files[ftype][series_name] = []
                        
                    self._raw_files[ftype][series_name].append(file_name)

                else: 
                    if 'restricted' in file_name:

                        if self._restricted:
                            if series_name not in self._raw_files['cont']['restricted']:
                                self._raw_files['cont']['restricted'][series_name] = []
                            self._raw_files['cont']['restricted'][series_name].append(
                                file_name
                            )
                    else:
                        if not self._restricted:
                            if series_name not in self._raw_files['cont']['open']:
                                self._raw_files['cont']['open'][series_name] = []
                            self._raw_files['cont']['open'][series_name].append(
                                file_name
                            )

        
    def _load_metadata(self):
        """
        get metadata
        """
        h5 = h5io.H5Reader()


        for data_type, data_dict in  self._raw_files.items():

            # check if data avaliable
            if not data_dict or not isinstance(data_dict, dict):
                continue

            # let's loop open/restricted
            for atype in ['open', 'restricted']:

                # get dictionary
                data = data_dict
                if data_type == 'cont':
                    if not data_dict[atype]:
                        continue
                    data = data_dict[atype]
                    
                # loop series
                series_metadata_dict = {}
                
                for series, file_list in  data.items():

                    # check if available
                    if not file_list or not isinstance(file_list, list):
                        continue

                    # get metadata
                    metadata =  copy.deepcopy(h5.get_metadata(file_list[0]))
                    
                    # adc
                    if 'adc_list' not in metadata.keys():
                        raise ValueError(
                            'ERROR: unrecognized file format!'
                        )
                    adc_id = metadata['adc_list'][0]
                    metadata_adc = metadata['groups'][adc_id]
                    metadata['sample_rate'] = float(metadata_adc['sample_rate'])
                    metadata['nb_samples'] = int(metadata_adc['nb_samples'])
                    metadata.pop('groups')
                    
                    # detector channel
                    config = h5.get_detector_config(file_list[0])
                    channel_list = list(config.keys())
                    
                    # save
                    series_metadata_dict[series] = {'channel_list': channel_list,
                                                    'detector_config': config,
                                                    'overall': metadata}

                
                # save
                if data_type == 'cont':
                    self._raw_metadata['cont'][atype] = series_metadata_dict
                else:
                    self._raw_metadata[data_type] = series_metadata_dict
                    break

