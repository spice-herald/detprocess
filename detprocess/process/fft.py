import yaml
import warnings
from pathlib import Path
import numpy as np
import vaex as vx
import importlib
import sys
import os
from glob import glob
from multiprocessing import Pool
from itertools import repeat
from datetime import datetime
import stat
import time
import astropy
import pytesdaq.io as h5io
import copy
from humanfriendly import parse_size
from detprocess.process.processing_data  import ProcessingData
from detprocess.process.config import YamlConfig
from detprocess.core.eventbuilder import EventBuilder
from detprocess.utils import utils
from detprocess.core.rawdata import RawData
from distutils.util import strtobool
import pyarrow as pa
warnings.filterwarnings('ignore')

vx.settings.main.thread_count = 1
vx.settings.main.thread_count_io = 1
pa.set_cpu_count(1)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


__all__ = [
    'FFTProcessing'
]


class FFTProcessing:
    """
    Class to perform pre-processing FFTs of an entire dataset.
    Calculates FFT for each "event" in the dataset.
    Dataframe can be saved in hdf5 using vaex framework.
    """

    def __init__(self, raw_data, config_data,
                 series=None,
                 processing_id=None,
                 restricted=False,
                 calib=False,
                 verbose=True):
        """
        Initialize FFT Processing
        
        Parameters
        ----------
        
        raw_data : str or RawData object
           raw data group directory 
           OR RawData object
           Only a single raw data group allowed

                 
        config_data : str  or  YamlConfig object
           Full path and file name to the YAML settings for the
           processing or YamlConfig object

        series : str or list of str, optional
            series to be process, disregard other data from raw_path
    
        processing_id : str, optional
            an optional processing name. This is used to be build 
            output subdirectory name and is saved as a feature in DetaFrame 
            so it can then be used later during 
            analysis to make a cut on a specific processing when mutliple 
            datasets/processing are added together.
        
        restricted : boolean
            if True, use restricted data 
            if False (default), exclude restricted data

        calib : boolean
           if True, use only "calib" files
           if False, no calib files included
    
        verbose : bool, optional
            if True, display info

            
        Return
        ------
        None
        """

        # display
        self._verbose = verbose

        # processing id
        self._processing_id = processing_id

        # restricted
        self._restricted = restricted

        # calibration data
        data_type = 'cont'
        self._calib = calib
        if calib:
            self._restricted = False
            data_type = 'calib'
        
        # extract input file list
        rawdata_inst = None
        if isinstance(raw_data, str):
      
            rawdata_inst = RawData(raw_data,
                                   data_type=data_type,
                                   series=series,
                                   restricted=self._restricted)
        else:

            if 'RawData' not in str(type(raw_data)):
                raise ValueError(
                    'ERROR: raw data argument should be either '
                    'a directory or RawData object'
                )
            
            rawdata_inst = raw_data # raw_data passed is a RawData object, so just use that

            if rawdata_inst.restricted != self._restricted:
                raise ValueError(f'ERROR: Unable to use RawData object.'
                                 f'It needs requirement restricted = '
                                 f'{self._restricted}!')
            
            # get file list
            rawdata_files = copy.deepcopy(
                rawdata_inst.get_data_files(data_type=data_type,
                                            series=series)
            )

            if not rawdata_files:
                raise ValueError('No files were found! Check configuration...')
            
            # get metadata list
            rawdata_metadata = rawdata_inst.get_data_config(data_type=data_type,
                                                            series=series)
            
            self._series_list = list(rawdata_files.keys())
            self._input_base_path = rawdata_inst.get_base_path()
            self._input_group_name = rawdata_inst.get_group_name()

            # available channels
            available_channels = None
            for it, it_config in rawdata_metadata.items():
                available_channels = it_config['channel_list']
                break

            # config file
            config_dict = {}
            if isinstance(config_data, str):

                if not os.path.isfile(config_data):
                    raise ValueError(f'ERROR: argument "{config_data}" '
                                 f'should be a file or YamlConfig object!')
                
                yaml = YamlConfig(config_data, available_channels=available_channels)
                config_dict = yaml.get_config('fft')

            else:

                if 'YamlConfig' not in str(type(config_data)):
                    raise ValueError(
                        'ERROR: raw data argument should be either '
                        'a directory or YamlConfig object'
                    )
                
                config_dict = config_data.get_config('fft')

            self._trigger_config = copy.deepcopy(config_dict['channels'])
            self._evtbuilder_config = copy.deepcopy(config_dict['overall'])
            self._trigger_channels = copy.deepcopy(config_dict['channel_list'])

            if not 'fiter_file' in config_dict['overall']:
                raise ValueError('ERROR: Filter file missing in yaml file!')
            
            # check channels to be processed
            if not self._trigger_channels:
                raise ValueError('No trigger channels to be processed! ' +
                                 'Check configuration...')
            
            # initialize output path
            self._output_group_path = None

            # instantiate processing data
            self._processing_data_inst = ProcessingData(
                self._input_base_path,
                rawdata_files,
                group_name=self._input_group_name,
                filger_file=config_dict['overall']['filter_file'],
                available_channels=available_channels,
                verbose=verbose
            )
            
    def get_output_path(self):
        """
        Getter for the output group path
        """
        return self._output_group_path
    
    def process(self,
                lgc_save=False,
                lgc_output=False,
                save_path=None,
                output_group_name=None,
                ncores=1,
                memory_limit='1GB'):
        """
        Process data (compute FFTs)

        Parameters
        ----------

        lgc_save : bool, optional
           if True, save dataframe in hdf5 files
           (dataframe not returned)
           if False, return dataframe (memory limit applies
           so not all events may be processed)
           Default: True

        
        lgc_output : bool, optional
            if True, returns dataframe after processing

        save_path : str, optional
           base directory where output group will be saved
           default: same base path as input data

        output_group_name : 

        ncores : int, optional
            nubmer of cores that will be used for processing
            default: 1

        memory_limit : str or float, optional
            memory limit per file, example '2GB', '2MB'
            if float, then unit is byte
        """

        # check input
        if (ncores>1 and ntriggers>-1):
            raise ValueError('ERROR: Multi cores processing only allowed when '
                             + 'processing ALL events!')
        
        # check number cores allowed
        if ncores>len(self._series_list):
            ncores = len(self._series_list)
            if self._verbose:
                print('INFO: Changing number of cores to '
                      + str(ncores) + ' (number of series)!')
        
        # create output directory
        output_group_path = None
        output_series_num = None

        if lgc_save:
            if save_path is None:
                save_path = self._input_base_path + '/processed'
                if '/raw/processed' in save_path:
                    save_path = save_path.replace('/raw/processed', '/processed')

            # add group name
            if self._input_group_name not in save_path:
                save_path = save_path + '/' + self._input_group_name

            output_group_path, output_series_num = (
                self.create_output_directory(
                    save_path,
                    self._processing_data_inst.get_facility(),
                    output_group_name=output_group_name,
                    restricted=self._restricted,
                    calib=self._calib
                )
            )
            if self._verbose:
                print(f'INFO: Processing output group path: {output_group_path}')
            
            # keep
            self._output_group_path = output_group_path
            self._output_series_num = output_series_num

            # convert memory usage in bytes
            if isinstance(memory_limit, str):
                memory_limit = parse_size(memory_limit)

            # initialize output
            output_df = None

            # case only 1 node used for processing
            if ncores == 1:
                output_df = self._process(1,
                                          self._series_list,
                                          lgc_save,
                                          lgc_output,
                                          output_series_num,
                                          output_group_path,
                                          memory_limit)
            else:
                ### Enable multiprocessing

                # disable vaex multi-threading
                vx.settings.main.thread_count = 1
                vx.settings.main.thread_count_io = 1
                pa.set_cpu_count(1)

                # split data
                series_list_split = self._split_series(ncores)

                # For multi-core processing, we need to decrease the
                # max memory limit so it fits in RAM
                memory_lmiit /= ncores
                
                # launch pool processing
                if self._verbose:
                    print(f'INFO: Processsing will be split between {ncores} cores!')

                node_nums = list(range(ncores+1))[1:]
                pool = Pool(processes=ncores)
                output_df_list = pool.starmap(self._process,
                                              zip(
                                                  node_nums,
                                                  series_list_split,
                                                  repeat(lgc_save),
                                                  repeat(lgc_output),
                                                  repeat(output_series_num),
                                                  repeat(output_group_path),
                                                  repeat(memory_limit)
                ))
                pool.close()
                pool.join()

                # concatenate output
                if lgc_output:
                    df_list = list()
                    for df in output_df_list:
                        if df is not None:
                            df_list.append(df)
                    if df_list:
                        output_df = vx.concat(df_list)

        # processing done
        if self._verbose:
            print('INFO: FFT processing done!')

        if lgc_output:
            return output_df

    def _process(self, node_num, series_list,
                 lgc_save, lgc_output, output_series_num,
                 output_group_path, memory_limit):
        """
        Process data
        (actually perform calculations)

        Parameters
        ----------

        node_num : int
            node id number, used for display in multi-core processing
        
        series_list : str
            list of series names to be processed

        lgc_save : bool
            if True, save dataframe in hdf5 files
            (dataframe not returned)
            if False, return dataframe (memory limit applies 
            so not all events may be processed)
            Default: True

        lgc_output : bool

        output_series_num : int

        output_group_path : str
            base directory where output feature file will be saved
            Default: same base path as input data

        memory_limit : float
            memory limit per file in bytes
            (and/or if return_df=True, max dataframe size)
        """

        return