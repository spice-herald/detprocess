import yaml
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import vaex as vx
import importlib
import sys
import os
from glob import glob
from pprint import pprint
from multiprocessing import Pool
from itertools import repeat
from datetime import datetime
import stat
import time
import astropy
from humanfriendly import parse_size
from itertools import groupby
import copy
from detprocess.core.algorithms  import FeatureExtractors
from detprocess.process.processing_data  import ProcessingData
from detprocess.utils import utils
from detprocess.process.config import YamlConfig
from detprocess.core.rawdata import RawData



import pytesdaq.io as h5io
import gc
warnings.filterwarnings('ignore')
import pyarrow as pa

vx.settings.main.thread_count = 1
vx.settings.main.thread_count_io = 1
pa.set_cpu_count(1)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


__all__ = [
    'FeatureProcessing'
]

class FeatureProcessing:
    """
    Class to manage data processing and 
    extract features, dataframe can be saved
    in hdf5 using vaex framework and returned
    as a pandas dataframe (not both)

    Multiple nodes can be used if data splitted in 
    different series

    """

    def __init__(self, raw_data, config_data,
                 series=None,
                 trigger_dataframe_path=None,
                 trigger_series=None,
                 external_file=None,
                 processing_id=None,
                 restricted=False,
                 calib=False,
                 salting_dataframe=None,
                 verbose=True):
        """
        Intialize data processing 
        
        Parameters
        ---------
    
        raw_path : str or list of str 
           data group directory containing data OR full path to HDF5  file 
           (or list of files). Data can be either raw data or 
           vaex dataframes
            
        config_file : str 
           Full path and file name to the YAML settings for the
           processing.

        output_path : str, optional (default=No saved file)
           base directory where output feature file will be saved
     

        external_file : str, optional  (default=no external files)
           The path to a .py file with a FeatureExtractors class 
           to add algorithms developed by users. The name of the static
           functions cannot be same as in _features (no duplicate).  This is 
           meant for rapid development of features without
           needing to rebuild the package.


        series : str or list of str, optional
            series to be process, disregard other data from raw_path

        processing_id : str, optional
            an optional processing name. This is used to be build output subdirectory name 
            and is saved as a feature in DetaFrame so it can then be used later during 
            analysis to make a cut on a specific processing when mutliple 
            datasets/processing are added together.
 
        restricted : boolean
            if True, use restricted data 
            if False (default), exclude restricted data

        calib : boolean
           if True, use only "calib" files
           if False, no calib files included

        salting_dataframe : str or vaex dataframe
           str if path to vaex hdf5 file or directly a dataframe 
        

        verbose : bool, optional
            if True, display info

        Return
        ------
        None
        """

        # feature processing data type
        self._feature_processing_type = ['cont','rand','thresh',
                                         'exttrig']
        # processing id
        self._processing_id = processing_id

        # restricted data
        self._restricted = restricted

        # display
        self._verbose = verbose
        
        # calibration data
        data_type = 'cont'
        self._calib = calib
        if calib:
            self._restricted = False
            data_type = 'calib'
        elif trigger_dataframe_path is None:
            data_type = 'thresh'
        
         
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
            
            rawdata_inst = raw_data

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
        self._available_channels = None
        self._sample_rate = None
        for it, it_config in rawdata_metadata.items():
            self._available_channels = it_config['channel_list']
            self._sample_rate = it_config['overall']['sample_rate']
            break;

    
        
        # config file
        config_dict = {}
        if  isinstance(config_data, str):
            
            if not os.path.isfile(config_data):
                raise ValueError(f'ERROR: argument "{config_data}" '
                                 f'should be a file or YamlConfig object!')

            yaml = YamlConfig(config_data, self._available_channels,
                              sample_rate=self._sample_rate)
            
            config_dict = copy.deepcopy(yaml.get_config('feature'))

        else:
            
            if 'YamlConfig' not in str(type(config_data)):
                raise ValueError(
                    'ERROR: raw data argument should be either '
                    'a directory or YamlConfig object'
                )
            
            config_dict = copy.deepcopy(config_data.get_config('feature'))

        # save info
        self._processing_config = config_dict['channels']
        self._selected_channels = config_dict['channel_list']
        self._traces_config =  config_dict['traces_config']
        self._weights =  config_dict['weights']
        self._filter_file = config_dict['overall']['filter_file']
        
        # check channels to be processed
        if not self._selected_channels:
            raise ValueError('ERROR: No channels to be processed! ' +
                             'Check configuration...')
        
        # Dataframe file list
        trigger_files = None
        trigger_path = None
        trigger_group_name = None
        
        if trigger_dataframe_path is not None:
                   
            trigger_files, trigger_path, trigger_group_name = (
                self._get_file_list(trigger_dataframe_path,
                                    series=trigger_series)
            )

            if not trigger_files:
                raise ValueError(f'No dataframe files were found! '
                                 f'Check configuration...')
            
            self._series_list = list(trigger_files.keys())
               
        # External feature extractors
        self._external_file = None
        if external_file is not None:
            if not os.path.exists(external_file):
                raise ValueError('External feature extractors file "'
                                 + external_file
                                 + '" not found!')
            self._external_file = external_file
            if self._verbose:
                print('INFO: External feature extractor = '
                      + self._external_file)
                
        # get list of available features and check for duplicate
        self._algorithm_list, self._ext_algorithm_list = (
            self._extract_algorithm_list()
        )

        # instantiate processing data
        self._processing_data_inst = ProcessingData(
            self._input_base_path,
            rawdata_files,
            group_name=self._input_group_name,
            trigger_files=trigger_files,
            trigger_group_name=trigger_group_name,
            filter_file=self._filter_file,
            available_channels=self._available_channels,
            salting_dataframe=salting_dataframe,
            verbose=verbose)

        # cleaup filter data tags 
        self._processing_config = (
            self._processing_data_inst.check_filter_data_tags(
                self._processing_config,
                default_tag='default'
            )
        )
        
    def process(self,
                nevents=-1,
                lgc_save=False, save_path=None,
                lgc_output=False, 
                ncores=1,
                memory_limit='1GB'):
        
        """
        Process data 
        
        Parameters
        ---------
       
        nevents : int, optional
           number of events to be processed
           if not all events, requires ncores = 1
           Default: all available (=-1).
        
        lgc_save : bool, optional
           if True, save dataframe in hdf5 files
           Default: False

        lgc_output : bool, optional
           if True, return dataframe 
           Default: False

        save_path : str, optional
           base directory where output group will be saved
           default: same base path as input data
    
        ncores: int, optional
           number of cores that will be used for processing
           default: 1

        memory_limit : str or float, optional
           memory limit per file, example '2GB', '2MB'
           if float, then unit is byte

        """

             
        # check input
        if (ncores>1 and nevents>-1):
            raise ValueError('ERROR: Multi cores processing only allowed when '
                             + 'processing ALL events!')

        if lgc_output and lgc_save:
            raise ValueError('ERROR: Unable to save and output datafame '
                             + 'at the same time. Set either lgc_output '
                             + 'or lgc_save to False.')

        # check number cores allowed
        if ncores>len(self._series_list):
            ncores = len(self._series_list)
            if self._verbose:
                print('INFO: Changing number cores to '
                      + str(ncores) + ' (maximum allowed)')
                
        # create output directory
        output_group_path = None
        output_series_num = None
        
        if lgc_save:
            if  save_path is None:
                save_path  = self._processing_data_inst.get_raw_path()
                if '/raw' in save_path:
                    save_path = save_path.replace('/raw','/processed')
                if 'processed' not in save_path:
                    save_path += '/processed'

            # add group name
            if self._input_group_name not in save_path:
                save_path = save_path + '/' + self._input_group_name

                    
            output_group_path, output_series_num = (
                self._create_output_directory(
                    save_path,
                    self._processing_data_inst.get_facility(),
                    restricted=self._restricted,
                    calib=self._calib
                )
            )
            
            if self._verbose:
                print(f'INFO: Processing output group path: {output_group_path}')

                
        # instantiate OF object
        # (-> calculate FFT template/noise, optimum filter)
        if self._verbose:
            print('INFO: Instantiate OF base for each channel!')

        self._processing_data_inst.instantiate_OF_base(self._processing_config)
            
        # convert memory usage in bytes
        if isinstance(memory_limit, str):
            memory_limit = parse_size(memory_limit)   

        # initialize output
        output_df = None
        
        # case only 1 node used for processing
        if ncores == 1:
            output_df = self._process(1,
                                      self._series_list,
                                      nevents,
                                      lgc_save,
                                      lgc_output,
                                      output_series_num,
                                      output_group_path,
                                      memory_limit)

        else:
            
            # split data
            series_list_split = self._split_series(ncores)
        
            # for multi-core processing, we need to decrease the
            # max memory so it fits in RAM
            memory_limit /= ncores

              
            # lauch pool processing
            if self._verbose:
                print(f'INFO: Processing with be split between {ncores} cores!')

            node_nums = list(range(ncores+1))[1:]
            pool = Pool(processes=ncores)
            output_df_list = pool.starmap(self._process,
                                          zip(node_nums,
                                              series_list_split,
                                              repeat(nevents),
                                              repeat(lgc_save),
                                              repeat(lgc_output),
                                              repeat(output_series_num),
                                              repeat(output_group_path),
                                              repeat(memory_limit)))
            pool.close()
            pool.join()

            # concatenate
            output_df = pd.concat(output_df_list)
            
        # processing done
        if self._verbose:
            print('INFO: Feature processing done!') 
                
        
        if lgc_output:
            return output_df 
        
           
    def _process(self, node_num,
                 series_list, nevents,
                 lgc_save,
                 lgc_output,
                 output_series_num,
                 output_group_path,
                 memory_limit):
        """
        Process data
        
        Parameters
        ---------

        node_num :  int
          node id number, used for display
        
        series_list : str
          list of series name to be processed

        nevents : int, optional
           number of events to be processed
           if not all events, requires ncores = 1
           Default: all available (=-1)
        
        lgc_save : bool, optional
           if True, save dataframe in hdf5 files
           (dataframe not returned)
           if False, return dataframe (memory limit applies
           so not all events may be processed)
           Default: True

        save_path : str, optional
           base directory where output feature file will be saved
           default: same base path as input data
    
        ncores: int, optional
           number of cores that will be used for processing
           default: 1

        memory_limit : float, optionl
           memory limit per file 
           (and/or if return_df=True, max dataframe size)
   
        """
                 
        # node string (for display)
        node_num_str = str()
        if node_num>-1:
            node_num_str = ' Node #' + str(node_num)

        # load salting dataframe
        self._processing_data_inst.load_salting_dataframe()
                    
        # feature extractors
        FE = FeatureExtractors
        FE_ext = None
        if self._external_file is not None:
            FE_ext = self._load_external_extractors(self._external_file)

        # output file name base
        output_base_file = None
        if lgc_save:
            
            file_prefix = 'feature'
            if self._processing_id is not None:
                file_prefix = self._processing_id + '_feature'
            if self._restricted:
                file_prefix += '_restricted'
            elif self._calib:
                file_prefix += '_calib'
                
            series_name = h5io.extract_series_name(
                int(output_series_num+node_num)
            )
                    
            output_base_file = (output_group_path
                                + '/' + file_prefix
                                + '_' + series_name)
                
        # intialize counters
        dump_counter = 1
        event_counter = 0

        # initialize data frame
        feature_df = pd.DataFrame()
        event_rows = []
        
        # loop series
        for series in series_list:

            if self._verbose:
                print('INFO' + node_num_str
                      + ': starting processing series '
                      + series)

            
            # set file list
            self._processing_data_inst.set_series(series)

            # loop events
            do_stop = False
            start_time = time.time()
            while (not do_stop):

                # -----------------------
                # Check number events
                # and memory usage
                # -----------------------

                nevents_limit_reached = (nevents>0 and event_counter>=nevents)
                
                # flag memory limit reached
                memory_limit_reached = False
                if event_rows:
                    memory_usage = sum(sys.getsizeof(row) for row in event_rows)
                    memory_limit_reached =  memory_usage  >= memory_limit
        
                # display
                if self._verbose:
                    
                    if (event_counter%100==0 and event_counter!=0):
                        print('INFO' + node_num_str
                              + ': Local number of events = '
                              + str(event_counter)
                              + ' (memory = ' + str(memory_usage/1e6) + ' MB)')

                        time_running = time.time() - start_time
                        speed = event_counter/time_running
                        print('INFO' + node_num_str
                              + ': Speed = '
                              + f'{speed:.4f}' + ' events per second')
                        
                # -----------------------
                # Read next event
                # -----------------------

                success = self._processing_data_inst.read_next_event(
                    channels=self._selected_channels,
                    traces_config=self._traces_config
                )

                # end of file
                if not success:
                    do_stop = True
            
                # -----------------------
                # Handle stop or
                # memory limit
                # -----------------------
                
                # now let's handle case we need to stop
                # memory/nb events limit reached
                
                if (do_stop
                    or nevents_limit_reached
                    or memory_limit_reached):

                    # convert to pandas
                    feature_df = pd.DataFrame(event_rows)
                    event_rows = []
                    
                    # save file if needed
                    if lgc_save:
                        
                        # build hdf5 file name
                        dump_str = str(dump_counter)
                        dump_str = '_F' + dump_str.zfill(4)
                        file_name =  output_base_file +  dump_str + '.hdf5'
                    
                        # convert to vaex
                        feature_df = feature_df.reset_index(drop=True)

                        # disable thread
                        vx.settings.main.thread_count = 1
                        vx.settings.main.thread_count_io = 1
                        pa.set_cpu_count(1)

                        # convert from pandas
                        feature_vx = vx.from_pandas(
                            feature_df,
                            copy_index=False)

                        # export
                        feature_vx.export_hdf5(file_name,
                                               mode='w')
                        feature_vx.close()
                        
                        # increment dump
                        dump_counter += 1
                        if not do_stop and self._verbose:
                            print('INFO' + node_num_str
                                  + ': Incrementing dump number')

                        # clean up
                        del feature_df,  feature_vx
                        gc.collect()

                        # initialize 
                        feature_df = pd.DataFrame()
                                            
                    # case maximum number of events reached
                    # -> processing done!
                    if nevents_limit_reached:
                        if self._verbose:
                            print('INFO' + node_num_str
                                  + ': Requested nb events reached. '
                                  + 'Stopping processing!')
                            
                        return feature_df

                    # case memory limit reached and not saving file
                    # -> processing done
                    # case memory limit reached
                    # -> processing needs to stop!
                    if lgc_output and memory_limit_reached:
                        raise ValueError(
                            'ERROR: memory limit reached! '
                            + 'Change memory limit or only save hdf5 files '
                            +'(lgc_save=True AND lgc_output=False) '
                        )

                # Now let's stop or increment event counter....
                if do_stop:
                    break
                else:
                    event_counter += 1
                   
                # -----------------------
                # Features calculation
                # -----------------------

                # initialize event features dictionary
                event_features = dict()

                # update signal trace in OF base objects
                # -> calculate FFTs, etc.
                self._processing_data_inst.update_signal_OF(
                    weights=self._weights
                )
              
                # Processing id   
                event_features.update(
                    {'processing_id': self._processing_id}
                )

                # admin data
                event_features.update(
                    self._processing_data_inst.get_event_admin()
                )

                # Detector settings
                for channel in self._processing_config.keys():
                    event_features.update(
                    self._processing_data_inst.get_channel_settings(channel)
                    )

          
                # Pulse features
            
                # loop channels from configuration file (including sum
                # of channels and multiple channels algoritms)  
                for channel, algorithms in self._processing_config.items():

                                        
                    # check channel has any parameters
                    if not isinstance(algorithms, dict):
                        continue

                    # check if feature channel name
                    # is changed by user
                    feature_channel = channel
                    if 'feature_channel' in algorithms.keys():
                        feature_channel = algorithms['feature_channel']

                    # number of samples raw data
                    nb_samples = self._processing_data_inst.get_nb_samples()
                    nb_pretrigger_samples = (
                        self._processing_data_inst.get_nb_pretrigger_samples()
                    )

                    # weights
                    weights_chan = None
                    if channel in self._weights:
                        weights_chan = self._weights[channel]
                                   
                    # loop algorithms to extact features
                    for algorithm, algorithm_params in algorithms.items():

                        # skip if "feature_channel"
                        if not isinstance(algorithm_params, dict):
                            continue
                                                                       
                        # skip if algorithm disable
                        if not algorithm_params['run']:
                            continue
                        
                        # check if derived algorithm
                        base_algorithm = algorithm
                        if 'base_algorithm' in algorithm_params.keys():
                            base_algorithm = algorithm_params['base_algorithm']

                        # number samples from configuration
                        nb_samples_algorithm = (
                            algorithm_params['nb_samples']
                        )
                        nb_pretrigger_samples_algorithm = (
                            algorithm_params['nb_pretrigger_samples']
                        )

                        if nb_samples_algorithm is not None:
                            nb_samples = nb_samples_algorithm
                        if nb_pretrigger_samples_algorithm is not None:
                            nb_pretrigger_samples = (
                                nb_pretrigger_samples_algorithm
                            )
                        
                        # get feature extractor
                        extractor = None
                        if base_algorithm in self._algorithm_list:
                            extractor = getattr(FE, base_algorithm)
                        elif base_algorithm in self._ext_algorithm_list:
                            extractor = getattr(FE_ext, base_algorithm)
                        else:
                            raise ValueError(
                                f'ERROR: Cannot find algorithm '
                                f'"{base_algorithm}" anywhere. '
                                f'Check feature extractor exists!')

                        
                        # extractor arguments (removing run)
                        # add parameter if needed
                        kwargs = {key: value
                                  for (key, value) in algorithm_params.items()
                                  if key != 'run'}

                        # add various parameters that may be needed
                        # by the algoithm
                        kwargs['fs'] = (
                            self._processing_data_inst.get_sample_rate()
                        )

                        if 'nb_samples' not in kwargs:
                            kwargs['nb_samples'] = nb_samples
                            
                        if 'nb_pretrigger_samples' not in kwargs:
                            kwargs['nb_pretrigger_samples'] = (
                                nb_pretrigger_samples
                            )
                  
                        window_min, window_max = (
                            self._get_window_indices(**kwargs)
                        )
                        
                        kwargs['window_min_index'] = window_min
                        kwargs['window_max_index'] = window_max
                            
                        # base feature name = algorithm name
                        kwargs['feature_base_name'] = algorithm
                        
                        # calculate features and get output dictionary 
                        extracted_features = dict()
                        
                        # For OF algorithms, get OB base object
                        key_tuple = (nb_samples_algorithm,
                                     nb_pretrigger_samples_algorithm)
                        
                        OF_base = self._processing_data_inst.get_OF_base(
                            key_tuple, algorithm)
                                                  
                        if OF_base is not None:
                            extracted_features = extractor(channel,
                                                           OF_base,
                                                           **kwargs)
                        else:
                            trace = self._processing_data_inst.get_channel_trace(
                                channel,
                                nb_samples=nb_samples_algorithm,
                                nb_pretrigger_samples=(
                                    nb_pretrigger_samples_algorithm
                                ),
                                weights=weights_chan
                            )
                            
                            extracted_features = extractor(trace, **kwargs)
                                                        
                        # append channel name and save in data frame
                        for feature_base_name in extracted_features:
                            feature_name = f'{feature_base_name}_{feature_channel}'
                            event_features.update(
                                {feature_name: extracted_features[feature_base_name]}
                            )

                            
                # done processing event! 
                # append event dictionary to list
                event_rows.append(event_features)
                                         
        # return features
        print(f'INFO {node_num_str}: Processing done. Returning dataframe!')
        return feature_df
       
        
    def _get_file_list(self, dataframe_path, series=None):
    
        """
        Get file list from path. Return as a dictionary
        with key=series and value=list of files

        Parameters
        ----------

        dataframe_path : str or list of str 
           raw data group directory OR full path to HDF5  file 
           (or list of files). Only a single raw data group 
           allowed 
        
        series : str or list of str, optional
            series to be process, disregard other data from raw_path

        Return
        -------
        
        series_dict : dict 
          list of files for splitted inot series

        base_path :  str
           base path of the raw data

        group_name : str
           group name of raw data

        """

        # convert dataframe_path to list 
        if isinstance(dataframe_path, str):
            dataframe_path = [dataframe_path]
            
            
        # initialize
        file_list = list()
        base_path = None
        group_name = None


        # loop files 
        for a_path in dataframe_path:
                   
            # case path is a directory
            if os.path.isdir(a_path):

                # check a single directory
                if len(dataframe_path) != 1:
                    raise ValueError('Only single directory allowed! ' +
                                     'No combination files and directories')

                if base_path is None:
                    base_path = str(Path(a_path).parent)
                    group_name = str(Path(a_path).name)
                            
                if series is not None:
                    if not isinstance(series, list):
                        series = [series]
                    for it_series in series:
                        file_name_wildcard = '*' + it_series + '_*.hdf5'
                        file_list.extend(glob(a_path + '/' + file_name_wildcard))
                else:
                    file_list = glob(a_path + '/*.hdf5')
                    
            # case file
            elif os.path.isfile(a_path):

                if base_path is None:
                    base_path = str(Path(a_path).parents[1])
                    group_name = str(Path(Path(a_path).parent).name)
                    
                if a_path.find('.hdf5') != -1:
                    if series is not None:
                        if not isinstance(series, list):
                            series = [series]
                        for it_series in series:
                            if a_path.find(it_series) != -1:
                                file_list.append(a_path)
                    else:
                        file_list.append(a_path)

            else:
                raise ValueError(f'File or directory "{a_path}" '
                                 f'does not exist!')
            
        if not file_list:
            msg = ('ERROR: No input dataframe vaex files found. '
                   'Check data path!')
            if series is not None:
                msg = (msg
                       + ' Or check "series" argument (it should be '
                       + '"series" of dataframe files, not raw data)')
            raise ValueError(msg)
            

        # sort
        file_list.sort()

      
        # convert to series dictionary so can be easily split
        # in multiple cores
        
        series_dict = dict()
        series_name = None
        file_counter = 0
        
        for file_name in file_list:

            # append file if series already in dictionary
            if (series_name is not None
                and series_name in file_name
                and series_name in series_dict.keys()):

                if file_name not in series_dict[series_name]:
                    series_dict[series_name].append(file_name)
                    file_counter += 1
                continue
            
            series_name =str(Path(file_name).name)
            sep_start = series_name.find('_I')
            sep_end = series_name.find('_F')
            series_name = series_name[sep_start+1:sep_end]

                
            if series_name not in series_dict.keys():
                series_dict[series_name] = list()

            # append
            if file_name not in series_dict[series_name]:
                series_dict[series_name].append(file_name)
                file_counter += 1
                

        if self._verbose:
            print(f'INFO: Found total of {file_counter} '
                  f'dataframe file (s) with {len(series_dict.keys())} '
                  f' different series number!')

      
        return series_dict, base_path, group_name

    
    def _load_external_extractors(self, external_file):
        """
        Helper function for loading an alternative SingleChannelExtractors
        class.
        
        Parameters
        ----------

        external_file :  str
           external feature file name 

        Return
        ------

        module : object
            FeatureExtractors module

        """
    
        module_name = 'detprocess.process'

        spec = importlib.util.spec_from_file_location(module_name,
                                                      external_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return module.FeatureExtractors



    def _create_output_directory(self, base_path, facility,
                                 restricted=False,
                                 calib=False):
        """
        Create output directory 

        Parameters
        ----------
        
        base_path :  str
           full path to base directory 
        
        facility : int
           id of facility 
    
        Return
        ------
          output_dir : str
            full path to created directory

        """

        now = datetime.now()
        series_day = now.strftime('%Y') +  now.strftime('%m') + now.strftime('%d') 
        series_time = now.strftime('%H') + now.strftime('%M')
        series_name = ('I' + str(facility) +'_D' + series_day + '_T'
                       + series_time + now.strftime('%S'))

        series_num = h5io.extract_series_num(series_name)
        
        # prefix
        prefix = 'feature'
        if self._processing_id is not None:
            prefix = self._processing_id + '_feature'
        if restricted:
            prefix += '_restricted'
        elif calib:
            prefix += '_calib'
            
        output_dir = base_path + '/' + prefix + '_' + series_name
        
        
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir)
                os.chmod(output_dir, stat.S_IRWXG | stat.S_IRWXU | stat.S_IROTH | stat.S_IXOTH)
            except OSError:
                raise ValueError('\nERROR: Unable to create directory "'+ output_dir  + '"!\n')
                
        return output_dir, series_num
        



    
    def _extract_algorithm_list(self):
        """
        Extract list for algorithms, check for duplicates

        Parameters
        ----------
        None


        Return
        ------
        algorithm_list : list
          list of available algorithm from _features.py FeatureExtractors class

        ext_algorithm_list : list
          list of availble algorithms from external files 

        """

        algorithm_list = list()
        ext_algorithm_list = list()

        
        # internal feature extractor
        FE = FeatureExtractors
        for attribute in dir(FE):
            if attribute[0] == '_':
                continue
            algorithm_list.append(attribute)

        # external feature extractor
        if self._external_file is not None:
            FE_ext = self._load_external_extractors(self._external_file)
            for attribute in dir(FE_ext):
                if attribute[0] == '_':
                    continue
                # check for duplicate
                if attribute in algorithm_list:
                    raise ValueError('External algorithm ' + attribute
                                     + ' is a duplicate from internal feature extractor!'
                                     + ' This is nto allowed. You need to change name...')

                else:
                    ext_algorithm_list.append(attribute)

                    
        return algorithm_list, ext_algorithm_list


    def _split_series(self, ncores):
        """
        Split data  between nodes
        following series


        Parameters
        ----------

        ncores : int
          number of cores

        Return
        ------

        output_list : list
           list of dictionaries (length=ncores) containing 
           data
         

        """


        output_list = list()
        
        # split series
        series_split = np.array_split(self._series_list, ncores)

        # remove empty array
        for series_sublist in series_split:
            if series_sublist.size == 0:
                continue
            output_list.append(list(series_sublist))

        return output_list

    

    
    def _get_channel_list(self, file_dict):
        """ 
        Get the list of channels from raw data file
        
        Parameters
        ----------
        file_dict  : dict
           directionary with list of files for each series

        Return
        -------
        channels: list
          List of channels
    
        """

        # let's get list of channels available in file
        # first file
        file_name = str()
        for key, val in file_dict.items():
            file_name = val[0]
            break
        
        # get list from configuration
        h5 = h5io.H5Reader()
        detector_settings = h5.get_detector_config(file_name=file_name)
        return list(detector_settings.keys())


    
    def _get_sample_rate(self, file_dict):
        """ 
        Get the list of channels from raw data file
        
        Parameters
        ----------
        file_dict  : dict
           directionary with list of files for each series

        Return
        -------
        sample_rate : float or NoneType
          return sample rate
    
        """
        fs = None
        
        # let's get list of channels available in file
        # first file
        file_name = str()
        for key, val in file_dict.items():
            file_name = val[0]
            break
        
        # get list from configuration
        h5 = h5io.H5Reader()
        metadata = h5.get_metadata(file_name=file_name,
                                   include_dataset_metadata=False)

        adc_name = metadata['adc_list'][0]
        data_info = metadata['groups'][adc_name]
        if 'sample_rate' in  data_info:
            fs = float(data_info['sample_rate'])

        return fs


    def _get_window_indices(self, nb_samples,
                            nb_pretrigger_samples, fs,
                            window_min_from_start_usec=None,
                            window_min_to_end_usec=None,
                            window_min_from_trig_usec=None,
                            window_max_from_start_usec=None,
                            window_max_to_end_usec=None,
                            window_max_from_trig_usec=None,
                            **kwargs):
        """
        Calculate window index min and max from various types
        of window definition

        Parameters
        ---------

        nb_samples : int
          total number of samples 

        nb_pretrigger_samples : int
           number of pretrigger samples

        window_min_from_start_usec : float, optional
           OF filter window start in micro seconds defined
           from beginning of trace

        window_min_to_end_usec : float, optional
           OF filter window start in micro seconds defined
           as length to end of trace
       
        window_min_from_trig_usec : float, optional
           OF filter window start in micro seconds from
           pre-trigger (can be negative if prior pre-trigger)


        window_max_from_start_usec : float, optional
           OF filter window max in micro seconds defined
           from beginning of trace

        window_max_to_end_usec : float, optional
           OF filter window max in micro seconds defined
           as length to end of trace


        window_max_from_trig_usec : float, optional
           OF filter window end in micro seconds from
           pre-trigger (can be negative if prior pre-trigger)
      
        Return:
        ------

        min_index : int
            trace index window min

        max_index : int 
            trace index window max
        """


        # min window
        min_index = 0
        if  window_min_from_start_usec is not None:
            min_index = int(window_min_from_start_usec*fs*1e-6)
        elif window_min_to_end_usec is not None:
            min_index = (nb_samples
                         - abs(int(window_min_to_end_usec*fs*1e-6))
                         - 1)
        elif window_min_from_trig_usec is not None:
                        min_index = (nb_pretrigger_samples 
                                     + int(window_min_from_trig_usec*fs*1e-6))

        # check
        if min_index<0:
            min_index=0
        elif min_index>nb_samples-1:
            min_index=nb_samples-1
            
        # max index
        max_index = nb_samples -1
        if  window_max_from_start_usec is not None:
            max_index = int(window_max_from_start_usec*fs*1e-6)
        elif window_max_to_end_usec is not None:
            max_index = (nb_samples
                         - abs(int(window_max_to_end_usec*fs*1e-6))
                         - 1)
        elif window_max_from_trig_usec is not None:
            max_index =  (nb_pretrigger_samples 
                          + int(window_max_from_trig_usec*fs*1e-6))

        # check
        if max_index<0:
            max_index=0
        elif max_index>nb_samples-1:
            max_index=nb_samples-1

        
        if max_index<min_index:
            raise ValueError('ERROR window calculation: '
                             + 'max index smaller than min!'
                             + 'Check configuration!')        

        return min_index, max_index
        
    
        
        
    
