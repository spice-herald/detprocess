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

from detprocess.process._features  import FeatureExtractors
from detprocess.process._processing_data  import ProcessingData
import pytesdaq.io as h5io



__all__ = [
    'Processing'
]




class Processing:
    """
    Class to manage data processing and 
    extract features, dataframe can be saved
    in hdf5 using vaex framework and returned
    as a pandas dataframe (not both)

    Multiple nodes can be used if data splitted in 
    different series

    """

    def __init__(self, raw_path, config_file,
                 external_file=None, series=None,
                 processing_id=None,
                 verbose=True):
        """
        Intialize data processing 
        
        Parameters
        ---------
    
        raw_path : str or list of str 
           raw data group directory OR full path to HDF5  file 
           (or list of files). Only a single raw data group 
           allowed 
            
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

        verbose : bool, optional
            if True, display info



        Return
        ------
        None
        """

        # display
        self._verbose = verbose

        # processing ID 
        self._processing_id = processing_id

        # extract input file list
        input_file_dict, input_base_path, group_name = (
            self._get_file_list(raw_path, series=series)
        )

        if not input_file_dict:
            raise ValueError('No files were found! Check configuration...')

        self._input_base_path = input_base_path
        self._series_list = list(input_file_dict.keys())
                  
        # processing configuration
        if not os.path.exists(config_file):
            raise ValueError('Configuration file "' + config_file
                             + '" not found!')
        available_channels = self._get_channel_list(input_file_dict)
        config, filter_file, selected_channels = (
            self._read_config(config_file, available_channels)
        )
        self._processing_config = config
        self._selected_channels = selected_channels
        
        # check channels to be processed
        if not self._selected_channels:
            raise ValueError('No channels to be processed! ' +
                             'Check configuration...')
        
        
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
        self._algorithm_list, self._ext_algorithm_list = self._extract_algorithm_list()



        # instantiate processing data 
        self._processing_data = ProcessingData(input_file_dict,
                                               group_name=group_name,
                                               filter_file=filter_file,
                                               verbose=verbose)
                                

        
    def process(self, nevents=-1,
                save_hdf5=True, output_path=None,
                ncores=1,
                memory_limit_MB=2000):
        
        """
        Process data 
        
        Parameters
        ---------

        nevents : int, optional
           number of events to be processed
           if not all events, requires ncores = 1
           Default: all available (=-1)
            

        save_hdf5 : bool, optional
           if True, save dataframe in hdf5 files
           (dataframe not returned)
           if False, return dataframe (memory limit applies
           so not all events may be processed)
           Default: True

        output_path : str, optional
           base directory where output group will be saved
           default: same base path as input data
    
        ncores: int, optional
           number of cores that will be used for processing
           default: 1

        memory_limit_MB : float, optionl
           memory limit per file  in MB
           (and/or if saved_hdf5=False, max dataframe size)
        

        """


        # check input
        if (ncores>1 and nevents>-1):
            raise ValueError('ERROR: Multi cores processing only allowed when '
                             + 'processing ALL events!')
        
        # check number cores allowed
        if ncores>len(self._series_list):
            ncores = len(self._series_list)
            if self._verbose:
                print('INFO: Changing number cores to '
                      + str(ncores) + ' (maximum allowed)')

                
        # create output directory
        output_group_path = None
        
        if save_hdf5:
            if  output_path is None:
                output_path  = self._input_base_path
            
            output_group_path = self._create_output_directory(
                output_path,
                self._processing_data.get_facility()
            )
            
            if self._verbose:
                print(f'INFO: Processing output group path: {output_group_path}')

                
        # instantiate OF object
        # (-> calculate FFT template/noise, optimum filter)
        if self._verbose:
            print('INFO: Instantiate OF base for each channel!')
        
        self._processing_data.instantiate_OF_base(self._processing_config)
        
                

        # initialize output
        output_df = None
        
        # case only 1 node used for processing
        if ncores == 1:
            output_df = self._process(-1,
                                      self._series_list,
                                      nevents,
                                      save_hdf5,
                                      output_group_path,
                                      memory_limit_MB)

        else:
            
            # split data
            series_list_split = self._split_series(ncores)
            
            # for multi-core processing, we need to decrease the
            # max memory so it fits in RAM
            memory_limit_MB /= ncores

              
            # lauch pool processing
            if self._verbose:
                print(f'INFO: Processing with be split between {ncores} cores!')

            node_nums = list(range(ncores+1))[1:]
            pool = Pool(processes=ncores)
            output_df_list = pool.starmap(self._process,
                                          zip(node_nums,
                                              series_list_split,
                                              repeat(nevents),
                                              repeat(save_hdf5),
                                              repeat(output_group_path),
                                              repeat(memory_limit_MB)))
            pool.close()
            pool.join()

            # concatenate
            output_df = pd.concat(output_df_list)



            
        # processing done
        if self._verbose:
            print('INFO: Feature processing done!') 
                
        
        if not save_hdf5:
            return output_df 
        
            

        

            
        
    def _process(self, node_num,
                 series_list, nevents,
                 save_hdf5, output_group_path,
                 memory_limit_MB):
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
        
        save_hdf5 : bool, optional
           if True, save dataframe in hdf5 files
           (dataframe not returned)
           if False, return dataframe (memory limit applies
           so not all events may be processed)
           Default: True

        output_path : str, optional
           base directory where output feature file will be saved
           default: same base path as input data
    
        ncores: int, optional
           number of cores that will be used for processing
           default: 1

        memory_limit_MB : float, optionl
           memory limit per file 
           (and/or if return_df=True, max dataframe size)
   
        """

        # node string (for display)
        node_num_str = str()
        if node_num>-1:
            node_num_str = ' Node #' + str(node_num)
        

        # feature extractors
        FE = FeatureExtractors
        FE_ext = None
        if self._external_file is not None:
            FE_ext = self._load_external_extractors(self._external_file)

            
        # intialize event counter
        # (only used to check maximum
        # numberof events
        event_counter = 0


        # initialize output data frame
        feature_df = pd.DataFrame()
            
        # loop series
        for series in series_list:


            if self._verbose:
                print('INFO' + node_num_str
                      + ': starting processing series '
                      + series)

            
            # set file list
            self._processing_data.set_series(series)

            # output file name base (if saving data)
            output_base_file = None
            if save_hdf5:
                
                file_prefix = 'feature'
                if self._processing_id is not None:
                    file_prefix = self._processing_id + '_feature'
                       
                output_base_file = (output_group_path
                                    + '/' + file_prefix
                                    + '_' + series)

       
            # initialize dump counter
            dump_couter = 1
                        
            # loop events
            do_stop = False
            while (not do_stop):

                # -----------------------
                # Check number events
                # and memory usage
                # -----------------------

                nevents_limit_reached = (nevents>0 and event_counter>=nevents)
                
                # flag memory limit reached
                memory_usage_MB = feature_df.memory_usage(deep=True).sum()/1e6
                memory_limit_reached =  memory_usage_MB  >= memory_limit_MB
                

                # display
                if self._verbose:
                    
                    if (event_counter%500==0 and event_counter!=0):
                        print('INFO' + node_num_str
                              + ': Local number of events = '
                              + str(event_counter)
                              + ' (memory = ' + str(memory_usage_MB) + ' MB)')
                        
                # -----------------------
                # Read next event
                # -----------------------

                success = self._processing_data.read_next_event(
                    channels=self._selected_channels
                )
                
                if not success:
                    do_stop = True

            
                # -----------------------
                # save file if needed
                # -----------------------
                
                # now let's handle case we need to stop
                # or memory/nb events limit reached
                if (do_stop
                    or nevents_limit_reached
                    or memory_limit_reached):
                    
                    # save file if needed
                    if save_hdf5:
                        
                        # build hdf5 file name
                        dump_str = '_F'
                        for ii in range(0,4-len(str(dump_couter))):
                            dump_str += '0'
                        dump_str += str(dump_couter)
                        file_name =  output_base_file +  dump_str + '.hdf5'
                    
                        # convert to vaex
                        feature_vx = vx.from_pandas(feature_df,
                                                    copy_index=True)


                        # export
                        feature_vx.export_hdf5(file_name,
                                               mode='w')
                        
                        # increment dump
                        dump_couter += 1

                        # reset dataframe
                        del feature_df
                        feature_df = pd.DataFrame()


                    # case maximum number of events reached
                    # -> processing done!
                    if nevents_limit_reached:
                        print('INFO' + node_num_str + ': Requested nb events reached. '
                                      + 'Stopping processing!')
                        return feature_df

                    # case memory limit reached and not saving file
                    # -> processing done
                    if memory_limit_reached:
                        print('INFO' + node_num_str + ': Memory limit reached!')
                        if save_hdf5:
                            print('INFO' + node_num_str + ': Starting a new  dump for series '
                                  + series)
                        else:
                            if success:
                                print('WARNING' + node_num_str + ': Stopping procssing. '
                                      +' Not all events have been processed!')
                            return feature_df




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
                self._processing_data.update_signal_OF()

              
                # Processing id   
                event_features.update(
                    {'processing_id': self._processing_id}
                )

                # admin data
                event_features.update(
                    self._processing_data.get_event_admin()
                )

                # Detector settings
                for channel in self._processing_config.keys():
                    event_features.update(
                    self._processing_data.get_channel_settings(channel)
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
                    if 'feature_channel' in algorithms:
                        feature_channel = algorithms['feature_channel']
                        
                    # loop algorithms to extact features
                    for algorithm, algorithm_params in algorithms.items():


                        # skip if "feature_channel"
                        if algorithm=='feature_channel':
                            continue
                        
                        # skip if algorithm disable
                        if not algorithm_params['run']:
                            continue
                        
                        # check if derived algorithm
                        base_algorithm = algorithm
                        if 'base_algorithm' in algorithm_params:
                            base_algorithm = algorithm_params['base_algorithm']

                     
                        # get feature extractor
                        extractor = None
                        if base_algorithm in self._algorithm_list:
                            extractor = getattr(FE, base_algorithm)
                        elif base_algorithm in self._ext_algorithm_list:
                            extractor = getattr(FE_ext, base_algorithm)
                        else:
                            raise ValueError('ERROR: Cannot find algorithm "'
                                             + base_algorithm
                                             + '" anywhere. '
                                             + 'Check feature extractor exists!')

                        
                        # extractor arguments (removing run)
                        # add parameter if needed
                        kwargs = {key: value
                                  for (key, value) in algorithm_params.items()
                                  if key!='run'}

                        # add various parameters that may be needed
                        # by the algoithm
                        kwargs['fs'] = self._processing_data.get_sample_rate()
                        kwargs['nb_samples_pretrigger'] = (
                            self._processing_data.get_nb_samples_pretrigger()-1
                        )
                        kwargs['nb_samples'] = self._processing_data.get_nb_samples()
                        
                        kwargs['window_min_index'], kwargs['window_max_index'] = (
                            self._get_window_indices(**kwargs)
                        )

                        # base feature name = algorithm name
                        kwargs['feature_base_name'] = algorithm
                        
                        # calculate features and get output dictionary 
                        extracted_features = dict()
                        
                        # For OF algorithms, get OB base object
                        OF_base = self._processing_data.get_OF_base(channel,
                                                                    base_algorithm)
                        if OF_base is not None:
                            extracted_features = extractor(OF_base, **kwargs)
                        else:
                            trace = self._processing_data.get_channel_trace(channel)
                            extracted_features = extractor(trace, **kwargs)

                                                        
                        # append channel name and save in data frame
                        for feature_base_name in extracted_features:
                            feature_name = f'{feature_base_name}_{feature_channel}'
                            event_features.update(
                                {feature_name: extracted_features[feature_base_name]}
                            )




                # done processing event!
                # append event dictionary to dataframe
                feature_df = feature_df.append(event_features,
                                               ignore_index=True)
           
        # return features
        return feature_df
       


        
        
    def _get_file_list(self, file_path, series=None):
        """
        Get file list from path. Return as a dictionary
        with key=series and value=list of files

        Parameters
        ----------

        file_path : str or list of str 
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
        
        # loop file path
        if not isinstance(file_path, list):
            file_path = [file_path]

        # initialize
        file_list = list()
        base_path = None
        group_name = None

        # loop files 
        for a_path in file_path:

                   
            # case path is a directory
            if os.path.isdir(a_path):

                if base_path is None:
                    base_path = str(Path(a_path).parent)
                    group_name = str(Path(a_path).name)
                            
                if series is not None:
                    if series == 'even' or series == 'odd':
                        file_name_wildcard = series + '_*.hdf5'
                        file_list = glob(a_path + '/' + file_name_wildcard)
                    else:
                        if not isinstance(series, list):
                            series = [series]
                        for it_series in series:
                            file_name_wildcard = '*' + it_series + '_*.hdf5'
                            file_list.extend(glob(a_path + '/' + file_name_wildcard))
                else:
                    file_list = glob(a_path + '/*.hdf5')
                
                # check a single directory
                if len(file_path) != 1:
                    raise ValueError('Only single directory allowed! ' +
                                     'No combination files and directories')
                
                    
            # case file
            elif os.path.isfile(a_path):

                if base_path is None:
                    base_path = str(Path(a_path).parents[1])
                    group_name = str(Path(Path(a_path).parent).name)
                    
                if a_path.find('.hdf5') != -1:
                    if series is not None:
                        if series == 'even' or series == 'odd':
                            if a_path.find(series) != -1:
                                file_list.append(a_path)
                        else:
                            if not isinstance(series, list):
                                series = [series]
                            for it_series in series:
                                if a_path.find(it_series) != -1:
                                    file_list.append(a_path)
                    else:
                        file_list.append(a_path)

            else:
                raise ValueError('File or directory "' + a_path
                                 + '" does not exist!')
            
        if not file_list:
            raise ValueError('ERROR: No raw input data found. Check arguments!')

        # sort
        file_list.sort()
      
        # get list of series
        series_dict = dict()
        h5reader = h5io.H5Reader()
        series_name = None
        file_counter = 0
        for file_name in file_list:

            # skip if filter file
            if 'filter' in file_name:
                continue
            
            # append file if series already in dictionary
            if (series_name is not None
                and series_name in file_name
                and series_name in series_dict.keys()):

                if file_name not in series_dict[series_name]:
                    series_dict[series_name].append(file_name)
                    file_counter += 1
                continue
            
            # get metadata
            metadata = h5reader.get_metadata(file_name)
            series_name = h5io.extract_series_name(metadata['series_num'])
            if series_name not in series_dict.keys():
                series_dict[series_name] = list()

            # append
            if file_name not in series_dict[series_name]:
                series_dict[series_name].append(file_name)
                file_counter += 1
                

        if self._verbose:
            print('INFO: Found total of '
                  + str(file_counter)
                  + ' files from ' + str(len(series_dict.keys()))
                  + ' different series number!')

      
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



    
    def _read_config(self, yaml_file, available_channels):
        """
        Read and check yaml configuration
        file 

        Parameters
        ----------

        yaml_file : str
          yaml configuraton file name (full path)

        available_channels : list
          list of channels available in the file


        Return
        ------
        
        processing_config : dict 
           dictionary with  processing configuration


        filter_file : str
            filter file name (full path)
        
        selected_channels : list
            list of all channels to be processed
           
        
        """

        # initialize output
        processing_config = dict()
        filter_file = None
        selected_channels = list()


        # open configuration file
        yaml_dict = dict()
        with open(yaml_file) as f:
            yaml_dict = yaml.safe_load(f)
        if not yaml_dict:
            raise ValueError('Unable to read processing configuration!')


        # Let's check for unallowed duplication
        key_list = list()
        key_list_duplicate = list()
        for key in yaml_dict.keys():
            if ',' in key:
                key_split = key.split(',')
                for key_sep in key_split:
                    if key_sep in key_list:
                        key_list_duplicate.append(key_sep)
                    else:
                        key_list.append(key_sep)
            else:
                if key in key_list:
                    key_list_duplicate.append(key)
                else:
                    key_list.append(key)


        if key_list_duplicate:
            raise ValueError('Duplicate key/channel(s) found in configuration file: '
                             + str(key_list_duplicate))
    
        
        # filter file
        if 'filter_file' in yaml_dict.keys():
            filter_file = yaml_dict['filter_file']
            
                
        # case 'all' channels key available
        if 'all' in yaml_dict.keys():
            for chan in available_channels:
                processing_config[chan] = yaml_dict['all'].copy()
                

        # loop config to load individual channels
        for key in yaml_dict.keys():
                     
            # skip "all" and "filter_file"
            # (already taking into account)
            if (key=='filter_file' or key=='all'):
                continue

            
            # check if key contains a comma 
            # -> need to split 
            key_split = list()
            if ',' in key:
                key_split = key.split(',')
            else:
                key_split.append(key)

            # loop single channels
            for channel in key_split:
                
                # If disable -> skip!
                if ('disable' in yaml_dict[key].keys()
                    and yaml_dict[key]['disable']):
                    
                    if channel in processing_config.keys():
                        processing_config.pop(channel)

                    continue
                            
                # check if chan already exist
                # if so, add/replace feature config
                if channel in processing_config.keys():
                    for item_key, item_val in yaml_dict[key].items():

                        if isinstance(item_val, dict):
                            item_val = item_val.copy()
                        
                        if not isinstance(processing_config[channel], dict):
                            processing_config[channel] = dict()
                            
                        processing_config[channel][item_key] = item_val
                else:
                    processing_config[channel] = yaml_dict[key].copy()

                # remove disable if needed
                if 'disable' in processing_config[channel]:
                    processing_config[channel].pop('disable')
                
                
        # Check if configuration channel  exist in file
        channel_list_temp = list()
        for key in processing_config.keys():

            chans = list()
            if ',' in key:
                chans = key.split(',')
            elif '|' in key:
                chans = key.split('|')
            elif '+' in key:
                chans = key.split('+')
            elif key != 'filter_file':
                chans.append(key)
                
            for chan in chans:
                if chan not in available_channels:
                    raise ValueError('Channel "' + chan
                                     + '" do not exist in '
                                     + 'raw data! Check yaml file!')
                else:
                    channel_list_temp.append(chan)

        # make list unique
        for chan in available_channels:
            if chan in channel_list_temp:
                selected_channels.append(chan)
            
                    
        # return
        return processing_config, filter_file, selected_channels


    def _create_output_directory(self, base_path, facility):
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
        
        # prefix
        prefix = 'feature'
        if self._processing_id is not None:
            prefix = self._processing_id + '_feature'
        output_dir = base_path + '/' + prefix + '_' + series_name
        
        
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir)
                os.chmod(output_dir, stat.S_IRWXG | stat.S_IRWXU | stat.S_IROTH | stat.S_IXOTH)
            except OSError:
                raise ValueError('\nERROR: Unable to create directory "'+ output_dir  + '"!\n')
                
        return output_dir
        



    
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




    def _get_window_indices(self, nb_samples,
                            nb_samples_pretrigger, fs,
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

        nb_samples_pretrigger : int
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
                        min_index = (nb_samples_pretrigger 
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
            max_index =  (nb_samples_pretrigger 
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
