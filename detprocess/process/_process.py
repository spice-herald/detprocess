import yaml
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import importlib
import sys
import os
from glob import glob
from pprint import pprint
from multiprocessing import Pool
from itertools import repeat

from detprocess.process._features  import FeatureExtractors
from detprocess.process._processing_data  import ProcessingData
import pytesdaq.io as h5io



__all__ = [
    'Processing'
]




class Processing:
    """
    Class to manage data processing and 
    extract features
    """

    def __init__(self, raw_path, config_file,
                 external_file=None, series=None,
                 processing_id=None,
                 verbose=True):
        """
        Intialize data processing 
        
        Arguments
        ---------
    
        raw_path: str or list of str (required)
           raw data group directory OR full path to HDF5  file 
           (or list of files). Only a single directory 
           allowed 
        
        
        config_file: str (required)
           Full path and file name to the YAML settings for the
           processing.

        output_path: str (optional, default=No saved file)
           base directory where output feature file will be saved
     

        external_file : str,  (optional,  default=no external files)
           The path to a .py file with a FeatureExtractors class 
           to add algorithms developed by users. The name of the static
           functions cannot be same as in _features (no duplicate).  This is 
           meant for rapid development of features without
           needing to rebuild the package.


        series: str or list of str (optional)
            series to be process, disregard other data from raw_path

        processing_id: str (optional)
            an optional processing name. This is used to be build output subdirectory name 
            and is saved as a feature in DetaFrame so it can then be used later during 
            analysis to make a cut on a specific processing when mutliple 
            datasets/processing are added together. 
     
        """


        self._verbose = verbose


        # Input file list
        self._input_file_dict, self._input_base_path = (
            self._get_file_list(raw_path, series=series)
        )
        
        if not self._input_file_dict:
            raise ValueError('No files were found! Check configuration...')
        
          
        
        # processing configuration
        if not os.path.exists(config_file):
            raise ValueError('Configuration file "' + config_file
                             + '" not found!')
        config, filter_file, channels = self._read_config(config_file)
        
        self._filter_filename = filter_file
        self._processing_config = config
        self._selected_channels = channels

        # check channels to be processed
        if not channels:
            raise ValueError('No channels to be processed! Check configuration...')

        # maximum number of nodes (can only split series, not files)
        self._nb_nodes_max = len(self._input_file_dict.keys())
                                  
        # processing ID 
        self._processing_id = processing_id
        
      
        # External feature extractors
        self._external_file = None
        if external_file is not None:
            if not os.path.exists(external_file):
                raise ValueError('External feature extractors file "'
                                 + external_file
                                 + '" not found!')
            self._external_file = external_file
            if self._verbose:
                print('INFO: external feature extractor = ' + self._external_file)
                
        # get list of available features and check for duplicate
        self._algorithm_list, self._ext_algorithm_list = self._extract_algorithm_list()

            
        # filter data
        self._filter_data = None
        if self._filter_filename is not None:
            filter_inst = h5io.FilterH5IO(self._filter_filename)
            self._filter_data = filter_inst.load()
            if self._verbose:
                print('INFO: Filter file '
                      + self._filter_filename
                      + ' has been loaded!')


        # ADC info
        self._adc_info = self._get_adc_info()
        

                
                
        
    def process(self, nevents=-1,
                save_hdf5=True, output_path=None,
                nb_cores=1, memory_limit_MB=2000):
        
        """
        Process data
        
        Arguments
        ---------
        
        output_path: str (optional, default=same base path as input data)
           base directory where output feature file will be saved
     
        nb_cores: int (optional, default=1)
           number of cores that will be used for processing
        """


        # check input
        if (nb_cores>1 and nevents>-1):
            raise ValueError('ERROR: Multi cores processing only allowed when '
                             + 'processing ALL events!')
        

        
        # create output directory
        output_group_path = None
        
        if save_hdf5:
            if  output_path is None:
                output_path  = self._input_base_path
                
            output_group_path = self._create_output_directory(ouput_path,
                                                              self._processing_id)
            if self._verbose:
                print(f'INFO: Processing output group path: {output_group_path}')

                
        # check number cores allowed
        if nb_cores>self._nb_nodes_max:
            nb_cores = self._nb_nodes_max
            if self._verbose:
                print('INFO: Changing number cores to '
                      + str(nb_cores) + ' (maximum allowed)')
                    

        # initialize output
        output_df = None
        
        # case only 1 node used for processing
        if nb_cores == 1:
            output_df = self._process(self._input_file_dict,
                                      nevents,
                                      save_hdf5,
                                      output_group_path,
                                      memory_limit_MB)

        else:
            
            # split data
            input_file_list = self._split_data(nb_cores)
             
            # for multi-core processing, we need to decrease the
            # max memory so it fits in RAM
            memory_limit_MB /= nb_cores

              
            # lauch pool processing
            if self._verbose:
                print(f'INFO: Processing with be split between {nb_cores} cores!')

            pool = Pool(processes=nb_cores)
            output_df = pool.starmap(self._process,
                                   zip(input_file_list,
                                       repeat(nevents),
                                       repeat(save_hdf5),
                                       repeat(output_group_path),
                                       repeat(memory_limit_MB)))
            pool.close()
            pool.join()


        # processing done
        if self._verbose:
            print('INFO: Feature processing done!') 
                
        
        if not save_hdf5:
            return output_df 
        
            

        

            
        
    def _process(self, file_list, nevents,
                 save_hdf5, output_group_path,
                 memory_limit_MB):
        """
        Process data
        
        Arguments
        ---------

        """

        
        # instantiate processing data
        processing_data = ProcessingData(self._processing_config,
                                         self._filter_data,
                                         self._selected_channels)

        # instantiate "OptimumFilter" objects
        # for all channels
        processing_data.instantiate_OF(
            sample_rate = self._adc_info['sample_rate']
        )


        # feature extractors
        FE = FeatureExtractors
        FE_ext = None
        if self._external_file is not None:
            self._FE_ext = self._load_external_extractors(self._external_file)

            
        # intialize event counter
        # (only used to check maximum
        # numberof events
        event_counter = 0


        # initialize output data frame
        feature_df = pd.DataFrame()
            
        # loop series
        for series, series_files in file_list.items():

            # set file list
            processing_data.set_files(series_files)

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
                    if (event_counter % 100 == 0):
                        print('INFO: Number of events = '
                              + str(event_counter)
                              + ' (memory = ' + str(memory_usage_MB)
                              + ' MB)')
                                        
                # -----------------------
                # Read next event
                # -----------------------

                success = processing_data.read_next_event()
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
                    
                        # save
                        feature_df.to_hdf(file_name,
                                          key='features',
                                          mode='w')
                        # increment dump
                        dump_couter += 1

                        # reset dataframe
                        del feature_df
                        feature_df = pd.DataFrame()


                    # case maximum number of events reached
                    # -> processing done!
                    if nevents_limit_reached:
                        print('INFO: Requested nb events reached. '
                                      + 'Stopping processing!')
                        return feature_df

                    # case memory limit reached and not saving file
                    # -> processing done
                    if memory_limit_reached:
                        print('INFO: Memory limit reached!')
                        if save_hdf5:
                            print('INFO: Starting a new  dump for series '
                                  + series)
                        else:
                            if success:
                                print('WARNING: Stopping procssing. '
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

                # update signa trace in OF objects
                # -> calculate FFTs, etc.
                processing_data.update_signal_OF()

              
                # Processing id   
                event_features.update(
                    {'processing_id': self._processing_id}
                )

                # admin data
                event_features.update(
                    processing_data.get_event_admin()
                )

                # Detector settings
                for channel in self._processing_config.keys():
                    event_features.update(
                    processing_data.get_channel_settings(channel)
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
                            raise ValueError('Cannot find algorithm "'
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
                        kwargs['fs'] = self._adc_info['sample_rate']
                        kwargs['nb_samples_pretrigger'] = self._adc_info['nb_samples_pretrigger']-1
                        kwargs['nb_samples'] = self._adc_info['nb_samples']

                        kwargs['min_index'], kwargs['max_index'] = (
                            self._get_window_indices(**kwargs)
                        )

                        # base feature name = algorithm name
                        kwargs['feature_base_name'] = algorithm
                        
                        # calculate features and get output dictionary 
                        extracted_features = dict()
                        
                        # case OF 1x1
                        if  base_algorithm in processing_data.get_algorithms_OF(
                                of_type='1x1'):
                            
                            # get psd/template tag
                            template_tag = 'default'
                            if 'template_tag' in algorithm_params:
                                template_tag = algorithm_params['template_tag']
                            psd_tag = 'default'
                            if 'psd_tag' in algorithm_params:
                                psd_tag = algorithm_params['psd_tag']

                            # get OF object
                            tag = psd_tag + '_' + template_tag
                            OF1x1 =  processing_data.get_OF(channel, tag, of_type='1x1')
                                                   
                            # extract
                            extracted_features = extractor(OF1x1, **kwargs)
                        
                        # other pulse algorithms
                        else:
                            trace = processing_data.get_channel_trace(channel)
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
        Get file list from directory
        path
        """
        
        # loop file path
        if not isinstance(file_path, list):
            file_path = [file_path]

        # initialize
        file_list = list()
        base_path = None


        # loop files 
        for a_path in file_path:
            
            # case path is a directory
            if os.path.isdir(a_path):

                base_path = Path(a_path).parent
                
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
                    base_path = Path(a_path).parents[1]
                                
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

      
        return series_dict, str(base_path)



    
    
    def _load_external_extractors(self, external_file):
        """
        Helper function for loading an alternative SingleChannelExtractors
        class.
        
        """
    
        module_name = 'detprocess.process'

        spec = importlib.util.spec_from_file_location(module_name,
                                                      external_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return module.FeatureExtractors



    
    def _read_config(self, yaml_file):
        """
        Read and check yaml configuration
        file 
        """

        # initialize output
        processing_config = dict()
        filter_file = None
        selected_channels = list()
           
        # let's get list of channels available in file
        # first file
        file_name = str()
        for key, val in self._input_file_dict.items():
            file_name = val[0]
            break
        available_channel_list  = self._get_channel_list(file_name)
            
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
            for chan in available_channel_list:
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
                if chan not in available_channel_list:
                    raise ValueError('Channel "' + chan
                                     + '" do not exist in '
                                     + 'raw data! Check yaml file!')
                else:
                    channel_list_temp.append(chan)

        # make list unique
        for chan in available_channel_list:
            if chan in channel_list_temp:
                selected_channels.append(chan)
            
                    
        # return
        return processing_config, filter_file, selected_channels


    def _create_output_directory(self, base_path):
        """
        Create series name
    
        Return
          Series name: string
        """

        now = datetime.now()
        series_day = now.strftime('%Y') +  now.strftime('%m') + now.strftime('%d') 
        series_time = now.strftime('%H') + now.strftime('%M')
        series_name = ('I' + str(self._facility) +'_D' + series_day + '_T'
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


    def _split_data(self, nb_cores):
        """
        Split data in between nodes
        following series
        """

        # get series
        series_keys =  list(self._input_file_dict.keys())
        nb_series = len(series_keys)

        # split series
        series_split = np.array_split(series_keys, nb_cores)

        # build list of dict
        file_dict_list = list()
        for series_array in series_split:
            if series_array.size == 0:
                continue
            file_dict = dict()
            for series in series_array:
                file_dict[str(series)] = self._input_file_dict[str(series)]

            # append
            file_dict_list.append(file_dict)

        return file_dict_list

    

    
    def _get_channel_list(self, file_name):
        """ 
        Get the list of channels from raw data file
        
        Parameters
        ----------
        file_name : str
        The full file_name (including path) to the data to be loaded.
        
        Returns
        -------
        channels: list
          List of channels
    
        """

        # get list from configuration
        h5 = h5io.H5Reader()
        detector_settings = h5.get_detector_config(file_name=file_name)
        return list(detector_settings.keys())


    def _get_adc_info(self):
        """
        Get ADC info
        """

        adc_info = None
        
        if not self._input_file_dict:
            raise ValueError('No file available to get sample rate!')

        h5 = h5io.H5Reader()
        for series, files in self._input_file_dict.items():
            metadata = h5.get_metadata(file_name=files[0],
                                       include_dataset_metadata=False)

            adc_name = metadata['adc_list'][0]
            adc_info = metadata['groups'][adc_name]
            break

        if adc_info is None:
            raise ValueError('ERROR: No ADC info in file. Something wrong...')

        return adc_info



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
        Calculate window index min and max

        Arguments:
        ---------


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
