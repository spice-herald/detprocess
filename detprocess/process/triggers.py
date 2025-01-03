import yaml
import warnings
from pathlib import Path
import numpy as np
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
import pytesdaq.io as h5io
import copy
from humanfriendly import parse_size
from detprocess.process.processing_data  import ProcessingData
from detprocess.process.config import YamlConfig
from detprocess.core.eventbuilder import EventBuilder
from detprocess.core.oftrigger import OptimumFilterTrigger
from detprocess.utils import utils
from detprocess.core.rawdata import RawData

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
    'TriggerProcessing'
]


class TriggerProcessing:
    """
    Class to manage trigger processing and 
    extract features, dataframe can be saved
    in hdf5 using vaex framework
    
    Multiple nodes can be used if data splitted in 
    different series

    """

    def __init__(self, raw_data, config_data,
                 series=None,
                 processing_id=None,
                 restricted=False,
                 calib=False,
                 salting_dataframe=None,
                 verbose=True):
        """
        Intialize data processing 
        
        Parameters
        ---------
    
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

        salting_dataframe : str or vaex dataframe
           str if path to vaex hdf5 file or directly a dataframe 

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
        available_channels = None
        for it, it_config in rawdata_metadata.items():
            available_channels = it_config['channel_list']
            break;
        
        # config file
        config_dict = {}
        if  isinstance(config_data, str):
            
            if not os.path.isfile(config_data):
                raise ValueError(f'ERROR: argument "{config_data}" '
                                 f'should be a file or YamlConfig object!')

            yaml = YamlConfig(config_data, available_channels)
            config_dict = yaml.get_config('trigger')

        else:
            
            if 'YamlConfig' not in str(type(config_data)):
                raise ValueError(
                    'ERROR: raw data argument should be either '
                    'a directory or YamlConfig object'
                )
            
            config_dict = config_data.get_config('trigger')
            
        self._trigger_config = copy.deepcopy(config_dict['channels'])
        self._evtbuilder_config = copy.deepcopy(config_dict['overall'])
        self._trigger_channels = copy.deepcopy(config_dict['channel_list'])
      
        if not 'filter_file' in config_dict['overall']:
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
            filter_file=config_dict['overall']['filter_file'],
            salting_dataframe=salting_dataframe,
            verbose=verbose
        )


    def get_output_path(self):
        """
        Get output group path
        """
        return self._output_group_path
    
        
    def process(self, ntriggers=-1,
                lgc_save=False,
                lgc_output=False,
                save_path=None,
                output_group_name=None,
                ncores=1,
                memory_limit='1GB'):
        
        """
        Process data 
        
        Parameters
        ---------

        ntriggers : int, optional
           number of events to be processed
           if not all events, requires ncores = 1
           Default: all available (=-1)

        lgc_save : bool, optional
           if True, save dataframe in hdf5 files
           (dataframe not returned)
           if False, return dataframe (memory limit applies
           so not all events may be processed)
           Default: True

        output_group_path : str, optional
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
        if (ncores>1 and ntriggers>-1):
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
        output_series_num = None
        
        if lgc_save:
            if  save_path is None:
                save_path  = self._input_base_path + '/processed'
                if '/raw/processed' in save_path:
                    save_path = save_path.replace('/raw/processed','/processed')
                    
            # add group name
            if self._input_group_name not in save_path:
                save_path = save_path + '/' + self._input_group_name
                
            output_group_path, output_series_num  = (
                self._create_output_directory(
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
                                      ntriggers,
                                      lgc_save,
                                      lgc_output,
                                      output_series_num,
                                      output_group_path,
                                      memory_limit)

        else:

            
            # disable vaex multi-threading
            vx.settings.main.thread_count = 1
            vx.settings.main.thread_count_io = 1
            pa.set_cpu_count(1)
                    
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
                                              repeat(ntriggers),
                                              repeat(lgc_save),
                                              repeat(lgc_output),
                                              repeat(output_series_num),
                                              repeat(output_group_path),
                                              repeat(memory_limit)))
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
            print('INFO: Trigger processing done!') 
                
        
        if lgc_output:
            return output_df 
        
           
    def _process(self, node_num,
                 series_list, ntriggers,
                 lgc_save, lgc_output,
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

        ntriggers : int, optional
           number of events to be processed
           if not all events, requires ncores = 1
           Default: all available (=-1)
        
        lgc_save : bool, optional
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

        memory_limit : float, optionl
           memory limit per file in bytes
           (and/or if return_df=True, max dataframe size)
   
        """

        
        # disable vaex multi-threading
        vx.settings.main.thread_count = 1
        vx.settings.main.thread_count_io = 1
        pa.set_cpu_count(1)
               
        # check argument
        if lgc_output and lgc_save:
            raise ValueError('ERROR: Unable to save and output datafame '
                             + 'at the same time. Set either lgc_output '
                             + 'or lgc_save to False.')

        # node string (for display)
        node_num_str = str()
        if node_num>-1:
            node_num_str = ' node #' + str(node_num)


        # salting dataframe
        self._processing_data_inst.load_salting_dataframe()
      
        # instantiate event builder
        evtbuilder_inst = EventBuilder()
              
        # instantiate OF trigger and add to EventBuilder'
        trigger_config = copy.deepcopy(self._trigger_config)
        nb_trigger_chans = len(list(trigger_config.keys()))
        for trig_chan, trig_data in trigger_config.items():

            # channel name
            channel_name = trig_data['channel_name']
            
            # get template
            template_tag = 'default'
            if 'template_tag' in trig_data:
                template_tag = trig_data['template_tag']

            template, template_metadata = (
                self._processing_data_inst.get_template(
                    channel_name,
                    tag=template_tag)   
            )

            nb_pretrigger_samples = None
            if  'nb_pretrigger_samples' in template_metadata.keys():
                nb_pretrigger_samples = (
                    template_metadata['nb_pretrigger_samples']
                )
            else:
                # back compatibility
                if 'pretrigger_length_samples' in template_metadata.keys():
                    nb_pretrigger_samples = (
                        template_metadata['pretrigger_length_samples']
                    )
                elif 'pretrigger_samples' in template_metadata.keys():
                    nb_pretrigger_samples = (
                        template_metadata['pretrigger_samples']
                    )
            if nb_pretrigger_samples is None:
                raise ValueError('ERROR: Template metadata needs to contain '
                                 '"nb_pretrigger_samples" value')

            # Get noise spectrum (CSD/PSD)
            noise_tag = 'default'
            if 'noise_tag' in trig_data:
                noise_tag = trig_data['noise_tag']
            elif 'csd_tag' in trig_data:
                noise_tag = trig_data['csd_tag']
            elif 'psd_tag' in trig_data:
                noise_tag = trig_data['psd_tag']

            csd, csd_freqs, csd_metadata = (
                self._processing_data_inst.get_noise(
                    channel_name,
                    tag=noise_tag)
            )
            
            # sample rate
            fs = self._processing_data_inst.get_sample_rate()

            # instantiate optimal filter trigger
            oftrigger_inst = OptimumFilterTrigger(
                channel_name, fs, template, csd,
                nb_pretrigger_samples,
                trigger_name=trig_chan
            )

            # add in EventBuilder
            evtbuilder_inst.add_trigger_object(
                trig_chan, oftrigger_inst)
            
        # output file name base (if saving data)
        output_base_file = None
        
        if lgc_save:

            file_prefix = 'threshtrig'
            if self._processing_id is not None:
                file_prefix = self._processing_id +'_' + file_prefix

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
        trigger_counter = 0

        # intialize output dataframe
        process_df = None
        
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
            while (not do_stop):

                # -----------------------
                # Check number events
                # and memory usage
                # -----------------------
                ntriggers_limit_reached = (ntriggers>0
                                           and trigger_counter>=ntriggers)
                
                # flag memory limit reached
                memory_usage = 0
                memory_limit_reached = False
                if process_df is not None:
                    memory_usage = process_df.shape[0]*process_df.shape[1]*8
                    memory_limit_reached =  memory_usage  >= memory_limit
                                  
                # display
                if self._verbose:
                    if (trigger_counter%100==0 and trigger_counter!=0):
                        print('INFO' + node_num_str
                              + ': Local number of events = '
                              + str(trigger_counter) 
                              + ' (memory = ' + str(memory_usage/1e6) + ' MB)')
                        
                # -----------------------
                # Read next event
                # -----------------------
                success = self._processing_data_inst.read_next_event(
                    channels=self._trigger_channels
                )

                # end of file or raw data issue
                if not success:
                    print('INFO' + node_num_str
                          + ': '
                          + str(trigger_counter) 
                          + ' events counted, triggering processing done')
                    do_stop = True
                                           
                # -----------------------
                # Handle stop or
                # nb trigger/memory limit
                # reached
                # -----------------------

                                
                # let's handle case we need to stop
                # or memory/nb events limit reached
                if (do_stop
                    or ntriggers_limit_reached
                    or memory_limit_reached):


                    # case nb triggers reached
                    if ntriggers_limit_reached:
                        nextra = trigger_counter-ntriggers
                        nkeep = len(process_df)-nextra
                        if nkeep>0:
                            process_df = process_df[0:nkeep]
                       
                           
                    # save file if needed
                    if lgc_save and  process_df is not None:
                        
                        # build hdf5 file name
                        dump_str = str(dump_counter)
                        file_name =  (output_base_file + '_F' + dump_str.zfill(4)
                                      + '.hdf5')
                            
                        # export to hdf5 
                        try:
                            # export
                            process_df.export_hdf5(file_name, mode='w')
                            process_df.close()
                            
                        except Exception as e:
                            
                            print('WARNING: Export failed with error: ', e)
                            print('Will try again...')
                            
                            # let's try one more time
                            df_pandas = process_df.to_pandas_df()
                            process_df = vx.from_pandas(
                                df_pandas,
                                copy_index=False
                            )
                            
                            # export
                            process_df.export_hdf5(file_name, mode='w')
                            process_df.close()
                            
                        # increment dump
                        dump_counter += 1
                        if self._verbose and not do_stop and not ntriggers_limit_reached:
                            if trigger_counter > 1e5:
                                print('INFO' + node_num_str
                                      + ': Incrementing dump number, '
                                      + f'{trigger_counter:.3e}' + ' total events triggered' ) 
                            else:
                                print('INFO' + node_num_str
                                      + ': Incrementing dump number, '
                                      + str(trigger_counter) + ' total events triggered' ) 

                        # initialize
                        del process_df
                        process_df = None
                            

                    # case maximum number of events reached
                    # -> processing done!
                    if ntriggers_limit_reached:
                        if self._verbose:
                            print('INFO' + node_num_str
                                  + ': Requested nb events reached. '
                                  + 'Stopping processing!')
                        return process_df

                    # case memory limit reached
                    # -> processing needs to stop!
                    if lgc_output and memory_limit_reached:
                        raise ValueError(
                            'ERROR: memory limit reached! '
                            + 'Change memory limit or only save hdf5 files '
                            +'(lgc_save=True AND lgc_output=False) '
                        )
                    
                                
                # check if stop
                if do_stop:
                    break
               
                                 
                # -----------------------
                # process triggers
                # -----------------------

                # clear event
                evtbuilder_inst.clear_event()

                
                # loop trigger channels
                for trig_chan, trig_data in trigger_config.items():

                    # channel
                    channel_name =  trig_data['channel_name']
                 
                    # get threshold
                    threshold = None
                    if 'threshold_sigma' in trig_data.keys():
                        threshold = float(trig_data['threshold_sigma'])
                    elif 'threshold' in trig_data.keys():
                        threshold = float(trig_data['threshold'])
                    else:
                        raise ValueError(
                            'ERROR: "treshold_sigma" missing in '
                            + 'yaml configuration file')
                        
                    # pileup window
                    pileup_window_msec = None
                    if 'pileup_window_msec' in trig_data.keys():
                        pileup_window_msec = float(
                            trig_data['pileup_window_msec']
                        )

                    pileup_window_samples = None
                    if 'pileup_window_samples' in trig_data.keys():
                        pileup_window_samples = int(
                            trig_data['pileup_window_samples'])

                    # positive pulse
                    positive_pulses = True
                    if 'positive_pulses' in trig_data.keys():
                        positive_pulses = trig_data['positive_pulses']
                        
                    # get trace (If multiple channels, need to follow order)
                    trace = self._processing_data_inst.get_channel_trace(
                        channel_name
                    )

                    # acquire trigger
                    evtbuilder_inst.acquire_triggers(
                        trig_chan,
                        trace,
                        threshold,
                        pileup_window_msec=pileup_window_msec,
                        pileup_window_samples=pileup_window_samples,
                        positive_pulses=positive_pulses)


                # -----------------------
                # build event
                # merge coincident triggers
                # -----------------------

                coincident_window_msec = None
                if 'coincident_window_msec' in self._evtbuilder_config.keys():
                    coincident_window_msec = (
                        self._evtbuilder_config['coincident_window_msec'])
                    
                coincident_window_samples = None
                if 'coincident_window_samples' in self._evtbuilder_config.keys():
                    coincident_window_samples = (
                        self._evtbuilder_config['coincident_window_samples'])
                    
                # get event metadata
                event_info = self._processing_data_inst.get_event_admin()
              
                # build event
                evtbuilder_inst.build_event(
                    event_info,
                    fs=fs,
                    coincident_window_msec=coincident_window_msec,
                    coincident_window_samples=coincident_window_samples,
                    nb_trigger_channels=nb_trigger_chans
                )


                # get trigger data
                event_df = evtbuilder_inst.get_event_df()

                # check if triggers
                if (event_df is None or len(event_df)==0):
                    continue

                             
                # increment counter
                nb_triggers = len(event_df)
             
                trigger_counter += nb_triggers
                

                # -----------------------
                # Add metadata
                # -----------------------
                
                # add processing id
                event_df['processing_id'] = np.array(
                    [self._processing_id]*nb_triggers
                )

                # done processing event!
                # append event dictionary to dataframe
                if process_df is None:
                    process_df = event_df
                else:
                    process_df = vx.concat([process_df, event_df])



        # cleanup
        del evtbuilder_inst
                    
        # return features
        return process_df

    
    def _create_output_directory(self, base_path, facility,
                                 output_group_name=None,
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
    
         restricted : boolean
          if True, create directory name that includes "restricted"
          default: False

        Return
        ------
          output_dir : str
            full path to created directory

        """

        time.sleep(1)
        
        # create series name/number
        now = datetime.now()
        series_day = now.strftime('%Y') +  now.strftime('%m') + now.strftime('%d') 
        series_time = now.strftime('%H') + now.strftime('%M')
        series_name = ('I' + str(facility) +'_D' + series_day + '_T'
                       + series_time + now.strftime('%S'))
        series_num = int(h5io.extract_series_num(series_name))
      
        # build full path
        if output_group_name is None:
            prefix = 'trigger'
            if self._processing_id is not None:
                prefix = self._processing_id + '_trigger'
            if restricted:
                prefix += '_restricted'
            elif calib:
                prefix += '_calib'
                
            output_dir = base_path + '/' + prefix + '_' + series_name
        else:
            if output_group_name not in base_path:
                output_dir = base_path + '/' + output_group_name
            else:
                output_dir = base_path
                
        # create directory
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(str(output_dir))
                os.chmod(str(output_dir),
                         stat.S_IRWXG | stat.S_IRWXU | stat.S_IROTH | stat.S_IXOTH)
            except OSError:
                raise ValueError('\nERROR: Unable to create directory "'
                                 + output_dir  + '"!\n')
    
        return output_dir, series_num


    

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
        h5.clear()
        
        return list(detector_settings.keys())


    def _get_facility(self, file_dict):
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
        metadata = h5.get_metadata(file_name=file_name)
        facility = int(metadata['facility'])
        h5.clear()
        
        return facility
