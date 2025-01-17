import numpy as np
import warnings
import vaex as vx
from pathlib import Path
import sys
import os
from glob import glob
from pprint import pprint
from multiprocessing import Pool
from itertools import repeat
from datetime import datetime
import stat
import time
import pytesdaq.io as h5io
from math import ceil, floor
import random
import copy
from humanfriendly import parse_size
import pyarrow as pa
from detprocess.core.rawdata import RawData

warnings.filterwarnings('ignore')

vx.settings.main.thread_count = 1
vx.settings.main.thread_count_io = 1
pa.set_cpu_count(1)

__all__ = [
    'Randoms'
]

class Randoms:
    """
    Class to manage acquisitions of randoms from 
    continuous data. The randoms metadata are 
    saved in vaex dataframe for further processing.
    """

    def __init__(self, raw_data,
                 series=None,
                 processing_id=None,
                 restricted=False,
                 calib=False,
                 verbose=True):
        """
        Initialize randoms acquisition
        
        Parameters
        ---------
        
        raw_data : str or dictionary
           raw data group directory 
           OR RawData object
           Only a single raw data group allowed
            
        series : str or list of str, optional
            series to be process, disregard other data from raw_path

        processing_id :  str, optional
          
            
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

        # calibration
        self._calib = calib
        data_type = 'cont'
        if calib:
            self._restricted = False
            data_type = 'calib'


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
                raise ValueError(f'ERROR: Unable to use RawData '
                                 f'object. It needs requirement restricted = '
                                 f'{self._restricted}!')
                

        # check
        #rawdata_inst.describe()

        # get file list
        data_dict = rawdata_inst.get_data_files(data_type=data_type,
                                                series=series)
        if not data_dict:
            raise ValueError('No files were found! Check configuration...')

        # get metadata list 
        data_config = rawdata_inst.get_data_config(data_type=data_type,
                                                   series=series)
        # save info
        self._series_dict = copy.deepcopy(data_dict)
        self._series_metadata_dict = copy.deepcopy(data_config)
        self._input_base_path = rawdata_inst.get_base_path()
        self._input_group_name = rawdata_inst.get_group_name()
        self._facility = rawdata_inst.get_facility()
        self._duration, self._nb_events = (
            rawdata_inst.get_duration(data_type=data_type,
                                      series=series,
                                      include_nb_events=True)
        )

        print(f'INFO: Total raw data duration = '
              f'{self._duration/60} minutes ({self._nb_events} events)')
        
        # initialize output path
        self._output_group_path = None
    
    @property
    def verbose(self):
        return self._verbose
        
    @verbose.setter
    def verbose(self, value):
        self._verbose=value


    def get_series_dict(self):
        return  self._series_dict

    def get_base_path(self):
        return self._input_base_path

    def get_group_name(self):
        return self._input_group_name

    def get_output_path(self):
        """
        Get output group path
        """
        return self._output_group_path

    
    def process(self, random_rate=None,
                nrandoms=None,
                min_separation_msec=20,
                edge_exclusion_msec=20,
                ncores=1,
                lgc_save=False,
                lgc_output=False,
                save_path=None,
                output_group_name=None,
                memory_limit='2GB'):
        
        """
        Acquire random trigger using specified rate (and minimum
        separation)
        """
        
        # data are split based on series so
        # check number cores requested is possible
        nseries = len(self._series_dict.keys())
        if ncores>nseries:
            ncores = nseries
            if self._verbose:
                print('INFO: Changing number cores to '
                      + str(ncores) + ' (maximum possible)')

        # convert to seconds
        min_separation_sec = min_separation_msec/1000
        edge_exclusion_sec = edge_exclusion_msec/1000

        # random rate
        if (random_rate is not None
            and nrandoms is not None):
            print('ERROR: Use either "random_rate" or "nrandoms", '
                  + 'not both!')
            return

        # if input is number of randoms, let's
        # calculate random_rate (slightly increased so
        # we end up with enough events)
        self._nrandoms = nrandoms
        if nrandoms is not None:
            random_rate = 1.05*float(nrandoms)/self._duration

            
        # average time between randoms
        random_length_sec = 1/random_rate

        if random_length_sec < min_separation_sec:
            min_separation_sec = random_length_sec * 0.75
            print('WARNING: Changed min separation to '
                  + str(1e3*min_separation_sec)
                  + ' milliseconds to allow requested (high) '
                  + 'random rate!')


        if min_separation_sec > edge_exclusion_sec:
            edge_exclusion_sec = min_separation_sec

        # If rate is low, we can increase minimum seperation
        # (up to 50% time between randoms) 
        #if (random_length_sec/2>min_separation_sec):
        #    min_separation_sec = random_length_sec/2
        
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
                save_path += '/' + self._input_group_name
                
            output_group_path, output_series_num  = (
                self._create_output_directory(
                    save_path,
                    self._facility,
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

        # initialize output
        output_df = None

        # convert memory usage in bytes
        if isinstance(memory_limit, str):
            memory_limit = parse_size(memory_limit)

        # case only 1 node used for processing
        if ncores == 1:
            series_list = list(self._series_dict.keys())
            output_df = self._process(1,
                                      series_list,
                                      random_length_sec,
                                      min_separation_sec,
                                      edge_exclusion_sec,
                                      output_series_num,
                                      output_group_path,
                                      lgc_save,
                                      lgc_output,
                                      memory_limit)
        else:
            
            # split data
            series_list_split = self._split_series(ncores)
            
            # max memory so it fits in RAM
            memory_limit /= ncores
                  
            # lauch pool processing
            if self._verbose:
                print(f'INFO: Processing with be split between {ncores} cores!')

            node_nums = list(range(ncores+1))[1:]
             
            with  Pool(processes=ncores) as pool:
                output_df_list = pool.starmap(self._process,
                                              zip(node_nums,
                                                  series_list_split,
                                                  repeat(random_length_sec),
                                                  repeat(min_separation_sec),
                                                  repeat(edge_exclusion_sec),
                                                  repeat(output_series_num),
                                                  repeat(output_group_path),
                                                  repeat(lgc_save),
                                                  repeat(lgc_output),
                                                  repeat(memory_limit))
                )
                pool.close()
                pool.join()

            # concatenate
            if lgc_output:
                output_df = vx.concat(output_df_list)
                 
        # processing done
        if self._verbose:
            print('INFO: Randoms acquisition done!') 

        if lgc_output and self._nrandoms is not None:
            if len(output_df)>self._nrandoms:
                output_df = output_df.sample(n=self._nrandoms)
                            
        return output_df

    
    def _process(self, node_num,
                 series_list,
                 random_length_sec,
                 min_separation_sec,
                 edge_exclusion_sec,
                 output_series_num,
                 output_group_path,
                 lgc_save,
                 lgc_output,
                 memory_limit):
        """
        Acquire random trigger using specified rate (and minimum
        separation)
        """

        # disable multithreading
        vx.settings.main.thread_count = 1
        vx.settings.main.thread_count_io = 1
        pa.set_cpu_count(1)
            
        # node string (for display)
        node_num_str = ' Node #' + str(node_num)

        # trigger prod group name
        trigger_prod_group_name = np.nan
        if output_group_path is not None:
            trigger_prod_group_name = (
                str(Path(output_group_path).name)
            )

        # copy series dict
        series_dict = copy.deepcopy(self._series_dict)
        series_metadata_dict = copy.deepcopy(self._series_metadata_dict)
            
        # set output dataframe to None
        # only used if dataframe returned
        process_df = None

        # output file name base (if saving data)
        output_base_file = None
        
        if lgc_save:

            file_prefix = 'rand'
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

        # loop series
        dump_counter = 1
        nb_series = len(series_list)
        for series_count, series  in enumerate(series_list):

            if self._verbose:
                print('INFO' + node_num_str
                      + ': Acquiring randoms for series '
                      + series)

            # metadata
            series_metadata = series_metadata_dict[series]['overall']
                
            
            # flag memory limit reached (on a series by series basis)
            # if memory reached, save file 
            memory_usage = 0
            memory_limit_reached = False
            if process_df is not None:
                memory_usage = process_df.shape[0]*process_df.shape[1]*8
                memory_limit_reached =  memory_usage  >= memory_limit
                
            # series num
            series_num = series_metadata['series_num']
                      
            # trace length in second:
            sample_rate = series_metadata['sample_rate']
            nb_samples = series_metadata['nb_samples']
            trace_length_sec = nb_samples/sample_rate
        
            # timing
            fridge_run_start_time = np.nan
            series_start_time = np.nan
            group_start_time =  np.nan
            
            if 'fridge_run_start' in  series_metadata:
                fridge_run_start_time = series_metadata['fridge_run_start']
                series_start_time = series_metadata['series_start']
                group_start_time =  series_metadata['group_start']
            elif 'fridge_run_start_time' in  series_metadata:
                fridge_run_start_time = series_metadata['fridge_run_start_time']
                series_start_time = series_metadata['series_start_time']
                group_start_time =  series_metadata['group_start_time']
                
            # nb random triggers per event
            nb_rand_trig_per_event =  int(
                round(trace_length_sec/random_length_sec)
            )
            if nb_rand_trig_per_event<1:
                nb_rand_trig_per_event = 1

            # number of samples that will be used for random sample,
            # taking into account that we will need to add space between
            # randoms (as well as space at the beginning/end trace)
            min_separation_samples = int(
                ceil(sample_rate*min_separation_sec)
            )

            edge_exclusion_samples = int(
                ceil(sample_rate*edge_exclusion_sec)
            )
            
            nb_samples_reduced = (
                nb_samples - 2*edge_exclusion_samples -(
                    (nb_rand_trig_per_event-1)*min_separation_samples
                )
            )
            
            # build list with samples  
            samples_list =  list(range(nb_samples_reduced))
                                     
            # Fraction of events that will be needed
            #  1: find randoms every events
            # <1: finds randoms every 1/"event_fraction" events
            event_fraction = 1
            if random_length_sec>trace_length_sec:
                event_fraction = trace_length_sec/random_length_sec

         
            # loop files
            current_event_time = None
            trigger_id = 0
            total_event_counter = 0
        
            for file_name in series_dict[series]:

                # initialize feature dictionary
                feature_dict = {'series_number': list(),
                                'event_number': list(),
                                'dump_number': list(),
                                'event_time': list(),
                                'series_start_time': list(),
                                'group_start_time': list(),
                                'fridge_run_start_time': list(),
                                'fridge_run_number':list(),
                                'trigger_index': list(),
                                'trigger_time': list(),
                                'trigger_type': list(),
                                'data_type': list(),
                                'group_name':list(),
                                'processing_id':list(),
                                'trigger_prod_id': list(),
                                'trigger_prod_group_name':list()}
                
                # get file metadata 
                h5reader = h5io.H5Reader()
                metadata = h5reader.get_metadata(file_name,
                                                 include_dataset_metadata=True)

                

                # find ADC id 
                if 'adc_list' not in metadata.keys():
                    raise ValueError(
                        'ERROR: unrecognized file format for file : '
                        + afile)

                adc_id = metadata['adc_list'][0]
                metadata_adc = metadata['groups'][adc_id]
                
                # nb of events in file
                nb_events = 0
                if 'nb_events' in  metadata_adc.keys():
                    nb_events = metadata_adc['nb_events']
                elif  'nb_datasets' in  metadata_adc.keys():
                    nb_events = metadata_adc['nb_datasets']
                else:
                    raise ValueError('ERROR: Unknow file format. Unable '
                                     'to get number of events')
                    
                total_event_counter += nb_events

                # number of randoms requested
                nb_random_events = int(round(nb_events*event_fraction))
        
                if nb_random_events == 0:
                    nb_random_events = 1
                    #print('WARNING: Modifying random rate to have a least '
                    #      ' one event per dump! To be fixed soon...')

                # event list (all events continuous data)
                # and random event list
                event_list = list(range(1, nb_events+1))
                rand_event_list = event_list.copy()
                if nb_random_events<nb_events:
                    rand_event_list = random.sample(event_list, nb_random_events)
                    rand_event_list.sort()
                    
                # loop all events
                for evend_id in event_list:

                    dataset_metadata = (
                        metadata_adc['datasets']['event_' + str(evend_id)]
                    )

                    event_num = dataset_metadata['event_num']
                    event_time = dataset_metadata['event_time']

                    if (current_event_time is None
                        or event_time>current_event_time):
                        current_event_time = event_time
                    else:
                        current_event_time += trace_length_sec
                                                              
                    # continue loop if  event not needed
                    if evend_id not in rand_event_list:
                        continue
                                    
                    # randomly pick trigger from indices
                    trigger_indices = np.array(
                        random.sample(samples_list,
                                      nb_rand_trig_per_event)
                    )
                    
                    trigger_indices = np.sort(trigger_indices)

                    # add min space between randoms
                    trigger_indices = trigger_indices + (
                        edge_exclusion_samples + (
                            np.arange(nb_rand_trig_per_event)
                            * min_separation_samples
                        )
                    )
                    
                    # loop triggers and fill dataframe
                    for trigger_index in trigger_indices:

                        # increment trigger id
                        trigger_id +=1

                        # trigger time
                        trigger_time = trigger_index/sample_rate
                        event_time_trigger = int(
                            round(current_event_time+trigger_time))
                                          
                        # fill dictionary                        
                        feature_dict['series_number'].append(int(series_num))
                        feature_dict['event_number'].append(int(event_num))
                        feature_dict['dump_number'].append(int(metadata['dump_num']))
                        feature_dict['event_time'].append(event_time_trigger)
                        feature_dict['series_start_time'].append(
                            event_time_trigger-series_start_time
                        )
                        feature_dict['group_start_time'].append(
                            event_time_trigger-group_start_time
                        )
                        feature_dict['fridge_run_start_time'].append(
                            event_time_trigger-fridge_run_start_time
                        )
                        feature_dict['trigger_index'].append(int(trigger_index))
                        feature_dict['trigger_time'].append(trigger_time)
                        feature_dict['trigger_type'].append(3)
                        feature_dict['data_type'].append(metadata['run_type'])
                        feature_dict['fridge_run_number'].append(int(metadata['fridge_run']))
                        feature_dict['trigger_prod_id'].append(trigger_id)
                        feature_dict['trigger_prod_group_name'].append(trigger_prod_group_name)
                        feature_dict['group_name'].append(metadata['group_name'])
                        processing_id = np.nan
                        if self._processing_id is not None:
                            processing_id = self._processing_id
                        feature_dict['processing_id'].append(processing_id)
                      
                # convert to vaex
                df = vx.from_dict(feature_dict)

                # concatenate
                if process_df is None:
                    process_df = df
                else:
                    process_df = vx.concat([process_df, df])

                # close file
                h5reader.close()

                                
            # save file
            if (lgc_save
                and (nb_series==series_count+1
                     or memory_limit_reached)):
                
                # build hdf5 file name
                dump_str = str(dump_counter)
                file_name =  (output_base_file + '_F' + dump_str.zfill(4)
                              + '.hdf5')

                # check if nrandoms provided
                if (self._nrandoms is not None
                    and len(process_df)>self._nrandoms):
                    process_df = process_df.sample(n=self._nrandoms)
                
                # export
                process_df.export_hdf5(file_name, mode='w')
                        
                # increment dump
                dump_counter += 1

                # delete df
                if memory_limit_reached:
                    del process_df
                    process_df = None
                    
            # case memory limit reached
            # -> processing needs to stop!
            if lgc_output and memory_limit_reached:
                raise ValueError(
                    'ERROR: memory limit reached! '
                    + 'Change memory limit or only save hdf5 files '
                    +'(lgc_save=True AND lgc_output=False) '
                )
                                       
        return process_df
            
    
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
        series_list = list(self._series_dict.keys())
        series_split = np.array_split(series_list, ncores)
      
        # remove empty array
        for series_sublist in series_split:
            if series_sublist.size == 0:
                continue
            output_list.append(list(series_sublist))
            

        return output_list

    
    def _create_output_directory(self, base_path, facility,
                                 output_group_name=None,
                                 restricted=False, calib=False):
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
    
        calib : boolean
          if True,  create directory name that includes "calib"

        Return
        ------
          output_dir : str
            full path to created directory

        """


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
                os.makedirs(output_dir)
                os.chmod(output_dir, stat.S_IRWXG | stat.S_IRWXU | stat.S_IROTH | stat.S_IXOTH)
            except OSError:
                raise ValueError('\nERROR: Unable to create directory "'+ output_dir  + '"!\n')
    
        return output_dir, series_num
        
  
    
