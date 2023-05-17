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

__all__ = [
    'Randoms'
]



class Randoms:
    """
    Class to manage acquisitions of randoms from 
    continuous data. The randoms metadata are 
    saved in vaex dataframe for further processing.
    """

    def __init__(self, raw_path, series=None,
                 verbose=True):
        """
        Initialize randoms acquisition
        
        Parameters
        ---------
        
        raw_path : str or list of str 
           raw data group directory OR full path to HDF5  file 
           (or list of files). Only a single raw data group 
           allowed 
            
        series : str or list of str, optional
            series to be process, disregard other data from raw_path

        verbose : bool, optional
            if True, display info

        Return
        ------
        None

        """

        # display
        self._verbose = verbose

      
        # extract input file list
        input_file_dict, input_base_path, group_name, facility = (
            self._get_file_list(raw_path, series=series)
        )

        if not input_file_dict:
            raise ValueError('No files were found! Check configuration...')

        self._input_base_path = input_base_path
        self._series_dict =input_file_dict
        self._group_name = group_name
        self._facility = facility
    

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
        return self._group_name




    
    def acquire(self, random_rate,
                min_separation_msec=100,
                output_path=None,
                ncores=4,
                lgc_save=True,
                lgc_output=False):
        
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


        # minimum separation (convert to seconds)
        min_separation_sec = min_separation_msec/1000
        
        # average time between randoms
        random_length_sec = 1/random_rate
        if random_length_sec<min_separation_sec:
            print('ERROR: Unable to have a minimum separation '
                  + 'of ' + str(min_separation_msec) + 'ms '
                  + 'between randoms for the requested rate. '
                  + 'Please, change "min_separation_msec" value!')
            
            return

        # If rate is low, we can increase minimum seperation
        # (up to 50% time between randoms) 
        if (random_length_sec/2>min_separation_sec):
            min_separation_sec = random_length_sec/2


        # create output directory
        output_processing_path = None
        
        if lgc_save:
            
            if output_path is None:
                output_path  = self._input_base_path
                dir_name = str(Path(output_path).name)
                if dir_name == 'raw':
                    output_path = str(Path(output_path).parent)

                output_path += '/processed'
                           
            if self._group_name not in output_path:
                output_path += '/' + self._group_name

                                    
            output_processing_path = self._create_output_directory(
                output_path,
                self._facility
            )
            
            if self._verbose:
                print(f'INFO: Processing output group path: {output_processing_path}')
            
           
                     

        # initialize output
        output_df = None
        
        # case only 1 node used for processing
        if ncores == 1:
            series_list = list(self._series_dict.keys())
            output_df = self._acquire(-1,
                                      series_list,
                                      random_length_sec,
                                      min_separation_sec,
                                      output_processing_path,
                                      lgc_save,
                                      lgc_output)
        else:
            
            # split data
            series_list_split = self._split_series(ncores)
            
            # lauch pool processing
            if self._verbose:
                print(f'INFO: Processing with be split between {ncores} cores!')

            node_nums = list(range(ncores+1))[1:]

             
            #pool = Pool(processes=ncores)

            with  Pool(processes=ncores) as pool:
                output_df_list = pool.starmap(self._acquire,
                                              zip(node_nums,
                                                  series_list_split,
                                                  repeat(random_length_sec),
                                                  repeat(min_separation_sec),
                                                  repeat(output_processing_path),
                                                  repeat(lgc_save),
                                                  repeat(lgc_output))
                )
                pool.close()
                pool.join()

            # concatenate
            if lgc_output:
                output_df = vx.concat(output_df_list)
       

            
        # processing done
        if self._verbose:
            print('INFO: Randoms acquisition done!') 

            
        return output_df
                
       
    
    def _acquire(self, node_num,
                 series_list,
                 random_length_sec,
                 min_separation_sec,
                 output_processing_path,
                 lgc_save,
                 lgc_output):
        """
        Acquire random trigger using specified rate (and minimum
        separation)
        """

        # node string (for display)
        node_num_str = str()
        if node_num>-1:
            node_num_str = ' node #' + str(node_num)


        # trigger prod group name
        trigger_prod_group_name = np.nan
        if output_processing_path is not None:
            trigger_prod_group_name = (
                str(Path(output_processing_path).name)
            )


            
        
        # set output dataframe to None
        # only used if dataframe returned
        output_df = None
        
        # loop series
        for series  in series_list:


            if self._verbose:
                print('INFO' + node_num_str
                      + ': Acquiring randoms for series '
                      + series)

            # set series dataframe to None
            series_df = None

            # series num
            series_num = self._series_dict[series]['series_num']
                 


            # trigger file name
            trigger_prod_file_name = np.nan
            if lgc_save:
                trigger_prod_file_name = ('rand_vaex_'
                                     + series + '.hdf5')
                

            
            # trace length in second:
            sample_rate = self._series_dict[series]['sample_rate']
            nb_samples = self._series_dict[series]['nb_samples']
            trace_length_sec = nb_samples/sample_rate

        
            # timing
            fridge_run_start_time = self._series_dict[series]['fridge_run_start_time']
            series_start_time = self._series_dict[series]['series_start_time']
            group_start_time =  self._series_dict[series]['group_start_time']
            
            # nb random triggers per event
            nb_random_triggers_per_event =  int(
                round(trace_length_sec/random_length_sec)
            )
            if nb_random_triggers_per_event<1:
                nb_random_triggers_per_event = 1

            # build list of trigger indices
            nb_samples_per_chunk = nb_samples
            if nb_random_triggers_per_event>1:
                nb_samples_per_chunk = int(
                    round(nb_samples/nb_random_triggers_per_event)
                )

            trigger_indices = list(
                range(round(nb_samples_per_chunk/2), nb_samples,
                      nb_samples_per_chunk)
            )
                
                
            # number of samples we will randomly choose trigger time
            # from center of the random chunk (to avoid to be phase locked)
            
            nb_samples_from_center = int(
                floor(sample_rate*min_separation_sec/2)
            )
            
                        
            # Fraction of events that will be needed
            #  1: find randoms every events
            # <1: finds randoms every 1/"event_fraction" events
            event_fraction = 1
            if random_length_sec>trace_length_sec:
                event_fraction = trace_length_sec/random_length_sec

            # loop files
            current_event_time = None
            nb_same_event_time = 0
            trigger_id = 0
            total_event_counter = 0
            for file_name in self._series_dict[series]['files']:


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
                                'trigger_id': list(),
                                'trigger_prod_id': list(),
                                'trigger_prod_group_name':list(),
                                'trigger_prod_file_name':list()}

                
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
                nb_events = metadata_adc['nb_events']
                total_event_counter += nb_events

                # number of randoms requested
                nb_random_events = int(round(nb_events*event_fraction))
                if nb_random_events == 0:
                    nb_random_events = 1
                    print('WARNING: Modifying random rate to have a least '
                          ' one event per dump!')

                # event list (continuous data) and trigger list
                event_list = list(range(1, nb_events+1))
                trigger_list = event_list.copy()
                if nb_random_events<nb_events:
                    trigger_list = random.sample(event_list, nb_random_events)
                    trigger_list.sort()
                    
                # loop events
                for evend_id in event_list:

                    dataset_metadata = (
                        metadata_adc['datasets']['event_' + str(evend_id)]
                    )

                    event_num = dataset_metadata['event_num']
                    event_time = dataset_metadata['event_time']

                    # check if same event time as previous event
                    if (current_event_time is not None
                        and event_time == current_event_time):
                        nb_same_event_time +=1
                    else:
                        nb_same_event_time = 0

                    # save current event time
                    current_event_time = event_time
                  
                    # skip event if needed
                    if evend_id not in trigger_list:
                        continue

                    event_time_adjusted = (
                        event_time + nb_same_event_time*trace_length_sec
                    )
                                    
                    # loop triggers
                    for ind in trigger_indices:

                        # increment trigger id
                        trigger_id +=1

                        # randomly select trigger index around
                        # center
                        index_range = list(
                            range(ind-nb_samples_from_center,
                                  ind+nb_samples_from_center)
                        )

                        trigger_index = random.choice(index_range)
                        trigger_time = trigger_index/sample_rate
                        event_time_trigger = int(round(event_time_adjusted + trigger_time))
                                          
                        # fill dictionary                        
                        feature_dict['series_number'].append(series_num)
                        feature_dict['event_number'].append(event_num)
                        feature_dict['dump_number'].append(metadata['dump_num'])
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
                        feature_dict['trigger_index'].append(trigger_index)
                        feature_dict['trigger_time'].append(trigger_time)
                        feature_dict['trigger_type'].append(3)
                        feature_dict['data_type'].append(metadata['run_type'])
                        feature_dict['fridge_run_number'].append(metadata['fridge_run'])
                        feature_dict['trigger_id'].append(trigger_id)
                        feature_dict['trigger_prod_id'].append(trigger_id)
                        feature_dict['trigger_prod_group_name'].append(trigger_prod_group_name)
                        feature_dict['trigger_prod_file_name'].append(trigger_prod_file_name)
                        feature_dict['group_name'].append(metadata['group_name'])
                      
                # convert to vaex
                df = vx.from_dict(feature_dict)

                # concatenate
                if series_df is None:
                    series_df = df
                else:
                    series_df = vx.concat([series_df, df])

                # close file
                h5reader.close()



            # display rate
            if self._verbose:
                nb_events = len(series_df)
                rate = nb_events/(trace_length_sec*total_event_counter)
                print('INFO' + node_num_str
                      + ': Randoms acquisition for ' + series
                      + ' done! Final rate = '
                      + str(rate)
                      + ' Hz')
                


                
            # save file
            if lgc_save:

                output_file = (
                    output_processing_path
                    + '/'
                    + trigger_prod_file_name
                )

                # export
                series_df.export_hdf5(output_file,
                                      mode='w')
                print('INFO ' + node_num_str
                      + ': Saving vaex dataframe in '
                      + output_file)
                
            # output dataframe
            if lgc_output:
                
                # concatenate
                if output_df is None:
                    output_df = series_df
                else:
                    output_df = vx.concat([output_df, series_df])
            else:
                del series_df
                
                                   
        return output_df
            


    
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
        prefix = 'randoms'
        output_dir = base_path + '/' + prefix + '_' + series_name
        
        
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir)
                os.chmod(output_dir, stat.S_IRWXG | stat.S_IRWXU | stat.S_IROTH | stat.S_IXOTH)
            except OSError:
                raise ValueError('\nERROR: Unable to create directory "'+ output_dir  + '"!\n')
                
        return output_dir
        


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
        facility = None
        
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
                            file_list.extend(
                                glob(a_path + '/' + file_name_wildcard))
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
        
                if file_name not in series_dict[series_name]['files']:
                    series_dict[series_name]['files'].append(file_name)
                    file_counter += 1
                continue
            
            # get metadata
            metadata = h5reader.get_metadata(file_name)
        
            # get series name
            series_num = metadata['series_num']
            series_name = h5io.extract_series_name(series_num)
            if series_name not in series_dict.keys():
                series_dict[series_name] = dict()
                series_dict[series_name]['files'] = list()
                series_dict[series_name]['series_num'] = series_num
            
                
            facility = metadata['facility']
                
            # append file
            if file_name not in series_dict[series_name]:
                series_dict[series_name]['files'].append(file_name)
                file_counter += 1
                
            # get other ADC info
            if 'adc_list' not in metadata.keys():
                raise ValueError(
                    'ERROR: unrecognized file format!'
                )

            
            adc_id = metadata['adc_list'][0]
            metadata_adc = metadata['groups'][adc_id]
            series_dict[series_name]['sample_rate'] = metadata_adc['sample_rate']
            series_dict[series_name]['nb_samples'] = metadata_adc['nb_samples']
        
            # time since start of run
            fridge_run_start_time = np.nan
            if 'fridge_run_start' in metadata:
                fridge_run_start_time = metadata['fridge_run_start']
            elif 'fridge_run_start_time' in metadata:
                fridge_run_start_time = metadata['fridge_run_start_time']

            series_dict[series_name]['fridge_run_start_time'] = fridge_run_start_time

            # time since start of series
            if 'series_start_time' in metadata:
                series_dict[series_name]['series_start_time'] = metadata['series_start_time']
            else:
                series_dict[series_name]['series_start_time'] = np.nan

            # time since start of group
            if 'group_start_time' in metadata:
                series_dict[series_name]['group_start_time'] = metadata['group_start_time']
            else:
                series_dict[series_name]['group_start_time'] = np.nan

            # close
            h5reader.close()
                
                
        if self._verbose:
            print('INFO: Found total of '
                  + str(file_counter)
                  + ' files from ' + str(len(series_dict.keys()))
                  + ' different series number!')

      
        return series_dict, base_path, group_name, facility


