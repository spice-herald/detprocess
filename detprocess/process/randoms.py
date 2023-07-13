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
                 processing_id=None,
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

        # processing id
        self._processing_id = processing_id
      
        # extract input file list
        input_file_dict, input_base_path, input_group_name, facility = (
            self._get_file_list(raw_path, series=series)
        )

        if not input_file_dict:
            raise ValueError('No files were found! Check configuration...')
        
        self._input_base_path = input_base_path
        self._input_group_name = input_group_name
        self._series_dict = input_file_dict
        self._facility = facility

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
                min_separation_msec=100,
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


        # minimum separation (convert to seconds)
        min_separation_sec = min_separation_msec/1000


        # random rate
        if (random_rate is not None
            and nrandoms is not None):
            print('ERROR: Use either "random_rate" or "nrandoms", '
                  + 'not both!')
            return


        # if input is number of randoms, let's calculate
        # approximately random_rate (slightly increased so
        # we end up with enough events)
        self._nrandoms = nrandoms
        if nrandoms is not None:

            # let's get approximate random rate
            nb_events = 0
            nb_samples = 0
            sample_rate = 0
            for series,file_dict in self._series_dict.items():
                nb_samples = file_dict['nb_samples']
                sample_rate = file_dict['sample_rate']
                nb_events_series = file_dict['nb_events_first_file']
                nb_files = len(file_dict['files'])
                if nb_files>1:
                    nb_files -= 1
                nb_events  += (nb_events_series*nb_files)

            # increasing nrandoms by 20%
            nrandoms_increased = float(nrandoms)*1.2
            random_rate = nrandoms_increased/(nb_events*nb_samples/sample_rate)
          

        
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
                output_df = output_df[0:1000]
            
        return output_df
                
       
    
    def _process(self, node_num,
                 series_list,
                 random_length_sec,
                 min_separation_sec,
                 output_series_num,
                 output_group_path,
                 lgc_save,
                 lgc_output,
                 memory_limit):
        """
        Acquire random trigger using specified rate (and minimum
        separation)
        """

        # node string (for display)
        node_num_str = ' node #' + str(node_num)


        # trigger prod group name
        trigger_prod_group_name = np.nan
        if output_group_path is not None:
            trigger_prod_group_name = (
                str(Path(output_group_path).name)
            )
        
        # set output dataframe to None
        # only used if dataframe returned
        process_df = None

        # output file name base (if saving data)
        output_base_file = None
        
        if lgc_save:

            file_prefix = 'rand'
            if self._processing_id is not None:
                file_prefix = self._processing_id +'_' + file_prefix 
                
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

            
            # flag memory limit reached (on a series by series basis)
            # if memory reached, save file 
            memory_usage = 0
            memory_limit_reached = False
            if process_df is not None:
                memory_usage = process_df.shape[0]*process_df.shape[1]*8
                memory_limit_reached =  memory_usage  >= memory_limit
                

                
            # series num
            series_num = self._series_dict[series]['series_num']
      
                
            # trace length in second:
            sample_rate = self._series_dict[series]['sample_rate']
            nb_samples = self._series_dict[series]['nb_samples']
            trace_length_sec = nb_samples/sample_rate
        
            # timing
            fridge_run_start_time = (
                self._series_dict[series]['fridge_run_start_time'])
            series_start_time = self._series_dict[series]['series_start_time']
            group_start_time =  self._series_dict[series]['group_start_time']
            
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
                
            nb_samples_reduced = (
                nb_samples - (
                    (nb_rand_trig_per_event+1)*min_separation_samples
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
                nb_events = metadata_adc['nb_events']
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
                    trigger_indices += (
                        min_separation_samples + (
                            np.arange(nb_rand_trig_per_event)
                            *min_separation_samples)
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
                        feature_dict['data_type'].append(int(metadata['run_type']))
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
                                 output_group_name=None):
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
        for afile in file_list:

            file_name = str(Path(afile).name)
                        
            # skip if filter file
            if 'filter' in file_name:
                continue

            # skip didv
            if 'didv_' in file_name:
                continue

            if 'treshtrig_' in file_name:
                continue

            # append file if series already in dictionary
            if (series_name is not None
                and series_name in afile
                and series_name in series_dict.keys()):
        
                if afile not in series_dict[series_name]['files']:
                    series_dict[series_name]['files'].append(afile)
                    file_counter += 1
                continue
            
            # get metadata
            metadata = h5reader.get_metadata(afile)
        
            # get series name
            series_num = metadata['series_num']
            series_name = h5io.extract_series_name(series_num)
            if series_name not in series_dict.keys():
                series_dict[series_name] = dict()
                series_dict[series_name]['files'] = list()
                series_dict[series_name]['series_num'] = series_num
            
                
            facility = metadata['facility']
                
            # append file
            if afile not in series_dict[series_name]:
                series_dict[series_name]['files'].append(afile)
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
            series_dict[series_name]['nb_events_first_file'] = metadata_adc['nb_events']



            # time since start of run
            fridge_run_start_time = np.nan
            if 'fridge_run_start' in metadata:
                fridge_run_start_time = metadata['fridge_run_start']
            elif 'fridge_run_start_time' in metadata:
                fridge_run_start_time = metadata['fridge_run_start_time']

            series_dict[series_name]['fridge_run_start_time'] = fridge_run_start_time

            # time since start of series
            if 'series_start' in metadata:
                series_dict[series_name]['series_start_time'] = metadata['series_start']
            else:
                series_dict[series_name]['series_start_time'] = np.nan

            # time since start of group
            if 'group_start' in metadata:
                series_dict[series_name]['group_start_time'] = metadata['group_start']
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


