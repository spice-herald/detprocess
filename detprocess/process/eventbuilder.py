import numpy as np
import os
import vaex as vx
import pandas as pd

class EventBuilder:
    """
    Class for storing trigger data from single continuous trace, 
    finding coincident event,  and constructing event(s) information
    """

    def __init__(self):
        """
        Intialization
        """

        # trigger id
        self._current_trigger_id = 0
        self._current_event_time = 0
        

        # Initialize trigger data 
        self._trigger_df = None
        self._trigger_objects = None
        self._trigger_channels = None

    def clear_trigger_data(self):
        """
        clear 
        """
        self._trigger_df = None
        self._trigger_channels = None

        # FIXME clear object...
        
               
    def add_trigger_object(self, trigger_channel, trigger_object):
        """
        Add trigger object 
        """

        if self._trigger_objects is None:   
            self._trigger_objects = dict()


        if trigger_channel in self._trigger_objects.keys():
            raise ValueError(
                'ERROR: Trigger object "' + trigger_channel
                + ' already stored!')

        # store in dictionary
        self._trigger_objects[trigger_channel] = trigger_object


        
    
    def add_trigger_data(self, trigger_channel, trigger_data):
        """
        Add trigger data dictionary for a specific
        trigger channel
        """

        # intialize if needed
        if self._trigger_channels is None:
            self._trigger_channels = list()

        
        # check if trigger channel already saved
        if trigger_channel in self._trigger_channels:
            raise ValueError('ERROR: Trigger data for channel '
                             + trigger_channel + ' already added!')
            
        # add to list 
        self._trigger_channels.append(trigger_channel)

        # dataframe 
        if self._trigger_df is None:
            self._trigger_df = trigger_data
        else:
            self._trigger_df = vx.concat(
                [self._trigger_df, trigger_data]
            )

        # sort by trigger index
        self._trigger_df = self._trigger_df.sort('trigger_index')


        
    def acquire_trigger(self, trigger_channel, trace, thresh,
                        merge_window_msec=None, merge_window_samples=None,
                        positive_pulses=True):
        """
        calc
        """

        # find trigger object
        if trigger_channel not in self._trigger_objects.keys():
            raise ValueError(
                'ERROR: Trigger object ' + trigger_channel
                + ' not found!')

        trigger_obj = self._trigger_objects[trigger_channel]

        # update trace
        trigger_obj.update_trace(trace)

        # find triggers
        trigger_obj.find_triggers(
            thresh,
            merge_window_msec=merge_window_msec,
            merge_window_samples=merge_window_samples,
            positive_pulses=positive_pulses)

        # store trigger data
        df = trigger_obj.get_trigger_data_df()
        self.add_trigger_data(trigger_channel, df)
        
        

    def build_event(self, event_metadata=None,
                    coincident_window_msec=None,
                    coincident_window_samples=None):
        """
        Function to merge coincident 
        events based on user defined window (in msec or samples)
        """

        # metadata
        if event_metadata is None:
            event_metadata = dict()
        
        # sample rate
        fs = None
        if 'sample_rate' in event_metadata.keys():
            fs = event_metadata['sample_rate']

        # trace length in seconds
        trace_length_sec = 0
        if (fs is not None
            and 'nb_samples' in event_metadata.keys()):
            trace_length_sec = event_metadata['nb_samples']/fs
            
        
        # event time
        event_time_start = np.nan
        event_time_end = np.nan
        if 'event_time' in event_metadata.keys():
            event_time_data = event_metadata['event_time']
            if (event_time_data>=self._current_event_time):
                event_time_start =  event_time_data
            else:
                event_time_start = self._current_event_time
            event_time_end = int(event_time_start + trace_length_sec)

        # store event time
        self._current_event_time = event_time_end
        
        # check if any triggers
        if (self._trigger_df is None or len(self._trigger_df)==0):
            return
    
        # merge coincident events
        self._merge_coincident_triggers(
            coincident_window_msec=coincident_window_msec,
            coincident_window_samples=coincident_window_samples)


        # number of triggers (after merging coincident events)
        nb_triggers = len(self._trigger_df)

        #  add metadata
        default_val = np.array([np.nan]*nb_triggers)
        metadata_dict = {'series_number': default_val,
                         'event_number': default_val,
                         'dump_number': default_val,
                         'series_start_time': default_val,
                         'group_start_time': default_val,
                         'fridge_run_start_time': default_val,
                         'fridge_run_number': default_val,
                         'data_type': default_val,
                         'group_name':default_val}

        # replace value if available
        for key in metadata_dict.keys():
            if key in event_metadata.keys():
                metadata_dict[key] = np.array(
                    [event_metadata[key]]*nb_triggers)

        # some parameters have different names
        if 'series_num' in event_metadata.keys():
            metadata_dict['series_number'] = np.array(
                [event_metadata['series_num']]*nb_triggers)    
        if 'event_num' in event_metadata.keys():
            metadata_dict['event_number'] = np.array(
                [event_metadata['event_num']]*nb_triggers)    
        if 'dump_num' in event_metadata.keys():
            metadata_dict['dump_number'] = np.array(
                [event_metadata['dump_num']]*nb_triggers)    
        if 'run_type' in event_metadata.keys():
            metadata_dict['data_type'] = np.array(
                [event_metadata['run_type']]*nb_triggers)   
        if 'fridge_run' in event_metadata.keys():
            metadata_dict['fridge_run_number'] = np.array(
                [event_metadata['fridge_run']]*nb_triggers)   

                
        # event times
        trigger_times = self._trigger_df['trigger_time'].values
        event_times = trigger_times + event_time_start
        event_times_int = np.around(event_times).astype(int)

        # add new parameters in dictionary
        metadata_dict['event_time'] = event_times_int
        metadata_dict['series_start_time'] = (
            event_times_int-metadata_dict['series_start_time'])
        metadata_dict['group_start_time'] = (
            event_times_int-metadata_dict['group_start_time'])
        metadata_dict['fridge_run_start_time'] = (
            event_times_int-metadata_dict['fridge_run_start_time'])

        # trigger id
        metadata_dict['trigger_id'] = (
            np.array(range(nb_triggers))
            + self._current_trigger_id
            + 1)
        
        self._current_trigger_id = metadata_dict['trigger_id'][-1]
        
        
        # add to dataframe
        for key,val in metadata_dict.items():
            self._trigger_df[key] = val
            
            
        

                
    def _merge_coincident_triggers(self, fs=None,
                                  coincident_window_msec=None,
                                  coincident_window_samples=None):
        """
        Function to merge coincident 
        events based on user defined window (in msec or samples)
        """
        
        # check
        if self._trigger_df is None:
            raise ValueError('ERROR: No trigger data '
                             + 'available')  
        

        # check if any triggers
        if len(self._trigger_df)==0:
            return
                
        # merge window
        merge_window = 0
        if coincident_window_msec is not None:
            if fs is None:
                raise ValueError(
                    'ERROR: sample rate "fs" needs to be provided!'
                )
            merge_window = int(coincident_window_msec*fs/1000)
        elif coincident_window_samples is not None:
            merge_window = coincident_window_samples
            

        # let's convert vaex dataframe to pandas so we can modify it
        # more easily
        df_pandas = self._trigger_df.to_pandas_df()
        
            
        # get trigger index and amplitude
        trigger_indices =  np.array(df_pandas['trigger_index'].values)
        trigger_amplitudes =  np.array(df_pandas['trigger_amplitude'].values)
        trigger_channels = np.array(df_pandas['trigger_channel'].values)
    
  
        # find list of indices within merge_window
        # then store in list of index ranges
        lgc_coincident = np.diff(trigger_indices) < merge_window
        lgc_coincident = np.concatenate(([0], lgc_coincident, [0]))
        lgc_coincident_diff = np.abs(np.diff(lgc_coincident))
        coincident_ranges = np.where(lgc_coincident_diff == 1)[0].reshape(-1, 2)

      
        # let's first loop through ranges
        # then
        #  - disregard if only one channel (=pileups)
        #  - further split if combination of pile-ups and coincidents
        # then save indices
        coincident_indices = list()
        for range_it in coincident_ranges:
            indices = np.arange(range_it[0], range_it[1]+1)
            channels = trigger_channels[indices]
            channels_unique = np.unique(channels)
            
            # case single channel -> pileup
            if len(channels_unique) == 1:
                continue

            # case only coincident
            if len(channels_unique) == len(channels):
                coincident_indices.append(indices)
                
            # case mix coincident/pileup -> split
            
            # let's do something simple. Group indices
            # in function of time so each sublist have
            # single channels

            if len(channels_unique) < len(channels):

                # initialize list split
                indices_split = list()

                # loop channels
                current_chan_list  = list()
                current_ind_list = list()
                for chan_it in range(len(channels)):
                    
                    chan = channels[chan_it]
                    chan_ind = indices[chan_it]

                    if chan in current_chan_list:
                        # save current indices
                        indices_split.append(current_ind_list)
                        
                        # reset list
                        current_chan_list  = list()
                        current_ind_list = list()

                    # add to list
                    current_chan_list.append(chan)
                    current_ind_list.append(chan_ind)
    
        

                # save last iteration
                if current_ind_list:
                    indices_split.append(current_ind_list)

                # loop indices  and save if not single value
                for inds in indices_split:
                    # if single channel -> not coincident
                    if len(inds)==1:
                        continue
                    else:
                        coincident_indices.append(inds) 
                                            
                        
        # Loop coincident indices then
        #  - find the one with maximum amplitude -> primary trigger
        #  - merge other channels trigger value to primary and remove row
        column_inds_to_drop = list()
        for inds in coincident_indices:

            # amplitudes
            amps = trigger_amplitudes[inds]
            max_index = amps.argmax()
                     
            # primary trigger
            primary_index = int(inds[max_index])
            primary_channel = trigger_channels[primary_index]
                    
            # other triggers 
            other_indices = inds[inds!=primary_index]
            other_channels = trigger_channels[other_indices]

            if not isinstance(other_indices, np.ndarray):
                other_indices = [other_indices]
                other_channels = [other_channels]
            
     
            # loop other channels and add trigger specific parameters
            # into primary channel row
            for other_it in range(len(other_indices)):

                other_index = int(other_indices[other_it])
                other_chan = str(other_channels[other_it])
                
                # find channel specific columns
                column_names = np.array(
                    df_pandas.columns[df_pandas.iloc[other_index].notnull()]).astype('U')
             
                matching_elements = np.char.find(column_names, other_chan) >= 0
                column_names = column_names[matching_elements]

                # replace values
                for column_name in column_names:
                    df_pandas[column_name][primary_index] =  df_pandas[column_name][other_index]

                # add to drop list
                column_inds_to_drop.append(other_index)
               

        # drop rows non primary trigger channels 
        if column_inds_to_drop:
            df_pandas = df_pandas.drop(column_inds_to_drop)

        # convert back to vaex
        self._trigger_df = vx.from_pandas(df_pandas, copy_index=False)
                
            
            
        
                
                
                
            
                
        
            
        

