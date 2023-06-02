import numpy as np
import os
import vaex as vx
import pandas as pd

class EventBuilder:
    """
    Class for storing trigger data from single continuous trace, 
    finding coincident event,  and constructing event(s) information
    """

    def __init__(self, fs):
        """
        Intialization
        """

        # sample rate
        self._fs = fs

        # Initialize trigger data 
        self._trigger_df = None
        self._trigger_channels = list()

        
    
    def add_trigger_data(self, trigger_channel, trigger_data):
        """
        Add trigger data dictionary for a specific
        trigger channel
        """

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
        
        
             

    def merge_coincident(self,
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

        # coincident window
        if (coincident_window_msec is None
            and coincident_window_samples is None):
            print('WARNING: No coincident window defined. '
                  + 'No merging  will be done')
            return
        
        # merge window
        merge_window = 0
        if coincident_window_msec is not None:
            merge_window = int(coincident_window_msec*self._fs/1000)
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
        lgc_coincident = np.diff(trigger_indices) <= merge_window
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
               


        if column_inds_to_drop:
            df_pandas = df_pandas.drop(column_inds_to_drop)
        self._trigger_df = vx.from_pandas(df_pandas, copy_index=False)
                
            
            
        
                
                
                
            
                
        
            
        

