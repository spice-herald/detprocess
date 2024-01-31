import os
import sys
import numpy as np
from scipy.optimize import curve_fit



__all__ = ['split_channel_name', 'extract_window_indices',
           'find_linear_segment']



def split_channel_name(channel_name,
                       available_channels,
                       separator=None):
    """
    Split channel name and return
    list of individual channels and separator
    """
    
    # intialize output
    channel_list = list()
        
    # case already an individual channel
    if (channel_name in available_channels
        or channel_name == 'all'
        or (separator is not None
            and separator not in channel_name)):

        channel_list.append(channel_name)
        return channel_list, None
        
    # keep copy
    channel_name_orig = channel_name
        
        
    # split
    if separator is not None:
        channel_split = channel_name.split(separator)
        for chan in channel_split:
            if chan:
                channel_list.append(chan.strip())

    else:

        # remove all known channels from string
        for chan in available_channels:
            if chan in channel_name:
                channel_list.append(chan)
                channel_name = channel_name.replace(chan, '')

        # find remaining separator
        separator_list = list()
        channel_name = channel_name.strip()
        for sep in channel_name:
            if sep not in separator_list:
                separator_list.append(sep)
        if len(separator_list) == 1:
            separator = separator_list[0]
        else:
            raise ValueError('ERROR: Multiple separators found in "'
                             + channel_name_orig + '"!"')
        

        # check if channel available in raw data
        for chan in channel_list:
            if chan not in available_channels:
                raise ValueError('ERROR: Channel "' + chan
                                 + '" does not exist in '
                                 + 'raw data! Check yaml file!')
            
    return channel_list, separator
        
   


def extract_window_indices(nb_samples,
                           nb_samples_pretrigger, fs,
                           window_min_from_start_usec=None,
                           window_min_to_end_usec=None,
                           window_min_from_trig_usec=None,
                           window_max_from_start_usec=None,
                           window_max_to_end_usec=None,
                           window_max_from_trig_usec=None):
    """
    Calculate window index min and max from various types
    of window definition
    
    Parameters
    ---------

        nb_samples : int
          total number of samples 

        nb_samples_pretrigger : int
           number of pretrigger samples

        fs: float
           sample rate

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
    
    # ------------
    # min window
    # ------------
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


    # -------------
    # max index
    # -------------
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


def find_linear_segment(x, y, tolerance=0.05):
    """
    Find linear segment within tolerance using first 3 points
    fit (distance based on standardized X and Y using 
    first 3 points mean/std). 
    """
    # check length
    if len(x)<3:
        print('WARNING: Not enough points to check linearity!')
        return []

    if len(x) != len(y):
        raise ValueError('ERROR: X and Y arrays should have same length!')
    
    # standardize data using mean/std first 3 points
    xmean = np.mean(x[:3])
    xstd = np.std(x[:3])
    x = (x - xmean) / xstd
    
    ymean = np.mean(y[:3])
    ystd = np.std(y[:3])
    y = (y - ymean) / ystd
    
    # Use only the first three points to fit a linear
    # regression line
    slope, intercept = np.polyfit(x[:3], y[:3], 1)
    
    # Calculate fitted values for all points
    y_fit = slope * x + intercept
    
    # Compute deviations for all points
    deviations = np.abs(y - y_fit)


    # get linear index list
    # the deviation for the first 3 points used for the fit
    # should be very small. Will use tolerance/10
    index_list = list()
    nb_points = len(deviations)
    for idx in range(nb_points):
        deviation = deviations[idx]
        if (idx<3 and deviation>tolerance/10):
            return []
        if deviation>tolerance:
            if nb_points>idx+1:
                if deviations[idx+1]>tolerance:
                    break
            else:
                break    
        else:
            index_list.append(idx)
        
    return index_list
