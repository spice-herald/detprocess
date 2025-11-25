import os
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import yaml
import copy
from yaml.loader import SafeLoader
import re
from pprint import pprint
from pytesdaq.io import convert_length_msec_to_samples
from qetpy.utils import convert_channel_name_to_list, convert_channel_list_to_name
import vaex as vx
from datetime import datetime
import stat
from glob import glob


__all__ = ['split_channel_name', 'extract_window_indices',
           'find_linear_segment', 'create_directory', 'create_series_name',
           'get_dataframe_series_list', 'get_ind_freq_ranges',
           'is_empty','unique_list','estimate_sampling_rate' ,'find_psd_peaks',
           'get_trigger_template_info']


    
def create_series_name(facility):
    """
    Create output directory 
    
    Parameters
    ----------
    
    facility : int
       facility number
    
    
    Return
    ------
          
    name : str
        
    """

    now = datetime.now()
    series_day = now.strftime('%Y') +  now.strftime('%m') + now.strftime('%d') 
    series_time = now.strftime('%H') + now.strftime('%M')
    series_name = ('I' + str(facility) +'_D' + series_day + '_T'
                   + series_time + now.strftime('%S'))
    
    return series_name
        

def create_directory(directory_path):
    """
    create (sub) directory
    """

    if not os.path.isdir(directory_path):
        try:
            os.makedirs(directory_path)
            os.chmod(directory_path, stat.S_IRWXG | stat.S_IRWXU | stat.S_IROTH | stat.S_IXOTH)
        except OSError:
            raise ValueError('\nERROR: Unable to create directory "'+ directory_path  + '"!\n')
        
        


def split_channel_name(channel_name,
                       available_channels=None,
                       separator=None,
                       label=None):
    """
    Split channel name after various checks and return
    list of individual channels and separator(s)
    """
    
    # allowed separators
    allowed_separators = [',', '|', '+' ,'-']

    # strip whitespace
    channel_name = channel_name.replace(' ','')

    # check if separator allowed
    if (separator is not None
        and separator not in allowed_separators):
        raise ValueError(
            f'ERROR: separator "{separator}" not '
            f'recognized. Allowed separator '
            f'{allowed_separators} ')
    
    # check if channel_name has any separators
    has_separator = False
    for sep in allowed_separators:
        if sep in channel_name:
             has_separator = True
             break
    if not has_separator:
        return [channel_name], None

    
    # case available_channels is None
    if  available_channels is None:

        if separator is None:
            raise ValueError(
                'ERROR: separator required when '
                '"available_channels" not provided! ')

        if separator == '-':
            raise ValueError(
                'ERROR: "available_channels" required '
                'when using separator "-"')

        if ((separator == '-' or separator == '+')
              and (',' in channel_name or '|' in channel_name)):
            raise ValueError(
                f'ERROR: Channels cannot be split with '
                f'{separator} before channels split with '
                f'"," and "|"')
    
        channel_list = channel_name.split(separator)
        return channel_list, separator

    # from this point available_channels has been provided

    # case already an individual channel
    # or no separator found
    if (channel_name in available_channels
        or channel_name == 'all'):
        return [channel_name], None

    # get list of separators
    channel_check = channel_name
    channel_list = []
    for chan in available_channels:
        if chan in channel_check:
            channel_check = channel_check.replace(chan, '')
            channel_list.append(chan)
            
    separator_list =  [x for x in channel_check]
    separator_list =  list(set(separator_list))
       
    # check if any channels are  unavailable
    non_separator_list = []
    for sep in separator_list:
        if sep not in allowed_separators:
            non_separator_list.append(sep)
    if non_separator_list:
        raise ValueError(
            f'ERROR: Unidentified channel "{channel_name}" in yaml file! '
            f'Perhaps not in raw data? Available channels = {available_channels}')

    # if no separator 
    if separator is None:
        
        if len(separator_list) == 1:
            separator_list = separator_list[0]
            if separator_list != '-':
                channel_list = channel_name.split(separator_list)
                
        return channel_list, separator_list

    # case separator provided
    
    # check if separator in channe_name
    if separator not in channel_name:
        return [channel_name], None

    # case not '-'
    if separator != '-':
        channel_list = channel_name.split(separator)
        return channel_list.copy(), separator


    if ('|' in channel_name
        or '+' in channel_name
        or ',' in channel_name):

        raise ValueError(f'Multiple separators available, split first '
                         f'with other separators before "-"')
    else:
        return  channel_list.copy(), separator




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

def is_empty(param):
    """
    check if empty
    """
    
    if param is None:
        return True
    try:
        return len(param) == 0
    except TypeError:
        return False


def get_dataframe_series_list(file_path):
    """
    Get list of series of all files in data_path
        
    Parameters
    ----------

    file_path : str
       path to dataframe(s) 


    Return
    -------
    
     series_list : list of series name
    
    """
    
    # check argument
    if not os.path.isdir(file_path):
        raise ValueError('ERROR: Expecting a directory!')

    
    # initialize output
    series_list = []

    # get all files
    file_list =  glob(file_path + '/*.hdf5')
    if not file_list:
        raise ValueError(f'ERROR: No HDF5 files found in {self._raw_path}')
    
    # make unique and sort
    file_list = list(set(file_list))
    file_list.sort()
        
    # loop file
    for afile in file_list:
        aname = str(Path(afile).name)
        sep_start = aname.find('_I')
        sep_end = aname.find('_F')
        series_name = aname[sep_start+1:sep_end]
        
        if series_name not in series_list:
            series_list.append(series_name)
            
    return series_list

def unique_list(alist):
    """
    make list unique
    """

    if not isinstance(alist, (list, np.ndarray)):
        alist = [alist]
    
    seen = set()
    unique_items = []
    
    for item in alist:
        if item not in seen:
            unique_items.append(item)
        seen.add(item)
        
    return unique_items


def cleanup_freq_ranges(f_lims):
    """
    cleanup frequency range for psd_peaks, also return feature 
    base names
    """
    if not isinstance(f_lims, list):
        f_lims = [f_lims]

    # loop
    freq_ranges = list()
    range_names = list()
    for freq_range in f_lims:

        # case single number
        if (isinstance(freq_range, float)
            or isinstance(freq_range, int)):
            freq_range = [freq_range]

        f_low = abs(freq_range[0])
        if len(freq_range) == 2:
            f_high = abs(freq_range[1])
            if f_low > f_high:
                f_low, f_high = f_high, f_low
            name = f'{round(f_low)}_{round(f_high)}'
            if name not in range_names:
                freq_ranges.append([f_low, f_high])
                range_names.append(f'{round(f_low)}_{round(f_high)}')
        else:
            name = f'{round(f_low)}'
            if name not in range_names:
                freq_ranges.append([f_low])
                range_names.append(f'{round(f_low)}')

    return freq_ranges, range_names




def get_ind_freq_ranges(freq_ranges, freqs):
    """
    Return index list
    """
    
    idx_ranges = list()
    for freq_range in freq_ranges:
        f_low = abs(freq_range[0])
        ind_low = int(np.argmin(np.abs(freqs - f_low)))
        ind_high = ind_low + 1
        if len(freq_range) == 2:
            f_high =  abs(freq_range[1])
            ind_high = int(np.argmin(np.abs(freqs - f_high)))
            
        # Ensure order
        if ind_low > ind_high:
            ind_low, ind_high = ind_high, ind_low
            
        # Handle identical indices 
        if ind_low == ind_high:
            if ind_high < len(freqs) - 1:
                ind_high += 1
            elif ind_low > 0:
                ind_low -= 1
            else:
                raise ValueError("Frequency range too narrow or outside bounds.")
        idx_ranges.append([ind_low, ind_high])
        
            
    return idx_ranges


def estimate_sampling_rate(freq_array):
    """
    Estimate the sampling rate from a frequency array that may be:
      - Double-sided (fftfreq)
      - Single-sided (rfftfreq)
      
    Parameters
    ----------
    freq_array : array-like
        Array of frequencies (e.g., as returned by fftfreq or rfftfreq).
        
    Returns
    -------
    fs : float
        Estimated sampling rate.
        
    Notes
    -----
    - For a double-sided array of length N, the bin spacing is fs/N,
      and there are negative frequencies in the array.
    - For a single-sided array of length M = (N//2) + 1, the bin spacing
      is still fs/N, but the array contains only [0, fs/N, 2fs/N, ..., fs/2].
   
    """
    freq_array = np.asarray(freq_array)
    
    # Sort (in case the input is not sorted) and remove duplicates
    freq_sorted = np.unique(np.sort(freq_array))
    
    # Find the smallest positive frequency (this is our bin spacing, df)
    positive_mask = freq_sorted > 0
    if not np.any(positive_mask):
        raise ValueError("No positive frequencies found; cannot infer sampling rate.")
        
    df = freq_sorted[positive_mask][0]  # first positive frequency
    
    # Check if we have negative frequencies (i.e. double-sided)
    if freq_sorted[0] < 0:
        # Double-sided array (e.g., fftfreq)
        N = len(freq_array)
    else:
        # Single-sided array (e.g., rfftfreq)
        # For real-valued time-domain signals,
        #   length of rfftfreq array = N//2 + 1.
        # => N = 2 * (len(freq_array) - 1) if N is even
        # (also works for odd N in practice because rfftfreq definition.)
        N = 2 * (len(freq_array) - 1)
    
    fs = N * df
    return fs


def find_psd_peaks(
        f, psd,
        fmin=100.0, fmax=1000.0,
        npeaks=1,
        min_separation_hz=0.0,
        min_prominence=None):
    
    """
    Find up to `npeaks` highest peaks in a PSD between fmin and fmax.
    
    Parameters
    ----------
    f : array-like
        Frequency array (Hz).
    psd : array-like
        PSD values. Linear units by default. If `use_db=True`, provide in dB.
    fmin, fmax : float
        Frequency search band (Hz).
    npeaks : int
        Number of peaks to return (top-N by amplitude within band).
    min_separation_hz : float
        Enforce a minimum separation between detected peaks (in Hz).
    min_prominence : float or None
        Minimum prominence (same units as `Pxx`, or dB if `use_db=True`).
        Leave None to let `find_peaks` decide automatically.
     
    Returns
    -------
    peaks : list of dict
        Each dict has: 'freq', 'amplitude', 'prominence', 'left_base_freq', 'right_base_freq', 'index'
    """
    
    f = np.asarray(f)
    y = np.asarray(psd)

    # Restrict to search band
    band_mask = (f >= fmin) & (f <= fmax)
    
    if not np.any(band_mask):
        # No bins fall strictly inside the band: choose the closest bin
        # Strategy: compute distance to the interval [fmin, fmax]
        # If f[i] < fmin -> distance = fmin - f[i]
        # If f[i] > fmax -> distance = f[i] - fmax
        # If f[i] inside -> distance = 0 (but this case is already excluded here)
        distances = np.where(f < fmin, fmin - f,
                             np.where(f > fmax, f - fmax, 0.0))
        idx = np.argmin(distances)
        f_band = np.array([f[idx]])
        y_band = np.array([y[idx]])
        return [{
            "freq": float(f_band[0]),
            "amplitude": float(y_band[0]),
            "prominence": None,
            "left_base_freq": None,
            "right_base_freq": None,
            "index": int(idx),
        }]

        
    f_band = f[band_mask]
    y_band = y[band_mask]
    base = int(np.where(band_mask)[0][0])
    
    # Case only one bin in range
    if len(f_band) == 1:
        return [{
            "freq": float(f_band[0]),
            "amplitude": float(y_band[0]),
            "prominence": None,
            "left_base_freq": None,
            "right_base_freq": None,
            "index": int(base),
        }]
    
    # Frequency resolution -> convert min separation (Hz) to bins
    df = np.median(np.diff(f_band)) if len(f_band) > 1 else np.inf
    if np.isfinite(df) and min_separation_hz > 0:
        distance_bins = max(1, int(np.ceil(min_separation_hz / df)))
    else:
        distance_bins = 0   # safe default


    # Peak finding
    # Prominence is given in the same units as y_band (linear or dB)
    distance_arg = distance_bins if distance_bins >= 1 else None
    peaks_idx, props = find_peaks(y_band, prominence=min_prominence, distance=distance_arg)

    if peaks_idx.size == 0:

        # Fallback: take up to `npeaks` largest bins, optionally separated by `distance_bins`
        y_work = y_band.copy()
        picked = []

        for _ in range(min(npeaks, len(y_work))):
            j = int(np.nanargmax(y_work))
            if not np.isfinite(y_work[j]):
                break
            picked.append(j)

            # Suppress neighbors so we don;t pick adjacent bins of the same line
            if distance_bins > 0:
                lo = max(0, j - distance_bins)
                hi = min(len(y_work), j + distance_bins + 1)
                y_work[lo:hi] = -np.inf
            else:
                y_work[j] = -np.inf  # just exclude the chosen bin

        # Build results, ordered by amplitude (desc)
        results = [{
            "freq": float(f_band[j]),
            "amplitude": float(y_band[j]),
            "prominence": None,
            "left_base_freq": None,
            "right_base_freq": None,
            "index": int(base + j),
        } for j in picked]

        results.sort(key=lambda d: d["amplitude"], reverse=True)
        return results


    # Sort by height (amplitude) descending and take top-N
    sort_order = np.argsort(y_band[peaks_idx])[::-1]
    top = sort_order[:npeaks]

    results = []
    for i in top:
        idx = peaks_idx[i]
        amp = float(y_band[idx])
        freq = float(f_band[idx])
        prom = None
        left_base_freq = None
        right_base_freq = None
        if props:
            if 'prominence' in props:
                prom = float(props['prominence'][i])
            if 'left_bases' in props:
                lb = props['left_bases'][i]
                left_base_freq = float(f_band[lb])
            if 'right_bases' in  props:
                rb = props['right_bases'][i]
                right_base_freq = float(f_band[rb])
        results.append({
            'freq': freq,
            'amplitude': amp,
            'prominence': prom,
            'left_base_freq':left_base_freq,
            'right_base_freq': right_base_freq,
            'index': int(base + idx)  # index in original arrays
        })
    # Sort the returned list by frequency 
    # results.sort(key=lambda d: d['freq'])
    return results



def get_trigger_template_info(trigger_config, filter_data_inst):
    """
    Check template length/pretrigger 
    for deadtime estimate
    """

    trigger_info = dict()
    posttrigger_list = list()
    pretrigger_list = list()
    
    # loop channels

    for trigger_chan, trigger_dict in trigger_config['channels'].items():

        if not trigger_dict['run']:
            continue
            
        chan = trigger_dict['channel_name']
        template_tag = trigger_dict['template_tag']
        template, _, template_metadata =  filter_data_inst.get_template(
            chan,
            tag=template_tag,
            return_metadata=True)

        # get info
        sample_rate = template_metadata['sample_rate']
        nb_pretrigger_samples = template_metadata['nb_pretrigger_samples']
        nb_samples = template_metadata['nb_samples']
        nb_posttrigger_samples =  nb_samples - nb_pretrigger_samples

        # convert to msec
        pretrigger_length_msec = 1e3*nb_pretrigger_samples/sample_rate
        posttrigger_length_msec = 1e3*nb_posttrigger_samples/sample_rate
        trace_length_msec = 1e3*nb_samples/sample_rate


        # save
        trigger_info[trigger_chan] = {
            'nb_pretrigger_samples': nb_pretrigger_samples,
            'nb_posttrigger_samples': nb_posttrigger_samples,
            'nb_samples': nb_samples,
            'pretrigger_length_msec': pretrigger_length_msec,
            'posttrigger_length_msec': posttrigger_length_msec,
            'trace_length_msec': trace_length_msec
        }


        posttrigger_list.append(posttrigger_length_msec)
        pretrigger_list.append(pretrigger_length_msec)

        

    # find min/max
    trigger_info['min_posttrigger_length_msec'] = min(posttrigger_list)
    trigger_info['max_posttrigger_length_msec'] = max(posttrigger_list)
    trigger_info['min_pretrigger_length_msec'] = min(pretrigger_list)
    trigger_info['max_pretrigger_length_msec'] = max(pretrigger_list)
    trigger_info['min_edge_exclusion'] = min(trigger_info['min_posttrigger_length_msec'],
                                             trigger_info['min_pretrigger_length_msec'])
    trigger_info['max_edge_exclusion'] = max(trigger_info['max_posttrigger_length_msec'],
                                             trigger_info['max_pretrigger_length_msec'])
    
 
    return trigger_info
