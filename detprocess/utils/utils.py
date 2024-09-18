import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import yaml
import copy
from yaml.loader import SafeLoader
import re
from pprint import pprint
from pytesdaq.io import convert_length_msec_to_samples
from qetpy.utils import convert_channel_name_to_list, convert_channel_list_to_name


__all__ = ['split_channel_name', 'extract_window_indices',
           'find_linear_segment', 'read_config']



def split_channel_name(channel_name,
                       available_channels=None,
                       separator=None):
    """
    Split channel name and return
    list of individual channels and separator
    """

    # allowed separators
    separators = [',','+','-','|']

    # case available_channels is None
    if  available_channels is None:

        if separator is None:
            raise ValueError(
                'ERROR: separator required when '
                'available_channels not provided!')
        else:
            channel_list = channel_name.split(separator)
            return channel_list, separator
                          
    # case already an individual channel
    # or no separator found
    if (channel_name in available_channels
        or channel_name == 'all'):
        return [channel_name], None
      
    # Let's first find the separator if None
    if separator is None:

        separator_check = channel_name

        # remove all channels
        for chan in available_channels:
            if chan in separator_check:
                separator_check = separator_check.replace(chan, '')
                
        separator_check = separator_check.strip()

        # convert to list
        separator_list = [x for x in separator_check]
        separator_list = list(set(separator_list))
        
        if len(separator_list) == 1:
            separator = separator_list[0]
        else:
            raise ValueError(
                f'ERROR: Multiple separators found! '
                f'Only one allowed from {separators}! '
            )


    # check separator
    if separator not in separators:
        raise ValueError(
            f'ERROR: separator "{separator}" not '
            f'recognized. Allowed separator '
            f'{separators} ')

    # check if separator in channe_name
    if separator not in channel_name:
        return [channel_name], None

    # now let's split channel name
    pattern = f"([{re.escape(separator)}])"
    split_parts = re.split(pattern, channel_name)
      
    channel_list = []
    current_name = ''
    for part in split_parts:
        
        if part in available_channels:

            # add current_name if constructed
            if current_name:
                channel_list.append(current_name)
            current_name = ''
            
            # add part ot list
            channel_list.append(part)


        elif part == separator:

            if (current_name
                and  current_name in available_channels):
                channel_list.append(current_name)
                current_name = ''     
            elif current_name:
                current_name += part
        else:
            current_name += part
        
    if current_name and  current_name in available_channels:
        channel_list.append(current_name)

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



def read_config(yaml_file, available_channels, sample_rate=None):
    """
    Read configuration (yaml) file 
    
    Parameters
    ----------

    yaml_file : str
        yaml configuraton file name (full path)

    available_channels : list
        list of all individual channels in the raw data

    sample_rate : float (optinal)
        sample rate 
 
    Return
    ------
        
    processing_config : dict 
        dictionary with  processing configuration
        
    """

    # obsolete keys
    obsolete_keys = {'nb_samples': 'trace_length_samples',
                     'nb_pretrigger_samples': 'pretrigger_length_samples'}

    # configuration types
    configuration_types = ['global', 'feature',
                           'didv', 'noise',
                           'template', 'trigger']
    
    
    # available global config
    global_parameters = ['filter_file']

    # global trigger parameters
    global_trigger_parameters = ['coincident_window_msec',
                                 'coincident_window_samples']        
    # available channel separator
    separators = [',', '+', '-', '|']

    # available channels
    if isinstance(available_channels, str):
        available_channels =  [available_channels]
                    
    # load yaml file
    yaml_dict = yaml.load(open(yaml_file, 'r'),
                          Loader=_UniqueKeyLoader)

    if not yaml_dict:
        raise ValueError('ERROR: No configuration loaded'
                         'Something went wrong...')

    if 'include' in  yaml_dict:
        include_files = yaml_dict['include']
        if isinstance(include_files, str):
            include_files = [include_files]
        for afile in include_files:
            yaml_dict.update(yaml.load(open(afile, 'r'),
                                       Loader=_UniqueKeyLoader))
        yaml_dict.pop('include')
            

        
    # let's split configuration based on know type of processing
    config_dicts = dict()
    for config_name  in configuration_types:
        
        # set to None
        config_dicts[config_name] = dict()
      
        # add if available
        if config_name in yaml_dict.keys():

            # add copy
            config_dicts[config_name] = copy.deepcopy(
                yaml_dict[config_name]
            )
            
            # remove from yaml file
            yaml_dict.pop(config_name)

    # global config based on  hard coded list
    for param in global_parameters:
        config_dicts['global'][param] = None
        if param in yaml_dict.keys():
            config_dicts['global'][param] = copy.deepcopy(
                yaml_dict[param]
            )
            yaml_dict.pop(param)
                

    # the rest of parameter are for  feature processing
    for param in  yaml_dict.keys():
        config_dicts['feature'][param] = copy.deepcopy(
            yaml_dict[param]
        )

    # rename obsolete keys
    for old_key, new_key in obsolete_keys.items():
        config_dicts = _rename_key_recursively(config_dicts, old_key, new_key)

     
    # intialize output
    processing_config = dict()
        
    # Loop configuration and check/cleanup parameters
    for config_name  in configuration_types:

        # check if there is anything available
        if not config_dicts[config_name]:
            continue
        
        # initialize  output
        processing_config[config_name] = dict()

        # dictionary
        config_dict = config_dicts[config_name]

        # global parameters
        if config_name == 'global':
            processing_config[config_name] = config_dict.copy()
            continue

        # configuration for 'all' (individual) channels
        # -> enable all
        if 'all' in config_dict.keys():
            
            # loop available channels and copy parameters
            for chan in available_channels:
                
                processing_config[config_name][chan] = copy.deepcopy(
                    config_dict['all']
                )
                
            # remove from dict    
            config_dict.pop('all')

        # let's split channels that are separated
        # by a comma and check duplicate
        parameter_list = list()
        iter_list = list(config_dict.keys())
        for chan in iter_list:
            
            if ',' in chan:
                
                # split channels
                split_channels ,_ = split_channel_name(
                    chan, available_channels, separator=','
                )

                # loop and add config for split channels
                for split_chan in split_channels:

                    # error if multiple times defined
                    if split_chan in parameter_list:
                        raise ValueError(f'ERROR: channel {split_chan} '
                                         f'defined multiple times in the '
                                         f'{config_name} configuration. '
                                         f'This is not allowed to avoid mistake.'
                                         f'Check yaml file!')

                    # copy dict 
                    config_dict[split_chan] = copy.deepcopy(
                        config_dict[chan]
                    )
                    
                    parameter_list.append(split_chan)

                # remove from config
                config_dict.pop(chan)
            
            else:

                if chan in parameter_list:
                    raise ValueError(f'ERROR: parameter or channel {chan} '
                                     f'defined multiple times in the '
                                     f'{config_name} configuration. '
                                     f'This is not allowed to avoid mistake!'
                                     f'Check yaml file!')

                parameter_list.append(chan)

        # check duplication of "length" parameters
        if ('coincident_window_msec' in parameter_list
            and 'coincident_window_samples' in  parameter_list):
            raise ValueError(f'ERROR: Found both "coincident_window_msec" '
                             f'and "coincident_window_samples" in '
                             f'{config_name} configuration. Choose between '
                             f'msec or samples!')
    
            
        # loop channels/keys and add to output configuration
        for chan, config in config_dict.items():

            # check if empty 
            if not config:
                raise ValueError(
                    f'ERROR: empty channel/parameter '
                    f'{chan} for {config_name} configuration!')

            # case individual channels
            if chan in  available_channels:
               
                if not isinstance(config, dict):
                    raise ValueError(f'ERROR: Empty channel {chan} in the '
                                     f'{config_name} configuration. Check '
                                     f'yaml file!')
                # check if disabled
                if ('disable' in config and config['disable']
                    or 'run' in config and not config['run']):

                    # remove if needed
                    if chan in processing_config[config_name]:
                        processing_config[config_name].pop(chan)
                        
                else:
                    # add
                    if chan in processing_config[config_name]:
                        processing_config[config_name][chan].update(
                            copy.deepcopy(config)
                        )
                    else:
                        processing_config[config_name][chan] = (
                            copy.deepcopy(config)
                        )

                    if 'disable' in processing_config[config_name][chan]:
                        processing_config[config_name][chan].pop('disable')

                continue

            # check if non-channel parameter
            if (config_name == 'trigger'
                and chan in global_trigger_parameters):
                processing_config[config_name][chan] = config
                continue
            
            # check if channel contains with +,-,| separator
            split_channels, separator = split_channel_name(
                chan, available_channels, separator=None
            )

            if separator in separators:
                
                # check if disabled
                if ('disable' in config and config['disable']
                    or 'run' in config and not config['run']):
                    if chan in processing_config[config_name]:
                        processing_config[config_name].pop(chan)
                else:
                    processing_config[config_name][chan] = (
                        copy.deepcopy(config)
                    )
                    
                    if 'disable' in processing_config[config_name][chan]:
                        processing_config[config_name][chan].pop('disable')

                continue

            # at this point, parameter is unrecognized
            raise ValueError(f'ERROR: Unrecognized parameter '
                             f'{chan} in the {config_name} '
                             f'configuration. Perhaps a channel '
                             f'not in raw data?')

        
    # Feature processing specific config
    if 'feature' in processing_config:
        
        chan_list = list(processing_config['feature'].keys())

        for chan in chan_list:

            chan_config = copy.deepcopy(
                processing_config['feature'][chan]
            )

            # check channel has any parameters
            if not isinstance(chan_config, dict):
                raise ValueError(
                    f'ERROR: Channel {chan} has '
                    f'no configuration! Remove '
                    f'from yaml file or disable it!'
                )
    
            # channel list
            chan_list, separator = split_channel_name(
                chan, available_channels
            )

            # trace length
            nb_samples = None
            nb_pretrigger_samples = None

            # check if in global:
            if 'trace_length_samples' in processing_config['global']:
                nb_samples  = (
                    processing_config['global']['trace_length_samples']
                )
            elif 'trace_length_msec' in processing_config['global']:
                if sample_rate is None:
                    raise ValueError(
                        'ERROR: sample rate is required '
                        'when trace length is in msec. '
                        )
                trace_length_msec = (
                    processing_config['global']['trace_length_msec']
                )
                nb_samples  = (
                    convert_length_msec_to_samples(trace_length_msec,
                                                   sample_rate)
                )
                
            if 'pretrigger_length_samples' in processing_config['global']:
                nb_pretrigger_samples = (
                    processing_config['global']['pretrigger_length_samples']
                )
            elif 'pretrigger_length_msec' in processing_config['global']:
                pretrigger_length_msec = (
                    processing_config['global']['pretrigger_length_msec']
                )
                nb_pretrigger_samples  = (
                    convert_length_msec_to_samples(pretrigger_length_msec,
                                                   sample_rate)
                )
                
            # Get trace/pretrigger length at the channel level
            if 'trace_length_samples' in chan_config.keys():
                nb_samples  = chan_config['trace_length_samples']
            elif 'trace_length_msec' in chan_config.keys():
                if sample_rate is None:
                    raise ValueError(
                        'ERROR: sample rate is required '
                        'when trace length is in msec. '
                        )
                trace_length_msec = chan_config['trace_length_msec']
                nb_samples  = (
                    convert_length_msec_to_samples(trace_length_msec,
                                                   sample_rate)
                )

            if 'pretrigger_length_samples' in chan_config.keys():
                nb_pretrigger_samples = chan_config['pretrigger_length_samples'] 
            elif 'pretrigger_length_msec' in chan_config.keys():
                pretrigger_length_msec = chan_config['pretrigger_length_msec']
                nb_pretrigger_samples  = (
                    convert_length_msec_to_samples(pretrigger_length_msec,
                                                   sample_rate)
                )
                
            if (nb_samples is not None
                and nb_pretrigger_samples is None):
                raise ValueError(
                    f'ERROR: Missing "pretrigger_length_samples" '
                    f'for channel {chan} !')
            elif (nb_samples is None
                  and nb_pretrigger_samples is not None):
                raise ValueError(
                    f'ERROR: Missing "trace_length_samples" '
                    f' for channel {chan} !'
                )

            # loop algorithms  
            algorithm_list = list()
            for algo, algo_config in chan_config.items():

                # check if algorithm dictionary
                if not isinstance(algo_config, dict):
                    continue
                
                if 'run' not in algo_config.keys():
                    raise ValueError(
                        f'ERROR: Missing "run" parameter for channel '
                        f'{chan}, algorithm {param}. Please fix the '
                        f'configuration yaml file')

                # remove from configuration if not run
                if not algo_config['run']:
                    processing_config['feature'][chan].pop(algo)
                    continue

                # add to list of algorithms
                algorithm_list.append(algo)

                # overwite nb samples for this particular algorithm
                nb_samples_alg =  nb_samples
                nb_pretrigger_samples_alg = nb_pretrigger_samples
                
                if 'trace_length_samples' in algo_config.keys():
                    nb_samples_alg = algo_config['trace_length_samples']
                elif 'trace_length_msec' in algo_config.keys():
                    trace_length_msec = algo_config['trace_length_msec']
                    nb_samples_alg  = (
                        convert_length_msec_to_samples(trace_length_msec,
                                                       sample_rate)
                )
                   
                    
                if 'pretrigger_length_samples' in algo_config.keys():
                    nb_pretrigger_samples_alg = (
                        algo_config['pretrigger_length_samples']
                    )
                elif 'pretrigger_length_msec' in algo_config.keys():
                    pretrigger_length_msec = algo_config['pretrigger_length_msec']
                    nb_pretrigger_samples_alg  = (
                        convert_length_msec_to_samples(pretrigger_length_msec,
                                                       sample_rate)
                    )

                # update algorithm with trace length
                processing_config['feature'][chan][algo]['nb_samples'] = (
                    nb_samples_alg
                )

                processing_config['feature'][chan][algo]['nb_pretrigger_samples'] = (
                    nb_pretrigger_samples_alg 
                )   
                
            # remove channel if no algorithm
            if not algorithm_list:
                processing_config['feature'].pop(chan)
            else:
                # remove trace length / weight
                if 'trace_length_samples' in processing_config['feature'][chan]:
                    processing_config['feature'][chan].pop('trace_length_samples')
                if 'pretrigger_length_samples' in processing_config['feature'][chan]:
                    processing_config['feature'][chan].pop('pretrigger_length_samples')
                                  
    # return
    return processing_config

def get_indices_from_freq_ranges(freqs, freq_ranges):
    """
    convert frequency ranges to index ranges. Return 
    also name freq[0]_freq[1]
    """

    name_list = list()
    index_ranges = list()
        
    for it, freq_range in enumerate(freq_ranges):
                        
        # ignore if not a range
        if len(freq_range) != 2:
            continue
            
        # low/high frequency
        f_low = abs(freq_range[0])
        f_high = abs(freq_range[1])
            
        if f_low > f_high:
            f_low = abs(freq_range[1])
            f_high = abs(freq_range[0])
                
                    
        # indices
        ind_low = np.argmin(np.abs(freqs - f_low))
        ind_high = np.argmin(np.abs(freqs - f_high))

        # check if proper range
        if ind_low == ind_high:
            if ind_low < len(freqs)-2:
                ind_high = ind_low + 1
            else:
                continue
            
                
        # store
        name = f'{round(f_low)}_{round(f_high)}'
        
        if name in name_list:
            continue
            
        name_list.append(name)
        index_ranges.append((ind_low, ind_high))

            
    return name_list, index_ranges

class _UniqueKeyLoader(SafeLoader):
    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None,
                'expected a mapping node, but found %s' % node.id,
                node.start_mark)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f'ERROR: Duplicate key "{key}" '
                                 f'found in the yaml file for same '
                                 f'channel and algorithm. '
                                 f'This is not allowed to avoid '
                                 f'unwanted configuration!')
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


def _rename_key_recursively(d, old_key, new_key):
    """
    Recursively renames a key in a dictionary and 
    all its sub-dictionaries.
    """

    # check if dictionary
    if not isinstance(d, dict):
        return d
    
    for key in list(d.keys()):  
        if isinstance(d[key], dict):
            _rename_key_recursively(d[key], old_key, new_key)
        if key == old_key:
            d[new_key] = d.pop(old_key)
    return d



