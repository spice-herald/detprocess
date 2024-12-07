import os
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
import yaml
import copy
from yaml.loader import SafeLoader
import re
from pytesdaq.io import convert_length_msec_to_samples
from qetpy.utils import convert_channel_name_to_list, convert_channel_list_to_name
from detprocess.utils import utils

__all__ = [
    'YamlConfig'
]

class YamlConfig:
    """
    Class to read and manage yaml configuration
    """

    def __init__(self, yaml_file, available_channels, verbose=True):
        """
        Initialize class

        Parameters:
        ----------

        verbose : bool, optional
          display information


        """

        # yaml file 
        self._yaml_file = yaml_file
        
        # initialize processing config
        self._processing_config = None

        # configuration types
        self._configuration_fields = ['global', 'salting',
                                      'feature', 'didv', 'noise',
                                     'template', 'trigger']

        # available global parameters
        self._global_parameters = ['filter_file', 'didv_file']
        self._global_trigger_parameters = ['coincident_window_msec',
                                           'coincident_window_samples'] 
        self._global_salting_parameters = ['dm_pdf_file',
                                           'energies',
                                           'nsalt'] 
        # modified/obsolete parameters
        # -> keep back-compatibility
        # obsolete keys
        self._obsolete_keys = {
            'nb_samples': 'trace_length_samples',
            'nb_pretrigger_samples': 'pretrigger_length_samples'
        }
    
        # available channel separators
        self._separators =  [',', '+', '-', '|']


        # available channels
        if isinstance(available_channels, str):
            available_channels =  [available_channels]
        self._available_channels = available_channels

        # read yaml file
        self._read_config()

    def get_config(self, processing_type=None):
        """
        Get config
        """

        if self._processing_config is None:
            return None

        
        config = {}
        if processing_type is not None:

            if processing_type not in self._configuration_fields:
                raise ValueError(f'ERROR: Configuration type '
                                 f'"{processing_type}" not found!')

            config = copy.deepcopy(self._processing_config[processing_type])
            
        else:
            config = copy.deepcopy(self._processing_config)

        return config

            
    def _read_config(self):
        """
        Read configuration (yaml) file 
        """
        
        # load yaml file
        yaml_dict = yaml.load(open(self._yaml_file, 'r'),
                              Loader=_UniqueKeyLoader)
        
        if not yaml_dict:
            raise ValueError('ERROR: No configuration loaded'
                             'Something went wrong...')

        # case multiple files 
        if 'include' in  yaml_dict:
            include_files = yaml_dict['include']
            if isinstance(include_files, str):
                include_files = [include_files]
            for afile in include_files:
                yaml_dict.update(yaml.load(open(afile, 'r'),
                                           Loader=_UniqueKeyLoader))
            yaml_dict.pop('include')
            

        
        # let's split configuration based on the known
        # type of processing
        config_dicts = dict()
        for config_name in self._configuration_fields:
        
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

        # global parameters
        for param in self._global_parameters:
            config_dicts['global'][param] = None
            if param in yaml_dict.keys():
                config_dicts['global'][param] = copy.deepcopy(
                    yaml_dict[param]
                )
                yaml_dict.pop(param)
                

        # the rest of parameter are for feature processing
        for param in  yaml_dict.keys():
            config_dicts['feature'][param] = copy.deepcopy(
                yaml_dict[param]
            )

        # rename obsolete keys
        for old_key, new_key in self._obsolete_keys.items():
            config_dicts = self._rename_key_recursively(
                config_dicts, old_key, new_key
            )
            

        # let's cleanup for all types:
        #    - separate channels with ','
        #    - replace "all" with actuall channel names
        
        # intialize 
        processing_config = dict()
        
        # Loop configuration and check/cleanup parameters
        for config_name  in self._configuration_fields:

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

            # convert 'all' (individual) channels to actual
            # available channels
            if 'all' in config_dict.keys():
            
                # loop available channels and copy parameters
                for chan in self._available_channels:
                
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
                    split_channels ,_ = utils.split_channel_name(
                        chan, self._available_channels, separator=','
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

                # check if non-channel parameter
                if (config_name == 'trigger'
                    and chan in self._global_trigger_parameters):
                    processing_config[config_name][chan] = config
                    continue
                
                if (config_name == 'salting'
                    and chan in self._global_salting_parameters):
                    processing_config[config_name][chan] = config
                    continue

                # case individual channels
                if chan in  self._available_channels:
               
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

                # case multi-channels
                split_channels, separator = utils.split_channel_name(
                    chan, self._available_channels, separator=None
                )

                if separator in self._separators:
                
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
            feature_dict = copy.deepcopy(processing_config['feature'])
            global_dict = copy.deepcopy(processing_config['global'])
            processing_config['feature'] = self._configure_features(
                feature_dict, global_dict
            )
            
        #if 'trigger' in processing_config:
        #     processing_config['trigger'] = self._configure_features(
        #         processing_config['trigger']
        #     )
        
        self._processing_config = processing_config
        
    def _configure_features(self, feature_config, global_config):
        """
        Feature specific configuration
        """

        # copy 
        feature_dict = copy.deepcopy(feature_config)

        # chan
        chan_list = list(feature_dict.keys())

        for chan in chan_list:

            chan_config = feature_dict[chan]

            # check channel has any parameters
            if not isinstance(chan_config, dict):
                raise ValueError(
                    f'ERROR: Channel {chan} has '
                    f'no configuration! Remove '
                    f'from yaml file or disable it!'
                )
    
            # channel list
            chan_list, separator = utils.split_channel_name(
                chan, self._available_channels
            )

            # trace length
            nb_samples = None
            nb_pretrigger_samples = None

            # check if in global:
            if 'trace_length_samples' in global_config:
                nb_samples  = (
                    global_config['trace_length_samples']
                )
            elif 'trace_length_msec' in global_config:
                if sample_rate is None:
                    raise ValueError(
                        'ERROR: sample rate is required '
                        'when trace length is in msec. '
                        )
                trace_length_msec = (
                    global_config['trace_length_msec']
                )
                nb_samples  = (
                    convert_length_msec_to_samples(trace_length_msec,
                                                   sample_rate)
                )
                
            if 'pretrigger_length_samples' in global_config:
                nb_pretrigger_samples = (
                    global_config['pretrigger_length_samples']
                )
            elif 'pretrigger_length_msec' in global_config:
                pretrigger_length_msec = (
                    global_config['pretrigger_length_msec']
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
                    feature_dict[chan].pop(algo)
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
                feature_dict[chan][algo]['nb_samples'] = (
                    nb_samples_alg
                )

                feature_dict[chan][algo]['nb_pretrigger_samples'] = (
                    nb_pretrigger_samples_alg 
                )   
                
            # remove channel if no algorithm
            if not algorithm_list:
                feature_dict.pop(chan)
            else:
                # remove trace length / weight
                if 'trace_length_samples' in feature_dict[chan]:
                    feature_dict[chan].pop('trace_length_samples')
                if 'pretrigger_length_samples' in feature_dict[chan]:
                    feature_dict[chan].pop('pretrigger_length_samples')
                                  
        # return
        return feature_dict
 

    def _rename_key_recursively(self, d, old_key, new_key):
        """
        Recursively renames a key in a dictionary and 
        all its sub-dictionaries.
        """

        # check if dictionary
        if not isinstance(d, dict):
            return d
    
        for key in list(d.keys()):  
            if isinstance(d[key], dict):
                self._rename_key_recursively(d[key], old_key, new_key)
            if key == old_key:
                d[new_key] = d.pop(old_key)
        return d



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


