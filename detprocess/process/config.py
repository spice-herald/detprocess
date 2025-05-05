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

    def __init__(self, yaml_file, available_channels, sample_rate=None,
                 verbose=True):
        """
        Initialize class

        Parameters:
        ----------

        verbose : bool, optional
          display information


        """

        # yaml file 
        self._yaml_file = yaml_file

        # sample rate
        self._sample_rate = sample_rate
        
        # initialize processing config
        self._processing_config = None

        # configuration types
        self._configuration_fields = ['salting', 'feature',
                                      'didv', 'noise',
                                      'template', 'trigger']

        # available global parameters
        self._overall_parameters  = {
            'global': ['filter_file', 'didv_file'],
            'trigger': ['coincident_window_msec',
                        'coincident_window_samples'] ,
            'salting': ['dm_pdf_file',
                        'coincident_salts',
                        'energies',
                        'nsalt'],
            'feature': ['trace_length_samples',
                        'pretrigger_length_samples',
                        'trace_length_msec',
                        'pretrigger_length_msec']
            }
            
            
        # modified/obsolete parameters
        # -> keep back-compatibility
        # obsolete keys
        self._obsolete_keys = {
            'trigger_name': 'trigger_channel',
            'nb_samples': 'trace_length_samples',
            'nb_pretrigger_samples': 'pretrigger_length_samples',
            'template_time_tags': 'template_group_ids'
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
            

        # rename obsolete keys
        for old_key, new_key in self._obsolete_keys.items():
            yaml_dict = self._rename_key_recursively(
                yaml_dict, old_key, new_key
            )
              
        # Initialize
        processing_configs = {'global':{}}
        for field in self._configuration_fields:
            processing_configs[field] = {'overall':{},
                                         'channels':{}}
            
            
        # global parameters
        for param in self._overall_parameters['global']:
            processing_configs['global'][param] = None
            if param in yaml_dict.keys():
                processing_configs['global'][param] = copy.deepcopy(
                    yaml_dict[param]
                )
                yaml_dict.pop(param)
                
        
        # let's split configuration based on the known
        # type of processing
        for field in self._configuration_fields:

            # check if field available
            if field not in yaml_dict.keys():
                continue
            
            # set to None
            field_map = {'overall': {},
                         'channels': {}}
    
            # overall parameters
            overall_params = []
            if field in self._overall_parameters:
                overall_params = self._overall_parameters[field]

            # let's get config dictionary
            config_dict = copy.deepcopy(yaml_dict[field])
            yaml_dict.pop(field)
            
            for config, config_items in config_dict.items():

                if config in overall_params:
                    field_map['overall'][config] = (
                        config_items
                    )
                elif field == 'feature' and config == 'global':
                    for param in config_items:
                        field_map['overall'][param] = (
                            config_items[param]
                        )
                else:
                    field_map['channels'][config] = (
                        config_items
                    )
            # save
            processing_configs[field] = field_map
            
        # the rest of parameters are for feature processing (without
        # "feature" field
        for param in  yaml_dict.keys():

            if param == 'global':
                processing_configs['feature']['overall'] = copy.deepcopy(
                    yaml_dict[param]
                )
            else:
                processing_configs['feature']['channels'][param] = copy.deepcopy(
                    yaml_dict[param]
                )
            

        # for all fields, let's split channels if separated by ','
        for field in self._configuration_fields:
            
            if (field not in processing_configs
                or 'channels' not in processing_configs[field]):
                continue
            
            new_channel_config = {}
            channels = processing_configs[field]['channels']
            for chan, chan_dict in channels.items():

                # check if disable
                if ('disable' in chan_dict and chan_dict['disable']
                    or 'run' in chan_dict and not chan_dict['run']):
                    continue
                
                if chan == 'all':
                    for single_chan in self._available_channels:
                        new_channel_config[single_chan] = (
                            copy.deepcopy(chan_dict)
                        )
                else:
                    # split channels
                    split_channels, _ = utils.split_channel_name(
                        chan,
                        available_channels=self._available_channels,
                        separator=',', label=field
                    )

                    for split_chan in split_channels:
                        new_channel_config[split_chan] = (
                            copy.deepcopy(chan_dict)
                        )
            # save
            processing_configs[field]['channels'] = new_channel_config
            

     
    
        # let's cleanup of each type of processing
        # feature
        configs = self._configure_features(processing_configs['feature'],
                                          processing_configs['global'])
        
        processing_configs['feature'] = copy.deepcopy(configs)
        
        # trigger
        configs = self._configure_triggers(processing_configs['trigger'],
                                           processing_configs['global'])
        
        processing_configs['trigger'] = copy.deepcopy(configs)
    

        # salting
        configs = self._configure_salting(processing_configs['salting'],
                                           processing_configs['global'])
        
        processing_configs['salting'] = copy.deepcopy(configs)


        
        self._processing_config = processing_configs


    def _configure_salting(self, salting_config, global_config):
        """
        Salting specific configuration
        """

        # copy 
        salting_dict = copy.deepcopy(salting_config)
        global_dict =  copy.deepcopy(global_config)

        # add global into feature_dict
        if global_config:
            for config, config_val in global_config.items():
                if config not in salting_dict['overall']:
                    salting_dict['overall'][config] = config_val
                    
        # loop channels
        split_channel_list = []
        chan_list = list(salting_dict['channels'].keys())
        for chan in chan_list:

            chan_config = copy.deepcopy(salting_dict['channels'][chan])
            
            # check channel has any parameters
            if not isinstance(chan_config, dict):
                raise ValueError(
                    f'ERROR: Channel {chan} has '
                    f'no configuration! Remove '
                    f'from yaml file or disable it!'
                )


            # get split channel list
            split_chans, separator = utils.split_channel_name(
                chan,
                available_channels=self._available_channels,
                label='salting'
            )

            split_channel_list.extend(split_chans)

        salting_dict['channel_list'] =  utils.unique_list(split_channel_list)

        return salting_dict

    def _configure_triggers(self, trigger_config, global_config):
        """
        Trigger specific configuration
        """

        # copy 
        trigger_dict = copy.deepcopy(trigger_config)
        global_dict =  copy.deepcopy(global_config)

        # add global into feature_dict
        if global_config:
            for config, config_val in global_config.items():
                if config not in trigger_dict['overall']:
                    trigger_dict['overall'][config] = config_val
                    
                    
        # loop channels
        split_channel_list = []
        trigger_channel_dict = {}
        chan_list = list(trigger_dict['channels'].keys())
      
        for chan in chan_list:

            chan_config = copy.deepcopy(trigger_dict['channels'][chan])
            
            # check channel has any parameters
            if not isinstance(chan_config, dict):
                raise ValueError(
                    f'ERROR: Channel {chan} has '
                    f'no configuration! Remove '
                    f'from yaml file or disable it!'
                )


            # get split channel list
            split_chans, separator = utils.split_channel_name(
                chan,
                available_channels=self._available_channels,
                label='trigger'
            )

            split_channel_list.extend(split_chans)
      
            # check if trigger channel
            trigger_channel = chan
            if 'trigger_channel' in chan_config:
                trigger_channel =  chan_config['trigger_channel']
                chan_config.pop('trigger_channel')
            
            # case no algorithm name
            if 'run' in chan_config:

                if not chan_config['run']:
                    continue

                chan_config['channel_name'] = chan
                
                # save
                trigger_channel_dict[trigger_channel] = chan_config

            else:
                
                for algo, algo_dict in chan_config.items():

                    if (not isinstance(algo_dict, dict)
                        or 'run' not in algo_dict):
                        raise ValueError(
                            f'ERROR: Missing "run" parameter for trigger '
                            f'channel {chan}'
                        )
                    
                    if not algo_dict['run']:
                        continue

                    algo_trigger_channel = f'{algo}_{trigger_channel}'

                    # save
                    algo_dict['channel_name'] = chan
                    trigger_channel_dict[algo_trigger_channel] = algo_dict
                    
        trigger_dict['channels'] = trigger_channel_dict
        trigger_dict['channel_list'] = utils.unique_list(split_channel_list)
        return trigger_dict
                
                
    def _configure_features(self, feature_config, global_config):
        """
        Feature specific configuration
        """

        # copy 
        feature_dict = copy.deepcopy(feature_config)
        global_dict =  copy.deepcopy(global_config)


        # add global into feature_dict
        if global_config:
            for config, config_val in global_config.items():
                if config not in feature_dict['overall']:
                    feature_dict['overall'][config] = config_val

        
        # loop channels
        #  - check channel is in raw data
        #  - remove disabled algorithm
        #  - add trace length
        split_channel_list = []
        chan_list = list(feature_dict['channels'].keys())
        for chan in chan_list:

            chan_config = copy.deepcopy(feature_dict['channels'][chan])

            # check channel has any parameters
            if not isinstance(chan_config, dict):
                raise ValueError(
                    f'ERROR: Channel {chan} has '
                    f'no configuration! Remove '
                    f'from yaml file or disable it!'
                )
    
            # channel list
            split_chans, separator = utils.split_channel_name(
                chan, self._available_channels, label='feature'
            )

            split_channel_list.extend(split_chans)


            # trace length
            nb_samples = None
            nb_pretrigger_samples = None

            # check if in global:
            if 'trace_length_samples' in feature_dict['overall']:
                nb_samples  = (
                    feature_dict['overall']['trace_length_samples']
                )
            elif 'trace_length_msec' in feature_dict['overall']:
                if self._sample_rate is None:
                    raise ValueError(
                        'ERROR: sample rate is required '
                        'when trace length is in msec. '
                        )
                trace_length_msec = (
                    feature_dict['overall']['trace_length_msec']
                )
                nb_samples  = (
                    convert_length_msec_to_samples(trace_length_msec,
                                                   self._sample_rate)
                )
                
            if 'pretrigger_length_samples' in feature_dict['overall']:
                nb_pretrigger_samples = (
                    feature_dict['overall']['pretrigger_length_samples']
                )
            elif 'pretrigger_length_msec' in feature_dict['overall']:
                pretrigger_length_msec = (
                    feature_dict['overall']['pretrigger_length_msec']
                )
                nb_pretrigger_samples  = (
                    convert_length_msec_to_samples(pretrigger_length_msec,
                                                   self._sample_rate)
                )
                
            # Get trace/pretrigger length at the channel level
            if 'trace_length_samples' in chan_config.keys():
                nb_samples  = chan_config['trace_length_samples']
            elif 'trace_length_msec' in chan_config.keys():
                if self._sample_rate is None:
                    raise ValueError(
                        'ERROR: sample rate is required '
                        'when trace length is in msec. '
                        )
                trace_length_msec = chan_config['trace_length_msec']
                nb_samples  = (
                    convert_length_msec_to_samples(trace_length_msec,
                                                   self._sample_rate)
                )

            if 'pretrigger_length_samples' in chan_config.keys():
                nb_pretrigger_samples = chan_config['pretrigger_length_samples'] 
            elif 'pretrigger_length_msec' in chan_config.keys():
                pretrigger_length_msec = chan_config['pretrigger_length_msec']
                nb_pretrigger_samples  = (
                    convert_length_msec_to_samples(pretrigger_length_msec,
                                                   self._sample_rate)
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
                    feature_dict['channels'][chan].pop(algo)
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
                                                       self._sample_rate)
                )
                   
                if 'pretrigger_length_samples' in algo_config.keys():
                    nb_pretrigger_samples_alg = (
                        algo_config['pretrigger_length_samples']
                    )
                elif 'pretrigger_length_msec' in algo_config.keys():
                    pretrigger_length_msec = algo_config['pretrigger_length_msec']
                    nb_pretrigger_samples_alg  = (
                        convert_length_msec_to_samples(pretrigger_length_msec,
                                                       self._sample_rate)
                    )

                # update algorithm with trace length
                feature_dict['channels'][chan][algo]['nb_samples'] = (
                    nb_samples_alg
                )

                feature_dict['channels'][chan][algo]['nb_pretrigger_samples'] = (
                    nb_pretrigger_samples_alg 
                )   
                
            # remove channel if no algorithm
            if not algorithm_list:
                feature_dict['channels'].pop(chan)
            else:
                # remove trace length / weight
                if 'trace_length_samples' in feature_dict['channels'][chan]:
                    feature_dict['channels'][chan].pop('trace_length_samples')
                if 'pretrigger_length_samples' in feature_dict['channels'][chan]:
                    feature_dict['channels'][chan].pop('pretrigger_length_samples')


        # add channel list
        feature_dict['channel_list'] = utils.unique_list(split_channel_list)

        # get weight and trace info
        traces_config = dict()
        weights = dict()

        # loop channels
        for chan, chan_config in feature_dict['channels'].items():

            # list of individual channels
            chan_list, separator = utils.split_channel_name(
                chan, feature_dict['channel_list']
            )
            
            # weights
            for chan_split in chan_list:
                param = f'weight_{chan_split}'
                if param in chan_config:
                    if chan not in weights:
                        weights[chan] = dict()
                    weights[chan][param] = chan_config[param]
            
            # now loop through algorithms, get/add trace length at the
            # algorithm level 
            for algo, algo_config in chan_config.items():
            
                if not isinstance(algo_config, dict):
                    continue
                
                if not algo_config['run']:
                    continue

                nb_samples =  algo_config['nb_samples']
                nb_pretrigger_samples = algo_config['nb_pretrigger_samples']
                trace_tuple = (nb_samples, nb_pretrigger_samples)
                
                if trace_tuple in traces_config:
                    traces_config[trace_tuple].extend(chan_list.copy())
                else:
                    traces_config[trace_tuple] = chan_list.copy()

      
        for key in traces_config.keys():
            traces_config[key] = utils.unique_list(traces_config[key])
            
        if not traces_config:
            traces_config = None

        feature_dict['traces_config'] = copy.deepcopy(traces_config)
        feature_dict['weights'] = copy.deepcopy(weights)
        
                    
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


