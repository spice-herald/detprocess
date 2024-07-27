import numpy as np
import pandas as pd
import qetpy as qp
import sys
from pprint import pprint
import pytesdaq.io as h5io
import vaex as vx
from glob import glob
from detprocess.utils import utils
from detprocess.core import FilterData
import copy


__all__ = [
    'ProcessingData'
]

class ProcessingData:
    """
    Class to manage data used for processing,
    in particular
      - raw data filew, event traces and metadata
      - filter file data
      - optimal filter objects containing template/PSD FFTs, etc.

    """

    def __init__(self, raw_path, raw_files,
                 group_name=None,
                 trigger_files=None,
                 trigger_group_name=None,
                 filter_file=None,
                 available_channels=None,
                 verbose=True):
        """
        Intialize data processing

        Parameters
        -----------

        input_files : list
           list of raw data files to be processed

        group_name : str, optional
           raw data group name
           default: None

        filter_file : str, optional
          full path to filter file
          default: None

        verbose :  bool, optional
          if True, display info
           Default: True

        """

        # verbose
        self._verbose = verbose

        # input raw files/path
        self._raw_path = raw_path
        self._raw_files = raw_files
        self._group_name = group_name

        # trigger dataframe
        self._trigger_files = trigger_files
        self._trigger_group_name = trigger_group_name

        # filter data
        self._filter_data = None
        if filter_file is not None:
            self._filter_data = FilterData()
            self._filter_data.load_hdf5(filter_file, overwrite=True)

         
        # initialize vaex dataframe
        self._dataframe = None
        self._is_dataframe = False
        if self._trigger_files is not None:
            self._is_dataframe = True


        # initialize OF containers
        self._OF_base_objs = dict()
        self._OF_base_algorithms = [
            'of1x1', 'of1x2', 'of1x3',
            'ofnxm', 'ofnxmx2', 'psd_amp'
        ]
        
        # initialize raw data reader
        self._h5 = h5io.H5Reader()
        
        # initialize current event traces and metadata
        self._current_traces  = None
        self._current_admin_info = None
        self._current_traces_data = None
        self._current_dataframe_index = -1
        self._current_dataframe_info = None
        self._current_event_number = None
        self._current_series_number = None
        self._current_trigger_index = None

        # get ADC and file info
        self._data_info = self._extract_data_info()
            
        # available channels
        self._available_channels =  available_channels
        
    @property
    def verbose(self):
        return self._verbose

    def instantiate_OF_base(self, processing_config, channel=None):
        """
        Instantiate QETpy OF base class, perform pre-calculations
        such as FFT, etc, check trace / pre-trigger length

        Parameters
        ----------

        processing_config : dict
         disctionary with processing user config (loaded
         from yaml file)

        channel : str,  optional
          channel name
          Default: all channels from config

        """

        # check if filter file data available
        if self._filter_data is None:
            if self._verbose:
                print('INFO: No filter file available. Skipping '
                      + ' OF instantiation!')
            return
        
        # check if channel argument, if None -> all channels
        if (channel is not None
            and channel not in processing_config.keys()):
            raise ValueError(f'No channel {channel}" '
                             f'found in configuration file!')
        # get sample rate
        sample_rate = self.get_sample_rate()

        # loop channels
        for chan, chan_config in processing_config.items():
                    
            # skip if filter file
            if (chan == 'filter_file'
                or not isinstance(chan_config, dict)):
                continue

            # skip if not selected channel
            if (channel is not None
                and channel != chan):
                continue

            # channel list
            chan_list, _ = utils.split_channel_name(
                chan,
                self._available_channels,
                separator='|'
            )

            nb_channels = len(chan_list)
                    
            # intialize a list of psd/csd tags
            # -> only one allowed
            psd_tags = dict()
            csd_tags = dict()
            
            # loop configuration and get list of templates
            for algo, algo_config in chan_config.items():

                # skip if not dictionary
                if not isinstance(algo_config, dict):
                    continue
                
                # skip if disable
                if not algo_config['run']:
                    continue

                # check if algorithm requires OF base
                is_of_base_used = False
                for prefix in self._OF_base_algorithms:
                    algo_base = algo
                    if 'base_algorithm' in algo_config:
                        algo_base = algo_config['base_algorithm']
                    if prefix in algo_base:
                        is_of_base_used = True

                if not is_of_base_used:
                    continue
                
                # number of samples
                nb_samples = algo_config['nb_samples']
                nb_pretrigger_samples =  algo_config['nb_pretrigger_samples']

                # instantiate OF base if needed
                key_tuple = (nb_samples, nb_pretrigger_samples)
                if key_tuple not in self._OF_base_objs:
                    self._OF_base_objs[key_tuple] = {
                        'OF': qp.OFBase(sample_rate, verbose=True),
                        'channels_split': chan_list,
                        'channels': [chan],
                        'algorithms': [algo]
                    }
                else:
                    self._OF_base_objs[key_tuple]['channels_split'].extend(chan_list)
                    self._OF_base_objs[key_tuple]['channels'].append(chan)
                    self._OF_base_objs[key_tuple]['algorithms'].append(algo)
                           
                # Multichannel: add CSD
                if nb_channels > 1:

                    tag = algo_config['csd_tag']
                        
                    csd, csd_freqs, csd_metadata = (
                        self._filter_data.get_csd(
                            chan,
                            tag=tag,
                            fold=False,
                            return_metadata=True)
                    )

                    if nb_samples not in csd_tags:
                        csd_tags[nb_samples] = list()  
                    csd_tags[nb_samples].append(tag)
                    
                    # check sample rate
                    if 'sample_rate' in csd_metadata:
                        fs = csd_metadata['sample_rate']
                        if fs != sample_rate:
                            raise ValueError(
                                f'Sample rate is not '
                                f'consistent between raw data '
                                f'and csd for channel {chan}!'
                            )
                    # check nb samples
                    if nb_samples != csd.shape[-1]:
                        raise ValueError(
                            f'Number of samples is not '
                            f'consistent between raw data '
                            f'and csd for channel {chan}, '
                            f'algorithm {algo}!'
                        )
                                                
                    # add in OF base
                    if self._OF_base_objs[key_tuple]['OF'].csd(chan) is None:
                        self._OF_base_objs[key_tuple]['OF'].set_csd(chan, csd)

                # Add PSD
                else:
                    tag = algo_config['psd_tag']
                    
                    psd, psd_freqs, psd_metadata = (
                        self._filter_data.get_psd(
                            chan,
                            tag=tag,
                            fold=False,
                            return_metadata=True)
                    )
                    
                    if nb_samples not in psd_tags:
                        psd_tags[nb_samples] = list()  
                    psd_tags[nb_samples].append(tag)
                                    
                    # check sample rate
                    if 'sample_rate' in psd_metadata:
                        fs = psd_metadata['sample_rate']
                        if fs != sample_rate:
                            raise ValueError(
                                f'Sample rate is not '
                                f'consistent between raw data '
                                f'and psd for channel {chan}!'
                            )
                    # check nb samples
                    if nb_samples != psd.shape[-1]:
                        raise ValueError(
                            f'Number of samples is not '
                            f'consistent between raw data '
                            f'and psd for channel {chan}, '
                            f'algorithm {algo}!'
                        )

                    # coupling
                    coupling = 'AC'
                    if 'coupling' in algo_config.keys():
                        coupling = algo_config['coupling']
                        
                    # add in OF base
                    if self._OF_base_objs[key_tuple]['OF'].psd(chan) is None:
                        self._OF_base_objs[key_tuple]['OF'].set_psd(
                            chan, psd, coupling=coupling
                        )


                # Add template (s)
                         
                # template integral or max
                integralnorm = False
                if 'integralnorm' in algo_config.keys():
                    integralnorm = algo_config['integralnorm']
                                
                template_tag = algo_config['template_tag']
                matrix_tag = None
                if nb_channels > 1:
                    matrix_tag = algo_config['template_matrix_tag']
                                   
                # create template with zeros
                template_zeros = np.zeros(nb_samples, dtype='float64')
                    
                # store template tags for each channels in
                # dictionary
                 
                template_tag_dict = dict()
                if nb_channels == 1:
                    if isinstance(template_tag, str):
                        template_tag = [template_tag]
                    template_tag_dict[chan] = template_tag
                elif matrix_tag is None:
                    for ichan, chan_split in enumerate(chan_list):
                        template_tag_dict[chan_split] = (
                            template_tag[ichan, :].tolist()
                        )
                # add individual templates
                for chan_tag, tags in template_tag_dict.items():
                    
                    for tag in tags:

                        if tag == 'None' or tag is None:
                            template = template_zeros
                            template_metadata = dict()
                        else:
                            # get template from filter file
                            template, template_time, template_metadata = (
                                self._filter_data.get_template(
                                    chan_tag, tag=tag,
                                    return_metadata=True)
                            )

                        # check samples
                        if nb_samples != template.shape[-1]:
                            raise ValueError(
                                f'Number of samples is not '
                                f'consistent between raw data '
                                f'and template (tag={tag} '
                                f'for channel {chan_tag}, '
                                f'algorithm {algo}!'
                            )

                        nb_pretrigger_template = None
                        if 'nb_pretrigger_samples' in template_metadata.keys():
                            nb_pretrigger_template = (
                                int(template_metadata['nb_pretrigger_samples'])
                            )
                        else:
                            nb_pretrigger_template = nb_pretrigger_samples
                            
                        #  add
                        self._OF_base_objs[key_tuple]['OF'].add_template(
                            chan_tag,
                            template,
                            template_tag=tag,
                            pretrigger_samples=nb_pretrigger_template,
                            integralnorm=integralnorm,
                            overwrite=True,
                        )

                # case matrix
                if matrix_tag is not None:

                    # get template matrix from filter file
                    template, template_time, template_metadata = (
                        self._filter_data.get_template(
                            chan, tag=matrix_tag,
                            return_metadata=True)
                    )
            
                    # pretrigger
                    nb_pretrigger_template = None
                    if 'nb_pretrigger_samples' in template_metadata.keys():
                        nb_pretrigger_template = (
                            int(template_metadata['nb_pretrigger_samples'])
                        )
                    else:
                        nb_pretrigger_template = nb_pretrigger_samples

                    # add
                    self._OF_base_objs[key_tuple]['OF'].add_template_many_channels(
                        chan, template, template_tag,
                        pretrigger_samples=nb_pretrigger_template,
                        integralnorm=integralnorm,
                        overwrite=True)

                # build matrix
                if nb_channels > 1:
                    self._OF_base_objs[key_tuple]['OF'].build_template_matrix(
                        chan, template_tag)
                                    
                    
            # check psd / csd tags
            for tag in psd_tags.keys():
                psd_tags[tag] = (
                    list(set(psd_tags[tag]))
                )
                if len(psd_tags[tag]) != 1:
                    raise ValueError(
                        f'ERROR: Only a single psd tag '
                        f'allowed for channel {chan}! '
                    )
                
            for tag in csd_tags.keys():
                csd_tags[tag] = (
                    list(set(csd_tags[tag]))
                )
                if len(csd_tags[tag]) != 1:
                    raise ValueError(
                        f'ERROR: Only a single csd tag '
                        f'allowed for channel {chan}! '
                    )

                
            # calculate phi for each OF base
            for key in self._OF_base_objs.keys():

                if chan not in self._OF_base_objs[key]['channels']:
                    continue
                
                if nb_channels == 1:
                    self._OF_base_objs[key]['OF'].calc_phi(chan)
                else:
                    self._OF_base_objs[key]['OF'].calc_phi_mat(chan)
                    self._OF_base_objs[key]['OF'].calc_weight_mat(chan)

                    
        # remove duplicated
        for key in self._OF_base_objs:
            self._OF_base_objs[key]['channels'] = list(
                set(self._OF_base_objs[key]['channels'])
            )
            
            self._OF_base_objs[key]['channels_split'] = list(
                set(self._OF_base_objs[key]['channels_split'])
            )
                    
            self._OF_base_objs[key]['algorithms'] = list(
                set(self._OF_base_objs[key]['algorithms'])
            )

    def get_OF_base(self, key_tuple, algo_name):
        """
        Get OF object

        Parameters
        ----------

        key_tuple : tuple
          (nb_samples, nb_pretrigger_samples)

        algo_name : string
           algorithm name
                

        Return
        ------

        OF : object
          optiomal filter base instance

        """

        OF = None
        if key_tuple in self._OF_base_objs:
            algorithms = (
                self._OF_base_objs[key_tuple]['algorithms']
            )
            if algo_name in algorithms:
                OF = self._OF_base_objs[key_tuple]['OF']
            
        return OF 

    def set_series(self, series):
        """
        Set file list for a specific
        series

        Parameters
        ----------

        series : str
          series name

        Return
        ------
        None

        """

        # open/set files

        # case dataframe
        if self._is_dataframe:
            if self._verbose:
                print('INFO: Loading dataframe for trigger series '
                      + series)
            file_list = self._trigger_files[series]
            self._dataframe = vx.open_many(file_list)
            self._current_dataframe_index = -1
        else:
            file_list = self._raw_files[series]
            self._h5.set_files(file_list)


    def get_raw_path(self):
        """
        """
        return self._raw_path


    def read_next_event(self, channels=None, traces_config=None):
        """
        Read next event
        """

        if not self._is_dataframe:

            # read the entire trace
            self._current_traces, self._current_admin_info = (
                self._h5.read_next_event(
                    detector_chans=channels,
                    adctoamp=True,
                    include_metadata=True
                )
            )

            # check size
            if self._current_traces.size == 0:
                return False

        else:

            # require traces_info
            if traces_config is None:
                raise ValueError(
                    'ERROR: No trace info available.'
                    'Something went wrong...'
                )


            # increment event pointer
            # and get event info
            self._current_dataframe_index +=1

            # check if still some events
            if self._current_dataframe_index >= len(self._dataframe):
                self._current_traces = []
                return False

            # get record
            self._current_dataframe_info = self._dataframe.to_records(
                index=self._current_dataframe_index
            )

            # get event/series number, and trigger index
            event_number = self._current_dataframe_info['event_number']
            dump_number = self._current_dataframe_info['dump_number']
            series_number = self._current_dataframe_info['series_number']
            trigger_index = self._current_dataframe_info['trigger_index']
            group_name = self._current_dataframe_info['group_name']

            # event index in file
            event_index = int(event_number%100000)

            # check if new file need dump needs
            # to be load
            if (self._current_series_number is None
                or series_number != self._current_series_number
                or dump_number != self._current_dump_number):

                # file name
                file_search = (
                    self._raw_path
                    + '/' + group_name
                    + '/*_' + h5io.extract_series_name(series_number)
                    + '_F' + str(dump_number).zfill(4) + '.hdf5'
                )

                file_list = glob(file_search)
                if len(file_list) != 1:
                    raise ValueError('ERROR: Unable to get raw data file. '
                                     + 'Something went wrong...')
                self._h5.set_files(file_list[0])


            # intialize
            self._current_admin_info = None
            self._current_traces_data = dict()

            for key_tuple, key_channels in traces_config.items():

                # nb samples
                nb_samples = int(key_tuple[0])
                nb_pretrigger_samples = int(key_tuple[1])
                
                # read traces
                traces, info = (
                    self._h5.read_single_event(
                        event_index,
                        detector_chans=key_channels,
                        trigger_index=trigger_index,
                        trace_length_samples=nb_samples,
                        pretrigger_length_samples=nb_pretrigger_samples,
                        adctoamp=True,
                        include_metadata=True
                    )
                )
                self._current_traces_data[key_tuple] = {
                    'traces': traces,
                    'channels': info['detector_chans']
                }

                if self._current_admin_info is None:
                    self._current_admin_info = info
                else:
                    self._current_admin_info['detector_config'].update(
                        info['detector_config']
                    )

            # update info
            self._current_event_number = event_number
            self._current_series_number = series_number
            self._current_dump_number = dump_number
            self._current_trigger_index = trigger_index


        # if end of file, traces will be empty
        return True


    def update_signal_OF(self, weights=None):
        """
        Update OF base object with traces

        Parameters
        ---------
        None

        Return
        ------
        None

        """

        # loop keys
        for key_tuple, key_dict in self._OF_base_objs.items():

            # clear
            self._OF_base_objs[key_tuple]['OF'].clear_signal()
            
            nb_samples = int(key_tuple[0])
            nb_pretrigger_samples = int(key_tuple[1])
            channels_trace = key_dict['channels_split']
            channels_algorithm = key_dict['channels']
            
            # loop channels and update traces
            for chan in channels_trace:
                
                weights_chan = None
                if (weights is not None
                    and chan in weights):
                    weights_chan = weights[chan]
                    
                # get trace
                trace = self.get_channel_trace(
                    chan,
                    nb_samples=nb_samples,
                    nb_pretrigger_samples=nb_pretrigger_samples,
                    weights=weights_chan
                )
                
                # add to OF base if not yet added
                self._OF_base_objs[key_tuple]['OF'].update_signal(
                    chan, trace,
                    calc_signal_filt=False,
                    calc_q_vector= False,
                    calc_signal_filt_td=False,
                    calc_chisq_amp=False,
                )

            # OF calculations
            for chan in channels_algorithm:

                # channel list
                chan_list, _ = utils.split_channel_name(
                    chan,
                    self._available_channels,
                    separator='|'
                )

                if len(chan_list) == 1:
                    self._OF_base_objs[key_tuple]['OF'].calc_signal_filt(chan)
                    self._OF_base_objs[key_tuple]['OF'].calc_chisq_amp(chan)
                else:
                    self._OF_base_objs[key_tuple]['OF'].calc_signal_filt_mat(chan)
                    self._OF_base_objs[key_tuple]['OF'].calc_signal_filt_mat_td(chan)
                    
                    
    def get_event_admin(self, return_all=False):
        """
        Get event admin info

        Parameters
        ---------
        None

        Return
        ------
        admin_dict : dict
          dictionnary with various admin and trigger variables

        """

        admin_dict = dict()

        # check if event has been read
        if self._current_admin_info is None:
            return admin_dict

        # case dataframe
        if self._is_dataframe:
            if self._current_dataframe_info is None:
                return admin_dict
            else:
                for key, val in self._current_dataframe_info.items():
                    if val is None:
                        val = np.nan
                    admin_dict[key] = val
                return admin_dict

        # return all
        if return_all:
            return self._current_admin_info

        # fill dictionary
        admin_dict['event_number'] = np.int64(self._current_admin_info['event_num'])
        admin_dict['event_index'] = np.int32(self._current_admin_info['event_index'])
        admin_dict['dump_number'] = np.int16(self._current_admin_info['dump_num'])
        admin_dict['series_number'] = np.int64(self._current_admin_info['series_num'])
        admin_dict['event_id'] = np.int32(self._current_admin_info['event_id'])
        admin_dict['event_time'] = np.int64(self._current_admin_info['event_time'])
        admin_dict['run_type'] = self._current_admin_info['run_type']
        admin_dict['data_type'] = self._current_admin_info['run_type']

        # group name
        if self._group_name is not None:
            admin_dict['group_name'] = self._group_name
        else:
            admin_dict['group_name'] = np.nan

        # trigger info
        if 'trigger_type' in self._current_admin_info:
            admin_dict['trigger_type'] = np.int16(self._current_admin_info['trigger_type'])
        else:
            data_mode =  self._current_admin_info['data_mode']
            data_modes = ['cont', 'trig-ext', 'rand', 'threshold']
            if  data_mode  in data_modes:
                admin_dict['trigger_type'] = data_modes.index(data_mode)+1
            else:
                admin_dict['trigger_type'] = np.nan

        if  'trigger_amplitude' in self._current_admin_info:
            admin_dict['trigger_amplitude'] = self._current_admin_info['trigger_amplitude']
        else:
            admin_dict['trigger_amplitude'] = np.nan

        if  'trigger_time' in self._current_admin_info:
            admin_dict['trigger_time'] = self._current_admin_info['trigger_time']
        else:
            admin_dict['trigger_time'] = np.nan


        # fridge run
        if 'fridge_run' in self._current_admin_info:
            admin_dict['fridge_run_number'] = np.int64(
                self._current_admin_info['fridge_run']
            )
        else:
            admin_dict['fridge_run'] = np.nan


        # start times
        if 'fridge_run_start' in self._current_admin_info:
            admin_dict['fridge_run_start_time'] = np.int64(
                self._current_admin_info['fridge_run_start']
            )
        else:
            admin_dict['fridge_run_start_time'] = np.nan


        if 'series_start' in self._current_admin_info:
            admin_dict['series_start_time'] = np.int64(
                self._current_admin_info['series_start']
            )
        else:
            admin_dict['series_start_time'] = np.nan

        if 'group_start' in self._current_admin_info:
            admin_dict['group_start_time'] = np.int64(
                self._current_admin_info['group_start']
            )
        else:
            admin_dict['group_start_time'] = np.nan


        return admin_dict



    def get_channel_settings(self, channel):
        """
        Get channel settings dictionary

        Parameters
        ---------

        channel : str
           channel can be a single channel
           or sum of channels "chan1+chan2"
           or multiple channels "chan1|chan2"


        Return
        ------
        settings_dict : dict
          dictionnary with various detector settings variable

        """

        # initialize output
        settings_dict = dict()


        # check info filled
        if (self._current_admin_info is None or
            'detector_config' not in self._current_admin_info):
            return admin_dict

        #  get channels list
        channels, separator = utils.split_channel_name(
            channel, self._current_admin_info['detector_chans']
        )

        # fill dictionary
        for chan in channels:
            settings_dict['tes_bias_' + chan] =  (
                self._current_admin_info['detector_config'][chan]['tes_bias'])
            settings_dict['output_gain_' + chan] =  (
                self._current_admin_info['detector_config'][chan]['output_gain'])

        return settings_dict



    def get_channel_trace(self, channel,
                          nb_samples=None,
                          nb_pretrigger_samples=None,
                          weights=None):
        """
        Get trace (s)

        Parameters
        ----------

        channel : str
           channel can be a single channel
           or sum of channels "chan1+chan2"
           or multiple channels "chan1|chan2"

        Return:
        -------
        array : ndarray
          array with trace values

        """

        array = None

        #  get channels for case + or | used
        channels, separator = utils.split_channel_name(
            channel, self._current_admin_info['detector_chans']
        )

        weights_array = None
        if weights is not None:
            
            weights_array = np.ones(len(channels))
            
            for ichan, chan in enumerate(channels):
                param = f'weight_{chan}'
                if param not in weights:
                    raise ValueError(
                        f'ERROR: Missing parameter weight {param} '
                        f'for channel {channel}!'
                    )
                val = weights[param]
                weights_array[ichan] = val

        # case full trace
        if nb_samples is None:

            # get array indices
            channel_indices = list()
            for chan in channels:
                channel_indices.append(
                    self._current_admin_info['detector_chans'].index(chan)
                )

            if not channel_indices:
                raise ValueError('Unable to get event  traces for '
                                 + channel)
            # get array
            array = self._current_traces[channel_indices,:]

        else:

            if nb_pretrigger_samples is None:
                raise ValueError(
                    'ERROR: "nb_pretrigger_samples" required!')

            key = (nb_samples, nb_pretrigger_samples)
            if (self._current_traces_data is None
                or key not in self._current_traces_data.keys()):
                raise ValueError('ERROR: Traces not available!')

            channel_indices = list()
            for chan in channels:
                channel_indices.append(
                    self._current_traces_data[key]['channels'].index(chan)
                )

            # get array
            array = self._current_traces_data[key]['traces'][channel_indices,:]


        # Build output
        if separator == '+':
            if weights is not None:
                weights_array = weights_array[:, np.newaxis]
                array = array * weights_array
            array = np.sum(array,axis=0)
            
        elif separator == '-':
            if weights is not None:
                array = (array[0,:]*weights_array[0]
                         - array[1,:]*weights_array[1])
            else:
                array = array[0,:] - array[1,:]
                
        elif separator is None:
            array = array[0,:]

        return array



    def get_template(self, channel, tag='default'):
        """
        Get template from filter file

        Parameters
        ----------

        channel : str
          channel name


        tag : str
          tag/ID of the template
          Default: None

        Return
        ------

        template : ndarray
          array with template values

        metadata : dict 
          psd metadata

        """

        # get template from filter file
        template, template_time, template_metadata = (
            self._filter_data.get_template(
                channel, tag=tag,
                return_metadata=True)
        )

        return template, template_metadata


    def get_psd(self, channel, tag='default'):
        """
        Get psd from filter file

        Parameters
        ----------

        channel : str
          channel name

        tag : str
          tag/ID of the template
          Default: None

        Return
        ------

        psd : ndarray
          array with psd values

        freq : ndarray
          array with psd frequencies

        metadata : dict 
          psd metadata

        """

        psd, psd_freqs, psd_metadata = (
            self._filter_data.get_psd(
                channel,
                tag=tag,
                fold=False,
                return_metadata=True)
        )
    
        return psd, psd_freqs, psd_metadata

    
    def get_facility(self):
        """
        Function to extract facility # from
        metadata

        Parameters
        ----------
        None

        Return
        ------

        facility : int
         facility number

        """
        facility = None
        if 'facility' in self._data_info.keys():
            facility = self._data_info['facility']

        return facility

    def get_sample_rate(self):
        """
        Function to extract sample rate from
        metadata

        Parameters
        ----------
        None

        Return
        ------

        sample_rate : float
         ADC sample rate used to take data


        """
        sample_rate = None
        if 'sample_rate' in self._data_info.keys():
            sample_rate = self._data_info['sample_rate']

        return sample_rate

    def get_nb_samples(self):
        """
        Function to extract number of samples information
        from metadata

        Parameters
        ----------
        None

        Return
        ------

        nb_samples : int
         number of samples of the traces

        """
        nb_samples = None
        if 'nb_samples' in self._data_info.keys():
             nb_samples = self._data_info['nb_samples']

        return nb_samples


    def get_nb_pretrigger_samples(self):
        """
        Function to extract number of pretrigger samples
        information  from metadata

        Parameters
        ----------
        None

        Return
        ------

        nb_samples : int
         number of pretrigger samples



        """
        nb_pretrigger_samples = None
        if 'nb_pretrigger_samples' in self._data_info.keys():
             nb_pretrigger_samples = self._data_info['nb_pretrigger_samples']

        return nb_pretrigger_samples



    def _extract_data_info(self):
        """
        Function to extract all metadata
        from raw data

        Parameters
        ----------
        None

        Return
        ------

        data_info : dict
         dictionary with ADC/data information


        """

        data_info = None

        if not self._raw_files:
            raise ValueError('No file available to get sample rate!')

        for series, files in self._raw_files.items():
            metadata = self._h5.get_metadata(file_name=files[0],
                                             include_dataset_metadata=False)

            adc_name = metadata['adc_list'][0]
            data_info = metadata['groups'][adc_name]
            data_info['comment'] = metadata['comment']
            data_info['facility'] = metadata['facility']
            data_info['run_purpose'] = metadata['run_purpose']
            break

        if data_info is None:
            raise ValueError('ERROR: No ADC info in file. Something wrong...')

        return data_info

    def check_filter_data_tags(self, processing_config,
                               default_tag='default'):
        """
        Filter data tags for OF 
        """

        # loop channels
        config = copy.deepcopy(processing_config)
        for chan, chan_config in config.items():
                   
            # skip if filter file
            if (chan == 'filter_file'
                or not isinstance(chan_config, dict)):
                continue

            # channel list
            chan_list, _ = utils.split_channel_name(
                chan,
                self._available_channels,
                separator='|'
            )

            nb_channels = len(chan_list)
            
            # loop configuration
            for algo, algo_config in chan_config.items():

                # skip if not dictionary
                if not isinstance(algo_config, dict):
                    continue
                
                # skip if disable
                if not algo_config['run']:
                    continue

                # check if algorithm requires OF base
                is_of_base_used = False
                for prefix in self._OF_base_algorithms:
                    algo_base = algo
                    if 'base_algorithm' in algo_config:
                        algo_base = algo_config['base_algorithm']
                    if prefix in algo_base:
                        is_of_base_used = True

                if not is_of_base_used:
                    continue


                if nb_channels == 1:

                    # single channel
                    
                    if 'psd_tag' not in algo_config:
                        processing_config[chan][algo]['psd_tag'] = (
                            default_tag
                        )
                        
                    if 'template_tag' not in algo_config:
                        processing_config[chan][algo]['template_tag'] = (
                            default_tag
                        )

                        
                else:

                    # multi channels
                    if 'csd_tag' not in algo_config:
                        processing_config[chan][algo]['csd_tag'] = (
                            default_tag
                        )

                    # initialize matrix tag
                    processing_config[chan][algo]['template_matrix_tag'] = (
                        None
                    )
                        
                    if 'template_tag' not in algo_config.keys():

                        template_array = None
                        for ichan, chan_tag in enumerate(chan_list):
                            
                            param = f'template_tag_{chan_tag}'
                            
                            if param not in algo_config.keys():
                                raise ValueError(
                                    f'ERROR in the yaml config: Expecting '
                                    f'"{param}" or "template_tag" parameter '
                                    f'for channel {chan}, '
                                    f'algorithm "{algo}"'
                                )
                        
                            tags = algo_config[param]
                            if isinstance(tags, str):
                                tags = [tags]
                                                      
                            if template_array is None:
                                template_array = np.zeros(
                                    (nb_channels, len(tags)),
                                    dtype='object'
                                )
                    
                            template_array[ichan,:] = np.array(tags)
                            
                            # remove 
                            processing_config[chan][algo].pop(
                                param
                            )

                        processing_config[chan][algo]['template_tag'] = (
                            template_array
                        )
                        
                    else:
                        
                        template_tag = algo_config['template_tag']
                    
                        if isinstance(template_tag, list):
                            
                            template_tag = np.array(template_tag)
                            if (template_tag.ndim != 2
                                or template_tag.shape[0] != nb_channels):
                                raise ValueError(
                                    f'ERROR in the yaml config: Expecting '
                                    f'"template_tag" for channel {chan} '
                                    f'to be a (Nchan, Mtemplates) array, '
                                    f'(algorithm "{algo}")'
                                )

                            processing_config[chan][algo]['template_tag'] = (
                                template_tag
                            )
                        
                        elif isinstance(template_tag, str):

                            # get template matrix from filter file
                            template, template_time = (
                                self._filter_data.get_template(
                                    chan, tag=template_tag,
                                    return_metadata=False)
                            )
            
                            if template.ndim != 3:
                                raise ValueError(
                                    f'ERROR: Expecting a 3D templates matrix '
                                    f'for channel {chan}, matrix tag '
                                    f'{template_tag}, '
                                    f'algorithm {algo}!')
                
                            
                            if nb_channels != template.shape[0]:
                                raise ValueError(
                                    f'ERROR: Expecting a templates matrix to have '
                                    f'shape[0] = {nb_channels} '
                                    f'for channel {chan}, '
                                    f'matrix tag {template_tag}, '
                                    f'algorithm {algo}!')
                            
                            nb_templates = template.shape[1]
                            
                            # build tag array
                            tag_array = np.zeros(
                                (nb_channels, nb_templates),
                                dtype='object'
                            )
                                
                            for ichan in range(nb_channels):
                                for itemp in range(nb_templates):
                                    mtag = f'{template_tag}_{algo}_{ichan}_{itemp}'
                                    tag_array[ichan, itemp] = mtag

                            # replace
                            processing_config[chan][algo]['template_matrix_tag'] = (
                                template_tag
                            )
                            processing_config[chan][algo]['template_tag'] = (
                                tag_array
                            )
        
        return processing_config
