import numpy as np
import pandas as pd
import qetpy as qp
import sys
from pprint import pprint
import pytesdaq.io as h5io
import vaex as vx
from glob import glob


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
            filter_inst = h5io.FilterH5IO(filter_file)
            self._filter_data = filter_inst.load()
            if self._verbose:
                print('INFO: Filter file '
                      + filter_file
                      + ' has been successfully loaded!')


        # initialize vaex dataframe
        self._dataframe = None
        self._is_dataframe = False
        if self._trigger_files is not None:
            self._is_dataframe = True

              
        # initialize OF containers
        self._OF_base_objs = dict()
        self._OF_algorithms = dict()
        
        # initialize raw data reader
        self._h5 = h5io.H5Reader()

        # initialize current event traces and metadata 
        self._current_traces  = None
        self._current_admin_info = None
        self._current_dataframe_index = -1
        self._current_dataframe_info = None
        self._current_event_number = None
        self._current_series_number = None
        self._current_trigger_index = None 
            
        # get ADC and file info
        self._data_info = self._extract_data_info()
      


    @property
    def verbose(self):
        return self._verbose

                
        
    def instantiate_OF_base(self, processing_config, channel=None):
        """
        Instantiate QETpy OF base class, perform pre-calculations 
        such as FFT, etc

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
    
        
        # check channel
        if (channel is not None
            and channel not in processing_config.keys()):
            raise ValueError('No channel ' + channel
                             + ' found in configuration file!')
        
        
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

            pretrigger_samples = None
            if 'nb_pretrigger_samples' in chan_config:
                pretrigger_samples = chan_config['nb_pretrigger_samples']

            
            # loop configuration and get list of templates
            for alg, alg_config in chan_config.items():

                # FIXME: only 1x1 OF 
                if (alg.find('of1x')==-1 and alg.find('psd_amp')==-1):
                    continue
                               
                # psd
                psd_tag = 'default'
                if 'psd_tag' in alg_config.keys():
                    psd_tag = alg_config['psd_tag']

                # Initialize dictionary (to store OF object)
                if chan not in self._OF_base_objs.keys():
                    self._OF_base_objs[chan] = dict()

                # instantiate
                if psd_tag not in self._OF_base_objs[chan].keys():
                    
                    # get psd
                    psd, psd_freqs, psd_metadata = self.get_psd(chan, psd_tag)
                    
                    # coupling
                    coupling = 'AC'
                    if 'coupling' in alg_config.keys():
                        coupling = alg_config['coupling']
                        
                        
                    # check sample rate
                    psd_fs = None
                    fs = self.get_sample_rate()
                    if (psd_metadata is not None
                        and 'sample_rate' in psd_metadata):
                        psd_fs = psd_metadata['sample_rate']
                    
                        if (psd_fs is not None
                            and  (psd_fs != fs)):
                            raise ValueError('Sample rate for PSD is '
                                             + 'inconsistent with data!')
                       
                    # instantiate
                    self._OF_base_objs[chan][psd_tag] = qp.OFBase(
                        fs,
                        pretrigger_samples=pretrigger_samples,
                        channel_name=chan
                    )

                    # set psd
                    self._OF_base_objs[chan][psd_tag].set_psd(
                        psd,
                        coupling=coupling,
                        psd_tag=psd_tag)
                        
                                                
                # Add template
                
                # check if there are template tag(s)
                tag_list = list()
                for key in alg_config.keys():
                    if key.find('template_tag')!=-1:
                        tag_list.append(
                            alg_config[key]
                        )

                if not tag_list:
                    tag_list = ['default']

                integralnorm=False
                if 'integralnorm' in alg_config.keys():
                    integralnorm = alg_config['integralnorm']
                        
                for tag in tag_list:
                    template_list = (
                        self._OF_base_objs[chan][psd_tag].template_tags()
                    )
                    if tag not in template_list:
                        # get template from filter file
                        template, template_metadata = (
                            self.get_template(chan, tag)
                        )

                        #  add
                        self._OF_base_objs[chan][psd_tag].add_template(
                            template,
                            template_tag=tag,
                            integralnorm=integralnorm
                        )


                        # template pretrigger
                        pretrigger_samples_temp = None
                        if 'nb_pretrigger_samples' in template_metadata.keys():
                            pretrigger_samples_temp = (
                                int(template_metadata['nb_pretrigger_samples'])
                            )
                        else:
                            # back compatibility....
                            if 'pretrigger_samples' in template_metadata.keys():
                                pretrigger_samples_temp = (
                                    int(template_metadata['pretrigger_samples'])
                                )
                            elif 'pretrigger_length_samples' in template_metadata.keys():
                                pretrigger_samples_temp = (
                                    int(template_metadata['pretrigger_length_samples'])
                                )

                        # check
                        if (pretrigger_samples is not None
                            and  pretrigger_samples_temp is not None):

                            if (pretrigger_samples!=pretrigger_samples_temp):
                                raise ValueError(
                                    'ERROR: template pretrigger length ('
                                    + str(pretrigger_samples_temp) + ') '
                                    + 'is different than pretrigger length defined in '
                                    + 'configuration (yaml) file ('
                                    + str(pretrigger_samples) + ').'
                                    + 'Unable to process!')

                        if pretrigger_samples_temp is not None:
                            pretrigger_samples =  pretrigger_samples_temp
                                

                        # FIXME: need to modify QETpy to add parameter
                        self._OF_base_objs[chan][psd_tag]._pretrigger_samples = (
                            pretrigger_samples
                        )
                        
                                          
                # save algorithm name and psd/signal tag
                if chan not in self._OF_algorithms:
                    self._OF_algorithms[chan] = dict()
                self._OF_algorithms[chan][alg] = psd_tag

            # calculate phi
            if chan in self._OF_base_objs.keys():
                for tag in self._OF_base_objs[chan].keys():
                    self._OF_base_objs[chan][tag].calc_phi()


                    
    def get_OF_base(self, channel, alg_name):
        """
        Get OF object 

        Parameters
        ----------

        channel : str
          channel name 

        alg_name : str
          algorithm name 

        Return
        ------
        
        OF : object 
          optiomal filter base instance

        """
        
        OF = None

        # get OF base
        tag = str()
        if (channel in self._OF_algorithms.keys()
            and alg_name in self._OF_algorithms[channel].keys()):

            tag = self._OF_algorithms[channel][alg_name]

            OF = self._OF_base_objs[channel][tag]

        return OF

        
                
                    
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


        """
        
        # check if channel in filter_file
        if channel not in self._filter_data:
            raise ValueError('No channel ' + channel
                             + ' found in filter file!')

        # parameter name 
        template_name = 'template_' + tag

        # Back compatibility
        if (tag == 'default'
            and template_name not in self._filter_data[channel]):
            psd_name = 'template'


        # check if exist
        if template_name not in self._filter_data[channel]:
            raise ValueError('No template with tag "' + tag
                             + '" found in filter file!'
                             + ' for channel ' + channel)
        

        template = self._filter_data[channel][template_name].values
        
        # metadata
        metadata = None
        metadata_key = template_name + '_metadata'
        if metadata_key in self._filter_data[channel]:
            metadata = self._filter_data[channel][metadata_key]

        return template, metadata


    
    def get_psd(self, channel, tag='default'):
        """
        Get psd from filter file

          
        Parameters
        ----------

        channel : str
          channel name 


        tag : str
          tag/ID of the PSD
          Default: None

        Return
        ------

        psd : ndarray
          array with psd values

        freq : ndarray
          array with psd frequencies


        """
        
        # check if channel in filter_file
        if channel not in self._filter_data:
            raise ValueError('No channel ' + channel
                             + ' found in filter file!')

        # parameter name 
        psd_name = 'psd_' + tag

        # Back compatibility
        if (tag == 'default'
            and psd_name not in self._filter_data[channel]):
            psd_name = 'psd'
            
        if psd_name not in self._filter_data[channel]:
            raise ValueError('No psd with tag "' + tag
                             + '" found in filter file!'
                             + ' for channel ' + channel)

        # get values
        psd_vals = self._filter_data[channel][psd_name].values
        psd_freqs = self._filter_data[channel][psd_name].index

        # metadata
        metadata = None
        metadata_key = psd_name + '_metadata'
        if metadata_key in self._filter_data[channel]:
            metadata = self._filter_data[channel][metadata_key]
            
        return psd_vals, psd_freqs, metadata
    


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

            
        
    def read_next_event(self, channels=None):
        """
        Read next event
        """

        if not self._is_dataframe:
            self._current_traces, self._current_admin_info = (
                self._h5.read_next_event(
                    detector_chans=channels,
                    adctoamp=True,
                    include_metadata=True
                )
            )

        else:

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

            # event index
            event_index = int(event_number%100000)
                     
            # check if new event needs to be loaded
            if (self._current_event_number is None
                or self._current_series_number is None
                or event_number!=self._current_event_number
                or series_number!= self._current_series_number):


                # Read new file if needed
                if (series_number!=self._current_series_number
                    or dump_number!=self._current_dump_number):
                                    
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

                # read
                self._current_traces, self._current_admin_info = (
                    self._h5.read_single_event(
                        event_index,
                        detector_chans=channels,
                        adctoamp=True,
                        include_metadata=True
                    )
                )

            # update info
            self._current_event_number = event_number
            self._current_series_number = series_number
            self._current_dump_number = dump_number
            self._current_trigger_index = trigger_index
                             
        # if end of file, traces will be empty
        if self._current_traces.size != 0:
            return True
        else:
            return False
        

    def update_signal_OF(self, config=None):
        """
        Update OF base object with traces

        Parameters
        ---------
        None

        Return
        ------
        None

        """
        
        # loop OF and update traces
        for channel, channel_dict in self._OF_base_objs.items():

            # get trace length / pre-trigger
            nb_samples = None
            nb_pretrigger_samples = None
            
            if (config is not None
                and channel in config):

                if 'nb_samples' in config[channel]:
                    nb_samples = config[channel]['nb_samples']
                if 'nb_pretrigger_samples' in config[channel]:
                    nb_pretrigger_samples = (
                        config[channel]['nb_pretrigger_samples']
                    )
                
            trace = self.get_channel_trace(
                channel,
                nb_samples=nb_samples,
                nb_pretrigger_samples=nb_pretrigger_samples
            )

            for tag, OF_base in channel_dict.items():
                OF_base.update_signal(trace)
                                

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
        admin_dict['run_type'] = np.int16(self._current_admin_info['run_type'])
        admin_dict['data_type'] = np.int16(self._current_admin_info['run_type'])

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

        
        # channel list 
        channels = list()  
        if '+' in channel:
            channels = channel.split('+')
        elif '|' in channel:
            channels = channel.split('|')
            # elif '-' in channel:
            # channels = channel.split('-')
        else:
            channels = [channel]

        # fill dictionary
        for chan in channels:
            settings_dict['tes_bias_' + chan] =  (
                self._current_admin_info['detector_config'][chan]['tes_bias'])
            settings_dict['output_gain_' + chan] =  (
                self._current_admin_info['detector_config'][chan]['output_gain'])
            
        return settings_dict

     
                
    def get_channel_trace(self, channel,
                          nb_samples=None,
                          nb_pretrigger_samples=None):
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

        array = []

        #  get channels for case + or | used 
        channels = list()
            
        if '+' in channel:
            channels = channel.split('+')
        elif '|' in channel:
            channels = channel.split('|')
            # elif '-' in channel:
            # channels = channel.split('-')
        else:
            channels = [channel]

        # get indicies
        channel_indices = list()
        for chan in channels:
            channel_indices.append(
                self._current_admin_info['detector_chans'].index(chan)
            )

        if not channel_indices:
            raise ValueError('Unable to get event  traces for '
                             + channel)
 
        # build array
        if '+' in channel:
            array = np.sum(self._current_traces[channel_indices,:],
                           axis=0)
        elif '|' in channel:
            array =  self._current_traces[channel_indices,:]
            # elif '-' in channel:
            # if len(channel_indices) != 2:
            #     raise ValueError('ERROR: Unable to calculate subtracted pulse. '
            #                      + ' Two channels needed. Found '
            #                      + str(len(channel_indices)) + ' traces')
            # array = (self._current_traces[channel_indices[0],:]
            #          -self._current_traces[channel_indices[1],:])
        else:
            array =  self._current_traces[channel_indices[0],:]

        # extract trigger
        if self._current_trigger_index is not None:
            
            if (nb_samples is None
                or nb_pretrigger_samples is None):
                raise ValueError('ERROR: Unknow number of '
                                 + '(pretrigger) samples ')

            # min/max index
            trace_min_index = (self._current_trigger_index
                               -nb_pretrigger_samples)
            trace_max_index = trace_min_index + nb_samples

            array = array[...,trace_min_index:trace_max_index]
                   
        return array

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

    
    def get_nb_samples_pretrigger(self):
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
        nb_samples_pretrigger = None
        if 'nb_samples_pretrigger' in self._data_info.keys():
             nb_samples_pretrigger = self._data_info['nb_samples_pretrigger']

        return nb_samples_pretrigger


    
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
