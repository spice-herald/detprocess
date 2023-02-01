import numpy as np
import pandas as pd
import qetpy as qp
import sys
from pprint import pprint
import pytesdaq.io as h5io



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

    def __init__(self, input_files, group_name=None,
                 filter_file=None, verbose=True):
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

        # input files
        self._input_file_dict = input_files

        # group_name
        self._group_name = group_name
        
        # filter data
        self._filter_data = None
        if filter_file is not None:
            filter_inst = h5io.FilterH5IO(filter_file)
            self._filter_data = filter_inst.load()
            if self._verbose:
                print('INFO: Filter file '
                      + filter_file
                      + ' has been successfully loaded!')
                
        
        # initialize OF containers
        self._OF_base_objs = dict()
        self._OF_algorithms = dict()
        
        # initialize raw data reader
        self._h5 = h5io.H5Reader()

        # initialize event traces and metadata 
        self._event_traces  = None
        self._event_info = None

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

        channel : str. optional
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

            
            # loop configuration and get list of templates
            for alg, alg_config in chan_config.items():

                if alg.find('of1x')==-1:
                    continue
                
                # psd
                psd_tag = 'default'
                if 'psd_tag' in alg_config.keys():
                    psd_tag = alg_config['psd_tag']

                # check if OF already instantiated
                if chan not in self._OF_base_objs.keys():
                    self._OF_base_objs[chan] = dict()

                # instantiate
                if psd_tag not in self._OF_base_objs[chan].keys():
                    
                    # get psd
                    psd, psd_fs = self.get_psd(chan, psd_tag)
                    
                    # coupling
                    coupling = 'AC'
                    if 'coupling' in alg_config.keys():
                        coupling = alg_config['coupling']
                        
                        
                    # check sample rate
                    sample_rate = self.get_sample_rate()
                    if (psd_fs is not None
                        and  (psd_fs != sample_rate)):
                        raise ValueError('Sample rate for PSD is '
                                         + 'inconsistent with data!')
                       

                    # get pretrigger samples
                    pretrigger_samples = self.get_nb_samples_pretrigger()


                    # instantiate
                    self._OF_base_objs[chan][psd_tag] = qp.OFBase(
                        sample_rate,
                        pretrigger_samples=pretrigger_samples,
                        channel_name=chan
                    )


                    # set psd
                    self._OF_base_objs[chan][psd_tag].set_psd(
                        psd,
                        coupling=coupling,
                        psd_tag=psd_tag)
                        
                                                

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
                    template_list = self._OF_base_objs[chan][psd_tag].template_tags()
                    if tag not in template_list:
                        # get template from filter file
                        template = self.get_template(chan, tag)

                        #  add
                        self._OF_base_objs[chan][psd_tag].add_template(
                            template,
                            template_tag=tag,
                            integralnorm=integralnorm
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

        
                
                    
    def get_template(self, channel, tag=None):
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

        # check if template in filter file
        template_name = 'template'
        if tag is not None and tag != 'default':
            template_name += '_' + tag

        if template_name not in self._filter_data[channel]:
            raise ValueError('No parameter "' + template_name
                             + '" found in filter file!'
                             + ' for channel ' + channel)

        return self._filter_data[channel][template_name].values


    
    def get_psd(self, channel, tag=None):
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

        # check if psd in filter file
        psd_name = 'psd'
        if tag is not None and tag != 'default':
            psd_name += '_' + tag

        

        if psd_name not in self._filter_data[channel]:
            raise ValueError('No parameter "' + psd_name
                             + '" found in filter file!'
                             + ' for channel ' + channel)

        # FIXME add fs
        return self._filter_data[channel][psd_name].values, None
    


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

        file_list = self._input_file_dict[series]
        
        # Set files
        self._h5.set_files(file_list)
        
    
        
    def read_next_event(self, channels=None):
        """
        Read next event
        """

        self._event_traces, self._event_info = self._h5.read_next_event(
            detector_chans=channels,
            adctoamp=True,
            include_metadata=True
        )    


        # if end of file, traces will be empty
        if self._event_traces.size != 0:
            return True
        else:
            return False
        

    def update_signal_OF(self):
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
            trace = self.get_channel_trace(channel)
            for tag, OF_base in channel_dict.items():
                OF_base.update_signal(trace)
                
                

    def get_event_admin(self):
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

        if self._event_info is None:
            return admin_dict
        
        # fill dictionary
        admin_dict['event_number'] = np.int64(self._event_info['event_num'])
        admin_dict['event_index'] = np.int32(self._event_info['event_index'])
        admin_dict['dump_number'] = np.int16(self._event_info['dump_num'])
        admin_dict['series_number'] = np.int64(self._event_info['series_num'])
        admin_dict['event_id'] = np.int32(self._event_info['event_id'])
        admin_dict['event_time'] = self._event_info['event_time']
        admin_dict['run_type'] = np.int16(self._event_info['run_type'])

        if self._group_name is not None:
            admin_dict['group_name'] = self._group_name
        else:
            admin_dict['group_name'] = np.nan
            
        # trigger info
        if 'trigger_type' in self._event_info:
            admin_dict['trigger_type'] = np.int16(self._event_info['trigger_type'])
        else:
            data_mode =  self._event_info['data_mode']
            data_modes = ['cont', 'trig-ext', 'rand', 'threshold']
            if  data_mode  in data_modes:
                admin_dict['trigger_type'] = data_modes.index(data_mode)+1
            else:
                admin_dict['trigger_type'] = np.nan
        
        if  'trigger_amplitude' in self._event_info:
            admin_dict['trigger_amplitude'] = self._event_info['trigger_amplitude']
        else:
            admin_dict['trigger_amplitude'] = np.nan

        if  'trigger_time' in self._event_info:
            admin_dict['trigger_time'] = self._event_info['trigger_time']
        else:
            admin_dict['trigger_time'] = np.nan 


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
        if (self._event_info is None or
            'detector_config' not in self._event_info):
            return admin_dict

        
        # channel list 
        channels = list()  
        if '+' in channel:
            channels = channel.split('+')
        elif '|' in channel:
            channels = channel.split('|')
        else:
            channels = [channel]

        # fill dictionary
        for chan in channels:
            settings_dict['tes_bias_' + chan] =  (
                self._event_info['detector_config'][chan]['tes_bias'])
            settings_dict['output_gain_' + chan] =  (
                self._event_info['detector_config'][chan]['output_gain'])
            
        return settings_dict

     
                
    def get_channel_trace(self, channel):
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
        else:
            channels = [channel]

        # get indicies
        trace_indices = list()
        for chan in channels:
            trace_indices.append(self._event_info['detector_chans'].index(chan))

        if not trace_indices:
            raise ValueError('Unable to get event  traces for '
                             + channel)

        # build array
        if '+' in channel:
            array = np.sum(self._event_traces[trace_indices,:],
                           axis = 0)
        elif '|' in channel:
            array =  self._event_traces[trace_indices,:]
        else:
            array =  self._event_traces[trace_indices[0],:]

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
        
        if not self._input_file_dict:
            raise ValueError('No file available to get sample rate!')

        for series, files in self._input_file_dict.items():
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
