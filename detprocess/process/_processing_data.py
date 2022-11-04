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
      - event traces and metadata
      - filter file data
      - Optimal filter objects containing template/PSD FFTs, etc.
    """

    def __init__(self, input_files,
                 filter_file=None, verbose=True):
        """
        Intialize data processing 
        
        Arguments
        ---------
        
        config_file: str (required)
           Full path and file name to the YAML settings for the
           processing.
    
        """

        # verbose
        self._verbose = verbose

        # input files
        self._input_file_dict = input_files
      
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
        self._OF1x1 = dict()
        self._OF1x1_algorithms = list()
        
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

                
        
    def instantiate_OF(self, processing_config, channel=None):
        """
        Instantiate QETpy OF class, perform pre-calculations such as FFT, etc
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
        
        
        # OF instantiations
        for chan, chan_config in processing_config.items():

            # skip if filter file
            if (chan == 'filter_file'
                or not isinstance(chan_config, dict)):
                continue

            # skip if not selected channel
            if (channel is not None
                and channel != chan):
                continue

            
            # loop configuration
            for alg, alg_config in chan_config.items():

                # Case OF 1x1
                # FIXME: find better way to identity single pulse OF
                # especially when adding OF, OF NxM, N nsnb, etc.
                if alg.find('of_') != -1:
                    
                    self._OF1x1_algorithms.append(alg)
                    
                    # template
                    template_tag = 'default'
                    if 'template_tag' in alg_config:
                        template_tag = alg_config['template_tag']

                    template = self.get_template(chan, template_tag)

                    # psd
                    psd_tag = 'default'
                    if 'psd_tag' in alg_config:
                        psd_tag = alg_config['psd_tag']

                    psd, psd_fs = self.get_psd(chan, psd_tag)


                    # check sample rate
                    sample_rate = self.get_sample_rate()
                    if (psd_fs is not None
                        and  (psd_fs != sample_rate)):
                        raise ValueError('Sample rate for PSD is inconsistent with data!')
                        
                    
                    # instantiate OF 
                    of_tag = psd_tag + '_' + template_tag
                    if chan not in self._OF1x1:
                        self._OF1x1[chan] = dict()

                    if of_tag not in self._OF1x1[chan]:
                        self._OF1x1[chan][of_tag] = qp.OptimumFilter(
                            template,
                            template,
                            psd,
                            sample_rate
                        )
                        

                    
                        
    def get_OF(self, channel, tag=None, of_type='1x1'):
        """
        Get OF object 
        """
        
        OF = None
        
        # check if available
        if tag is None:
            tag = 'default_defaut'

        # case OF 1x1
        if of_type == '1x1':
        
            if (channel not in self._OF1x1
                or (channel in self._OF1x1
                    and tag not in self._OF1x1[channel])):
                raise ValueError('OF1x1 object not available for channel='
                                 + channel + ', tag='
                                 + tag)
            else:
                OF = self._OF1x1[channel][tag]

        # other case not implemented
        
                
        return  OF 
            
    def get_OF_algorithms(self, of_type='1x1'):
        """
        Get list of algorithms
        """
        if of_type == '1x1':
            return self._OF1x1_algorithms
        
                
                    
    def get_template(self, channel, tag=None):
        """
        Get template from filter file
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
        Set file list, initialize 
        H5 reader 

        Argument:
        --------

        file_list : str or list of str 
          List of HDF5 files (full path) 
        

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
        Update OF with traces
        """
        
        # loop OF1x1 and update traces
        for channel, channel_dict in self._OF1x1.items():
            trace = self.get_channel_trace(channel)
            for tag, OF1x1 in channel_dict.items():
                OF1x1.update_signal(trace)
                
                

    def get_event_admin(self):
        """
        Get event admin info

        Arguments
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
        admin_dict['eventnumber'] = np.int64(self._event_info['event_num'])
        admin_dict['eventindex'] = np.int32(self._event_info['event_index'])
        admin_dict['dumpnumber'] = np.int16(self._event_info['dump_num'])
        admin_dict['seriesnumber'] = np.int64(self._event_info['series_num'])
        admin_dict['eventid'] = np.int32(self._event_info['event_id'])
        admin_dict['eventtime'] = self._event_info['event_time']
        admin_dict['runtype'] = np.int16(self._event_info['run_type'])
      
        
        # trigger info
        if 'trigger_type' in self._event_info:
            admin_dict['triggertype'] = np.int16(self._event_info['trigger_type'])
        else:
            data_mode =  self._event_info['data_mode']
            data_modes = ['cont', 'trig-ext', 'rand', 'threshold']
            if  data_mode  in data_modes:
                admin_dict['triggertype'] = data_modes.index(data_mode)+1
            else:
                admin_dict['triggertype'] = np.nan
        
        if  'trigger_amplitude' in self._event_info:
            admin_dict['triggeramp'] = self._event_info['trigger_amplitude']
        else:
            admin_dict['triggeramp'] = np.nan

        if  'trigger_time' in self._event_info:
            admin_dict['triggertime'] = self._event_info['trigger_time']
        else:
            admin_dict['triggertime'] = np.nan 


        return admin_dict



    def get_channel_settings(self, channel):
        """
        Get channel settings dictionary

        Arguments
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

        Arguments
        ----------

        channel : str
           channel can be a single channel
           or sum of channels "chan1+chan2"
           or multiple channels "chan1|chan2"     
   
        Return:
        -------

        array : ndarray

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
        """
        facility = None
        if 'facility' in self._data_info.keys():
            facility = self._data_info['facility']

        return facility

    def get_sample_rate(self):
        """
        """
        sample_rate = None
        if 'sample_rate' in self._data_info.keys():
            sample_rate = self._data_info['sample_rate']

        return sample_rate
    
    def get_nb_samples(self):
        """
        """
        nb_samples = None
        if 'nb_samples' in self._data_info.keys():
             nb_samples = self._data_info['nb_samples']

        return nb_samples

    
    def get_nb_samples_pretrigger(self):
        """
        """
        nb_samples_pretrigger = None
        if 'nb_samples_pretrigger' in self._data_info.keys():
             nb_samples_pretrigger = self._data_info['nb_samples_pretrigger']

        return nb_samples_pretrigger


    
    def _extract_data_info(self):
        """
        Get ADC info
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
