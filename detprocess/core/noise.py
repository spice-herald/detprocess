import os
import pandas as pd
import numpy as np
from pprint import pprint
import pytesdaq.io as h5io
import qetpy as qp
from glob import glob
import vaex as vx
from pathlib import Path
from detprocess.process.randoms import Randoms
from detprocess.core.filterdata import FilterData
from detprocess.utils import utils

class Noise(FilterData):
    """
    Class to manage noise calculation from 
    randoms for multiple channels
    """

    def __init__(self, verbose=True):
        """
        Initialize class

        Parameters:
        ----------

        verbose : bool, optional
          display information


        """

        # instantiate base class
        super().__init__(verbose=verbose)
        

        # initialize raw data dictionary
        self._raw_data = None
        self._group_name = None
        self._raw_base_path = None
        self._series_list = None
        self._detector_config = None

        
        # intialize randoms dataframe
        self._dataframe = None

        # intialize event list
        self._event_list = None

        # sample rate stored for convenience
        self._fs = None
    
        # offset
        self._offset = dict()
 
        
    def get_detector_config(self, channel):
        """
        get detector config
        """
        if self._detector_config is None:
            print('WARNING: No data has been set yet! '
                  'Returning None ')
            return None
        elif channel not in self._detector_config.keys():
            print(f'WARNING: No channel {channe}  found! '
                  f'Returning None ')
            return None
        return self._detector_config[channel]
                
    def get_sample_rate(self):
        """
        Get sample rate in Hz ("calc_psd" needs to be 
        called before)
        """

        return self._fs


    def get_offset(self, channel):
        """
        Get offset, return None if no offset
        available
        """
        offset = None
        if channel in self._offset:
            offset = self._offset[channel]
        else:
            print(f'WARNING: No offset available for channel '
                  f'{channel}. You need to calculate psd first! '
                  f'Returning None. ')
            
        return offset

    def clear_randoms(self):
        """
        Clear internal data, however
        keep self._filter_data
        """

        # clear data
        self._dataframe = None
        self._event_list = None
        self._raw_data = None
        self._group_name = None
        self._raw_base_path = None
        self._series_list = None
        self._detector_config = None
        self._fs = None
        self._offset = dict()
        
    def set_randoms(self, raw_path, series=None,
                    dataframe=None,
                    event_list=None):
        """
        Set raw data path and vaex dataframe 
        with randoms events (either dataframe directly
        or path to vaex hdf5 files)
        """
        
        # initialize data
        self.clear_randoms()
        

        # check arguments
        if (dataframe is not None
            and event_list is not None):
            raise ValueError('ERROR: choose between "dataframe" and '
                             '"event_list", not both')
        
        # get file list
        raw_data, output_base_path, group_name = (
            self._get_file_list(raw_path,
                                series=series)
        )
        
        if not raw_data:
            raise ValueError('No raw data files were found! '
                             + 'Check configuration...')
        
        # store as internal data
        self._raw_base_path = output_base_path
        self._raw_data = raw_data
        self._group_name = group_name
        self._series_list = list(raw_data.keys())
        self._detector_config = dict()

        # check dataframe
        if dataframe is not None:
            
            if isinstance(dataframe, vx.dataframe.DataFrame):
                if len(dataframe)<1:
                    raise ValueError('ERROR: No event found in the datafame!')
            else:
                dataframe = self._load_dataframe(dataframe)

            self._dataframe = dataframe

        elif event_list is not None:
            self._event_list = event_list

        # check filter data
        if self._filter_data:
            print('WARNING: Some noise data have been previously saved. '
                  'Use "describe()" to check. If needed clear data '
                  'using "clear_data(channels=None, tag=None)" function!')
            
            
    def generate_randoms(self, raw_path, series=None,
                         random_rate=None,
                         nevents=None,
                         min_separation_msec=100,
                         edge_exclusion_msec=50,
                         restricted=False,
                         ncores=1):
        """
        Generate randoms from continuous data
        """

        # initialize data
        self.clear_randoms()

        # get file list
        raw_data, output_base_path, group_name = (
            self._get_file_list(raw_path,
                                series=series,
                                restricted=restricted)
        )
        
        if not raw_data:
            raise ValueError('No raw data files were found! '
                             + 'Check configuration...')
        
        # store as internal data
        self._raw_data = raw_data
        self._group_name = group_name
        self._raw_base_path = output_base_path
        self._series_list = list(raw_data.keys())
        self._detector_config = dict()

        
        # generate randoms
        rand_inst = Randoms(raw_path, series=series, verbose=self._verbose)
        self._dataframe = rand_inst.process(
            random_rate=random_rate,
            nrandoms=nevents,
            min_separation_msec=min_separation_msec,
            edge_exclusion_msec=edge_exclusion_msec,
            lgc_save=False,
            lgc_output=True,
            ncores=ncores
        )

        # check filter data
        if self._filter_data:
            print('WARNING: Some noise data have been previously saved. '
                  'Use "describe()" to check. If needed clear data '
                  'using "clear_data(channels=None, tag=None)" function!')

            
    def calc_psd(self, channels, 
                 series=None,
                 trace_length_msec=None,
                 trace_length_samples=None,
                 pretrigger_length_msec=None,
                 pretrigger_length_samples=None,
                 nevents=None,
                 weights=None,
                 tag='default'):
        """
        Calculate two-sided and folded-over PSD in Amps^2/Hz
        
        """

        # --------------------------------
        # Check arguments
        # --------------------------------
              
        if nevents is None:
            raise ValueError('ERROR: Maximum number of randoms required!'
                             ' Add "nevents" argument.')
        
        # check raw data has been loaded
        if self._raw_data is None:
            raise ValueError('ERROR: No raw data available. Use '
                             + '"set_randoms()" function first!')

        # check trace length
        if (trace_length_msec is not None
            and trace_length_samples is not None):
            raise ValueError('ERROR: Trace length need to be '
                             'in msec OR samples, nto both')

        if (pretrigger_length_msec is not None
            and pretrigger_length_samples is not None):
            raise ValueError('ERROR: Pretrigger length need to be '
                             'in msec OR samples, nto both')


        # available channels
        available_channels = self._get_channel_list()
        
        # --------------------------------
        # Loop channels and calculate PSD
        # --------------------------------
        if isinstance(channels, str):
            channels = [channels]
            
        for chan in channels:

            if self._verbose:
                if series is None:
                    print('INFO: Processing PSD for channel '
                          +  chan)
                else:
                    print('INFO: Processing PSD for channel '
                          +  chan + ' using series '
                          + str(series))


            # let's check if sum of pulses 
            separator = None
            chan_list, separator = utils.split_channel_name(
                chan, available_channels
            )


            # weights
            weights_array = None
            if weights is not None:
            
                weights_array = np.ones(len(chan_list))
            
                for ichan, chan_split in enumerate(chan_list):
                    if chan_split in weights:
                        weights_array[ichan] = weights[chan_split]
            
            # let's do first overall all PSD
            traces, traces_metadata = self._get_traces(
                chan_list,
                nevents=nevents,
                trace_length_msec=trace_length_msec,
                trace_length_samples=trace_length_samples,
                pretrigger_length_msec=pretrigger_length_msec,
                pretrigger_length_samples=pretrigger_length_samples,
                series=series
            )

            self._fs = traces_metadata['sample_rate']

            if separator == '+':
                if weights_array is not None:
                    weights_array = weights_array[np.newaxis, :, np.newaxis]
                    traces = traces * weights_array
                traces = np.sum(traces, axis=1)
                           
            elif separator == '-':
                if weights_array is not None:
                    traces = (traces[:,0,:]*weights_array[0]
                              - traces[:,1,:]*weights_array[1])
                else:
                    traces = traces[:,0,:] - traces[:,1,:]

                print(f'trace shape for {chan} = {traces.shape}')
            elif separator is not None:
                raise ValueError('ERROR: PSD can only be calculated '
                                 'on single channels!')
                
            if traces.ndim==3:
                if traces.shape[1] != 1:
                    raise ValueError('ERROR: Multiple channels. Expecting '
                                     'only one. Something went wrong!')
                traces = traces[:,0,:]
                
            # autocut_noise
            cut = qp.autocuts_noise(traces, fs=self._fs)
        
            if np.sum(cut)==0:
                raise ValueError('ERROR: No events selected after noise autocut! '
                                 + 'Unable to calculate PSD')
                
            cut_eff = np.sum(cut)/len(cut)*100
            if self._verbose:
                print('INFO: Number of events after cuts = '
                      '{}, efficiency = '
                      '{:0.2f}%'.format(np.sum(cut), cut_eff))
                
            # calc PSD
            freqs, psd = qp.calc_psd(traces[cut],
                                     fs=self._fs,
                                     folded_over=False)
        
            # calc offsets
            self._offset[chan] = np.average(np.median(traces[cut], axis=-1))
                       
            # parameter name
            psd_name = 'psd' + '_' + tag
                     
            # save in filter dict as pandas series
            if chan not in self._filter_data.keys():
                self._filter_data[chan] = dict()
                
            self._filter_data[chan][psd_name] = (
                pd.Series(psd, freqs))
               
            # metadata
            traces_metadata['channel'] = chan
            traces_metadata['cut_efficiency'] = cut_eff

            if weights is not None:
                for wchan,wval in weights.items():
                    param = f'weights_{wchan}'
                    traces_metadata[param] = wval
          
            self._filter_data[chan][psd_name + '_metadata'] = traces_metadata
         
            
    def calc_csd(self, channels, 
                 series=None,
                 trace_length_msec=None,
                 trace_length_samples=None,
                 pretrigger_length_msec=None,
                 pretrigger_length_samples=None,
                 nevents=None,
                 tag='default'):
        """
        Calculate two-sided and folded CSD. 
       
        """

        # --------------------------------
        # Check arguments
        # --------------------------------

        if nevents is None:
            raise ValueError('ERROR: Maximum number of randoms required!'
                             ' Add "nevents" argument.')
        
        # check raw data has been loaded
        if self._raw_data is None:
            raise ValueError('ERROR: No raw data available. Use '
                             + '"set_randoms()" function first!')

        # check trace length
        if (trace_length_msec is not None
            and trace_length_samples is not None):
            raise ValueError('ERROR: Trace length need to be '
                             'in msec OR samples, nto both')

        if (pretrigger_length_msec is not None
            and pretrigger_length_samples is not None):
            raise ValueError('ERROR: Pretrigger length need to be '
                             'in msec OR samples, nto both')


        # channels
        if isinstance(channels, str):
            
            if '|' in channels:
                channels = channels.replace(' ','')
                channels = channels.split('|')
            else:
                raise ValueError(
                    'ERROR: At least 2 channels required to calculate csd'
                )

        # build name 
        channel_name = '|'.join(channels)

        
        # -------------
        # Get data and
        # apply cuts
        # -------------
        
        traces, traces_metadata = self._get_traces(
            channels,
            nevents=nevents,
            trace_length_msec=trace_length_msec,
            trace_length_samples=trace_length_samples,
            pretrigger_length_msec=pretrigger_length_msec,
            pretrigger_length_samples=pretrigger_length_samples,
            series=series
        )
        
        # check shape
        if traces.shape[1] != len(channels):
            raise ValueError('ERROR: No all channels found in raw data. ')
        
        self._fs = traces_metadata['sample_rate']
        
        # apply pileup cut
        cut = np.ones(traces.shape[0], dtype=bool)
        for ichan in range(len(channels)):
            
            traces_chan = traces[:, ichan,:]
                                
            # autocut_noise
            cut_chan = qp.autocuts_noise(traces_chan, fs=self._fs)
                      
            if np.sum(cut_chan) == 0:
                raise ValueError(f'ERROR: No events selected after pileup autocut '
                                 f'for channel {channels[ichan]} ')
            cut &= cut_chan
                      
        # check efficiency total cut 
        if np.sum(cut) == 0:
            raise ValueError(f'ERROR: No events selected after pileup cut!')

        cut_eff = np.sum(cut)/len(cut)*100
        if self._verbose:
            print('INFO: Number of events after cuts = '
                  '{}, efficiency = '
                  '{:0.2f}%'.format(np.sum(cut), cut_eff))
            
        # calc CSD two-sided
        freqs, csd = qp.calc_csd(traces[cut],
                                 fs=self._fs,
                                 folded_over=False)
        # parameter name
        csd_name = 'csd_' + tag
        csd_freqs_name = 'csd_freqs_' + tag
                
        # store internal data
        if channel_name  not in self._filter_data.keys():
            self._filter_data[channel_name ] = dict()
            
        self._filter_data[channel_name ][csd_name] = csd
        self._filter_data[channel_name ][csd_freqs_name] = freqs

        # metadata
        traces_metadata['channel'] = channel_name
        traces_metadata['cut_efficiency'] = cut_eff

        names_metadata = [csd_name + '_metadata',
                          csd_freqs_name + '_metadata']
        
        for name in names_metadata:
            self._filter_data[channel_name][name] = traces_metadata
   
      
            
    def _get_traces(self, channels,
                    series=None,
                    trace_length_msec=None,
                    trace_length_samples=None,
                    pretrigger_length_msec=None,
                    pretrigger_length_samples=None,
                    nevents=5000):
        """
        Get raw data traces
        """
        
        # channels
        if isinstance(channels, str):
            channels = [channels]
        nb_channels = len(channels)
        
        # series numbers
        first_series = self._series_list[0]
        series_num = None
        if series is not None:

            # case series number
            if (isinstance(series, int) or
                str(series).isdigit()):
                series_num = int(series)
                series = str(h5io.extract_series_name(series_num))
            else:
                series_num = int(h5io.extract_series_num(series))

            first_series = series

            # check if series in list
            if series not in self._raw_data.keys():
                raise ValueError(
                    f'ERROR: series {series} not found! '
                    f'Check raw data path input.')

        # instantiate data reader
        h5reader = h5io.H5Reader()

        # get ADC info from first file
        first_file =  self._raw_data[first_series][0]
        metadata = h5reader.get_metadata(file_name=first_file)
        adc_name = metadata['adc_list'][0]
        adc_info = metadata['groups'][adc_name]
        fs = adc_info['sample_rate']
        nb_samples_raw = adc_info['nb_samples']
           
        self._detector_config = h5reader.get_detector_config(
            file_name=first_file)
        
        # trace length
        nb_samples = None
        if trace_length_samples is not None:
            nb_samples = trace_length_samples
        elif trace_length_msec is not None:
            nb_samples = h5io.convert_length_msec_to_samples(
                trace_length_msec, fs)
        else:
            if self._dataframe is not None:
                raise ValueError('ERROR: number of samples required!')

        # pretrigger length
        nb_pretrigger_samples = None
        if pretrigger_length_samples is not None:
            nb_pretrigger_samples = pretrigger_length_samples
        elif pretrigger_length_msec is not None:
            nb_pretrigger_samples = h5io.convert_length_msec_to_samples(
                pretrigger_length_msec, fs)
                        
        event_list = None
        series_num_list = None
        if self._event_list is not None:

            # case input event list 
            
            # loop list
            event_list = list()
            for ievent in range(len(self._event_list)):
                event_dict = self._event_list[ievent].copy()
                if series_num is not None:
                    if event_dict['series_number'] == series_num:
                        event_list.append(event_dict)
                else:
                    series_name = str(h5io.extract_series_name(series_num))
                    if  series_name not in self._raw_data.keys():
                        raise ValueError(
                            f'ERROR: series {series_name} from input '
                            f'event_list not found in raw data path!'
                            f'Check input.')
                    else:
                        event_list.append(event_dict)

        elif self._dataframe is not None:
            
            # case dataframe input
            
            event_list = list()
            dataframe = self._dataframe.copy()
         
            # filter based on series
            if series_num is not None:
                cut = dataframe.series_number == series_num
                dataframe = dataframe.filter(cut)

            # filter randoms and edge 
            cut = dataframe.trigger_type == 3

            # cut edge events
            min_indices = dataframe.trigger_index - nb_pretrigger_samples
            max_indices = min_indices + nb_samples -1
            cut_edge = (min_indices >= 0) & (max_indices < nb_samples_raw)
            
            cut = cut & cut_edge
            
            dataframe = dataframe.filter(cut)
           
            # randomly pick randoms
            if len(dataframe)>nevents:
                dataframe = dataframe.sample(n=nevents)
        
            # loop dataframe
            for idx in range(len(dataframe)):

                # even record from dataframe
                event_record = dataframe.to_records(idx)


                # extract event parameters and stored in
                # dictionary
                event_dict = dict()
                event_dict['event_number'] = int(
                    event_record['event_number'])
                event_dict['series_number'] = int(
                    event_record['series_number'])
                event_dict['trigger_index'] = int(
                    event_record['trigger_index'])
                event_dict['group_name'] = str(
                    event_record['group_name'])
                
                # append to list
                event_list.append(event_dict)

        else:

            # case raw data directly
          
            # list of series
            series_num_list = list()
            if series_num is not None:
                series_num_list.append(series_num)
            else:
                for series_name in self._raw_data.keys():
                    series_num_list.append(
                        int(h5io.extract_series_num(series_name))
                    )
                          
        # check number of events
        if event_list is not None:

            if len(event_list) == 0:
                raise ValueError('ERROR: No events selected! Something '
                                 + 'went wrong!')
            
            if len(event_list)>nevents:
                event_list = event_list[:nevents]

            nevents = len(event_list)
      
        raw_path = self._raw_base_path + '/' + self._group_name
        trace_array = h5reader.read_many_events(
            filepath=raw_path,
            nevents=nevents,
            output_format=2,
            detector_chans=channels,
            event_list=event_list,
            series_nums = series_num_list,
            trace_length_samples=nb_samples,
            pretrigger_length_samples=nb_pretrigger_samples,
            include_metadata=False,
            adctoamp=True,
            baselinesub=False)
        
        if self._verbose:
            chan_string = str(channels)
            if len(channels)==1:
                chan_string = channels[0]
                print('INFO: ' + str(trace_array.shape[0])
                      + ' events found in raw data'
                      + ' for channel(s) '
                      + str(chan_string))

        # metadata
        if nb_samples is None:
            nb_samples = nb_samples_raw
        if nb_pretrigger_samples is None:
            nb_pretrigger_samples =  nb_samples//2
            
        trace_metadata = dict()
        trace_metadata['sample_rate'] = fs
        trace_metadata['nb_samples'] = nb_samples
        trace_metadata['nb_pretrigger_samples'] = nb_pretrigger_samples
        trace_metadata['nb_randoms'] = trace_array.shape[0]
        if 'fridge_run' in metadata.keys():
            trace_metadata['fridge_run'] =  metadata['fridge_run']
        if 'comment' in metadata.keys():
            trace_metadata['comment'] =  metadata['comment']

        return trace_array, trace_metadata
        
    
    def _load_dataframe(self, dataframe_path):
        """
        Load vaex dataframe
        """


        # get list of files
        files_dict, base_path, group_name = (
            self._get_file_list(dataframe_path,
                                is_raw=False)
        )

        file_list = list()
        for series,files in files_dict.items():
            if len(files)>0:
                file_list.extend(files)

        dataframe = None
        if file_list:
            dataframe = vx.open_many(file_list)
        else:
            raise ValueError('ERROR: No vaex file found. Check path!')
        
        return dataframe



            
    def _get_file_list(self, file_path,
                       series=None,
                       is_raw=True,
                       restricted=False):
        """
        Get file list from path. Return as a dictionary
        with key=series and value=list of files

        Parameters
        ----------

        file_path : str or list of str 
           raw data group directory OR full path to HDF5  file 
           (or list of files). Only a single raw data group 
           allowed 
        
        series : str or list of str, optional
            series to be process, disregard other data from raw_path

        restricted : boolean
            if True, use restricted data 
            if False (default), exclude restricted data

        Return
        -------
        
        series_dict : dict 
          list of files for splitted inot series

        base_path :  str
           base path of the raw data

        group_name : str
           group name of raw data

        """

        # convert file_path to list 
        if isinstance(file_path, str):
            file_path = [file_path]
            
            
        # initialize
        file_list = list()
        base_path = None
        group_name = None


        # loop files 
        for a_path in file_path:
                   
            # case path is a directory
            if os.path.isdir(a_path):

                if base_path is None:
                    base_path = str(Path(a_path).parent)
                    group_name = str(Path(a_path).name)
                            
                if series is not None:
                    if series == 'even' or series == 'odd':
                        file_name_wildcard = series + '_*.hdf5'
                        file_list = glob(a_path + '/' + file_name_wildcard)
                    else:
                        if not isinstance(series, list):
                            series = [series]
                        for it_series in series:
                            file_name_wildcard = '*' + it_series + '_*.hdf5'
                            file_list.extend(glob(a_path + '/' + file_name_wildcard))
                else:
                    file_list = glob(a_path + '/*.hdf5')
               
                # check a single directory
                if len(file_path) != 1:
                    raise ValueError('Only single directory allowed! ' +
                                     'No combination files and directories')
                
                    
            # case file
            elif os.path.isfile(a_path):

                if base_path is None:
                    base_path = str(Path(a_path).parents[1])
                    group_name = str(Path(Path(a_path).parent).name)
                    
                if a_path.find('.hdf5') != -1:
                    if series is not None:
                        if series == 'even' or series == 'odd':
                            if a_path.find(series) != -1:
                                file_list.append(a_path)
                        else:
                            if not isinstance(series, list):
                                series = [series]
                            for it_series in series:
                                if a_path.find(it_series) != -1:
                                    file_list.append(a_path)
                    else:
                        file_list.append(a_path)

            else:
                raise ValueError('File or directory "' + a_path
                                 + '" does not exist!')
            
        if not file_list:
            raise ValueError('ERROR: No raw input data found. Check arguments!')

        # sort
        file_list.sort()

      
        # convert to series dictionary so can be easily split
        # in multiple cores
        
        series_dict = dict()
        h5reader = h5io.H5Reader()
        series_name = None
        file_counter = 0
        
        for afile in file_list:

            file_name = str(Path(afile).name)
                        
            # skip if filter file
            if 'filter_' in file_name:
                continue

            # skip didv
            if 'didv_' in file_name:
                continue

            # skip iv
            if 'iv_' in file_name:
                continue
                      
            if 'treshtrig_' in file_name:
                continue

            # restricted
            if (restricted
                and 'restricted' not in file_name):
                continue

            # not restricted
            if (not restricted
                and 'restricted' in file_name):
                continue
                      
            # append file if series already in dictionary
            if (series_name is not None
                and series_name in afile
                and series_name in series_dict.keys()):

                if afile not in series_dict[series_name]:
                    series_dict[series_name].append(afile)
                    file_counter += 1
                continue
            
            # get metadata
            if is_raw:
                metadata = h5reader.get_metadata(afile)
                series_name = h5io.extract_series_name(metadata['series_num'])
            else:
                sep_start = file_name.find('_I')
                sep_end = file_name.find('_F')
                series_name = file_name[sep_start+1:sep_end]
                              
            if series_name not in series_dict.keys():
                series_dict[series_name] = list()

            # append
            if afile not in series_dict[series_name]:
                series_dict[series_name].append(afile)
                file_counter += 1
       
            
        if self._verbose:
            msg = ' raw data file(s) from '
            if not is_raw:
                msg = ' dataframe file(s) from '
                
            print('INFO: Found total of '
                  + str(file_counter)
                  + msg
                  + str(len(series_dict.keys()))
                  + ' different series number!')

      
        return series_dict, base_path, group_name

    
    def _get_channel_list(self):
        """ 
        Get the list of channels from raw data file
        
        Parameters
        ----------
        file_dict  : dict
           directionary with list of files for each series

        Return
        -------
        channels: list
          List of channels
    
        """

        if self._raw_data is None:
            raise ValueError('ERROR: Unable to get channel list '
                             'from raw data. No data dictionary '
                             'available!')

        # let's get list of channels available in file
        # first file
        file_name = str()
        for key, val in self._raw_data.items():
            file_name = val[0]
            break
        
        # get list from configuration
        h5 = h5io.H5Reader()
        detector_settings = h5.get_detector_config(file_name=file_name)
        return list(detector_settings.keys())

