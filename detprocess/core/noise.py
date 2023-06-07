import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from pytesdaq.io import FilterH5IO, H5Reader
import qetpy as qp
from glob import glob

class Noise:
    """
    Class to manage noise calculation from 
    randoms for multiple channels
    """

    def __init__(self, traces=None, channels=None,
                 fs=None, verbose=True):
        """
        Initialize class

        Parameters:
        ----------

        verbose : bool, optional
          display information


        """
        self._verbose = verbose


        # initialize containers 
        self._filter_data = dict()
        

    @property
    def verbose(self):
        return self._verbose
        
    @verbose.setter
    def verbose(self, value):
        self._verbose=value


        
    def describe(self):
        """
        Print filter data
         
        Parameters:
        ----------
        None

        Return
        -------
        None


        """

        if not self._filter_data:
            print('No filter data available!')
            return
        
        for chan, chan_dict in self._filter_data.items():
            print('Channel ' + chan + ':')
            for par_name, val in chan_dict.items():

                if isinstance(val, dict):
                    continue
                                
                msg = ' * ' + par_name + ': '
       
                if isinstance(val, pd.Series):
                    msg += 'pandas.Series '
                elif  isinstance(val, pd.DataFrame):
                    msg += 'pandas.DataFrame '
                else: 
                    msg += str(type(val)) + ' '

                msg += str(val.shape)
                             
                print(msg)
            


    def calc_psd(self,
                 file_prefix=None,
                 channels=None,
                 event_list=None,
                 tag=None,
                 invert_pulse=False,
                 nb_events=-1):
        """
        Calculate PSD
        
        """


        # check path
        if not os.path.isdir(raw_path):
            raise ValueError('ERRRO: Path '
                             + raw_path
                             + ' is not a directory!')
    
        
        # get all files
        file_search  = raw_path + '/*.hdf5'
        if file_prefix is not None:
            file_search = raw_path + '/' + file_prefix + '_*.hdf5'
            
        file_list = glob(file_search)
        if len(file_list)<1:
            raise ValueError('ERROR: Unable to find file in directory '
                             + raw_path)

        
        # Instantial data reader
        h5reader = H5Reader()  
        

        # get list of channels
        detector_config = h5reader.get_detector_config(
            file_name=file_list[0]
        )

               
        # metadata
        file_metadata = h5reader.get_metadata(
            file_name=file_list[0]
        )

        metadata = {'run_purpose': file_metadata['run_purpose'],
                    'run_type': file_metadata['run_type'],
                    'comment': file_metadata['comment'],
                    'series_num': file_metadata['series_num']}
        
     
        available_channels = detector_config.keys()
        if len(available_channels)==0:
            raise ValueError('ERROR: Problem with raw data. '
                             +'No channels available.')
        

        if channels is None:
            channels = available_channels
        elif isinstance(channels, str):
            channels = [channels]
            
        
        # loop channels
        for chan in channels:

            # verbose
            if self._verbose:
                print('INFO: Processing noise for channel '
                      + chan)
                

            
            # case sum of channels
            chan_list = [chan]
            if '+' in chan:
                chan_list = chan.split('+')
          
            # remove white space
            for ichan in range(len(chan_list)):
                chan_list[ichan] =  chan_list[ichan].strip()
                
                      
            # loop files and load data for specific channel
            trace_buffer = None
            sample_rate  = None
            for file_name in file_list:
        
                traces, info = h5reader.read_many_events(
                    filepath=file_name,
                    detector_chans=chan_list,
                    output_format=2,
                    include_metadata=True,
                    adctoamp=True)
                
                sample_rate   = info[0]['sample_rate']
                
                if trace_buffer is None:
                    trace_buffer = traces
                else:   
                    trace_buffer =  np.append(trace_buffer,
                                              traces,
                                              axis=0)
                    
                if  (nb_events>-1
                     and trace_buffer.shape[0]>=nb_events):
                    trace_buffer = trace_buffer[0:nb_events,:,:]
                    break

            if (trace_buffer is None
                or trace_buffer.shape[0]==0):
                print('WARNING: No events found for channel '
                      + chan)
                continue
            
            if self._verbose:
                print('INFO: Number of events before cuts = '
                      + str(trace_buffer.shape[0]))

                          
            traces = []
            if len(chan_list)==1:
                traces = trace_buffer[:,0,:]
            else:
                traces = np.sum(trace_buffer[:,:,:], axis=1)

            if invert_pulse:
                 traces *= -1
                 
            # cut
            cut = qp.autocuts(traces, fs=sample_rate)

            if np.sum(cut)==0:
                print('WARNING: No events selected for channel '
                      + chan + '! Skipping channel...')
                continue
                

            
            # verbose
            cut_eff = np.sum(cut)/len(cut)*100
            
            if self._verbose:
                print('INFO: Number of events after cuts = '
                      + str(np.sum(cut)) + ', efficiency = '
                      + str(cut_eff) +'%')
                        
            # calc PSD
            freq, psd = qp.calc_psd(traces[cut],
                                    fs=sample_rate,
                                    folded_over=False)
            
            freq_fold, psd_fold = qp.foldpsd(psd, fs=sample_rate)

          
            # parameter name
            psd_name = 'psd'
            if tag is not None:
                psd_name += '_' + tag

            psd_fold_name = 'psd_fold'
            if tag is not None:
                psd_fold_name += '_' + tag

            # save in filter dict as pandas series
            if chan not in self._filter_data.keys():
                self._filter_data[chan] = dict()
                
            self._filter_data[chan][psd_name] = pd.Series(psd, freq)
            self._filter_data[chan][psd_fold_name] = pd.Series(psd_fold,
                                                               freq_fold)
        
            # metadata (FIXME add parameters)
            self._filter_data[chan][psd_name + '_metadata'] = metadata
            self._filter_data[chan][psd_fold_name + '_metadata'] = metadata

            
            
        # clean up 
        h5reader.clear()
        

                

    
    def load_hdf5(self, file_name, overwrite=False):
        """
        Load filter data from file. Key may be overwritten if 
        already exist
 
        
        Parameters:
        ----------

        file_name : str
           filter file name 
        overwrite : boolean
           if True, overwrite exising data

        Return
        -------
        None

        """

        filter_io = FilterH5IO(file_name)
        data = filter_io.load()

        if self._verbose:
            print('INFO: Loading noise data from file '
                  + file_name)


        
        # update
        if overwrite or not self._filter_data:
            self._filter_data.update(data)
        else:
            for key, item in data.items():
                if key not in self._filter_data:
                    self._filter_data[key] = item
                    continue
                for par_name, value in item.items():
                    if par_name not in self._filter_data[key].keys():
                        self._filter_data[key][par_name] = (
                            data[key][par_name]
                        )

                        
    def save_hdf5(self, file_name, overwrite=False):
        """
        Save filter data to HDF file. Key may be overwritten if 
        already exist
 
        
        Parameters:
        ----------

        file_name : str
           filter file name 
        overwrite : boolean
           if True, overwrite exising data

        Return
        -------
        None
        """

        if self._verbose:
            print('INFO: Saving noise data to file '
                  + file_name)
            if overwrite:
                print('INFO: channel data with same tag may be overwritten')

        
        filter_io = FilterH5IO(file_name)
        filter_io.save_fromdict(self._filter_data,
                                overwrite=overwrite)

        

    def get_psd(self, channel, tag=None, fold=True):
        """
        Get PSD for a specific channel

        Parameters:
        ----------

        channel :  str 
           channel name
        
        tag : str, optional
            psd tag, default: No tag

        fold : boolean, option
            if True, return folded psd

        Return
        ------

        psd : ndarray, 
            psd in units of amps
        f  : ndarray
            psd frequencies

        """

        
        # parameter name
        par_name = 'psd'
        if fold:
            par_name += '_fold'
        if tag is not None:
            par_name += '_' + tag


        # check if available
        if ((channel not in self._filter_data.keys()
             or (channel in self._filter_data.keys()
                 and par_name not in self._filter_data[channel].keys()))):
            print('ERROR: No psd available for channel '
                  + channel)
            return None, None
        

        psd_series = self._filter_data[channel][par_name]
        freq = psd_series.index
        val = psd_series.values

        return val, freq
            
        
             

                        
        
    def plot_psd(self, channels, tag=None, fold=True, unit='pA'):
        """
        Plot PSD for specified channel(s)

        Parameters:
        ----------
        
        channels :  str or list of str (required)
           channel name or list of channels

        tag : str (optional)
              psd name suffix:  "psd_[tag]" or "psd_fold_[tag]"
              if tag is None, then "psd" or "psd_fold" is used  

        fold : bool (optional, default=False)
             if True, plot "psd_fold" parameter
             if False, plot "psd" parameter

        unit : str (optional, default='pA')
            plot in Amps ('A') or pico Amps 'pA')


        Return:
        -------
        None

        """

        if isinstance(channels, str):
            channels = [channels]

      
            
        # define fig size
        fig, ax = plt.subplots(figsize=(8, 5))
        
        
        for chan in channels:

            psd, freq = self.get_psd(chan, tag=tag, fold=fold)
            psd = psd**0.5
            if unit=='pA':
                psd *= 1e12
            ax.loglog(freq, psd, label=chan)
            
        # add axis
        ax.legend()
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.grid(which='minor', linestyle='dotted')
        ax.grid(which='major')
        ax.set_title('Noise PSD',fontweight='bold')
        ax.set_xlabel('Frequency [Hz]',fontweight='bold')
        if  unit=='pA':
            ax.set_ylabel('PSD [pA/rtHz]',fontweight='bold')
        else:
            ax.set_ylabel('PSD [A/rtHz]',fontweight='bold')
        
