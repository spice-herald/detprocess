import os
import pandas as pd
import numpy as np
from pprint import pprint
import pytesdaq.io as h5io
import matplotlib.pyplot as plt
from numpy.fft import fftfreq, rfftfreq

class FilterData:
    """
    Class to manage filter data 
    """

    def __init__(self, verbose=True):
        """
        Initialize class

        Parameters:
        ----------

        verbose : bool, optional
          display information


        """
        self._verbose = verbose
      
        # filter file data dictionary
        self._filter_data = dict()

      

    @property
    def verbose(self):
        return self._verbose
        
    @verbose.setter
    def verbose(self, value):
        self._verbose=value


        
        
    def describe(self):
        """
        Print filter data info
         
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

        # Let's first loop channel and get tags/display msg
        filter_display = dict()
        parameter_list = ['psd_fold', 'psd', 'template']

        for chan, chan_dict in self._filter_data.items():
            
            if chan not in  filter_display.keys():
                filter_display[chan] = dict()
                
            for par_name, val in chan_dict.items():

                # check if metadata
                if 'metadata' in par_name:
                    continue
                              
                for base_par in parameter_list:
                    
                    if base_par in par_name:
                        tag = par_name.replace(base_par, '')
                        if tag and tag[0] == '_':
                            tag  = tag[1:]
                        else:
                            tag = 'default'
                        
                        if tag not in filter_display[chan]:
                            filter_display[chan][tag] = list()
                            
                        msg = base_par
                        if isinstance(val, pd.Series):
                            msg += ': pandas.Series '
                        elif  isinstance(val, pd.DataFrame):
                            msg += ': pandas.DataFrame '
                        else: 
                            msg += str(type(val)) + ' '
                        msg += str(val.shape)

                        filter_display[chan][tag].append(msg)
                        break
                
        # loop and display
        for chan, chan_vals in filter_display.items():
            print('\nChannel ' + chan + ':')
            for tag, tag_info in chan_vals.items():
                print(' * Tag "' + tag + '":')
                if tag_info:
                    for msg in tag_info:
                        print('    ' + msg)
                                    


    def clear_data(self, channels=None, tag=None):
        """
        clear filter data
        """

        if (channels is None and tag is not None):
            raise ValueError(
                'ERROR: "channels" argument needed when '
                '"tag" is provided'
            )
        
        
        if (channels is None and tag is None):
            self._filter_data.clear()
            self._filter_data = dict()
            
        elif channels is not None:

            if isinstance(channels, str):
                channels = [channels]

            for chan in channels:

                # check if channel exist
                if chan not in self._filter_data.keys():
                    continue

                # remove specific item
                if tag is None:
                    self._filter_data.pop(chan)
                else:
                    key_list = list(
                        self._filter_data[chan].keys()
                    ).copy()
                    
                    for key in key_list:
                        if tag in key:
                            self._filter_data[chan].pop(key)
            
                    
                    
            

                        
                
    def load_hdf5(self, file_name, overwrite=True):
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

        filter_io = h5io.FilterH5IO(file_name)
        data = filter_io.load()

        if self._verbose:
            print('INFO: Loading filter data from file '
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

        # check file name
        if 'filter' not in file_name:
            raise ValueError(
                'ERROR: file name required to include "filter", '
                'for example, "xxxxxx_yyyy_filter.hdf5"')
        
        
        if self._verbose:
            print('INFO: Saving noise data to file '
                  + file_name)
            if overwrite:
                print('INFO: channel data with same tag may be overwritten')

        
        filter_io = h5io.FilterH5IO(file_name)
        filter_io.save_fromdict(self._filter_data,
                                overwrite=overwrite)

        


        
    def get_psd(self, channel, tag='default', fold=False):
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
        
        # check channel
        if (channel not in self._filter_data.keys()):
            msg = ('ERROR: Channel "'
                   + channel + '" not available!')
            
            if self._filter_data.keys():
                msg += ' List of channels in filter file:'
                msg += str(list(self._filter_data.keys()))
                
            raise ValueError(msg)

        
        # parameter name
        psd_name = 'psd' + '_' + tag
        if fold:
            psd_name = 'psd_fold' + '_' + tag

            
        # back compatibility
        if (tag=='default'
            and psd_name not in self._filter_data[channel].keys()):
            psd_name = 'psd'
            if fold:
                psd_name = 'psd_fold'
                

        # check available tag
        if psd_name not in self._filter_data[channel].keys():
            msg = 'ERROR: Parameter not found!'            
            if  self._filter_data[channel].keys():
                msg += ('List of parameters for channel '
                        + channel + ':')
                msg += str(list(self._filter_data[channel].keys()))
                
            raise ValueError(msg)
            
    
        psd_series = self._filter_data[channel][psd_name]
        freq = psd_series.index
        val = psd_series.values

        return val, freq
            
          
    
    def get_template(self, channel, tag='default'):
        """
        Get template for a specific channel

        Parameters:
        ----------

        channel :  str 
           channel name
        
        tag : str, optional
            psd tag, default: No tag

        Return
        ------

        template : ndarray, 
            psd in units of amps
        time  : ndarray
            time array
        """

        # check channel
        if (channel not in self._filter_data.keys()):
            msg = ('ERROR: Channel "'
                   + channel + '" not available! ')

            if self._filter_data.keys():
                msg += 'List of channels in filter file: '
                msg += str(list(self._filter_data.keys()))
                
            raise ValueError(msg)


        
        # parameter name
        template_name = 'template' + '_' + tag


        # back compatibility
        if (tag=='default'
            and template_name not in self._filter_data[channel].keys()):
            template_name = 'psd'
            
        # check available tag
        if template_name not in self._filter_data[channel].keys():
            msg = 'ERROR: Parameter not found!'            
            if self._filter_data[channel].keys():
                msg += ('List of parameters for channel '
                        + channel + ':')
                msg += str(list(self._filter_data[channel].keys()))
                
            raise ValueError(msg)


        template_series = self._filter_data[channel][template_name]
        time = template_series.index
        val = template_series.values

        return val, time



    def set_template(self, channels, array,
                     sample_rate=None,
                     pretrigger_length_msec=None,
                     pretrigger_length_samples=None,
                     metadata=None,
                     tag='default'):
        """
        set template array
        """

        #  check array type/dim
        if isinstance(array, list):
            array = np.array(array)
        elif not isinstance(array, np.ndarray):
            raise ValueError('ERROR: Expecting numpy array!')
        if array.ndim == 1:
            array = array[np.newaxis, :]

        # number of channels
        if isinstance(channels, str):
            channels = [channels]
        nb_channels = len(channels)

        # check array shape
        if (array.shape[0] !=  nb_channels):
            raise ValueError(
                'ERROR: Array shape is not consistent with '
                'number of channels')

        # sample rate / pretrigger length
        if sample_rate is None:
            raise ValueError('ERROR: "sample_rate" argument required!')

        if (pretrigger_length_msec is None
            and pretrigger_length_samples is None):
            raise ValueError('ERROR: pretrigger length (samples or msec)'
                             ' required!')

        if pretrigger_length_msec is not None:
            pretrigger_length_samples = int(
                round(pretrigger_length_msec*sample_rate*1e-3)
            )

            
        # time array
        dt = 1/sample_rate
        t =  np.asarray(list(range(array.shape[-1])))*dt
        

        # parameter name
        template_name = 'template' + '_' + tag

        # metadata
        if metadata is None:
            metadata = dict()
        metadata['sample_rate'] =  sample_rate
        metadata['nb_samples'] = array.shape[1]
        metadata['nb_pretrigger_samples'] = pretrigger_length_samples
                    
        
        # loop channels and store 
        for ichan in range(nb_channels):

            # channel name
            chan = channels[ichan]

                     
            # template
            template = array[ichan,:]

            # add channel
            if chan not in self._filter_data.keys():
                self._filter_data[chan] = dict()
            else:
                # check PSD has same length
                psd_name = 'psd' + '_' + tag
                if psd_name in self._filter_data[chan]:
                    psd = self._filter_data[chan][psd_name].values
                    if len(psd)!=len(template):
                        raise ValueError(
                            'ERROR: template and psd for channel '
                            + chan + ' are required to have same length for '
                            + 'tag ' + tag  + '. Clear data first using '
                            + '"clear_data(...)" or set '
                            + ' template length to ' + str(len(psd))
                        )
                
                
            self._filter_data[chan][template_name] = (
                pd.Series(template, t))
            
            # add channel nanme metadata
            metadata['channel'] = chan
            self._filter_data[chan][template_name + '_metadata'] = metadata
            


    def set_psd(self, channels, array, fold=False,
                sample_rate=None,
                pretrigger_length_msec=None,
                pretrigger_length_samples=None,
                metadata=None,
                tag='default'):
        """
        set psd array
        """
        
        #  check array type/dim
        if isinstance(array, list):
            array = np.array(array)
        elif not isinstance(array, np.ndarray):
            raise ValueError('ERROR: Expecting numpy array!')
        if array.ndim == 1:
            array = array[np.newaxis, :]
            
        # number of channels
        if isinstance(channels, str):
            channels = [channels]
        nb_channels = len(channels)

        # check array shape
        if (array.shape[0] != nb_channels):
            raise ValueError(
                'ERROR: Array shape is not consistent with '
                'number of channels')

        # sample rate 
        if sample_rate is None:
            raise ValueError('ERROR: "sample_rate" argument required!')

        # pretrigger length (not required)
        if pretrigger_length_msec is not None:
            pretrigger_length_samples = int(
                round(pretrigger_length_msec*sample_rate*1e-3)
            )

       
        # parameter name
        psd_name = 'psd' + '_' + tag
        if fold:
            psd_name = 'psd_fold' + '_' + tag
            
        # metadata
        if metadata is None:
            metadata = dict()
        metadata['sample_rate'] =  sample_rate
        metadata['nb_samples'] = array.shape[1]
        if pretrigger_length_samples is not None:
            metadata['nb_pretrigger_samples'] = pretrigger_length_samples
                    
        
        # loop channels and store 
        for ichan in range(nb_channels):

            # channel name
            chan = channels[ichan]

            # psd
            psd = array[ichan,:]
            freqs = None
            if fold:
                freqs = rfftfreq(len(psd), d=1.0/sample_rate)
            else:
                freqs = fftfreq(len(psd), d=1.0/sample_rate)
                

            # add channel
            if chan not in self._filter_data.keys():
                self._filter_data[chan] = dict()
            else:
                # check template/psd have  same length
                template_name = 'template' + '_' + tag
                if template_name in self._filter_data[chan]:
                    template = self._filter_data[chan][template_name].values
                    if len(psd)!=len(template):
                        raise ValueError(
                            'ERROR: template and psd for channel '
                            + chan + ' are required to have same length for '
                            + 'tag ' + tag  + '. Clear data first using '
                            + '"clear_data(...)" or set '
                            + ' psd length to ' + str(len(template))
                        )
                
                
            self._filter_data[chan][psd_name] = (
                pd.Series(psd, freqs))
            
            # add channel nanme metadata
            metadata['channel'] = chan
            self._filter_data[chan][psd_name + '_metadata'] = metadata
            


    
    def plot_template(self, channels, tag='default'):
        """
        Plot template for specified channel(s)

        Parameters:
        ----------
        
        channels :  str or list of str (required)
           channel name or list of channels

        tag : str (optional)
              psd name suffix:  "psd_[tag]" or "psd_fold_[tag]"
              if tag is None, then "psd" or "psd_fold" is used  

        Return:
        -------
        None

        """

        if isinstance(channels, str):
            channels = [channels]

      
            
        # define fig size
        fig, ax = plt.subplots(figsize=(8, 5))
        
        
        for chan in channels:
            template, t = self.get_template(chan, tag=tag)
            if template is None:
                continue
            ax.plot(t*1e3, template, label=chan)
            
        # add axis
        ax.legend()
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.grid(which='minor', linestyle='dotted')
        ax.grid(which='major')
        ax.set_title('Template',fontweight='bold')
        ax.set_xlabel('Time [msec]',fontweight='bold')
     


    def plot_psd(self, channels, tag='default', fold=True, unit='pA'):
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

            if psd is None:
                continue
            
            psd = psd**0.5
            if unit=='pA':
                psd *= 1e12
                
            if fold:
                ax.loglog(freq, psd, label=chan)
            else:
                ax.plot(freq, psd, label=chan)
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
        
