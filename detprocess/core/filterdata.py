import os
import pandas as pd
import numpy as np
from pprint import pprint
import pytesdaq.io as h5io
import matplotlib.pyplot as plt
from qetpy.utils import fftfreq, rfftfreq, fold_spectrum
from qetpy.utils import convert_channel_name_to_list, convert_channel_list_to_name
import qetpy.plotting as plotting
from qetpy import calc_corrcoeff_from_csd
import copy


class FilterData:
    """
    Class to manage Template, noise psd, csd and IV/dIdV data 
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
            print('No filter data available! '
                  'Perhaps you need to load data first '
                  '(function load_hdf5(file_name)')
            return

        # Let's first loop channel and get tags/display msg
        filter_display = dict()

        # list of possible parameters
        parameter_list = [
            'psd', 'template',
            'csd',
            'dpdi_2poles', 'dpdi_3poles',
            'ivsweep_data',
            'ivsweep_results_noise',
            'ivsweep_results_didv',
            'didv_results_2poles_fit',
            'didv_results_2poles_params',
            'didv_results_2poles_errors',
            'didv_results_2poles_biasparams',
            'didv_results_2poles_biasparams_infinite_lgain',
            'didv_results_2poles_smallsignalparams',
            'didv_results_2poles_ssp_light',
            'didv_results_3poles_fit',
            'didv_results_3poles_params',
            'didv_results_3poles_errors',
            'didv_results_3poles_biasparams',
            'didv_results_2poles_biasparams_infinite_lgain',
            'didv_results_3poles_smallsignalparams',
            'didv_results_3poles_ssp_light',
            'didv_processing',
            'noise_processing',
        ]
            
        for chan, chan_dict in self._filter_data.items():
            
            if chan not in  filter_display.keys():
                filter_display[chan] = dict()
                
            for par_name, val in chan_dict.items():
                
                # check if metadata
                if '_metadata' in par_name:
                    continue
                
                # check if metadata
                if '_inds' in par_name:
                    continue

                # find tag
                par_split = par_name.split('_')
                tag = par_split[-1]
                base_par = par_name[:-len(tag)-1]
                if (base_par not in parameter_list
                    and len(par_split)>=2):
                    tag = '_'.join(par_split[-2:])
                    base_par = par_name[:-len(tag)-1]
                    if base_par not in parameter_list:
                        tag = 'default'
                        base_par = par_name
                if tag not in filter_display[chan]:
                    filter_display[chan][tag] = list()
                            
                msg = base_par
                if isinstance(val, pd.Series):
                    msg += ': pandas.Series '
                elif  isinstance(val, pd.DataFrame):
                    msg += ': pandas.DataFrame  '
                elif isinstance(val, np.ndarray):
                    ndim = val.ndim
                    msg += f': {ndim}D numpy.array  ' 
                else:
                    msg += (str(type(val)) + ' ')
                            
                msg += str(val.shape)
                
                if (base_par == 'ivsweep_data'
                    and  isinstance(val, pd.DataFrame)
                    and 'state' in val.columns):
                    nb_norm = len(np.where(val['state']=='normal')[0])
                    if nb_norm == 0:
                        nb_norm = 'Unknown'
                    nb_sc = len(np.where(val['state']=='sc')[0])
                    if nb_sc == 0:
                        nb_sc = 'Unknown'
                    msg += '\n       Nb SC points: ' + str(nb_sc)
                    msg += '\n       Nb Normal points: ' + str(nb_norm)
                                                    
                filter_display[chan][tag].append(msg)
                
                
        # loop and display
        channels = list(filter_display.keys())
        print(f'List of channels: {channels}')
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

        if self._verbose:

            msg = ('INFO: Saving filter/TES data to file '
                   + file_name + '!')
            if overwrite:
                msg += ' (overwrite=True)'
            print(msg)
                    
        filter_io = h5io.FilterH5IO(file_name, verbose=False)
        filter_io.save_fromdict(self._filter_data,
                                overwrite=overwrite)

    def get_psd(self, channels, tag='default',
                fold=False,
                return_metadata=False):
        """
        Get PSD for a specific channel in unit of  Amps^2/Hz

        Parameters:
        ----------

        channels :  str  or list of string
           channel name
        
        tag : str, optional
            psd tag, default: No tag

        fold : boolean, option
            if True, return folded psd

        Return
        ------

        psd : ndarray, 
            psd [Amps^2/Hz]
        f  : ndarray
            psd frequencies
        metadata : dict 
        """

        if isinstance(channels, str):
            channels = [channels]

        output_psd = None
        output_psd_freqs = None
        output_metadata = dict()
        
        for chan in channels:
        
            psd, psd_freqs, metadata = self._get_param_array(
                'psd', chan, tag=tag, return_metadata=True)
            output_metadata[chan] = copy.deepcopy(metadata)
            
            if fold:
                sample_rate = None
                if 'sample_rate' in metadata:
                    sample_rate = float(metadata['sample_rate'])
                else:
                    sample_rate = 2*np.max(np.abs(psd_freqs))

                psd_freqs, psd = fold_spectrum(psd, sample_rate)
                
            if output_psd is None:
                output_psd = psd.copy()
                output_psd = output_psd[np.newaxis, :]
                output_psd_freqs = psd_freqs.copy()
            else:

                if (psd.shape[0] != output_psd.shape[-1]
                    or np.any(psd_freqs != output_psd_freqs)):
                    raise ValueError(
                        'ERROR: unable to retrieve psd '
                        'for multiple channels. Inconsistent '
                        'number of samples. Get psd one by one...')
                psd = psd[np.newaxis, :].copy()
                output_psd = np.concatenate((output_psd, psd), axis=0)

        if output_psd.shape[0] == 1:
            output_psd  = np.squeeze(output_psd[0,:])

        if return_metadata:
            if len(channels) == 1:
                output_metadata = output_metadata[channels[0]]
            return output_psd, output_psd_freqs, output_metadata
        else:
            return output_psd, output_psd_freqs
    
    
    def get_csd(self, channels, tag='default',
                fold=False,
                return_metadata=False):
        """
        Get CSD for a specified channel list or string with channels
        separated with "|" in unit of  Amps^2/Hz. Channel order is 
        important!

        Parameters:
        ----------

        channels :  str or list 
           channel list or channel string with "|" separation
        
        tag : str, optional
            psd tag, default: No tag

        fold : boolean, option
            if True, return folded psd

        Return
        ------

        csd : ndarray, 
            csd [Amps^2/Hz]
        f  : ndarray
            csd frequencies
        metadata : dict
          csd metadata if return_metadata=True
        """
        
        # check channel
        channel_list = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nb_channels = len(channel_list)

        if nb_channels < 2:
            raise ValueError(
                'ERROR: At least 2 channels required to calculate csd'
            )
        
        # get values
        output_metadata = dict()
        csd, csd_freqs, metadata = (
            self._get_param_array('csd',
                                  channel_name,
                                  tag=tag,
                                  return_metadata=True)
        )

        
        output_metadata = copy.deepcopy(metadata)


        if fold:
            sample_rate = None
            if 'sample_rate' in metadata:
                sample_rate = float(metadata['sample_rate'])
            else:
                sample_rate = 2*np.max(np.abs(csd_freqs))
                
            csd_freqs, csd = fold_spectrum(csd, sample_rate)

        if return_metadata:
            return csd, csd_freqs, output_metadata
        else:
            return csd, csd_freqs
            
    
    def get_template(self, channel, tag='default',
                     return_metadata=False):
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
        
        # return values
        return self._get_param_array(
            'template',
            channel,
            tag=tag,
            return_metadata=return_metadata)



    def get_dpdi(self, channel, poles, tag='default'):
        """
        Get dpdi for a specific channel in units of Volts

        Parameters:
        ----------

        channel :  str 
           channel name
        
        tag : str, optional
            dpdi tag, default: No tag

        Return
        ------
      
        dpdi : ndarray, 
            dpdi [Volts]

        f  : ndarray
            dpdi frequencies
        
    
        """

        if poles not in [2,3]:
            raise ValueError('ERROR: "poles" should be '
                             '2 or 3!')

        par_name = f'dpdi_{poles}poles'
        
        # return values
        return self._get_param_array(par_name,cchannel,
                                     tag=tag)
            

    def set_template(self, channels, template,
                     sample_rate=None,
                     pretrigger_length_msec=None,
                     pretrigger_length_samples=None,
                     metadata=None,
                     tag='default'):
        """
        set template array
        """

        #  check array type/dim
        if not isinstance(template, np.ndarray):
            raise ValueError('ERROR: "template" argument '
                             'should be a numpy array!')

        channel_list = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nb_channels = len(channel_list)

        # check array shape
        if (nb_channels == 1 and template.ndim != 1):
            raise ValueError(
                'ERROR: For single channel, expecting and 1D '
                'array '
            )
        
        if (nb_channels > 1 and template.ndim != 3):
            raise ValueError(
                'ERROR: For multiple channels, expecting 3D '
                'array [nchans, ntemps, nsamples] '
            )
        
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
        t =  np.asarray(list(range(template.shape[-1])))*dt
   
        # parameter name
        template_name = 'template' + '_' + tag

        # metadata
        if metadata is None:
            metadata = dict()
        metadata['sample_rate'] =  sample_rate
        metadata['nb_samples'] = template.shape[-1]
        metadata['nb_pretrigger_samples'] = pretrigger_length_samples    

        # store
        if channel_name not in self._filter_data.keys():
            self._filter_data[channel_name] = dict()   

        if template.ndim == 1:
            self._filter_data[channel_name][template_name] = (
                pd.Series(template, t)
            )
        else:
            self._filter_data[channel_name][template_name] = template

            # inds
            inds_name = template_name + '_inds'
            self._filter_data[channel_name][inds_name ] = t
                   
        # add channel nanme metadata
        metadata['channel'] = channel_name
        self._filter_data[channel_name][template_name + '_metadata'] = (
            metadata
        )
                

    def set_psd(self, channels, psd, psd_freqs,
                sample_rate=None,
                pretrigger_length_msec=None,
                pretrigger_length_samples=None,
                metadata=None,
                tag='default'):
        """
        set psd array
        """
        
        #  check array type/dim
        if isinstance(psd, list):
                psd = np.array(psd)
        elif not isinstance(psd, np.ndarray):
            raise ValueError('ERROR: Expecting numpy array!')
        if psd.ndim == 1:
            psd = psd[np.newaxis, :]

        # metadata
        if metadata is None:
            metadata = dict()
                
        # check is sample rate in metadata
        if (sample_rate is None and 'sample_rate' in metadata):
                sample_rate = float(metadata['sample_rate'])

        # check frequency array
        if isinstance(psd_freqs, list):
            psd_freqs = np.array(psd_freqs)
        elif not isinstance(psd_freqs, np.ndarray):
            raise ValueError('ERROR: Expecting numpy array '
                             'for "psd_freqs" argument')

        # add dimension if needed
        if psd_freqs.ndim == 1:
            psd_freqs = psd_freqs[np.newaxis, :]

        # check if folded -> NOT ALLOWED
        is_folded = not np.any(psd_freqs<0)
        if is_folded:
            raise ValueError('ERROR: psd needs to be two-sided!')


        sample_rate_array = 2*np.max(np.abs(psd_freqs))
        if sample_rate is None:
            sample_rate = sample_rate_array
        elif sample_rate_array != sample_rate:
            raise ValueError('ERROR: sample_rate is inconsistent with '
                             'frequency array!')
        
        # number of channels
        if isinstance(channels, str):
            channels = [channels]
        nb_channels = len(channels)

        # check array shape
        if psd.shape[0] != nb_channels:
            raise ValueError(
                'ERROR: psd shape is not '
                ' consistent with number of channels')
            
        if psd_freqs.shape[0] !=  nb_channels:
            if psd_freqs.shape[0] == 1:
                psd_freqs = np.repeat(psd_freqs, nb_channels, axis=0)
            else:
                raise ValueError(
                    'ERROR: psd_freqs shape is not '
                    ' consistent with number of channels')


        # parameter name
        psd_name = 'psd' + '_' + tag
                                   
        # add pretrigger length (not required)
        if pretrigger_length_msec is not None:
            pretrigger_length_samples = int(
                round(pretrigger_length_msec*sample_rate*1e-3)
            )
        
        metadata['sample_rate'] =  sample_rate
        metadata['nb_samples'] = psd.shape[1]
        if pretrigger_length_samples is not None:
            metadata['nb_pretrigger_samples'] = pretrigger_length_samples
                
        # loop channels and store 
        for ichan in range(nb_channels):

            # channel name
            chan = channels[ichan]

            # psd
            psd =  np.squeeze(psd[ichan,:])
            freqs  = np.squeeze(psd_freqs[ichan,:])
         
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
                            f'ERROR: template and psd for channel {chan} '
                            f'are required to have same length for '
                            f'tag {tag}. Use a different tag, clear previous data '
                            f'first using "clear_data(...)" or set '
                            f'psd length to {len(template)}')
                           
            self._filter_data[chan][psd_name] = (
                pd.Series(psd, freqs))
            
            # add channel nanme metadata
            metadata['channel'] = chan
            self._filter_data[chan][psd_name + '_metadata'] = metadata

            
    def set_csd(self, channels, csd, csd_freqs,
                sample_rate=None,
                pretrigger_length_msec=None,
                pretrigger_length_samples=None,
                metadata=None,
                tag='default'):
        """
        set csd array
        """

        # channels
        channel_list = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
        nb_channels = len(channel_list)

        
        # metadata
        if metadata is None:
            metadata = dict()
        
        #  check array type/dim
        if (not isinstance(csd, np.ndarray)
            or csd.ndim != 3):
            raise ValueError('ERROR: Expecting a 3D numpy array!')
        
        if (len(channels) != csd.shape[0]
            or csd.shape[0] != csd.shape[1]):
            raise ValueError('ERROR: Array shape not consistent with '
                             'number of channels!')

        # check frequency array
        if isinstance(csd_freqs, list):
            csd_freqs = np.array(csd_freqs)
        elif not isinstance(csd_freqs, np.ndarray):
            raise ValueError('ERROR: Expecting a numpy array '
                             'for "csd_freqs" argument')
        # squeeze
        if csd_freqs.ndim == 2:
            csd_freqs = np.squeeze(csd_freqs[0,:])
            
        # check if folded -> NOT ALLOWED
        is_folded = not np.any(csd_freqs<0)
        if is_folded:
            raise ValueError('ERROR: psd needs to be two-sided!')


        sample_rate_array = 2*np.max(np.abs(csd_freqs))
        if sample_rate is None:
            sample_rate = sample_rate_array
        elif sample_rate_array != sample_rate:
            raise ValueError('ERROR: sample_rate is inconsistent with '
                             'frequency array!')
                
        # parameter name
        csd_name = 'csd_' + tag
        csd_freqs_name = csd_name + '_inds'
        
        # add pretrigger length (not required)
        if pretrigger_length_msec is not None:
            pretrigger_length_samples = int(
                round(pretrigger_length_msec*sample_rate*1e-3)
            )
        metadata['sample_rate'] =  sample_rate
        metadata['nb_samples'] = csd.shape[1]
        if pretrigger_length_samples is not None:
            metadata['nb_pretrigger_samples'] = pretrigger_length_samples
        metadata['channel'] = channel_name           

        # add channel
        if channel_name not in self._filter_data.keys():
            self._filter_data[channel_name] = dict()
                         
        self._filter_data[channel_name][csd_name] = csd
        self._filter_data[channel_name][csd_name + '_metadata'] = metadata
        self._filter_data[channel_name][csd_freqs_name] = csd_freqs
          
            
    def set_dpdi(self, channels,
                 dpdi, dpdi_freqs,
                 poles,
                 sample_rate=None,
                 metadata=None,
                 tag='default'):
        """
        set dpdi array
        """

        # check poles
        if poles not in [2,3]:
            raise ValueError('ERROR: Poles should be '
                             '2 or 3!')
        
        #  check array type/dim
        if isinstance(dpdi, list):
            dpdi = np.array(dpdi)
        elif not isinstance(dpdi, np.ndarray):
            raise ValueError('ERROR: Expecting numpy array!')
        
        if dpdi.ndim == 1:
            dpdi = dpdi[np.newaxis, :]

        #  check array type/dim
        if isinstance(dpdi_freqs, list):
            dpdi_freqs = np.array(dpdi_freqs)
        elif not isinstance(dpdi_freqs, np.ndarray):
            raise ValueError('ERROR: Expecting numpy array!')
        
        if dpdi_freqs.ndim == 1:
            dpdi_freqs = dpdi_freqs[np.newaxis, :]
           
        # check if folded -> NOT ALLOWED
        is_folded = not np.any(dpdi_freqs<0)
        if is_folded:
            raise ValueError('ERROR: dpdi needs to be two-sided!')

        sample_rate_array = 2*np.max(np.abs(dpdi_freqs))
        if sample_rate is None:
            sample_rate = sample_rate_array
        elif sample_rate_array != sample_rate:
            raise ValueError(f'ERROR: sample rate ({sample_rate}) '
                             f'is inconsistent with sample rate from '
                             f'frequency array ({sample_rate_array})!')
        
        # number of channels
        if isinstance(channels, str):
            channels = [channels]
        nb_channels = len(channels)

        # check array shape
        if dpdi.shape[0] != nb_channels:
            raise ValueError(
                'ERROR: Array shape is not consistent with '
                'number of channels')
        
        if dpdi_freqs.shape[0] !=  nb_channels:
            if dpdi_freqs.shape[0] == 1:
                dpdi_freqs = np.repeat(dpdi_freqs, nb_channels, axis=0)
            else:
                raise ValueError(
                    'ERROR:dpdi_freqs shape is not '
                    ' consistent with number of channels')
            
        # parameter name
        dpdi_name = f'dpdi_{poles}poles_{tag}'
                           
        # metadata
        if metadata is None:
            metadata = dict()
        metadata['sample_rate'] =  sample_rate
        metadata['nb_samples'] = dpdi.shape[-1]
        metadata['poles'] = poles
        
        # loop channels and store 
        for ichan in range(nb_channels):

            # channel name
            chan = channels[ichan]

            # psd
            dpdi_chan = np.squeeze(dpdi[ichan,:])
            dpdi_freqs_chan = np.squeeze(dpdi_freqs[ichan,:])
            
            # add channel if needed
            if chan not in self._filter_data.keys():
                self._filter_data[chan] = dict()

            self._filter_data[chan][dpdi_name] = (
                pd.Series(dpdi_chan, dpdi_freqs_chan)
            )
            
            # add channel name metadata
            metadata['channel'] = chan
            self._filter_data[chan][dpdi_name + '_metadata'] = metadata

            
    def set_ivsweep_data(self, 
                         channel,
                         dataframe,
                         metadata=None,
                         tag='default'):
        """
        Set IV-dIdV Sweep processed dataframe
        """


        # check dataframe
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError(
                'ERROR: Input is not a pandas Datafame!')
                
        # create channel dictionary
        if channel not in self._filter_data.keys():
            self._filter_data[channel] = dict()
        
        # data 
        data_tag = 'ivsweep_data_' + tag
        self._filter_data[channel][data_tag] = dataframe

        # metadata
        if metadata is not None:
            metadata.update({'channel': channel})
        else:
            metadata = {'channel': channel}
        self._filter_data[channel][data_tag + '_metadata'] = metadata

        
    def set_ivsweep_data_from_dict(self, data_dict,
                                   tag='default'):
        """
        Set IV-dIdV sweep data from dictionary
        (key=channel name, value=datframe)
        """

        for chan, df in data_dict.items():
            self.set_ivsweep_data(chan, df, tag=tag)
            
    def get_ivsweep_data(self, 
                         channel,
                         tag='default'):
        """
        Get IV-dIdV Sweep processed dataframe
        """

        # check channels
        if channel not in self._filter_data.keys():
            raise ValueError(
                f'ERROR: no channel {channel} available! '
                'Did you load from file first?')
        
        data_tag = 'ivsweep_data_' + tag
        if data_tag not in self._filter_data[channel].keys():
            raise ValueError(
                f'ERROR: no sweep data for channel {channel} available! '
                'Did you load from file first?')

        return self._filter_data[channel][data_tag]
        
    
    def set_ivsweep_results(self, 
                           channel,
                           results,
                           iv_type,
                           metadata=None,
                           tag='default'):
        """
        Set IV-dIdV Sweep analysis results
        (independent of bias point)
        """

        # check input
        if isinstance(results, dict):
            results = pd.Series(results)  
        elif not isinstance(results, pd.Series):
            raise ValueError(
                'ERROR: Input is not a pandas Series!')
        
        # create channel dictionary
        if channel not in self._filter_data.keys():
            self._filter_data[channel] = dict()
        
        # data 
        data_tag = 'ivsweep_results_' + iv_type  + '_' + tag
        self._filter_data[channel][data_tag] = results

        # metadata
        if metadata is not None:
            metadata.update({'channel': channel})
        else:
            metadata = {'channel': channel}
        self._filter_data[channel][data_tag + '_metadata'] = metadata


    def get_ivsweep_results(self, 
                            channel,
                            iv_type='noise',
                            include_bias_parameters=False,
                            tes_bias=None,
                            lgc_return_series=False,
                            tag='default'):
        """
        Get IV-dIdV Sweep result
        """
        
        # check channels
        if channel not in self._filter_data.keys():
            raise ValueError(
                f'ERROR: no channel {channel} available! '
                'Did you load from file first?')

        # check argument
        if (include_bias_parameters
            and tes_bias is None):
            raise ValueError(
                f'ERROR: "tes_bias" needs to be provided '
                f'when "include_bias_parameters" = True!')
        
        # result data tag
        data_tag = 'ivsweep_results_' + iv_type  + '_' + tag
        if data_tag not in self._filter_data[channel].keys():
            iv_type_new = str()
            if iv_type == 'noise':
                iv_type_new = 'didv'
            else:
                iv_type_new == 'noise'
            data_tag = 'ivsweep_results_' + iv_type_new  + '_' + tag
            is_avaible = data_tag in self._filter_data[channel].keys()

            if is_avaible:
                raise ValueError(
                    f'ERROR: No sweep results for channel {channel} available '
                    f'using data type "{iv_type}"! Change "iv_type'
                    f'argument to {iv_type_new}')
            else:
                raise ValueError(
                    f'ERROR: No sweep results for channel {channel} available. Did '
                    'you run the sweep analysis?')

        results = self._filter_data[channel][data_tag].to_dict()
      
        # include bias parameters
        if include_bias_parameters:

            # get dataframe
            df = self.get_ivsweep_data(channel, tag=tag)
            absolute_difference = abs(df['tes_bias'] - tes_bias)
            closest_index = absolute_difference.idxmin()
            params = df.loc[closest_index].to_dict()
            
            # add parameters
            results['tes_bias'] = params['tes_bias']
            results['ibias'] = params['ibias_true_' + iv_type]
            results['ibias_err'] = params['ibias_true_err_' + iv_type]
            results['i0'] = params['i0_' + iv_type]
            results['i0_err'] = params['i0_err_' + iv_type]
            results['r0'] = params['r0_' + iv_type]
            results['r0_err'] = params['r0_err_' + iv_type]
            results['p0'] = params['p0_' + iv_type]
            results['p0_err'] = params['p0_err_' + iv_type]

            # add temperature
            temperature_list = ['mc','cp','still']
            for temp in temperature_list:
                temp_par = 'temperature_' + temp
                temp_val = np.nan
                if temp_par in params:
                    results[temp_par] = params[temp_par]
             
            # infinite loop gain
            if 'didv_3poles_r0_infinite_lgain' in params:

                results['i0_infinite_lgain'] = (
                    params['didv_3poles_i0_infinite_lgain']
                )
                results['i0_err_infinite_lgain'] = (
                    params['didv_3poles_i0_err_infinite_lgain']
                )
                
                results['r0_infinite_lgain'] = (
                    params['didv_3poles_r0_infinite_lgain']
                )
                results['r0_err_infinite_lgain'] = (
                     params['didv_3poles_r0_err_infinite_lgain']
                )
                
                results['p0_infinite_lgain'] = (
                    params['didv_3poles_p0_infinite_lgain']
                )
                results['p0_err_infinite_lgain'] = (
                    params['didv_3poles_p0_err_infinite_lgain']
                )
                    
            # ssp
            didv_parameters = ['chi2',
                               'tau+','tau-','tau3',
                               'l', 'l_err',
                               'beta', 'beta_err',
                               'gratio', 'gratio_err',
                               'tau0','tau0_err',
                               'L','L_err']
            
            for model_poles in [2,3]:

                for didv_par in didv_parameters:
                    par_name = f'didv_{model_poles}poles_{didv_par}'
                    if par_name in params:
                        results[par_name] = params[par_name]
                                  
            if 'resolution_dirac' in params:
                results['resolution_dirac'] =  params['resolution_dirac']
                results['resolution_collection_efficiency'] = (
                    params['resolution_collection_efficiency']
                )
            if 'resolution_template' in params:
                results['resolution_template'] =  params['resolution_template']
                
        if lgc_return_series:
            results = pd.Series(results)

        return results



    def set_didv_results(self, 
                         channel,
                         results,
                         poles,
                         metadata=None,
                         tag='default'):
        """
        Set results from dIdV fit
        """

        # check input
        if not isinstance(results, dict):
            raise ValueError(
                'ERROR: "results" argument should be a dictionary!')
        
        if not isinstance(poles, int):
            raise ValueError(
                'ERROR: "poles" argument should be an interger!')
        
        # create channel dictionary
        if channel not in self._filter_data.keys():
            self._filter_data[channel] = dict()

        # base name, poles
        base_name = 'didv_results_' + str(poles) + 'poles'

        # metadata
        if metadata is not None:
            metadata.update({'channel': channel})
        else:
            metadata = {'channel': channel}
        
        # sub-dictionaries
        subdict_list = ['biasparams', 'biasparams_infinite_lgain',
                        'errors','params', 'smallsignalparams',
                        'ssp_light']
        
        # fit results
        fit_data = dict()
        for par_name, par_val  in results.items():
            if not isinstance(par_val, dict):
                fit_data[par_name] = par_val

        if fit_data:
            pd_series  = pd.Series(fit_data)
            data_name =  base_name + '_fit_' + tag
            self._filter_data[channel][data_name] = pd_series
            self._filter_data[channel][data_name + '_metadata'] = (
                metadata
            )
            
        for keyname in subdict_list:

            if (keyname not in results.keys() or
                not isinstance(results[keyname], dict)):
                continue
            
            data_dict = results[keyname]
            if (keyname == 'ssp_light'
                and 'vals' in results['ssp_light'].keys()):
                data_dict = dict()
                data_dict['cov'] = results['ssp_light']['cov']
                data_dict.update(results['ssp_light']['vals'])
                data_dict.update(results['ssp_light']['sigmas'])
                
            pd_series  = pd.Series(data_dict)
            data_name =  base_name + '_' + keyname + '_' + tag
            self._filter_data[channel][data_name] = pd_series
            self._filter_data[channel][data_name + '_metadata'] = (
                metadata
            )
            
        
    def get_didv_results(self, 
                         channel,
                         poles,
                         tag='default'):
        """
        Get dIdV fit results
        """
        
        # check channels
        if channel not in self._filter_data.keys():
            raise ValueError(
                f'ERROR: no channel {channel} available! '
            )

        output_data = dict()
        
        # base name 
        base_name = 'didv_results_' + str(poles) + 'poles'
        
        # fit
        par_name =  base_name + '_fit_' + tag
        if par_name not in self._filter_data[channel].keys():
            raise ValueError(f'ERROR: No dIdV {poles}-poles fit results '
                             f'for channel {channel}!')

        output_data.update(self._filter_data[channel][par_name].to_dict())
        output_data['metadata'] = (
            self._filter_data[channel][par_name + '_metadata']
        )
                
        # other par list
        par_list = ['biasparams', 'biasparams_infinite_lgain',
                    'errors','params', 'smallsignalparams',
                    'ssp_light']

        for par in par_list:
            par_name =   base_name + '_' + par + '_' + tag
            if par_name in self._filter_data[channel].keys():
                output_data[par] = self._filter_data[channel][par_name].to_dict()


        if 'ssp_light' in output_data:
            ssp_light = {'vals':dict(), 'sigmas':dict()}
            for par, val in output_data['ssp_light'].items():
                if par == 'cov':
                    continue
                if 'sigma' in par:
                    ssp_light['sigmas'][par] = val
                else:
                    ssp_light['vals'][par] = val

            output_data['ssp_light'].update(ssp_light)
                        
        return  output_data

    def set_didv_dataframe(self, 
                           channel,
                           dataframe,
                           metadata=None,
                           tag='default'):
        """
        Set dIdV processing 
        """


        # check dataframe
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError(
                'ERROR: Input is not a pandas Datafame!')
                
        # create channel dictionary
        if channel not in self._filter_data.keys():
            self._filter_data[channel] = dict()
        
        # data 
        data_tag = 'didv_processing_' + tag
        self._filter_data[channel][data_tag] = dataframe

        # metadata
        if metadata is not None:
            metadata.update({'channel': channel})
        else:
            metadata = {'channel': channel}
            
        self._filter_data[channel][data_tag + '_metadata'] = metadata

    def get_didv_dataframe(self, 
                           channel,
                           tag='default'):
        """
        Get dIdV processed dataframe
        """

        # check channels
        if channel not in self._filter_data.keys():
            raise ValueError(
                f'ERROR: no channel {channel} available! '
                'Did you load from file first?')
        
        data_tag = 'didv_processing_' + tag
        if data_tag not in self._filter_data[channel].keys():
            raise ValueError(
                f'ERROR: no dIdV data for channel {channel} available! '
                'Did you load from file first?')

        return self._filter_data[channel][data_tag]

    
    def set_noise_dataframe(self, 
                           channel,
                           dataframe,
                           metadata=None,
                           tag='default'):
        """
        Set noise processing 
        """


        # check dataframe
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError(
                'ERROR: Input is not a pandas Datafame!')
                
        # create channel dictionary
        if channel not in self._filter_data.keys():
            self._filter_data[channel] = dict()
        
        # data 
        data_tag = 'noise_processing_' + tag
        self._filter_data[channel][data_tag] = dataframe

        # metadata
        if metadata is not None:
            metadata.update({'channel': channel})
        else:
            metadata = {'channel': channel}
            
        self._filter_data[channel][data_tag + '_metadata'] = metadata

        
    def get_noise_dataframe(self, 
                           channel,
                           tag='default'):
        """
        Get noise processed dataframe
        """

        # check channels
        if channel not in self._filter_data.keys():
            raise ValueError(
                f'ERROR: no channel {channel} available! '
                'Did you load from file first?')
        
        data_tag = 'noise_processing_' + tag
        if data_tag not in self._filter_data[channel].keys():
            raise ValueError(
                f'ERROR: no noise data for channel {channel} available! '
                'Did you load from file first?')

        return self._filter_data[channel][data_tag]
              
    
    def plot_template(self, channels,
                      xmin=None, xmax=None,
                      tag='default',
                      figsize=(8,5)):
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
        fig, ax = plt.subplots(figsize=figsize)
        
        
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
        ax.set_title('Template', fontweight='bold')
        ax.set_xlabel('Time [msec]', fontweight='bold')

        if (xmin is not None or xmax is not None):
            ax.set_xlim(xmin=xmin, xmax=xmax)
        

            
    def plot_psd(self, channels, tag='default',
                 unit='pA', figsize=(8,5)):
        """
        Plot PSD for specified channel(s)

        Parameters:
        ----------
        
        channels :  str or list of str (required)
           channel name or list of channels

        tag : str (optional)
              psd name suffix:  "psd_[tag]" or "psd_fold_[tag]"
              if tag is None, then "psd" or "psd_fold" is used  

        unit : str (optional, default='pA')
            plot in Amps ('A') or pico Amps 'pA')


        Return:
        -------
        None

        """

        if isinstance(channels, str):
            channels = [channels]
            
        # define fig size
        fig, ax = plt.subplots(figsize=figsize)

        for chan in channels:

            psd, freq = self.get_psd(chan, tag=tag, fold=True)

            if psd is None:
                continue

            # convert to A/rtHz
            psd = psd**0.5
            
            if unit=='pA':
                psd *= 1e12
                
            ax.loglog(freq, psd, label=chan)
            
        # add axis
        ax.legend()
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.grid(which='minor', linestyle='dotted')
        ax.grid(which='major')
        if tag != 'default':
            ax.set_title(f'Noise PSD ("{tag}")',
                         fontweight='bold')
        else:
            ax.set_title(f'Noise PSD',fontweight='bold')
        
        ax.set_xlabel('Frequency [Hz]',fontweight='bold')
        if  unit=='pA':
            ax.set_ylabel('PSD [pA/rtHz]',fontweight='bold')
        else:
            ax.set_ylabel('PSD [A/rtHz]',fontweight='bold')
        

    def plot_csd(self, channels, whichcsd=['01'], lgcreal=True,
                 lgcsave=False, savepath=None, figsize=(8,5),
                 tag='default'):
        """
        Plot CSD elements
        """

        csd, csd_freqs = self.get_csd(channels, tag=tag, fold=True)
        
        if isinstance(channels, str):
            channels = channels.split('|')

        plotting.plot_csd(csd=csd.copy(),
                          csd_freqs=csd_freqs.copy(),
                          channels=channels,
                          whichcsd=whichcsd, lgcreal=lgcreal,
                          lgcsave=lgcsave,
                          savepath=savepath,
                          figsize=figsize)
        
                    
    def plot_corrcoeff(self, channels, lgcsmooth=True, nwindow=7,
                       lgcsave=False, savepath=None, figsize=(8,5),
                       tag='default'):
        """
        Plot correlation coefficient
        """

        csd, csd_freqs = self.get_csd(channels, tag=tag, fold=True)
        corrcoeff = calc_corrcoeff_from_csd(csd.copy())


        if isinstance(channels, str):
            channels = channels.split('|')
            
        plotting.plot_corrcoeff(
            corrcoeff=corrcoeff.copy(),
            corrcoeff_freqs=csd_freqs.copy(),
            channels=channels,
            lgcsmooth=lgcsmooth,
            nwindow=nwindow, lgcsave=lgcsave,
            savepath=savepath, figsize=figsize)


    def plot_ivsweep_offset(self, channel, tag='default'):
        """
        Plot offset vs tes_bias with errors from IV and if available
        dIdV offset
        """


        # get data frame
        df =  self.get_ivsweep_data(channel=channel,
                                    tag=tag)


        bias = df['tes_bias_uA'].values
        offset_noise = None
        offset_noise_err = None
        offset_didv = None
        offset_didv_err = None
        if 'offset_noise' in df.columns:
            offset_noise = df['offset_noise'].values*1e6
            offset_noise_err = df['offset_err_noise'].values*1e6
        if 'offset_didv' in df.columns:
            offset_didv = df['offset_didv'].values*1e6
            offset_didv_err = df['offset_err_didv'].values*1e6

        # Plotting the data with error bars
        if offset_noise is not None:
            plt.errorbar(bias, offset_noise,
                         yerr=offset_noise_err,
                         fmt='o', color='b',capsize=5,
                         label='Offset from noise')
        if offset_didv is not None:
            plt.errorbar(bias, offset_didv,
                         yerr=offset_didv_err,
                         fmt='+', color='r', capsize=5,
                         label='Offset from didv')            

        # Some basic plot settings
        plt.title(f'TES bias sweep {channel}',fontweight='bold')
        plt.xlabel('TES bias [uA]',fontweight='bold')
        plt.ylabel('Offset [uA]', fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

        
    def _get_param_array(self, param_name, channels, tag='default',
                         return_metadata=False):
        """
        Get parameter stored in dictionary

        Parameters:
        ----------
        
        param_name : str
          name of parameter

        channel :  str or list of str
           channel name or list of channels
        
        tag : str, optional
            psd tag, default: No tag

        Return
        ------

        array : ndarray, 
           nD numpy array corresponding to parameters 
        f  : ndarray
            array frequencies

        """
        # check channel
        channel_list = convert_channel_name_to_list(channels)
        channel_name = convert_channel_list_to_name(channels)
              
        if (channel_name not in self._filter_data.keys()):
            msg = f'ERROR: Channel "{channel_name}" not available!'
            
            if self._filter_data.keys():
                msg += ' List of channels in filter file: '
                msg += str(list(self._filter_data.keys()))
                
            raise ValueError(msg)

        
        # parameter name
        data_name = param_name + '_' + tag
        metadata_name = data_name + '_metadata'
        inds_name = data_name + '_inds'
        
        # check available tag
        if data_name not in self._filter_data[channel_name].keys():
            raise ValueError(f'ERROR: Parameter {data_name} not found '
                             f'for channel {channel_name}!')

        data  = copy.deepcopy(self._filter_data[channel_name][data_name])
        vals_inds = None
        vals = None
        if isinstance(data, pd.Series):
            vals = data.values
            vals_inds = data.index
            vals_inds = vals_inds.to_numpy()
            
        elif isinstance(data, np.ndarray):
            vals  = data.copy()
            if inds_name in self._filter_data[channel_name]:
                vals_inds = self._filter_data[channel_name][inds_name]
                if isinstance(vals_inds, pd.Series):
                    vals_inds = vals_inds.values

        metadata = dict()
        if metadata_name in self._filter_data[channel_name].keys():
            metadata =  copy.deepcopy(
                self._filter_data[channel_name][metadata_name]
            )

        if return_metadata:
            return vals, vals_inds, metadata
        else:
            return vals, vals_inds

