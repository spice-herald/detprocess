import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import pytesdaq.io as h5io
import qetpy as qp
from glob import glob
import vaex as vx
from pathlib import Path
from detprocess.core.filterdata import FilterData


class Template(FilterData):
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
            and pretrigger_length_samples):
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
        metadata['trace_length_samples'] = array.shape[1]
        metadata['pretrigger_length_samples'] = pretrigger_length_samples
                    
        
        # loop channels and store 
        for ichan in range(nb_channels):

            # template
            template = array[ichan,:]

            # add channel
            chan = channels[ichan]
            if chan not in self._filter_data.keys():
                self._filter_data[chan] = dict()
                
            self._filter_data[chan][template_name] = (
                pd.Series(template, t))
            
            # add channel nanme metadata
            metadata['channel'] = chan
            self._filter_data[chan][template_name + '_metadata'] = metadata
            
   
                     
            

        

    def create_template(self, channels,
                        sample_rate=None,
                        trace_length_msec=None,
                        trace_length_samples=None,
                        pretrigger_length_msec=None,
                        pretrigger_length_samples=None,
                        A=1, B=None, C=None,
                        tau_r=None,
                        tau_f1=None, tau_f2=None, tau_f3=None,
                        tag='default'):
        """
        Create 2,3,4 poles functional forms

        
        2-poles:
        A*(exp(-t/\tau_f1)) - A*(exp(-t/\tau_r))

        3-poles:
        A*(exp(-t/\tau_f1)) + B*(exp(-t/\tau_f2)) - 
            (A+B)*(exp(-t/\tau_r))
        
        4-poles:
        A*(exp(-t/tau_f1)) + B*(exp(-t/tau_f2)) +
            C*(exp(-t/tau_f3)) - (A+B+C)*(exp(-t/tau_r))

        """

        # check arguments
        if sample_rate is None:
            raise ValueError('ERROR: "sample_rate" argument required')
                
        if (trace_length_msec is None
            and trace_length_samples is None):
            raise ValueError(
                'ERROR: Trace length required ("trace_length_msec" or '
                '"trace_length_samples")!')

        if (pretrigger_length_msec is None
            and pretrigger_length_samples is None):
            raise ValueError(
                'ERROR: Pretrigger length required ("pretrigger_length_msec"'
                ' or "pretrigger_length_samples")!')

        if (tau_r is None):
            raise ValueError('ERROR: "tau_r" argument required')
        
        if (A is None and B is None and C is None):
            raise ValueError('ERROR: "A" and/or "B" and/or "C"'
                             ' argument(s) required!')

        if (tau_f1 is None and tau_f2 is None and tau_f3 is None):
            raise ValueError('ERROR: "tau_f1" and/or "tau_f2" and/or "tau_f3"'
                             ' argument(s) required!')
        
        
        # define time axis
        if trace_length_samples is None:
            trace_length_samples = int(
                round(1e-3*trace_length_msec*sample_rate)
            )
    
        # define time axis
        if pretrigger_length_msec is None:
            pretrigger_length_msec = (
                1e3*pretrigger_length_samples/sample_rate
            )
        else:
            pretrigger_length_samples = int(
                round(1e-3*pretrigger_length_msec*sample_rate)
            )
            
        # time array
        dt = 1/sample_rate
        t0 = pretrigger_length_msec*1e-3
        t =  np.asarray(list(range(trace_length_samples)))*dt
         

        # initialize template
        template = None
        poles = None
        
        # create 4-polse template
        if (A is not None
            and B is not None
            and C is not None):
            

            if self._verbose:
                print('INFO: Creating 4-poles template (tag="'
                      + tag + '")')

            # save number of poles
            poles = 4  
                
            # check fall time
            if (tau_f1 is None or tau_f2 is None or tau_f3 is None):
                raise ValueError('ERROR: 4-poles template requires 3 fall'
                                 ' times: "tau_f1", "tau_f2" and "tau_f3"')

            
            template = qp.utils.make_template_fourpole(
                t, A, B, C, tau_r, tau_f1, tau_f2, tau_f3,
                t0=t0, fs=sample_rate, normalize=True
            )


        elif (A is not None
              and B is not None):

            if self._verbose:
                print('INFO: Creating 3-poles template (tag="'
                      + tag + '")')

            # save number of poles
            poles = 3 
                
            # check fall time
            if (tau_f1 is None or tau_f2 is None):
                raise ValueError('ERROR: 3-poles template requires 2 fall'
                                 ' times: "tau_f1" and  "tau_f2"')

            
            template = qp.utils.make_template_threepole(
                t, A, B, tau_r, tau_f1, tau_f2,
                t0=t0, fs=sample_rate,
                normalize=True)
            
        elif (A is not None):

            if self._verbose:
                print('INFO: Creating 2-poles template (tag="'
                      + tag + '")')
                
            # save number of poles
            poles = 2
            
            # check fall time
            if tau_f1 is None:
                raise ValueError('ERROR: 2-poles template requires 1 fall'
                                 ' time: "tau_f1"')

            template = qp.utils.make_template_twopole(
                t, A, tau_r, tau_f1,
                t0=t0, fs=sample_rate,
                normalize=True)
            
            
        else:
            raise ValueError('ERROR: Unrecognize arguments. '
                             'Unable to create template')


          
        # parameter name
        template_name = 'template' + '_' + tag

        # metadata
        metadata = {'sample_rate': sample_rate,
                    'trace_length_samples': trace_length_samples,
                    'pretrigger_length_samples': pretrigger_length_samples,
                    'nb_poles': poles,
                    'A': A, 'tau_r': tau_r, 'tau_f1': tau_f1}

        if B is not None:
            metadata['B'] = B
            metadata['tau_f2'] = tau_f2
        if C is not None:
            metadata['C'] = C
            metadata['tau_f3'] = tau_f3

        
        # loop channels
        if isinstance(channels, str):
            channels = [channels]
        for chan in channels:

            # add channel
            if chan not in self._filter_data.keys():
                self._filter_data[chan] = dict()
                
            self._filter_data[chan][template_name] = (
                pd.Series(template, t))
            
            # add channel nanme metadata
            metadata['channel'] = chan
            self._filter_data[chan][template_name + '_metadata'] = metadata
            

                        
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
            ax.plot(t*1e3, template, label=chan)
            
        # add axis
        ax.legend()
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.grid(which='minor', linestyle='dotted')
        ax.grid(which='major')
        ax.set_title('Template',fontweight='bold')
        ax.set_xlabel('Time [msec]',fontweight='bold')
     
