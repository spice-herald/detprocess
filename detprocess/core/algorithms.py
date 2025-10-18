import numpy as np
import qetpy as qp
from detprocess.utils import utils
import random
from pprint import pprint

__all__ = [
    'FeatureExtractors',
]


class FeatureExtractors:
    """
    A class that contains all of the possible feature extractors
    for a given trace, assuming processing on a single channel.
    Each feature extraction function is a staticmethod for processing
    convenience.

    """

    @staticmethod
    def ofnxmx2(channel,
                of_base,
                available_channels=None,
                feature_base_name='ofnxmx2',
                template_tag=None,
                amplitude_names=None,
                fit_window = None,
                **kwargs):
        """
        Feature extraction for the nxmx2 Optimum Filter.
        
        Parameters
        ----------
        channel : str
          channel with format 'chan1|chan2|chan3'
          (order matter)
        
        of_base : OFBase object
           OFBase QETpy object
           
        lowchi2_fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz. (NOT IMPLEMENTED)
        
        available_channels : list
           list of available channels
           
           
        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

           
        """
          
        # split channel name into list (same order)
        channel_list, separator = utils.split_channel_name(
            channel,
            available_channels=available_channels,
            separator='|')
            
        nchans = len(channel_list)
        
        
        if  template_tag is None:
            raise ValueError(f'ERROR: Missing "template_tag" argument '
                             f'for channel {channel}, '
                             f'algorithm "{feature_base_name}"')

        # time constraints tag is the name of algorithm
        time_constraints_tag = feature_base_name
     
        # check template available
        template = of_base.template(channel,
                                    template_tag=template_tag)
        if template is None:
            raise ValueError(f'ERROR: Missing template '
                             f'for channel {channel}, '
                             f'tag "{template_tag}", '
                             f'algorithm "{feature_base_name}"')


        ntmps = template.shape[1]
        
        
        # amplitude names
        if amplitude_names is None:
            amplitude_names = []
            for itmp in range(ntmps):
                amplitude_names.append(f'amp{itmp+1}')
                
        else:

            if isinstance(amplitude_names, str):
                amplitude_names = [amplitude_names]

            if len(amplitude_names) != ntmps:
                raise ValueError(
                    f'ERROR: Wrong length for "amplitude_names" '
                    f'argument. Expecting {ntmps} name '
                    f'for  channel {channel}, '
                    f'algorithm "{feature_base_name}"')



        # initalize dictionary
        retdict = dict()
        retdict[f'chi2_{feature_base_name}'] =  -999999.0,
        retdict[f'delta_t_{feature_base_name}'] =  -999999.0,
        for iamp, amp_name in enumerate(amplitude_names):
            retdict[f'{amp_name}_{feature_base_name}'] =  -999999.0
            
        # check if signal stored
        if not of_base.is_signal_stored(channel):
            return retdict
            
        # instantiate OF NxM
        OF = qp.OFnxmx2(of_base=of_base,
                        channels=channel,
                        template_tag=template_tag,
                        time_constraints_tag=time_constraints_tag,
                        verbose=False)

        # calc
        OF.calc()
        
        amps, deltat, chi2 = OF.get_fit()
        
        retdict[f'chi2_{feature_base_name}'] = chi2
        retdict[f'delta_t_{feature_base_name}'] = deltat
        for iamp, amp_name in enumerate(amplitude_names):
            retdict[f'{amp_name}_{feature_base_name}'] = amps[iamp]

        return retdict

    @staticmethod
    def ofnxm(channel, of_base,
              available_channels=None,
              feature_base_name='ofnxm',
              template_tag=None,
              amplitude_names=None,
              window_min_from_trig_usec=None,
              window_max_from_trig_usec=None,
              window_min_index=None,
              window_max_index=None,
              lgc_outside_window=False,
              lowchi2_fcutoff=10000,
              interpolate_t0=False,
              **kwargs):
        """
        Feature extraction for the NxM Optimum Filter.
        Returns both constrained and nodelay paramaters.


        Parameters
        ----------
        channel : str
          channel with format 'chan1|chan2|chan3'
          (order matter)

        of_base : OFBase object
           OFBase QETpy object 

        lowchi2_fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz.

        available_channels : list
           list of available channels

        feature_base_name : str, option
            output feature base name


        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """
        
        debug = False
        
        # split channel name into list (same order)
        channel_list, separator = utils.split_channel_name(
            channel,
            available_channels=available_channels,
            separator='|')
        
        nchans = len(channel_list)

        # check data
        if  template_tag is None: 
            raise ValueError(f'ERROR: Missing "template_tag" argument '
                             f'for channel {channel}, '
                             f'algorithm "{feature_base_name}"')

        template = of_base.template(channel,
                                    template_tag=template_tag)
        if template is None:
            raise ValueError(f'ERROR: Missing template '
                             f'for channel {channel}, '
                             f'tag "{template_tag}", '
                             f'algorithm "{feature_base_name}"')


        ntmps = template.shape[1]
        
        if amplitude_names is None:
            amplitude_names = []
            for itmp in range(ntmps):
                amplitude_names.append(f'amp{itmp+1}')
        else:

            if isinstance(amplitude_names, str):
                amplitude_names = [amplitude_names]

            if len(amplitude_names) != ntmps:
                raise ValueError(
                    f'ERROR: Wrong length for "amplitude_names" '
                    f'argument. Expecting {ntmps} name '
                    f'for  channel {channel}, '
                    f'algorithm "{feature_base_name}"')

        # Initialize output
        retdict = dict()
        retdict[f'chi2_{feature_base_name}_constrained'] = -999999.0
        retdict[f't0_{feature_base_name}_constrained'] = -999999.0
        for iamp, amp_name in enumerate(amplitude_names):
            retdict[f'{amp_name}_{feature_base_name}_constrained'] = -999999.0    
        retdict[f'chi2_{feature_base_name}_nodelay'] = -999999.0
        for iamp, amp_name in enumerate(amplitude_names):
            retdict[f'{amp_name}_{feature_base_name}_nodelay'] = -999999.0

        # check if signal stored
        if not of_base.is_signal_stored(channel):
            return retdict

            
        # instantiate OF NxM
        OF = qp.OFnxm(of_base=of_base,
                      channels=channel,
                      template_tag=template_tag,
                      verbose=False)

        # calc
        OF.calc()

        # get data
        amps_constrained, t0_constrained, chi2_constrained = OF.get_fit_withdelay(
            window_min_from_trig_usec=window_min_from_trig_usec,
            window_max_from_trig_usec=window_max_from_trig_usec,
            window_min_index=window_min_index,
            window_max_index=window_max_index,
            interpolate_t0=interpolate_t0,
            lgc_outside_window=lgc_outside_window
        )
        amps_nodelay, t0_nodelay, chi2_nodelay = OF.get_fit_nodelay()
        
        # store
        retdict[f'chi2_{feature_base_name}_constrained'] = chi2_constrained
        retdict[f't0_{feature_base_name}_constrained'] = t0_constrained
        for iamp, amp_name in enumerate(amplitude_names):
            retdict[f'{amp_name}_{feature_base_name}_constrained'] = amps_constrained[iamp]
            
        retdict[f'chi2_{feature_base_name}_nodelay'] = chi2_nodelay
        for iamp, amp_name in enumerate(amplitude_names):
            retdict[f'{amp_name}_{feature_base_name}_nodelay'] = amps_nodelay[iamp]

        return retdict
    

    @staticmethod
    def of1x1_nodelay(channel, of_base,
                      template_tag=None,
                      lowchi2_fcutoff=10000,
                      feature_base_name='of1x1_nodelay',
                      **kwargs):
        """
        Feature extraction for the no delay Optimum Filter.


        Parameters
        ----------
        channel : str
          channel name

        of_base : OFBase object
           OFBase QETpy object 

        template_tag : str
           tag of the template to be used for OF calculation,
        

        lowchi2_fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz.

        feature_base_name : str, option
            output feature base name

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        # check tag
        if template_tag is None:
            raise ValueError('ERROR: Template tag required for OF 1x1')


        # intialize variables
        retdict = {
            ('amp_' + feature_base_name): -999999.0,
            ('chi2_' + feature_base_name): -999999.0,
            ('lowchi2_' + feature_base_name): -999999.0
        }

        # check if signal stored
        if not of_base.is_signal_stored(channel):
            return retdict
        
        
        # instantiate OF 1x1
        OF = qp.OF1x1(of_base=of_base,
                      channel=channel,
                      template_tag=template_tag)
        
        # calc 
        OF.calc(lgc_fit_withdelay=False,
                lgc_fit_nodelay=True,
                lowchi2_fcutoff=lowchi2_fcutoff)

        # get results
        amp, t0, chi2, lowchi2 = OF.get_result_nodelay()

        # store features
        retdict = {
            ('amp_' + feature_base_name): amp,
            ('chi2_' + feature_base_name): chi2,
            ('lowchi2_' + feature_base_name): lowchi2
        }

        return retdict



    @staticmethod
    def of1x1_unconstrained(channel, of_base,
                            template_tag='default',
                            interpolate=False,
                            lowchi2_fcutoff=10000,
                            feature_base_name='of1x1_unconstrained',
                            **kwargs):
        """
        Feature extraction for the unconstrained Optimum Filter.


        Parameters
        ----------
        channel : str
          channel name

        of_base : OFBase object
           OFBase  

        template_tag : str, option
           tag of the template to be used for OF calculation,
           Default: 'default'

        interpolate : bool, optional
           if True, do delay interpolation
           default: False

        lowchi2_fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz.

        feature_base_name : str, option
            output feature base name



        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """
        
        # intialize features dict
        retdict = {
            ('amp_' + feature_base_name): -999999.0,
            ('t0_' + feature_base_name): -999999.0,
            ('chi2_' + feature_base_name): -999999.0,
            ('lowchi2_' + feature_base_name): -999999.0
        }
        
        # check if signal stored
        if not of_base.is_signal_stored(channel):
            return retdict

        # instantiate OF1x1
        OF = qp.OF1x1(of_base=of_base,
                      channel=channel,
                      template_tag=template_tag)
        # calc
        OF.calc(lowchi2_fcutoff=lowchi2_fcutoff,
                interpolate_t0=interpolate,
                lgc_fit_withdelay=True,
                lgc_fit_nodelay=False,
                lgc_plot=False)

        # get results
        amp, t0, chi2, lowchi2 = OF.get_result_withdelay()


        # store features
        retdict = {
            ('amp_' + feature_base_name): amp,
            ('t0_' + feature_base_name): t0,
            ('chi2_' + feature_base_name): chi2,
            ('lowchi2_' + feature_base_name): lowchi2
        }

        return retdict


    @staticmethod
    def of1x1_constrained(channel, of_base,
                          template_tag='default',
                          window_min_from_trig_usec=None,
                          window_max_from_trig_usec=None,
                          window_min_index=None,
                          window_max_index=None,
                          lgc_outside_window=False,
                          interpolate=False,
                          lowchi2_fcutoff=10000,
                          feature_base_name='of1x1_constrained',
                          **kwargs):
        """
        Feature extraction for the constrained Optimum Filter.

   
        Parameters
        ----------
        channel : str
          channel name

        of_base : OFBase object
           OFBase QETpy object 

        template_tag : str, optional
           tag of the template to be used for OF calculation,
           Default: 'default'

        window_min_from_trig_usec : float, optional
           OF filter window start in micro seconds from
           pre-trigger (can be negative if prior pre-trigger)
           Default: use "window_min_index"  or
                    or set to 0 if both parameters are None

        window_max_from_trig_usec : float, optional
           OF filter window end in micro seconds from
           pre-trigger (can be negative if prior pre-trigger)
           Default: use "window_max_index"  or set to end trace
                    if  "window_max_index" also None


        window_min_index : int, optional
           ADC index OF filter window start (alternative
           to "window_min_from_trig_usec")

           Default: use "window_min_from_trig_usec"  or
                    set to 0 if both parameters are None

        window_max_index : int, optional
           ADC index OF filter window end (alternative
           to "window_min_from_trig_usec")

           Default: use "window_max_from_trig_usec"  or
                    set to end of trace if both parameters
                     are None

        lgc_outside_window : bool, optional
           If True, define window to be outside [min:max]
           Default: False

        interpolate : bool, optional
           if True, do delay interpolation
           default: False

        lowchi2_fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz.


        feature_base_name : str, optional
            output feature base name



        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        # intialize dictionary
        retdict = {
            ('amp_' + feature_base_name): -999999.0,
            ('t0_' + feature_base_name): -999999.0,
            ('chi2_' + feature_base_name): -999999.0,
            ('lowchi2_' + feature_base_name): -999999.0,
            ('chi2nopulse_' + feature_base_name): -999999.0,
            ('ampres_' + feature_base_name): -999999.0,
            ('timeres_' + feature_base_name): -999999.0
        }

        # check if signal stored
        if not of_base.is_signal_stored(channel):
            return retdict

        
        # instantiate OF1x1
        OF = qp.OF1x1(of_base=of_base,
                      channel=channel,
                      template_tag=template_tag)

        # calc (signal needs to be None if set already)
        OF.calc(window_min_from_trig_usec=window_min_from_trig_usec,
                window_max_from_trig_usec=window_max_from_trig_usec,
                window_min_index=window_min_index,
                window_max_index=window_max_index,
                lowchi2_fcutoff=lowchi2_fcutoff,
                interpolate_t0=interpolate,
                lgc_outside_window=lgc_outside_window,
                lgc_fit_withdelay=True,
                lgc_fit_nodelay=False,
                lgc_plot=False)


        # get results
        amp, t0, chi2, lowchi2 = OF.get_result_withdelay()
              
        # get chi2 no pulse
        chi2_nopulse = OF.get_chisq_nopulse()

        # get OF resolution
        ampres = OF.get_energy_resolution()
        timeres = OF.get_time_resolution()

        retdict = {
            ('amp_' + feature_base_name): amp,
            ('t0_' + feature_base_name): t0,
            ('chi2_' + feature_base_name): chi2,
            ('lowchi2_' + feature_base_name): lowchi2,
            ('chi2nopulse_' + feature_base_name): chi2_nopulse,
            ('ampres_' + feature_base_name): ampres,
            ('timeres_' + feature_base_name): timeres,
        }

        return retdict
    
    @staticmethod
    def of1x2x2(channel, of_base,
                template_tag_1='Scintillation',
                template_tag_2='Evaporation',
                feature_base_name='of1x2x2',
                **kwargs):
        """
        Feature extraction for the one channel, two template Optimum Filter.

        Parameters
        ----------
        of_base : OFBase object, optional
           OFBase  if it has been instantiated independently

        template_tag_1: str, option
           tag of the template to be used for OF calculation of the scintillation part; please use Scintilation,
           Default: 'Scintillation'

        template_tag_2 : str, option
           tag of the template to be used for OF calculation of the evaporation part; please use Evaporation,
           Default: 'Evaporation'

        feature_base_name : str, option
            output feature base name



        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        # intialize dictionary
        retdict = {
            ('scintillation_amp_' + feature_base_name): -999999.0,
            ('evaporation_amp_' + feature_base_name): -999999.0,
            ('time_diff_' + feature_base_name): -999999.0,
            ('scintillation_time_index' + feature_base_name): -999999.0,
            ('evaporation_time_index' + feature_base_name): -999999.0
        }
        
        # check if signal stored
        if not of_base.is_signal_stored(channel):
            return retdict
        
        # instantiate OF1x2
        OF = qp.OF1x2(
            of_base=of_base,
            template_1_tag=template_tag_1,
            template_2_tag=template_tag_1,
            channel_name= channel,
        )

        # calc (signal needs to be None if set already)
        OF.calc(lgc_plot=False)

        # get results
        scintillation_amp = OF._amplitude[OF._template_1_tag]
        evaporation_amp = OF._amplitude[OF._template_2_tag]
        time_diff= OF._time_diff_two_Pulses
        Starting_time_first_pulse =  OF._time_first_pulse
        Starting_time_second_pulse =  OF._time_second_pulse


        # store features
        retdict = {
            ('scintillation_amp_' + feature_base_name): scintillation_amp,
            ('evaporation_amp_' + feature_base_name): evaporation_amp,
            ('time_diff_' + feature_base_name): time_diff,
            ('scintillation_time_index' + feature_base_name): Starting_time_first_pulse ,
            ('evaporation_time_index' + feature_base_name): Starting_time_second_pulse
        }

        return retdict


    @staticmethod
    def baseline(trace,
                 window_min_index=None, window_max_index=None,
                 feature_base_name='baseline',
                 **kwargs):
        """
        Feature extraction for the trace baseline.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.

        window_min_index : int, optional
            The minium index of the window used to average the trace.
            Default: 0

        window_max_index : int, optional
            The maximum index of the window used to average the trace.
            Default: end of trace

        feature_base_name : str, optional
            output feature base name

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        # check if trace is empty or None
        if (trace is None or trace.size==0):
            retdict = {
                feature_base_name: -999999.0,
            }

            return retdict   

        
        if window_min_index is None:
            window_min_index = 0

        if window_max_index is None:
            window_max_index = trace.shape[-1] - 1


        baseline = np.mean(trace[window_min_index:window_max_index])

        retdict = {
            feature_base_name: baseline,
        }

        return retdict



    @staticmethod
    def integral(trace, fs,
                 window_min_index=None, window_max_index=None,
                 feature_base_name='integral',
                 **kwargs):
        """
        Feature extraction for the pulse integral.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.

        fs : float
            The digitization rate of the data in trace.

        window_min_index : int, optional
            The minium index of the window used to integrate the trace.
            Default: 0

        window_max_index : int, optional
            The maximum index of the window used to integrate the trace.
            Default: end of trace

        feature_base_name : str, optional
            output feature base name

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        # check if trace is empty or None
        if (trace is None or trace.size==0):
            retdict = {
                feature_base_name: -999999.0,
            }

            return retdict  

        
        if window_min_index is None:
            window_min_index = 0

        if window_max_index is None:
            window_max_index = trace.shape[-1] - 1


        integral = np.trapz(trace[window_min_index:window_max_index]) / fs

        retdict = {
            feature_base_name: integral,
        }

        return retdict




    @staticmethod
    def maximum(trace,
                window_min_index=None, window_max_index=None,
                feature_base_name='maximum',
                **kwargs):
        """
        Feature extraction for the maximum pulse value.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.

        window_min_index : int, optional
            The minium index of the window used to find  maximum
            Default: 0

        window_max_index : int, optional
            The maximum index of the window used to find maximum
            Default: end of trace

        feature_base_name : str, optional
            output feature base name


        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        # check if trace is empty or None
        if (trace is None or trace.size==0):
            retdict = {
                feature_base_name: -999999.0,
            }

            return retdict  

        
        if window_min_index is None:
            window_min_index = 0

        if window_max_index is None:
            window_max_index = trace.shape[-1] - 1

        max_trace = np.amax(trace[window_min_index:window_max_index])

        retdict = {
            feature_base_name: max_trace,
        }

        return retdict




    @staticmethod
    def minimum(trace,
                window_min_index=None, window_max_index=None,
                feature_base_name='minimum',
                **kwargs):
        """
        Feature extraction for the minimum pulse value.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.


        window_min_index : int, optional
            The minium index of the window used to find  minimum
            Default: 0

        window_max_index : int, optional
            The maximum index of the window used to find minimum
            Default: end of trace

        feature_base_name : str, optional
            output feature base name


        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        # check if trace is empty or None
        if (trace is None or trace.size==0):
            retdict = {
                feature_base_name: -999999.0,
            }

            return retdict  
        
        
        if window_min_index is None:
            window_min_index = 0

        if window_max_index is None:
            window_max_index = trace.shape[-1] - 1


        min_trace = np.amin(trace[window_min_index:window_max_index])

        retdict = {
            feature_base_name: min_trace,
        }

        return retdict



    @staticmethod
    def energyabsorbed(trace,
                       fs, vb, i0, rl,
                       window_min_index=None, window_max_index=None,
                       feature_base_name='energyabsorbed',
                       **kwargs):
        """
        Feature extraction for the minimum pulse value.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.
        fs : float
            The digitization rate of the data in trace.
        vb : float
            Bias voltage applied to the TES.
        i0 : float
            Quiescent operating current of the TES.
        rl : float
            Load resistance in the TES circuit.

        window_min_index : int, optional
            The index of the trace to start the integration.
             Default: 0
        window_max_index : int, optional
            The index of the trace to end the integration.
            Default: end of trace

        feature_base_name : str, optional
            output feature base name

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        # check if trace is empty or None
        if (trace is None or trace.size==0):
            retdict = {
                feature_base_name: -999999.0,
            }

            return retdict 

        
        baseline = trace[:window_min_index].mean()
        i_trace = trace[window_min_index:window_max_index] - baseline

        p0 = i_trace * (vb - 2*i0*rl) - i_trace**2 * rl

        en_abs = np.trapz(p0, dx=1/fs, axis=-1)

        retdict = {
            feature_base_name: en_abs,
        }

        return retdict

    
    @staticmethod
    def psd_amp(channel, of_base,
                f_lims=[],
                feature_base_name='psd_amp',
                **kwargs):
        """
        Feature extraction for measuring the average amplitude of a
        ffted trace in a range of frequencies. Rather than recalculating
        the fft, this feature references the pre-calculated OF class.
        The arguments "trace", "template", "psd"
        (and associated parameters "fs", "nb_samples_pretrigger")
        should only be used if  not already added in OF base object.
        Otherwise keep as None

        Parameters
        ----------
        of_base : OFBase object, optional
            OFBase  if it has been instantiated independently

        f_lims : list of list of floats
            A list of [f_low, f_high]s between which the averaged PSD is
            calculated. For example, [[45.0, 65.0], [120.0, 130.0]]

        feature_base_name : str, option
            output feature base name

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        # check f_lims and define feature name
        if not f_lims:
            raise ValueError('ERROR: "f_lims" required for algorithm psd_amps')

        freq_ranges, range_names = utils.cleanup_freq_ranges(f_lims)
        
        # initialize output
        retdict = {}
        for var_base in range_names:
            var_name = f'{feature_base_name}_{var_base}'
            retdict[var_name] = -999999.0

        # check if signal available
        if not of_base.is_signal_stored(channel):
            return retdict
       
        # get fft
        trace_fft = of_base.signal_fft(channel, squeeze_array=True)
        trace_fft = trace_fft.copy()
        freqs = of_base.fft_freqs().copy()
        nbins = of_base.nb_samples()

        # sample rate
        fs = utils.estimate_sampling_rate(freqs)
        if 'fs' in kwargs:
            fs = kwargs['fs']

        # calculate psd
        psd = (np.abs(trace_fft)**2.0)*nbins/fs

        # fold
        freqs_fold, psd_fold = qp.utils.fold_spectrum(psd, fs)

        # remove DC
        psd_fold =  psd_fold[1:]
        psd_fold =  np.sqrt(psd_fold)
        freqs_fold = freqs_fold[1:]
          
        retdict = {}
        ind_ranges = utils.get_ind_freq_ranges(freq_ranges, freqs_fold)
        for it, ind_range in enumerate(ind_ranges):

            ind_low = ind_range[0]
            ind_high = ind_range[1]

            # take median
            psd_chunk = psd_fold[ind_low:ind_high]
                
            # smooth + max ?
            # nb_samples = len(psd_chunk)
            #psd_chunk_smooth  = np.convolve(psd_chunk, np.ones(3)/3,
            #                                mode='valid')
            # 
            psd_avg = np.average(psd_chunk)
                           
            # parameter name
            psd_amp_name = f'{feature_base_name}_{range_names[it]}'
            retdict[psd_amp_name] = psd_avg
     
        return retdict


    @staticmethod
    def psd_peaks(channel, of_base,
                  f_lims=[],
                  npeaks=1,
                  min_separation_hz=0.0,
                  average_range=False,
                  feature_base_name='psd_peaks',
                  **kwargs):
        """
        Feature extraction for finding peaks of a psd
        in a range of frequencies. Rather than recalculating
        the fft, this feature references the pre-calculated OF class.
      
        Parameters
        ----------
        of_base : OFBase object, optional
            OFBase  if it has been instantiated independently

        f_lims : list of list of floats
            A list of [f_low, f_high]s between which the PSD peak is
            calculated. For example, [[45.0, 65.0], [120.0, 130.0]]

        npeaks : number of peaks to search. Default=1

        min_separation_hz : minimum separation between peaks

        feature_base_name : str, option
            output feature base name

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """
        # check f_lims and define feature name
        if not f_lims:
            raise ValueError('ERROR: "f_lims" required for algorithm psd_amps')

        freq_ranges, range_names = utils.cleanup_freq_ranges(f_lims)
      
        # initialize output
        retdict = {}
        for i in range(1, npeaks + 1):
            for var_base in range_names:
                var_name_amp = f'{feature_base_name}_{var_base}_amp_{i}'
                var_name_freq = f'{feature_base_name}_{var_base}_freq_{i}'
                retdict[var_name_amp] = -999999.0
                retdict[var_name_freq] = -999999.0
        # DC
        retdict[f'{feature_base_name}_dc_amp'] = -999999.0
                
        # check if signal stored
        if not of_base.is_signal_stored(channel):
            return retdict
        
        trace_fft = of_base.signal_fft(channel, squeeze_array=True)
        trace_fft = trace_fft.copy()
        freqs = of_base.fft_freqs().copy()
        nbins = of_base.nb_samples()

        if trace_fft.ndim != 1:
            # multi-channels, not implemented
            raise ValueError(f'ERROR: "psd_peaks" not implemented for '
                             f'multi-channel. Remove algorithm for '
                             f'channel {channel}!')

        # sample rate
        fs = utils.estimate_sampling_rate(freqs)
        if 'fs' in kwargs:
            fs = kwargs['fs']

        # calculate psd
        psd = (np.abs(trace_fft)**2.0)*nbins/fs
        
        # fold
        freqs_fold, psd_fold = qp.utils.fold_spectrum(psd, fs)

        # store DC amp
        retdict[f'{feature_base_name}_dc_amp'] = np.sqrt(psd_fold[0])

        # remove DC
        psd_fold =  psd_fold[1:]
        psd_fold =  np.sqrt(psd_fold)
        freqs_fold = freqs_fold[1:]
              
        # loop range and find peaks
        ind_ranges = utils.get_ind_freq_ranges(freq_ranges, freqs_fold)
        for it, freq_range in enumerate(freq_ranges):

            ind_range = ind_ranges[it]
            var_base  =  range_names[it]

            # case single frequency or just take average range
            if ((ind_range[1] == ind_range[0]+1)
                or average_range):
                
                ind_low = ind_range[0]
                ind_high = ind_range[1]
                psd_chunk = psd_fold[ind_low:ind_high]
                psd_avg = np.average(psd_chunk)
                freq_avg = np.average(freqs_fold[ind_low:ind_high])
                var_name_amp = f'{feature_base_name}_{var_base}_amp_1'
                var_name_freq = f'{feature_base_name}_{var_base}_freq_1'

                retdict[var_name_amp] = psd_avg
                retdict[var_name_freq] = freq_avg

            else:
                

                # find peaks within interval
                result_list = utils.find_psd_peaks(
                    freqs_fold, psd_fold,
                    fmin=freq_range[0], fmax=freq_range[1],
                    npeaks=npeaks,
                    min_separation_hz=min_separation_hz,
                    min_prominence=None)
                     
                      
                # loop peaks
                for i in range(npeaks):
                    var_name_amp = f'{feature_base_name}_{var_base}_amp_{i+1}'
                    var_name_freq = f'{feature_base_name}_{var_base}_freq_{i+1}'
                    if i < len(result_list):
                        result = result_list[i]
                        retdict[var_name_amp] = result['amplitude']
                        retdict[var_name_freq] =  result['freq']
                    else:
                        retdict[var_name_amp] = -999999.
                        retdict[var_name_freq] = -999999.
                    
        # done
        return retdict
    
    @staticmethod
    def phase(channel, of_base,
              f_lims=[],
              npeaks=1,
              min_separation_hz=0.0,
              threshold_factor=1e-3,
              feature_base_name='phase',
              **kwargs):
        """
        Feature extraction for finding the phase signal for peaks of the psd
        in a range of frequencies. Rather than recalculating
        the fft, this feature references the pre-calculated OF class.
        Phase is returned in units of radians.
        
        Parameters
        ----------
        of_base : OFBase object, optional
            OFBase  if it has been instantiated independently

        f_lims : list of list of floats
            A list of [f_low, f_high]s that will be used to search for peaks in the PSD.
            The phase will be calculated and returned for frequencies 
            associated with those peaks.
            Example: [[45.0, 65.0], [120.0, 130.0]].

        npeaks : number of peaks to search. Default=1

        min_separation_hz : minimum separation between peaks

        threshold_factor : a threshold factor that will be applied in FFT units to avoid calculating the phase for noisy garbage.

        feature_base_name : str, option
            output feature base name

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

            Phase will be returned in radians.
        """

        # check f_lims and define feature name
        if not f_lims:
            raise ValueError('ERROR: "f_lims" required for algorithm psd_amps')

        freq_ranges, range_names = utils.cleanup_freq_ranges(f_lims)
    
        # initialize output
        retdict = {}
        for i in range(1, npeaks + 1):
            for var_base in range_names:
                var_name_amp = f'{feature_base_name}_{var_base}_phase_{i}'
                var_name_freq = f'{feature_base_name}_{var_base}_freq_{i}'
                retdict[var_name_amp] = -999999.0
                retdict[var_name_freq] = -999999.0

                
        # check if signal stored
        if not of_base.is_signal_stored(channel):
            return retdict
        
        trace_fft = of_base.signal_fft(channel, squeeze_array=True)
        trace_fft = trace_fft.copy()
        freqs = of_base.fft_freqs().copy()
        nbins = of_base.nb_samples()

        if trace_fft.ndim != 1:
            # multi-channels, not implemented
            raise ValueError(f'ERROR: "phase" not implemented for '
                             f'multi-channel. Remove algorithm for '
                             f'channel {channel}!')

        # sample rate
        fs = utils.estimate_sampling_rate(freqs)
        if 'fs' in kwargs:
            fs = kwargs['fs']

        # calculate psd
        psd = (np.abs(trace_fft)**2.0)*nbins/fs
        
        # fold
        freqs_fold, psd_fold = qp.utils.fold_spectrum(psd, fs)

        # remove DC
        psd_fold =  psd_fold[1:]
        psd_fold =  np.sqrt(psd_fold)
        freqs_fold = freqs_fold[1:]
     
        fft_cpy = np.array(trace_fft, copy=True)
        mag = np.abs(fft_cpy)
        
        # Phase reference shift for pretrigger
        nb_samples_pretrigger = kwargs.get('nb_samples_pretrigger', 0)
        t0 = nb_samples_pretrigger / fs  # seconds
        fft_cpy *= np.exp(1j * 2.0 * np.pi * freqs * t0)

        # Mask tiny-magnitude bins to avoid noisy phase
        thr = mag.max() * float(kwargs.get('threshold_factor', 0.0))
        phase = np.angle(fft_cpy)
        if thr > 0:
            phase = np.where(mag >= thr, phase, -999999.0)


        # Keep only positive frequencies (includes DC and Nyquist if N even)
        N = phase.shape[-1]
        pos_stop = N // 2 + 1
        phase_fold = phase[:pos_stop]
        freqs_fold = freqs[:pos_stop]

        # Fix Nyquist if negative (happens for even N)
        if freqs_fold[-1] < 0:
            freqs_fold[-1] = abs(freqs_fold[-1])
            
        # Remove DC
        phase_fold = phase_fold[1:]
        freqs_fold = freqs_fold[1:]
      
        # loop ranges
        ind_ranges = utils.get_ind_freq_ranges(freq_ranges, freqs_fold)
        for it, freq_range in enumerate(freq_ranges):

            ind_range = ind_ranges[it]
            var_base  =  range_names[it]

            # case single frequency or just take average range
            if (ind_range[1] == ind_range[0]+1):
                ind_low = ind_range[0]
                ind_high = ind_range[1]
                var_name_phase = f'{feature_base_name}_{var_base}_phase_1'
                var_name_freq = f'{feature_base_name}_{var_base}_freq_1'

                freq_val = float(freqs_fold[ind_low:ind_high].item())
                phase_val = float(phase_fold[ind_low:ind_high].item())
                
                retdict[var_name_freq] = freq_val
                retdict[var_name_phase] = phase_val
                                
            else:
                
                # find peaks within interval
                result_list = utils.find_psd_peaks(
                    freqs_fold, psd_fold,
                    fmin=freq_range[0], fmax=freq_range[1],
                    npeaks=npeaks,
                    min_separation_hz=min_separation_hz,
                    min_prominence=None)
                            
                # loop peaks
                for i in range(npeaks):
                    var_name_phase = f'{feature_base_name}_{var_base}_phase_{i+1}'
                    var_name_freq = f'{feature_base_name}_{var_base}_freq_{i+1}'
                    if i < len(result_list):
                        result = result_list[i]
                        retdict[var_name_freq] =  result['freq']
                        retdict[var_name_phase] = phase_fold[result['index']]
                    else:
                        retdict[var_name_phase] = -999999.
                        retdict[var_name_freq] = -999999.
                    
        # done
        return retdict
