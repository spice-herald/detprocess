import numpy as np
import qetpy as qp
from detprocess.utils import utils
import random


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
        Feature extraction for the no delay Optimum Filter.


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
        
        debug = True

        
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
        elif template_tag.ndim != 2:
            raise ValueError(f'ERROR: Expecting a 2D "template_tag" '
                             f'array  for channel {channel}, '
                             f'algorithm "{feature_base_name}"')
        
        nchans_array = template_tag.shape[0]
        ntmps =  template_tag.shape[1]
        
        if nchans != nchans_array:
            raise ValueError(f'ERROR: Expecting a 2D "template_tag" '
                             f'with shape[0] = {nchans} '
                             f'for channel {channel}, '
                             f'algorithm "{feature_base_name}"')

        
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
                                
        # instantiate OF NxM
        OF = qp.OFnxm(of_base=of_base,
                      channels=channel,
                      template_tags=template_tag,
                      verbose=False)

        # calc
        OF.calc()

        # get data
        amps, t0, chi2 = OF.get_fit_withdelay(
            window_min_from_trig_usec=window_min_from_trig_usec,
            window_max_from_trig_usec=window_max_from_trig_usec,
            window_min_index=window_min_index,
            window_max_index=window_max_index,
            interpolate_t0=interpolate_t0,
            lgc_outside_window=lgc_outside_window
        )
        
        # store
        retdict = dict()
        retdict[f'chi2_{feature_base_name}'] = chi2
        retdict[f't0_{feature_base_name}'] = t0
        for iamp, amp_name in enumerate(amplitude_names):
            retdict[f'{amp_name}_{feature_base_name}'] = amps[iamp]

        return retdict
    

    @staticmethod
    def of1x1_nodelay(channel, of_base,
                      template_tag='default',
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

        template_tag : str, option
           tag of the template to be used for OF calculation,
           Default: 'default'

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

        # instantiate OF 1x1
        OF = qp.OF1x1(of_base=of_base,
                      channel=channel,
                      template_tag=template_tag)
        
        # calc 
        OF.calc_nodelay(lowchi2_fcutoff=lowchi2_fcutoff)

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


        # instantiate OF1x1
        OF = qp.OF1x1(of_base=of_base,
                      channel=channel,
                      template_tag=template_tag)
        # calc
        OF.calc(lowchi2_fcutoff=lowchi2_fcutoff,
                interpolate_t0=interpolate,
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
                lgc_fit_nodelay=False,
                lgc_plot=False)


        # get results
        amp, t0, chi2, lowchi2 = OF.get_result_withdelay()
              
        # get chi2 no pulse
        chi2_nopulse = OF.get_chisq_nopulse()

        # get OF resolution
        ampres = OF.get_energy_resolution()
        timeres = OF.get_time_resolution(amp)

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
    def of1x2(channel, of_base,
              template_tag_1='Scintillation',
              template_tag_2='Evaporation',
              feature_base_name='of1x2',
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

        # get OF base data
        freqs = of_base.fft_freqs()
        trace_fft = of_base.signal_fft(channel)

        # sample rate
        fs = 2*np.max(np.abs(freqs))
        if 'fs' in kwargs:
            fs = kwargs['fs']

        # calculate psd
        psd = (np.abs(trace_fft)**2.0) * fs

        # fold
        freqs_fold, psd_fold = qp.utils.fold_spectrum(psd, fs)

        # remove DC
        psd_fold =  psd_fold[1:]
        freqs_fold = freqs_fold[1:]
    
        # index ranges
        name_list, ind_range_list = utils.get_indices_from_freq_ranges(
            freqs_fold, f_lims)
        
        retdict = {}
        for it, ind_range in enumerate(ind_range_list):

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
            psd_avg = np.sqrt(psd_avg)
                    
            # parameter name
            psd_amp_name = f'{feature_base_name}_{name_list[it]}'
            retdict[psd_amp_name ] = psd_avg

        return retdict
