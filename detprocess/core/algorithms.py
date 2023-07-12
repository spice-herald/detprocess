import numpy as np
import qetpy as qp
from numpy.fft import fftfreq, fft


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
    def of1x1_nodelay(of_base=None,  template_tag='default',
                      trace=None, template=None, psd=None,
                      fs=None, nb_samples_pretrigger=None,
                      lowchi2_fcutoff=10000,
                      coupling='AC', integralnorm=False,
                      feature_base_name='of1x1_nodelay',
                      **kwargs):
        """
        Feature extraction for the no delay Optimum Filter. 
        
        The arguments "trace", "template", "psd"
        (and associated parameters "fs", "nb_samples_pretrigger") 
        should only be used if  not already added in OF base object. 
        Otherwise keep as None


        Parameters
        ----------
        of_base : OFBase object, optional
           OFBase  if it has been instantiated independently


        template_tag : str, option
           tag of the template to be used for OF calculation,
           Default: 'default'
        
       
        trace : ndarray, optional
            An ndarray containing the raw data to extract the feature
            from. It is required if trace not already added in OFbase,
            otherwise keep it as None

        template : ndarray, optional
            The template to use for the optimum filter. It is required 
            if template not already added in OFbase,
            otherwise keep it as None

        psd : ndarray, optional
            The PSD to use for the optimum filter.It is required 
            if psd not already added in OFbase,
            otherwise keep it as None

        fs : float, optional
            The digitization rate of the data in trace, required
            if  "of_base" argument  is None, otherwise set to None

        nb_samples_pretrigger : int, optional
            Number of pre-trigger samples, required
            if "of_base" argument is None, otherwise set to None
            
        lowchi2_fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz.
        
        coupling : str, optional
            Only used if "psd" argument is not None. 
            "coupling" string etermines if the zero 
            frequency bin of the psd should be ignored (i.e. set to infinity) 
            when calculating the optimum amplitude. If set to 'AC', then the zero
            frequency bin is ignored. If set to anything else, then the
            zero frequency bin is kept. O

        integralnorm : bool, optional
            Only used if "template" argument is not None. 
            If set to True, then  template will be normalized 
            to an integral of 1, and any optimum filters will
            instead return the optimum integral in units of Coulombs.
            If set to False, then the usual optimum filter amplitudes
            will be returned (in units of Amps).

        feature_base_name : str, option
            output feature base name 



        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """


        # instantiate OF 1x1
        OF = qp.OF1x1(
            of_base=of_base,
            template_tag=template_tag,
            template=template,
            psd=psd,
            sample_rate=fs,
            pretrigger_samples=nb_samples_pretrigger,
            coupling=coupling,
            integralnorm=integralnorm,
        )
            

        # calc (signal needs to be None if set already)
        OF.calc_nodelay(signal=trace,
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
    def of1x1_unconstrained(of_base=None, template_tag='default',
                            interpolate=False,
                            trace=None, template=None, psd=None,
                            fs=None, nb_samples_pretrigger=None,
                            lowchi2_fcutoff=10000,
                            coupling='AC', integralnorm=False,
                            feature_base_name='of1x1_unconstrained',
                            **kwargs):
        """
        Feature extraction for the unconstrained Optimum Filter.
 
        The arguments "trace", "template", "psd"
        (and associated parameters "fs", "nb_samples_pretrigger") 
        should only be used if  not already added in OF base object. 
        Otherwise keep as None

        
        Parameters
        ----------
        of_base : OFBase object, optional
           OFBase  if it has been instantiated independently


        template_tag : str, option
           tag of the template to be used for OF calculation,
           Default: 'default'
        
       interpolate : bool, optional
           if True, do delay interpolation
           default: False
        
        trace : ndarray, optional
            An ndarray containing the raw data to extract the feature
            from. It is required if trace not already added in OFbase,
            otherwise keep it as None

        template : ndarray, optional
            The template to use for the optimum filter. It is required 
            if template not already added in OFbase,
            otherwise keep it as None

        psd : ndarray, optional
            The PSD to use for the optimum filter.It is required 
            if psd not already added in OFbase,
            otherwise keep it as None

        fs : float, optional
            The digitization rate of the data in trace, required
            if  "of_base" argument  is None, otherwise set to None

        nb_samples_pretrigger : int, optional
            Number of pre-trigger samples, required
            if "of_base" argument is None, otherwise set to None
            
        lowchi2_fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz.
        
        coupling : str, optional
            Only used if "psd" argument is not None. 
            "coupling" string etermines if the zero 
            frequency bin of the psd should be ignored (i.e. set to infinity) 
            when calculating the optimum amplitude. If set to 'AC', then ths zero
            frequency bin is ignored. If set to anything else, then the
            zero frequency bin is kept. O

        integralnorm : bool, optional
            Only used if "template" argument is not None. 
            If set to True, then  template will be normalized 
            to an integral of 1, and any optimum filters will
            instead return the optimum integral in units of Coulombs.
            If set to False, then the usual optimum filter amplitudes
            will be returned (in units of Amps).

        feature_base_name : str, option
            output feature base name 



        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """


        # instantiate OF1x1
        OF = qp.OF1x1(
            of_base=of_base,
            template_tag=template_tag,
            template=template,
            psd=psd,
            sample_rate=fs,
            pretrigger_samples=nb_samples_pretrigger,
            coupling=coupling,
            integralnorm=integralnorm,
        )
            
                    

        # calc (signal needs to be None if set already)
        OF.calc(signal=trace,
                lowchi2_fcutoff=lowchi2_fcutoff,
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
    def of1x1_constrained(of_base=None, template_tag='default',
                          window_min_from_trig_usec=None,
                          window_max_from_trig_usec=None,
                          window_min_index=None,
                          window_max_index=None,
                          lgc_outside_window=False,
                          interpolate=False,
                          trace=None, template=None, psd=None,
                          fs=None, nb_samples_pretrigger=None,
                          lowchi2_fcutoff=10000,
                          coupling='AC', integralnorm=False,
                          feature_base_name='of1x1_constrained',
                          **kwargs):
        """
        Feature extraction for the constrained Optimum Filter.

        The arguments "trace", "template", "psd"
        (and associated parameters "fs", "nb_samples_pretrigger") 
        should only be used if  not already added in OF base object. 
        Otherwise keep as None

        
        Parameters
        ----------
        of_base : OFBase object, optional
           OFBase  if it has been instantiated independently
        
        
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
        
        trace : ndarray, optional
            An ndarray containing the raw data to extract the feature
            from. It is required if trace not already added in OFbase,
            otherwise keep it as None

        template : ndarray, optional
            The template to use for the optimum filter. It is required 
            if template not already added in OFbase,
            otherwise keep it as None

        psd : ndarray, optional
            The PSD to use for the optimum filter.It is required 
            if psd not already added in OFbase,
            otherwise keep it as None

        fs : float, optional
            The digitization rate of the data in trace, required
            if  "of_base" argument  is None, otherwise set to None

        nb_samples_pretrigger : int, optional
            Number of pre-trigger samples, required
            if "of_base" argument is None, otherwise set to None
            
        lowchi2_fcutoff : float, optional
            The frequency (in Hz) that we should cut off the chi^2 when
            calculating the low frequency chi^2. Default is 10 kHz.
        
        coupling : str, optional
            Only used if "psd" argument is not None. 
            "coupling" string etermines if the zero 
            frequency bin of the psd should be ignored (i.e. set to infinity) 
            when calculating the optimum amplitude. If set to 'AC', then ths zero
            frequency bin is ignored. If set to anything else, then the
            zero frequency bin is kept. O

        integralnorm : bool, optional
            Only used if "template" argument is not None. 
            If set to True, then  template will be normalized 
            to an integral of 1, and any optimum filters will
            instead return the optimum integral in units of Coulombs.
            If set to False, then the usual optimum filter amplitudes
            will be returned (in units of Amps).

        feature_base_name : str, optional
            output feature base name 



        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        
        # instantiate OF1x1
        OF = qp.OF1x1(
            of_base=of_base,
            template_tag=template_tag,
            template=template,
            psd=psd,
            sample_rate=fs,
            pretrigger_samples=nb_samples_pretrigger,
            coupling=coupling,
            integralnorm=integralnorm,
        )
        
            
            
        # calc (signal needs to be None if set already)
        OF.calc(signal=trace,
                window_min_from_trig_usec=window_min_from_trig_usec,
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
    def psd_amp(of_base=None, trace=None, fs=None,
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
                   
        trace : ndarray, optional
            An ndarray containing the raw data to extract the feature
            from. It is required if trace not already added in OFbase,
            otherwise keep it as None
        
        fs : float, optional
            If trace is passed, used to construct fft of trace and fft
            frequencies array.
            
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
            
        if of_base is not None:
            freqs = of_base._fft_freqs
            trace_psd = np.abs(of_base._signal_fft)
            
        else:
            trace_length = len(trace[0])
            df = fs/trace_length
            freqs = fftfreq(trace_length, 1.0/fs)
            trace_psd = np.abs(fft(trace, axis=-1)/trace_length/df)
                          
        i = 0
        retdict = {}   
        while i < len(f_lims):
            f_low = f_lims[i][0]
            f_high = f_lims[i][1]
            
            #round the frequencies so we can more easily look up what
            #frequencies to average between
            freq_spacing = freqs[1] - freqs[0]
            f_low_mod = freq_spacing * np.ceil(f_low/freq_spacing)
            f_high_mod = freq_spacing * np.floor(f_high/freq_spacing)
            
            #get the indices of where to average the psd
            f_low_index = np.where(freqs == f_low_mod)[0][0]
            f_high_index = np.where(freqs == f_high_mod)[0][0]
            
            #calculate the average psd in that range
            av_psd = np.average(trace_psd[f_low_index:f_high_index])
            
            # store features
            retdict[feature_base_name + '_' + str(round(f_low)) + '_' + str(round(f_high))] = av_psd            
            i += 1
            
        return retdict
