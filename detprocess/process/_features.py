import numpy as np
import qetpy as qp


__all__ = [
    'FeatureExtractors',
]




class FeatureExtractors(object):
    """
    A class that contains all of the possible feature extractors
    for a given trace, assuming processing on a single channel.
    Each feature extraction function is a staticmethod for processing
    convenience.

    """

    @staticmethod
    def of_nodelay(OF=None,
                   trace=None, template=None, psd=None, fs=None,
                   nb_samples=None, nb_samples_pretrigger=None,
                   feature_base_name='of_nodelay',
                   **kwargs):
        """
        Feature extraction for the no delay Optimum Filter.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.
        template : ndarray
            The template to use for the optimum filter.
        psd : ndarray
            The PSD to use for the optimum filter.
        fs : float
            The digitization rate of the data in trace.

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        if OF is None:
            
            OF = qp.OptimumFilter(
                trace,
                template,
                psd,
                fs,
            )


        # window center (relative to 1/2 pulse length)
        window_center = nb_samples//2 - nb_samples_pretrigger
  

            
        ofamp_nodelay, ofchi2_nodelay = OF.ofamp_nodelay(
            windowcenter= window_center
        )
        
        retdict = {
            feature_base_name.replace('of','ofamp'): ofamp_nodelay,
            feature_base_name.replace('of','ofchi2'): ofchi2_nodelay,
        }

        return retdict


    @staticmethod
    def of_unconstrained(OF=None,
                         trace=None, template=None, psd=None, fs=None,
                         feature_base_name='of_unconstrained',
                         **kwargs):
        """
        Feature extraction for the unconstrained Optimum Filter.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.
        template : ndarray
            The template to use for the optimum filter.
        psd : ndarray
            The PSD to use for the optimum filter.
        fs : float
            The digitization rate of the data in trace.

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """
        
        if OF is None:
            OF = qp.OptimumFilter(
                trace,
                template,
                psd,
                fs,
            )

                    
            
        ofamp_unconstrained, oft0_unconstrained, ofchi2_unconstrained = (
            OF.ofamp_withdelay()
        )

        retdict = {
            feature_base_name.replace('of','ofamp'): ofamp_unconstrained,
            feature_base_name.replace('of','oft0'): oft0_unconstrained,
            feature_base_name.replace('of','ofchi2'): ofchi2_unconstrained,
        }

        return retdict


    @staticmethod
    def of_constrained(OF=None, 
                       min_index=None, max_index=None ,
                       nb_samples=None, nb_samples_pretrigger=None,
                       trace=None, template=None, psd=None,fs=None,
                       feature_base_name='of_constrained',
                       **kwargs):
        """
        Feature extraction for the constrained Optimum Filter.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.
        template : ndarray
            The template to use for the optimum filter.
        psd : ndarray
            The PSD to use for the optimum filter.
        fs : float
            The digitization rate of the data in trace.
        nconstrain : int
            The constraint window set by the processing file.
        windowcenter : int
            The window center of the constraint window.

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """
        if OF is None:
            OF = qp.OptimumFilter(
                trace,
                template,
                psd,
                fs,
            )

        # Nb bins window
        window_center = nb_samples//2 - nb_samples_pretrigger-1
        nconstrain = max_index-min_index
                 
            
        ofamp_constrained, oft0_constrained, ofchi2_constrained = OF.ofamp_withdelay(
            nconstrain=nconstrain,
            windowcenter=window_center,
        )

        retdict = {
            feature_base_name.replace('of','ofamp'): ofamp_constrained,
            feature_base_name.replace('of','oft0'): oft0_constrained,
            feature_base_name.replace('of','ofchi2'): ofchi2_constrained,
        }

        return retdict


    @staticmethod
    def baseline(trace, min_index, max_index,
                 feature_base_name='baseline',
                 **kwargs):
        """
        Feature extraction for the trace baseline.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.
        max_index : int
            The index of the trace to average the trace up to.

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """
     
        baseline = np.mean(trace[min_index:max_index])

        retdict = {
            feature_base_name: baseline,
        }
        
        return retdict



    @staticmethod
    def integral(trace, min_index, max_index, fs,
                 feature_base_name='integral',
                 **kwargs):
        """
        Feature extraction for the pulse integral.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.
        min_index : int
            The index of the trace to start the integration.
        max_index : int
            The index of the trace to end the integration.
        fs : float
            The digitization rate of the data in trace.

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """
                
        integral = np.trapz(trace[min_index:max_index]) / fs

        retdict = {
            feature_base_name: integral,
        }

        return retdict


    @staticmethod
    def maximum(trace, min_index, max_index,
                feature_base_name='maximum',
                **kwargs):
        """
        Feature extraction for the maximum pulse value.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.
        min_index : int
            The index of the trace to start searching for the max.
        max_index : int
            The index of the trace to end searching for the max.

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        max_trace = np.amax(trace[min_index:max_index])

        retdict = {
            feature_base_name: max_trace,
        }

        return retdict


    @staticmethod
    def minimum(trace, min_index, max_index,
                feature_base_name='minimum',
                **kwargs):
        """
        Feature extraction for the minimum pulse value.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.
        min_index : int
            The index of the trace to start searching for the min.
        max_index : int
            The index of the trace to end searching for the min.

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        min_trace = np.amin(trace[min_index:max_index])

        retdict = {
            feature_base_name: min_trace,
        }

        return retdict

    @staticmethod
    def energyabsorbed(trace, min_index, max_index, fs, vb, i0, rl,
                       feature_base_name='energyabsorbed',
                       **kwargs):
        """
        Feature extraction for the minimum pulse value.

        Parameters
        ----------
        trace : ndarray
            An ndarray containing the raw data to extract the feature
            from.
        min_index : int
            The index of the trace to start the integration.
        max_index : int
            The index of the trace to end the integration.
        fs : float
            The digitization rate of the data in trace.
        vb : float
            Bias voltage applied to the TES.
        i0 : float
            Quiescent operating current of the TES.
        rl : float
            Load resistance in the TES circuit.

        Returns
        -------
        retdict : dict
            Dictionary containing the various extracted features.

        """

        baseline = trace[:min_index].mean()
        i_trace = trace[min_index:max_index] - baseline

        p0 = i_trace * (vb - 2*i0*rl) - i_trace**2 * rl

        en_abs = np.trapz(p0, dx=1/fs, axis=-1)

        retdict = {
            feature_base_name: en_abs,
        }
            
        return retdict

