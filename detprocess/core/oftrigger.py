import numpy as np
import os
from math import log10, floor
import math
import array
from scipy.signal import correlate, oaconvolve
from scipy.fft import ifft, fft, next_fast_len
import vaex as vx
import qetpy as qp
from scipy import special, stats




__all__ = ['OptimumFilterTrigger']


def _getchangeslessthanthresh(x, threshold):
    """
    Helper function that returns a list of the start and ending indices
    of the ranges of inputted values that change by less than the
    specified threshold value.

    Parameters
    ----------
    x : ndarray
        1-dimensional of values.
    threshold : int
        Value to detect the different ranges of vals that change by
        less than this threshold value.

    Returns
    -------
    ranges : ndarray
        List of tuples that each store the start and ending index of
        each range. For example, vals[ranges[0][0]:ranges[0][1]] gives
        the first section of values that change by less than the
        specified threshold.
    vals : ndarray
        The corresponding starting and ending values for each range in
        x.

    """

    diff = x[1:]-x[:-1]
    a = diff>threshold
    inds = np.where(a)[0]+1

    start_inds = np.zeros(len(inds)+1, dtype = int)
    start_inds[1:] = inds

    end_inds = np.zeros(len(inds)+1, dtype = int)
    end_inds[-1] = len(x)
    end_inds[:-1] = inds

    ranges = np.array(list(zip(start_inds,end_inds)))

    if len(x)!=0:
        vals = np.array([(x[st], x[end-1]) for (st, end) in ranges])
    else:
        vals = np.array([])

    return ranges, vals


class OptimumFilterTrigger:
    """
    Class for applying a time-domain optimum filter to a long trace,
    which can be thought of as an FIR filter.

    Attributes
    ----------
    phi : ndarray 
        The optimum filter in time-domain, equal to the inverse FT of
        (FT of the template/power spectral density of noise)
    norm : float
        The normalization of the optimal amplitude.
    fs : float
        The sample rate of the data (Hz).
    traces : ndarray
        Trace(s) to be filtered, assumed to be an ndarray of
        shape 1D or 2D = (# of channels, # of trace bins). Should
        be in units of Amps.
    template : ndarray
        The template that will be used for the Optimum Filter.
    noisepsd : ndarray
        The two-sided noise PSD that will be used to create the Optimum
        Filter.
    filtered_trace : ndarray 
        The result of the FIR filter on each of the traces.
    resolution : float
        The expected energy resolution in Amps given by the template
        and the noisepsd, calculated from the Optimum Filter.


    """

    def __init__(self, trigger_channel,
                 fs, template, noisecsd,
                 pretrigger_samples,
                 trigger_name=None,
                 template_ttl=None):
        """
        Initialization of the FIR filter.
       
        ----------
        trigger_channel : str or list of str
            the channel name (s) of the trigger. In case of NxM trigger, 
            it can be a list of channels
        fs : float
            The sample rate of the data (Hz)
        template : ndarray
            The pulse template(s) to be used when creating the optimum
            filter (assumed to be normalized). We will do an NxM trigger.
            You can pass this in a few ways:
            3D array [channels, amplitudes, samples]
            1D array or 2D array with shape [1,S] or [S,1]
                - Will be reshaped into 1 x 1 x S
            2D array with two non-unity dimensions
                - Will raise an error because we don't know whether
                  to set N_chan = 1 or M_amp = 1
        noisecsd : ndarray
            The two-sided cross power spectral density in units of A^2/Hz
            Same shape requirements as "template", though the number of
            samples is replaced by the number of frequencies (which 
            should be equal, anyway).
         template_ttl : NoneType, ndarray, optional
            The template for the trigger channel pulse. If left as
            None, then the trigger channel will not be analyzed.
        
        """

        # save internal variables
        self._fs = fs
        self._pretrigger_samples = pretrigger_samples

        # trigger_channel might be a list of channels for the NxM trigger
        self._trigger_channel = qp.utils.convert_channel_list_to_name(trigger_channel)

        # trigger name
        self._trigger_name = trigger_name
        if trigger_name is None:
            self._trigger_name = self._trigger_channel
            
        
        # Reshape template if needed to [channels, amplitudes, samples]
        n_dims_template = len(template.shape)
        if n_dims_template == 1:
            self._template = np.reshape(template, (1, 1, len(template)))
        elif n_dims_template == 2:
            if template.shape[0] == 1:
                self._template = np.reshape(template, (1, 1, template.shape[1]))
            elif template.shape[1] == 1:
                self._template = np.reshape(template, (1, 1, template.shape[0]))
            else:
                raise ValueError(
                    f'Template is shaped as {template.shape}. ' + 
                    'It should be (N, M, samples) or (samples,) or ' + 
                    '(1, samples) or (samples, 1).'
                )
        elif n_dims_template == 3:
            self._template = template

        self._nb_samples = self._template.shape[-1]
        self._posttrigger_samples = self._nb_samples - self._pretrigger_samples


        # Reshape CSD if needed to [channels, amplitudes, frequencies]
        n_dims_csd = len(noisecsd.shape)
        if n_dims_csd == 1:
            self._noisecsd = np.reshape(noisecsd, (1, 1, len(noisecsd)))
        elif n_dims_csd == 2:
            if noisecsd.shape[0] == 1:
                self._noisecsd = np.reshape(noisecsd, (1, 1, noisecsd.shape[1]))
            elif noisecsd.shape[1] == 1:
                self._noisecsd = np.reshape(noisecsd, (1, 1, noisecsd.shape[0]))
            else:
                raise ValueError(
                    f'Noise CSD is shaped as {noisecsd.shape}. ' + 
                    'Should be (N, M, frequencies) or (frequencies,) or ' + 
                    '(1, frequencies) or (frequencies, 1).'
                )
        elif n_dims_template == 3:
            self._noisecsd = noisecsd

        # Save the number of channels, amplitudes, and frequencies/times
        self._n_channels, _, self._f_frequencies = self._noisecsd.shape
        self._m_amplitudes = self._template.shape[1]
        self._t_times = self._f_frequencies

        # trigger index shift if pretrigger not midpoint
        self._trigger_index_shift = self._pretrigger_samples - self._nb_samples//2
        
        # check array shapes
        #if isinstance(trigger_channel, str):   
        #    trigger_channel = [trigger_channel]

        # intitialize trigger data
        # dictionary 
        self._trigger_data = None
  
        # Create an OF Base object to run the OF pre-calculations
        self._of_base = qp.OFBase(fs)

        template_tags = np.full((self._n_channels, self._m_amplitudes), 'default_XX_XX')
        for i in range(self._n_channels):
            for j in range(self._m_amplitudes):
                template_tags[i,j] = f'default_{i}_{j}'

        self._template_tags = template_tags

        self._of_base.add_template_many_channels(
            self._trigger_channel,
            self._template,
            template_tags,
            pretrigger_samples=self._pretrigger_samples
        )
        
        self._of_base.set_csd(self._trigger_channel, self._noisecsd)
        self._of_base.calc_phi_matrix(self._trigger_channel, template_tags)
        self._of_base.calc_weight_matrix(self._trigger_channel, template_tags)

        self._iw_matrix = self._of_base.iw_matrix(self._trigger_channel, template_tags)
        self._w_matrix = np.linalg.inv(self._iw_matrix)

        # By default, phi is in the order [f_frequencies, n_channels, m_amplitudes]
        # so we transpose. Also we need to make a copy or we accidentally pass phi
        # by reference
        self._phi_fd = np.copy(self._of_base.phi(self._trigger_channel, template_tags))
        self._phi_fd = self._phi_fd.transpose(1,2,0)

        self._phi_fd[:,:,0] = 0 #ensure we do not use DC information
        self._phi_td = ifft(self._phi_fd, axis=2).real
        
        # calculate the normalization of the optimum filter
        self._norm = np.dot(self._phi_td[0,0], self._template[0,0])
        
        # calculate the expected energy resolution for each amplitude
        self._resolution = np.sqrt(np.diag(self._iw_matrix))
                            
        # TTL template and norm
        self._template_ttl = template_ttl
        self._norm_ttl = None
        if template_ttl is not None:
            self._norm_ttl = np.dot(template_ttl, template_ttl)
    

    def get_filtered_trace(self):
        """
        Get current filtered trace
        """

        return self._filtered_trace


    def get_filtered_trace_ttl(self):
        """
        Get current filtered trace
        """

        return self._filtered_trace_ttl


    
    def get_trigger_data(self):
        """
        Get current filtered trace
        """

        return self._trigger_data

    def get_trigger_data_df(self):
        """
        Get current filtered trace
        """
        df = None
        if self._trigger_data is not None:
            df = vx.from_dict(
                self._trigger_data[self._trigger_name]
            )
        return df
           

    def get_phi(self):
        """
        Get optimal filter 
        in time domain
        """

        return self._phi_td

    
    def get_norm(self):
        """
        Get normalization
        """

        return self._norm


    def get_resolution(self):
        """
        Get resolution
        """
        return self._resolution
    

            
    def update_trace(self, trace=None, filtered_trace=None,
                     filtered_trace_ttl=None):
        """
        Method to apply the FIR filter the inputted traces with
        specified times.

        Parameters
        ----------

        trace : ndarray, optional
            trigger channel trace(s) to be filtered
            units: Amps
                Pulses should be positive-going for consistency,
                but this technically doesn't matter
            Shape: 2D array [N_channels, samples] 
            Default: None (required "filtered_trace")
           
        filtered_trace : ndarray, optional
            OF filtered channel trace(s)
            units: Amps
            Shape: 2D array [M_amplitudes, samples] 

        filtered_trace_ttl: ndarray, optional
            OF filtered trigger channel trace(s) with TTL template
            units: Amps
            1D array or 2D array [#channels, #samples]
            For single pulse channel: #channels=1
        """

        # check input
        if trace is None and filtered_trace is None:
            raise ValueError(
                'ERROR: "trace" or "filtered_trace required!'
            )

        # update the traces, times, and ttl attributes
        if (trace is not None) and (trace.ndim == 1):
            self._raw_trace = np.reshape(trace, (1, len(trace)))
        else:
            self._raw_trace = trace

        if (filtered_trace is not None) and (filtered_trace.ndim == 1):
            self._filtered_trace = np.reshape(filtered_trace, (1, len(filtered_trace)))
        else:
            self._filtered_trace = filtered_trace

        # check dimension of trace(s) here
        if trace is not None and self._raw_trace.shape[0] != self._n_channels:
            raise ValueError(
                f'ERROR: "trace" has shape {trace.shape}, ' + 
                f'but we have {self._n_channels} channels!'
            )

        if filtered_trace is not None and self._filtered_trace.shape[0] != self._m_amplitudes:
            raise ValueError(
                f'ERROR: "filtered_trace" has shape {filtered_trace.shape}, ' + 
                f'but we have {self._m_amplitudes} channels!'
            )

        # filter trace 
        if self._filtered_trace is None:

            # V is the vector of convolutions; equivalently, the element in WA = V
            V_td = np.zeros((self._m_amplitudes, self._raw_trace.shape[-1]))
            for theta in range(self._m_amplitudes):
                V_td_per_channel = oaconvolve(self._raw_trace, self._phi_td[theta,:], mode='same', axes=-1)
                V_td[theta,:] = np.sum(V_td_per_channel, axis=0)

            self._filtered_trace = np.einsum('ij,jz->iz', self._iw_matrix, V_td).real
            
        # Calculate chi2 for the filtered trace(s)
        self._delta_chi2_trace = -2 * np.einsum('iz,ij,jz->z', self._filtered_trace, self._w_matrix, self._filtered_trace)
        
        # set the filtered values to zero near the edges,
        # so as not to use the padded values in the analysis
        cut_len = self._t_times
        self._delta_chi2_trace[:cut_len] = 0.0
        self._delta_chi2_trace[-(cut_len)+(cut_len+1)%2:] = 0.0
        
        # filtered with ttl template
        self._filtered_trace_ttl = filtered_trace_ttl
        if self._template_ttl is not None:
            self._filtered_trace_ttl = correlate(
                trace,
                self._template_ttl,
                mode="same",
                method='fft')/self._norm_ttl      

            self._filtered_trace_ttl[:cut_len] = 0.0
            self._filtered_trace_ttl[-(cut_len)
                                     + (cut_len+1)%2:] = 0.0
         
            
            

            
    def find_triggers(self, thresh, thresh_ttl=None,
                      pileup_window_msec=None, pileup_window_samples=None,
                      positive_pulses=False):
        """
        Method to detect events in the traces with an optimum amplitude
        greater than the specified threshold. Note that this may return
        duplicate events, so care should be taken in post-processing to
        get rid of such events.

        Parameters
        ----------
        thresh : float
            The number of standard deviations of the energy resolution
            to use as the threshold for which events will be detected
            as a pulse.
        thresh_ttl : NoneType, float, optional
            The threshold value (in units of the ttl channel) such
            that any amplitudes higher than this will be detected as
            ttl trigger event. If left as None, then only the pulses
            are analyzed.


        pileup_window_msec : float, optional

        pileup_window_samples : int, optional
        
        positive_pulses : boolean, optional
            Boolean flag for which direction the pulses go in the
            traces. If they go in the positive direction, then this
            should be set to True. If they go in the negative
            direction, then this should be set to False. Default is
            False for the chi2 trigger, since delta chi2 is negative
            by definition.

        """

        # check filtered trace
        if self._delta_chi2_trace is None:
            raise ValueError('ERROR: Filter trace not available. '
                             + ' Use "update_trace" first!')
        
        
        # intialize dictionary with list of
        # variables
        trigger_data = {
            'trigger_amplitude':list(),
            'trigger_time': list(),
            'trigger_index': list(),
            'trigger_pileup_window': list(),
            'trigger_threshold_sigma': list(),
            'trigger_type': list()}
        
        # Extra parameters if TTL trigger used
        if (self._filtered_trace_ttl is not None
            and thresh_ttl is not None):
            trigger_data.update(
                {'trigger_time_ttl': list(),
                 'trigger_index_ttl': list(),
                 'trigger_amplitude_ttl': list(),
                 'trigger_time_pulse': list(),
                 'trigger_index_pulse': list(),
                 'trigger_amplitude_pulse': list()}
            )
               
        # merge window
        pileup_window = 0
        if pileup_window_msec is not None:
            pileup_window = int(pileup_window_msec*self._fs/1000)
        elif pileup_window_samples is not None:
            pileup_window = pileup_window_samples
            
        # Trigger on chi2, using "thresh" in sigma units as representing
        # the two-sided survival fraction of a Gaussian. For example, if
        # thresh = 1, we will trigger on 32% of samples by random chance;
        # corresponding to the 32% of probability density outside the
        # 1-sigma bands in a Gaussian. If thresh = 5, we will trigger on
        # 5.733e-07 of samples by random chance; again, the integrated
        # probability beyond +/- 5 sigma in a Gaussian.
        # This is done so that the trigger behaves roughly as it used to,
        # while accounting for the difference between a chi2 and Gaussian
        # distribution.
        survival_fraction = stats.norm.sf(thresh) * 2
        chi2_threshold = -1 * special.gammainccinv(self._m_amplitudes / 2, survival_fraction) * 4
        triggers_mask = self._delta_chi2_trace < chi2_threshold

        triggers = np.where(triggers_mask)[0]
        
        # check if any left over detected triggers are within
        # the specified pulse_range from each other
        trigger_ranges, trigger_vals = _getchangeslessthanthresh(
            triggers, pileup_window)
        
        # set the trigger type to pulses
        rangetypes = np.zeros((len(trigger_ranges), 3), dtype=bool)
        rangetypes[:,1] = True
        

        # TTL trigger
        if (self._filtered_trace_ttl is not None
            and thresh_ttl is not None):
            
            # make a boolean mask of the ranges of the events in the trace
            # from the pulse triggering
            pulse_mask = np.zeros(self._filtered_trace.shape, dtype=bool)
            for evt_range in trigger_ranges:
                if evt_range[1]>evt_range[0]:
                    trigger_inds = triggers[evt_range[0]:evt_range[1]]
                    pulse_mask[trigger_inds] = True
                    
            # find where the ttl trigger has an optimum amplitude
            # greater than the specified threshold
            triggers_mask_ttl = self._filtered_trace_ttl>thresh_ttl

            # get the mask of the total events, taking the
            # or of the pulse and ttl trigger events
            tot_mask = np.logical_or(triggers_mask_ttl, pulse_mask)
            triggers = np.where(tot_mask)[0]
            trigger_ranges, trigger_vals = _getchangeslessthanthresh(
                triggers,  pileup_window)
            
            tot_types = np.zeros(len(tot_mask), dtype=int)
            tot_types[triggers_mask] = 1
            tot_types[triggers_mask_ttl] = 2
            
            # given the ranges, determine the trigger type based on if
            # the total ranges overlap with
            # the pulse events and/or the ttl trigger events
            rangetypes = np.zeros((len(trigger_ranges), 3), dtype=bool)
            for ival, vals in enumerate(trigger_vals):
                if np.any(tot_types[vals[0]:vals[1]]==1):
                    rangetypes[ival, 1] = True
                if np.any(tot_types[vals[0]:vals[1]]==2):
                    rangetypes[ival, 2] = True


                    
        # for each range with changes less than the merge_windw
        # keep only the bin with the largest amplitude
        for irange, evt_range in enumerate(trigger_ranges):
                
            if evt_range[1]>evt_range[0]:

                evt_inds = triggers[evt_range[0]:evt_range[1]]

                # find index maximum amplitude
                evt_ind = None
                if positive_pulses:
                    evt_ind = evt_inds[np.argmax(self._delta_chi2_trace[evt_inds])]
                else:
                    evt_ind = evt_inds[np.argmin(self._delta_chi2_trace[evt_inds])]

                evt_ind_shift = evt_ind + self._trigger_index_shift

                    
                # Case TTL is used
                if (self._filtered_trace_ttl is not None
                    and thresh_ttl is not None):

                    # TTL index
                    ttl_ind = evt_inds[np.argmax(self._filtered_trace_ttl[evt_inds])]
                    ttl_ind_shift = ttl_ind + self._trigger_index_shift
                        
                    # case trigger TTL -> primary
                    if rangetypes[irange][2]:

                        # primary = TTL
                        trigger_data['trigger_index'].extend([ttl_ind_shift])
                        trigger_data['trigger_time'].extend([ttl_ind_shift/self._fs])
                        trigger_data['trigger_amplitude'].extend(
                            [self._filtered_trace[ttl_ind]])
                        trigger_data['trigger_type'].extend([5])

                        
                        # TTL                    
                        trigger_data['trigger_index_ttl'].extend([ttl_ind_shift])
                        trigger_data['trigger_time_ttl'].extend([ttl_ind_shift/self._fs])
                        trigger_data['trigger_amplitude_ttl'].extend(
                            [self._filtered_trace_ttl[ttl_ind]])
                        
                        # pulse
                        if rangetypes[irange][1]:
                            # pulse also triggered
                            trigger_data['trigger_index_pulse'].extend([evt_ind_shift])
                            trigger_data['trigger_time_pulse'].extend([evt_ind_shift/self._fs])
                            trigger_data['trigger_amplitude_pulse'].extend(
                                [self._filtered_trace[evt_ind]])
                        else:
                            # no trigger
                            trigger_data['trigger_index_pulse'].extend([np.nan])
                            trigger_data['trigger_time_pulse'].extend([np.nan])
                            trigger_data['trigger_amplitude_pulse'].extend([np.nan])

                    # case only pulse triggered
                    else:
                        # primary = pulse
                        trigger_data['trigger_index'].extend([evt_ind_shift])
                        trigger_data['trigger_time'].extend([evt_ind_shift/self._fs])
                        trigger_data['trigger_amplitude'].extend(
                            [self._filtered_trace[evt_ind]])
                        trigger_data['trigger_type'].extend([4])

                        
                        # pulse also triggered
                        trigger_data['trigger_index_pulse'].extend([evt_ind_shift])
                        trigger_data['trigger_time_pulse'].extend([evt_ind_shift/self._fs])
                        trigger_data['trigger_amplitude_pulse'].extend(
                            [self._filtered_trace[evt_ind]])
                        
                        #ttl
                        trigger_data['trigger_index_ttl'].extend([np.nan])
                        trigger_data['trigger_time_ttl'].extend([np.nan])
                        trigger_data['trigger_amplitude_ttl'].extend([np.nan])

                # Case no TTL
                else:
                    trigger_data['trigger_index'].extend([evt_ind_shift])
                    trigger_data['trigger_time'].extend([evt_ind_shift/self._fs])
                    trigger_data['trigger_amplitude'].extend([self._delta_chi2_trace[evt_ind]])
                    trigger_data['trigger_type'].extend([4])
                    
                # extra parameter both TTL and pulse threshold
                trigger_data['trigger_threshold_sigma'].extend([thresh])
                trigger_data['trigger_pileup_window'].extend([pileup_window])


        # duplicate key channel name
        self._trigger_data = dict()
        self._trigger_data[self._trigger_name] = trigger_data.copy()
        
        for key, val in trigger_data.items():
            newkey = key + '_' + self._trigger_name
            self._trigger_data[self._trigger_name][newkey] = val

        #  add channels
        nb_events = len(trigger_data['trigger_index'])
        chan_list = list()
        if nb_events>0:
            chan_list = [self._trigger_name]*nb_events
        self._trigger_data[self._trigger_name]['trigger_channel'] = chan_list
        
        
        
        
