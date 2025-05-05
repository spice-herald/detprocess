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
import copy
import warnings
import pyarrow as pa

vx.settings.main.thread_count = 1
vx.settings.main.thread_count_io = 1
pa.set_cpu_count(1)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

__all__ = ['OptimumFilterTrigger',
           'shift_templates_to_match_chi2',
           'combine_trigger_data']


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



def _getchangeslessthandynamicthresh(x, amplitudes, threshold_function):
    """
    Helper function that returns a list of the start and ending indices
    of ranges in x where consecutive values are separated by less than
    a dynamically determined threshold.

    Parameters
    ----------
    x : ndarray
        1-dimensional array of time values.
    amplitudes : ndarray
        1-dimensional array of values at each time point, same shape as x.
    amplitudes_interp : ndarray
        1-dimensional interpolation array that maps values from `vals` to `threshold` values.
        Must be monotonically increasing!
    threshold_interp : ndarray
        1-dimensional interpolation array that maps values from `vals` to `threshold` values.
    log_interp : bool
        Whether to interpolate in log space (if False, then default to linear)

    Returns
    -------
    ranges : ndarray
        List of tuples that each store the start and ending index of each range.
        For example, x[ranges[0][0]:ranges[0][1]] gives the first section of values
        that change by less than the dynamically determined threshold.
    vals : ndarray
        The corresponding starting and ending values for each range in x.
    """

    # Initialize lists for storing start and end indices of each range
    start_inds = []
    end_inds = []
    
    # Initialize the first region starting at index 0
    current_start = 0
    
    for i in range(1, len(x)):
        # Calculate the dynamic threshold for the current range
        max_amplitude = np.max(amplitudes[current_start:i+1])  # Maximum value in current region of `amplitudes`
        dynamic_threshold = threshold_function(max_amplitude)  # Calculate dynamic threshold

        # Check if the difference exceeds the dynamic threshold
        if (x[i] - x[i - 1]) > dynamic_threshold:
            # Mark the end of the current range
            start_inds.append(current_start)
            end_inds.append(i)
            
            # Start a new range
            current_start = i

    # Add the final range
    start_inds.append(current_start)
    end_inds.append(len(x))

    # Combine start and end indices into ranges
    ranges = np.array(list(zip(start_inds, end_inds)))

    # Extract start and end values for each range
    if len(x) != 0:
        vals = np.array([(x[st], x[end - 1]) for (st, end) in ranges])
    else:
        vals = np.array([])

    return ranges, vals



def shift_templates_to_match_chi2(fs, primary_template, secondary_templates, noisecsd, relative_amplitudes=None):  
    """
    Shift "secondary_template" in time such that running a
    "primary_template" trigger on both templates results in chi2 maximized
    at the same time. Assumes noise is the same between templates.
    
    ----------
    fs : float
        The sample rate of the data (Hz)
    primary_template : ndarray
        The pulse template(s) to be used when creating the optimum
        filter (assumed to be normalized). We will do an NxM trigger.
        You can pass this in a few ways:
        3D array [channels, amplitudes, samples]
        1D array or 2D array with shape [1,S] or [S,1]
            - Will be reshaped into 1 x 1 x S
        2D array with two non-unity dimensions
            - Will raise an error because we don't know whether
                to set N_chan = 1 or M_amp = 1
    secondary_templates : list of ndarrays
        The pulse template(s) to shift in time such that a
        primary_template NxM trigger running on this template will trigger
        at the same time as the same trigger running on a primary_template
        template. 
        Shape requirements for each ndarray are the same as primary_template.
        Note that the secondary templates must have the same N and M as the
        primary template!
    noisecsd : ndarray
        The two-sided cross power spectral density in units of A^2/Hz
        Same shape requirements as "primary_template", though the number of
        samples is replaced by the number of frequencies (which 
        should be equal, anyway).
    relative_amplitudes : ndarray
        The relative scaling factors between the amplitude degrees of freedom.
        Must be 1D array of length M_amp.
        Defaults to an array of ones if not provided.
    ----------
    Returns:
    secondary_templates_shifted : list of 3D ndarrays
        The shifted templates such that when the primary trigger is run on them,
        delta chi2 is maximized at the same time.
    time_shift_samples : ndarray
        For each secondary template, the number of samples that we shifted
    
    """
    
    # Container to return objects
    secondary_templates_shifted = [None for _ in range(len(secondary_templates))]
    time_shift_samples = np.zeros(len(secondary_templates), dtype=int)
    
    # Loop over templates
    for i, secondary_template in enumerate(secondary_templates):
        
        # Reshape secondary template if needed to [channels, amplitudes, samples]
        # This is now renamed s_template after reshaping
        n_dims_template = len(secondary_template.shape)
        if n_dims_template == 1:
            s_template = np.reshape(secondary_template, (1, 1, len(secondary_template)))
        elif n_dims_template == 2:
            if secondary_template.shape[0] == 1:
                s_template = np.reshape(secondary_template, (1, 1, secondary_template.shape[1]))
            elif template.shape[1] == 1:
                s_template = np.reshape(secondary_template, (1, 1, secondary_template.shape[0]))
            else:
                raise ValueError(
                    f'Primary template is shaped as {secondary_template.shape}. ' +
                    'It should be (N, M, samples) or (samples,) or ' +
                    '(1, samples) or (samples, 1).'
                )
        elif n_dims_template == 3:
            s_template = np.copy(secondary_template)

        n_channels, m_amplitudes, f_frequencies = s_template.shape
        
        # Set relative amplitudes if not given
        if relative_amplitudes is None:
            relative_amplitudes = np.ones(m_amplitudes)

        # Create primary OF trigger
        primary_oftrigger = OptimumFilterTrigger(
            '|'.join([f'channel_{j}' for j in range(n_channels)]),
            fs, primary_template, noisecsd, int(f_frequencies/2))
         
        # Convert primary template to a trace by summing amplitudes in
        # the appropriate ratios
        primary_trace = np.zeros((n_channels, f_frequencies))
        for iamp in range(m_amplitudes):
            primary_trace += primary_template[:,iamp,:] * relative_amplitudes[iamp]
            
        # Run the primary trigger on the primary template
        primary_oftrigger.update_trace(primary_trace, padding=False)

        # Determine what time the primary template delivers its max delta chi2        
        primary_delta_chi2 = primary_oftrigger.get_filtered_delta_chi2()
        time_max_amplitude_primary_samples = np.argmax(primary_delta_chi2)

        # Convert secondary template to a trace by summing amplitudes in
        # the appropriate ratios
        secondary_trace = np.zeros((n_channels, f_frequencies))
        for iamp in range(m_amplitudes):
            secondary_trace += s_template[:,iamp,:] * relative_amplitudes[iamp]

        # Run the primary trigger on the primary template
        primary_oftrigger.update_trace(secondary_trace, padding=False)

        # Determine what time the primary template delivers its max delta chi2        
        secondary_delta_chi2 = primary_oftrigger.get_filtered_delta_chi2()
        time_max_amplitude_secondary_samples = np.argmax(secondary_delta_chi2)
        
        # Determine the time shift and apply it to the secondary template
        time_shift_samples[i] = time_max_amplitude_primary_samples - time_max_amplitude_secondary_samples
        secondary_templates_shifted[i] = np.roll(s_template, time_shift_samples[i])
    
    return secondary_templates_shifted, time_shift_samples

    
def combine_trigger_data(
    original_trigger_data, new_trigger_data, original_triggers, new_triggers
):
    """
    Combines the dictionaries `original_trigger_data` and `new_trigger_data`
    without replicating entries, appending only values corresponding to entries
    in `new_triggers` that are not in `original_triggers`.

    Parameters
    ----------
    original_trigger_data : dict
        Dictionary containing the original trigger data.
    new_trigger_data : dict
        Dictionary containing the new trigger data.
    original_triggers : list
        List of integers representing original triggers.
    new_triggers : list
        List of integers representing new triggers.

    Returns
    -------
    dict
        Combined dictionary with new non-duplicate entries appended.
    """
    
    # Find entries in `new_triggers` not in `original_triggers`
    unique_new_triggers = set(new_triggers) - set(original_triggers)
    trigger_name = list(original_trigger_data.keys())[0]

    # Get the original and new trigger name keys
    appended_inner_dict = copy.deepcopy(original_trigger_data[trigger_name])
    new_inner_dict = copy.deepcopy(new_trigger_data[trigger_name])

    # Loop through each key in the inner dictionaries
    for key in new_inner_dict.keys():
        # Keys that are named similarly are identical
        # in reference, not just in value. I.e. trigger_index and
        # trigger_index_ch1|ch2 are the same object.
        if ('_' + trigger_name) in key:
            continue
        # Append values corresponding to unique triggers from new_trigger_data
        for idx, trigger in enumerate(new_triggers):
            if trigger in unique_new_triggers:
                appended_inner_dict[key].append(new_inner_dict[key][idx])

    return {trigger_name: appended_inner_dict}



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
                 trigger_name=None):
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

        # tag 
        template_tag = 'default'

        self._of_base.add_template(
            self._trigger_channel,
            self._template,
            template_tag=template_tag,
            pretrigger_samples=self._pretrigger_samples
        )
        
        self._of_base.set_csd(self._trigger_channel, self._noisecsd)
        self._of_base.calc_phi(self._trigger_channel,
                               template_tag=template_tag,
                               calc_weight=True)
        
        self._iw_matrix = self._of_base.iweight(self._trigger_channel, template_tag)
        self._w_matrix = self._of_base.weight(self._trigger_channel, template_tag)
        
        # Get phi then take inverse fourier transform
        # phi shape in OF base (n_channels, m_amplitudes, f_frequencies)      
        self._phi_fd = np.copy(self._of_base.phi(self._trigger_channel, template_tag))
        self._phi_fd[:,:,0] = 0 #ensure we do not use DC information
        self._phi_td = ifft(self._phi_fd, axis=2).real
        
        # calculate the normalization of the optimum filter
        self._norm = np.dot(self._phi_td[0,0], self._template[0,0])
        
        # calculate the expected energy resolution for each amplitude
        self._resolution = np.sqrt(np.diag(self._iw_matrix))
                            
     
    def get_filtered_trace(self):
        """
        Get current filtered trace
        """

        return self._filtered_trace


    def get_filtered_delta_chi2(self):
        """
        Get current delta chi2 trace
        """

        return self._delta_chi2_trace
    
    
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
                     padding=True):
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
            
        padding : bool
            if True, set the filtered values to zero near the edges

        """

        # check input
        if trace is None and filtered_trace is None:
            raise ValueError(
                'ERROR: "trace" or "filtered_trace required!'
            )

        # reshape if needed
        if (trace is not None) and (trace.ndim == 1):
            self._raw_trace = np.reshape(trace, (1, len(trace)))
        else:
            self._raw_trace = trace

        self._raw_trace_LPF_50kHz = np.copy(self._raw_trace)
        for ich in range(self._n_channels):            
            self._raw_trace_LPF_50kHz[ich] = qp.utils.lowpassfilter(
                self._raw_trace[ich],
                cut_off_freq=50e3,
                fs=self._fs
            )

        if (filtered_trace is not None) and (filtered_trace.ndim == 1):
            self._filtered_trace = np.reshape(filtered_trace,
                                              (1, len(filtered_trace)))
        else:
            self._filtered_trace = filtered_trace

        # check dimension of trace(s) here
        if trace is not None and self._raw_trace.shape[0] != self._n_channels:
            raise ValueError(
                f'ERROR: "trace" has shape {trace.shape}, ' + 
                f'but we have {self._n_channels} channels!'
            )

        if (filtered_trace is not None
            and self._filtered_trace.shape[0] != self._m_amplitudes):
            raise ValueError(
                f'ERROR: "filtered_trace" has shape {filtered_trace.shape}, ' + 
                f'but we have {self._m_amplitudes} channels!'
            )

        # filter trace 
        if self._filtered_trace is None:
            
            # V is the vector of convolutions; equivalently, the element in WA = V
            V_td = np.zeros((self._m_amplitudes, self._raw_trace.shape[-1]))
            for theta in range(self._m_amplitudes):
                V_td_per_channel = oaconvolve(self._raw_trace,
                                              self._phi_td[:,theta,:],
                                              mode='same', axes=-1)
                V_td[theta,:] = np.sum(V_td_per_channel, axis=0)

            self._filtered_trace = np.einsum('ij,jz->iz', self._iw_matrix, V_td).real
            
        # Calculate chi2 for the filtered trace(s)
        self._delta_chi2_trace = np.einsum('iz,ij,jz->z',
                                           self._filtered_trace,
                                           self._w_matrix,
                                           self._filtered_trace)
        
        # set the filtered values to zero near the edges,
        # so as not to use the padded values in the analysis
        if padding:
            cut_len = self._t_times
            self._delta_chi2_trace[:cut_len] = 0.0
            self._delta_chi2_trace[-(cut_len)+(cut_len+1)%2:] = 0.0
        
      
    def find_triggers(self, thresh, 
                      pileup_window_msec=None, pileup_window_samples=None,
                      positive_pulses=True, dynamic=False,
                      dynamic_threshold_function=None, residual=False,
                      saturation_amplitudes_LPF_50kHz=None,
                      return_trigger_data=False):
        """
        Method to detect events in the traces with a delta chi2 amplitude
        greater than the specified threshold. Note that this may return
        duplicate events, so care should be taken in post-processing to
        get rid of such events.
        
        Make sure to also read the documentation for find_triggers_once().
        
        If residual == False, then this simply calls find_triggers_once().
        
        If residual == True, then we call find_triggers_once(). We then
        subtract the best-fit triggered pulses from the trace (in delta
        chi2 space), excluding saturated pulses which we know won't match
        the template. Then, we retrigger on the residual trace. If any new
        triggers above threshold are found, we combine those with the
        first-pass triggers and save all the triggers in the
        self._trigger_data dictionary.

        Parameters
        ----------
                
        thresh : float
            See find_triggers_once() for documentation.
        pileup_window_msec : float, optional
            See find_triggers_once() for documentation.
        pileup_window_samples : int, optional
            See find_triggers_once() for documentation.
        dynamic : bool, optional
            See find_triggers_once() for documentation.
        dynamic_threshold_function : function, optional
            See find_triggers_once() for documentation.
        residual : bool, optional
            If True, we search for triggers on the residual chi2 trace
            after subtracting the best-fit pulses from the first-pass
            trigger round.
        saturation_amplitudes_LPF_50kHz : array_like, optional
            If provided, this should be an N-length array or list, where
            N is the number of channels. Each element represents the
            saturation amplitude in each channel, in A.
            The raw trace is passed through a 50 kHz low-pass filter.
            Then, if the LPF-filtered trace passes above the saturation
            amplitude within template_length/4 samples of a given trigger,
            this trigger is considered to be a saturated pulse. The
            residual algorithm won't run on this trigger.
            If residual == True, but saturation_amplitudes_LPF_50kHz is
            not provided, we just set the saturation amplitude to inf or
            negative inf, based on positive_pulses; equivalently,
            disabling the check saturation.
        return_trigger_data : bool, optional
            If True, return four objects: the first-pass trigger dictionary,
            the first-pass delta chi2 trace, the second-pass trigger
            dictionary, and the second-pass delta chi2 trace. If False,
            return nothing. Regardless of this argument's value, all the
            triggers will be saved in self._trigger_data, but setting this
            True is the only way to determine if an individual trigger
            was from the first-pass or second-pass triggering.
 
        """

        if residual:
            
            # Set the saturation amplitude to +/- infinity if not given
            if saturation_amplitudes_LPF_50kHz is None:
                if positive_pulses:
                    saturation_amplitudes_LPF_50kHz = [np.inf for _ in self._n_channels]
                else:
                    saturation_amplitudes_LPF_50kHz = [-1 * np.inf for _ in self._n_channels]

            # Do first pass of triggers
            self.find_triggers_once(thresh,
                                    pileup_window_msec, pileup_window_samples,
                                    dynamic, dynamic_threshold_function)

            # Save the first-pass triggers so we don't lose them
            original_triggers = np.copy(self._trigger_data[self._trigger_name]['trigger_index'])
            original_trigger_data = copy.deepcopy(self._trigger_data)
            original_delta_chi2_trace = np.copy(self._delta_chi2_trace)
            
            # Loop over first-pass triggers
            for trigger_index in original_triggers:
                
                # For each trigger, check if the pulse is saturated. If so,
                # ignore this trigger and move on to the next trigger.
                saturated = False
                for ch_index in range(self._n_channels):
                    pulse_amplitude_A = self._raw_trace_LPF_50kHz[ch_index][trigger_index - int(self._t_times / 4) : trigger_index + int(self._t_times / 4)]
                    if positive_pulses:
                        if sum(pulse_amplitude_A > saturation_amplitudes_LPF_50kHz[ch_index]) > 0:
                            saturated = True
                    else:
                        if sum(pulse_amplitude_A < - 1 * saturation_amplitudes_LPF_50kHz[ch_index]) > 0:
                            saturated = True

                if saturated:
                    continue

                # Based on the M filtered amplitude(s) at the location of
                # the trigger, construct the "best" chi2 shape vs. time,
                # assuming the templates are correct. This is stored in
                # delta_chi2_trace.
                trigger_amplitudes = self._filtered_trace[:,trigger_index]
                
                trigger_trace = np.zeros((self._n_channels, self._t_times))
                for iamp in range(self._m_amplitudes):
                    trigger_trace += self._template[:, iamp, :] * trigger_amplitudes[iamp]

                V_td = np.zeros((self._m_amplitudes, self._t_times))
                for theta in range(self._m_amplitudes):
                    V_td_per_channel = oaconvolve(trigger_trace, self._phi_td[theta,:],
                                                  mode='same', axes=-1)
                    V_td[theta,:] = np.sum(V_td_per_channel, axis=0)

                filtered_trace = np.einsum('ij,jz->iz', self._iw_matrix, V_td).real
            
                delta_chi2_trace = np.einsum('iz,ij,jz->z',
                                             filtered_trace,
                                             self._w_matrix,
                                             filtered_trace)
                
                # Subtract delta_chi2_trace for this trigger from the original
                # full self._delta_chi2_trace.
                j_samples = np.argmax(delta_chi2_trace)
                self._delta_chi2_trace[trigger_index - j_samples : trigger_index - j_samples + self._t_times] -= (
                    delta_chi2_trace
                )

            # Retrigger on the residual self._delta_chi2_trace
            self.find_triggers_once(thresh,
                                    pileup_window_msec, pileup_window_samples,
                                    dynamic, dynamic_threshold_function)
            
            # Save our second-pass triggers so we don't lose them
            new_triggers = np.copy(self._trigger_data[self._trigger_name]['trigger_index'])
            new_trigger_data = copy.deepcopy(self._trigger_data)

            # Return self._delta_chi2_trace to its original value
            # because right now, it is saving the residual trace.
            self._residual_delta_chi2_trace = np.copy(self._delta_chi2_trace)
            self._delta_chi2_trace = np.copy(original_delta_chi2_trace)
            
            # Combine all triggers into a single dictionary
            combined_trigger_data = combine_trigger_data(
                original_trigger_data, new_trigger_data, original_triggers, new_triggers)
            self._trigger_data = copy.deepcopy(combined_trigger_data)
            
            # If the user wants the first-pass and second-pass triggers
            # separately, return those
            if return_trigger_data:
                return original_trigger_data, original_delta_chi2_trace, new_trigger_data, new_delta_chi2_trace


        else:
            
            # If residual algorithm is disabled, just run find_triggers_once()
            self.find_triggers_once(thresh,
                      pileup_window_msec, pileup_window_samples,
                      dynamic, dynamic_threshold_function)



            
    def find_triggers_once(self, thresh,
                      pileup_window_msec=None, pileup_window_samples=None,
                      dynamic=False,
                      dynamic_threshold_function=None):
        """
        Method to detect events in the traces with a delta chi2 amplitude
        greater than the specified threshold. Note that this may return
        duplicate events, so care should be taken in post-processing to
        get rid of such events.

        Parameters
        ----------
        thresh : float
            The number of standard deviations of the energy resolution
            to use as the threshold for which events will be detected
            as a pulse.
        pileup_window_msec : float, optional
            A static pileup window that coalesces regions of time with
            delta chi2 above threshold into a single trigger, as long
            as they are separated by less than the window.
        pileup_window_samples : int, optional
            Same as pileup_window_msec, but in samples
        dynamic : bool, optional
            If True, then pileup_window_msec and pileup_window_samples
            will be ignored. Instead the pileup window will be
            trigger amplitude-dependent. If True, you must also pass,
            dynamic_threshold_function.
        dynamic_threshold_function : function, optional
            A function (probably lambda/anonymous, but doesn't have
            to be) that converts a delta chi2 value to a pileup window
            in samples. This function should accept one scalar float as
            an argument and output one scalar float.
        
        """

        # check filtered trace
        if self._delta_chi2_trace is None:
            raise ValueError('ERROR: Filter trace not available. '
                             + ' Use "update_trace" first!')
        
        
        # intialize dictionary with list of
        # variables
        trigger_data = {
            'trigger_delta_chi2':list(),
            'trigger_time': list(),
            'trigger_index': list(),
            'trigger_pileup_window': list(),
            'trigger_threshold_sigma': list(),
            'trigger_type': list()}
        for iamp in range(self._m_amplitudes):
            trigger_data[f'trigger_amplitude_{iamp}'] = list()
        if self._m_amplitudes == 1:
            trigger_data[f'trigger_amplitude'] = list()
                       
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
        
        #stats.norm.sf can't give arbitrarilly small values, need to switch
        #to a different calculation method around threshold = 30. We'll just
        #assume k = 1 so chi^2 = sigma^2
        
        if thresh < 25:
            survival_fraction = stats.norm.sf(thresh) * 2
            chi2_threshold = special.gammainccinv(self._m_amplitudes / 2, survival_fraction) * 2
        else:
            chi2_threshold = thresh**2
            if self._m_amplitudes > 1:
                warnings.warn('You have asked for an amplitude threshold of ' + 
                        f'{thresh} sigma, but this is too high for us ' + 
                        'calculate the equivalent chi2 threshold. We are ' + 
                        'going to use the result for M = 1, which is a ' +
                        f'threshold of chi2 = {chi2_threshold}.')
        
        triggers_mask = self._delta_chi2_trace > chi2_threshold

        triggers = np.where(triggers_mask)[0]
        
        # check if any left over detected triggers are within
        # the specified pulse_range from each other
        
        if dynamic:
            trigger_ranges, trigger_vals = _getchangeslessthandynamicthresh(
                triggers, self._delta_chi2_trace[triggers_mask],
                dynamic_threshold_function)
        else:
            trigger_ranges, trigger_vals = _getchangeslessthanthresh(
                triggers, pileup_window)
        
        # set the trigger type to pulses
        rangetypes = np.zeros((len(trigger_ranges), 3), dtype=bool)
        rangetypes[:,1] = True
                    
        # for each range with changes less than the merge_windw
        # keep only the bin with the largest amplitude
        for irange, evt_range in enumerate(trigger_ranges):
                
            if evt_range[1]>evt_range[0]:

                evt_inds = triggers[evt_range[0]:evt_range[1]]

                # find index maximum amplitude
                evt_ind = None
                evt_ind = evt_inds[np.argmax(self._delta_chi2_trace[evt_inds])]
                evt_ind_shift = evt_ind + self._trigger_index_shift

                # fill dictionary
                trigger_data['trigger_index'].extend([evt_ind_shift])
                trigger_data['trigger_time'].extend([evt_ind_shift/self._fs])
                trigger_data['trigger_delta_chi2'].extend([self._delta_chi2_trace[evt_ind]])
                trigger_data['trigger_type'].extend([4])
                for iamp in range(self._m_amplitudes):
                    trigger_data[f'trigger_amplitude_{iamp}'].extend([self._filtered_trace[iamp][evt_ind]])
                if self._m_amplitudes == 1:
                    trigger_data[f'trigger_amplitude'].extend([self._filtered_trace[0][evt_ind]])
                    
                # extra parameters 
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
            if '\0' in self._trigger_name:
                self._trigger_name = self._trigger_name.replace('\0', '')
            self._trigger_data[self._trigger_name]['trigger_channel'] = (
                pa.array([self._trigger_name]*nb_events,
                         type=pa.string())
            )
                        
        
        
