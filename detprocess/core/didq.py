import os
import numpy as np
import pandas as pd
import qetpy as qp
import pytesio as h5io
import vaex as vx
from glob import glob
from pathlib import Path

from detprocess.core.filterdata import FilterData


def identify_didq_channels(detector_config,
                           heater_channel=None,
                           thermometer_channel=None):
    """
    Identify the heater and thermometer channels of a dIdQ measurement.

    The heater is the channel into which the square wave is injected, which the
    data acquisition records as signal_gen_source equal to "tes". The
    thermometer is the remaining channel.

    Parameters
    ----------
    detector_config : dict
        Detector configuration keyed by detector channel name, as returned by
        pytesio H5Reader.get_detector_config.
    heater_channel : str, optional
        Explicit heater channel name, bypassing auto-detection.
    thermometer_channel : str, optional
        Explicit thermometer channel name, bypassing auto-detection.

    Return
    ------
    heater : str
        Heater channel name.
    thermometer : str
        Thermometer channel name.
    """

    channels = list(detector_config.keys())

    for name, label in ((heater_channel, 'heater'),
                        (thermometer_channel, 'thermometer')):
        if (name is not None) and (name not in channels):
            raise ValueError(
                f'ERROR: requested {label} channel "{name}" is not present '
                f'in the data. Available channels: {channels}'
            )

    heater = heater_channel
    if heater is None:
        candidates = list()
        for channel in channels:
            source = str(detector_config[channel].get('signal_gen_source', ''))
            if source.lower() == 'tes':
                candidates.append(channel)

        if len(candidates) == 0:
            raise ValueError(
                'ERROR: no heater channel found. No channel has '
                'signal_gen_source set to "tes". Use the heater_channel '
                'argument to set it explicitly.'
            )

        if len(candidates) > 1:
            raise ValueError(
                f'ERROR: more than one heater candidate found {candidates}. '
                f'Use the heater_channel argument to disambiguate.'
            )

        heater = candidates[0]

    thermometer = thermometer_channel
    if thermometer is None:
        remaining = [chan for chan in channels if chan != heater]

        if len(remaining) != 1:
            raise ValueError(
                f'ERROR: expecting exactly one thermometer channel, found '
                f'{remaining}. Use the thermometer_channel argument to set '
                f'it explicitly.'
            )

        thermometer = remaining[0]

    return heater, thermometer


def resolve_drive_parameters(detector_config,
                             heater_channel,
                             duty_cycle=0.5):
    """
    Resolve the square wave drive parameters for a dIdQ measurement.

    Every value is read from the heater channel's detector configuration as
    recorded by the data acquisition, so a fit always uses the parameters the
    hardware actually ran with. The duty cycle is the one exception, since it
    is not recorded.

    The drive amplitude is a current because that is what the dIdV model needs.
    The signal generator is configured in volts, but the acquisition divides by
    the signal generator series resistance and records the resulting current in
    signal_gen_current.

    Parameters
    ----------
    detector_config : dict
        Detector configuration keyed by detector channel name.
    heater_channel : str
        Heater channel name.
    duty_cycle : float, optional
        Square wave duty cycle. Not recorded in the raw data, so it defaults
        to 0.5.

    Return
    ------
    params : dict
        Dictionary with keys sgfreq_hz, sgamp_amps, rshunt_ohms and
        duty_cycle, all floats.
    """

    if heater_channel not in detector_config:
        raise ValueError(
            f'ERROR: heater channel "{heater_channel}" is not present in the '
            f'detector configuration.'
        )

    config = detector_config[heater_channel]

    def read_required(key):
        """
        Read one drive parameter recorded by the data acquisition.

        Parameters
        ----------
        key : str
            Metadata key to read.

        Return
        ------
        value : float
            Recorded parameter value, in the units the acquisition wrote.
        """
        if key not in config:
            raise ValueError(
                f'ERROR: "{key}" not found in the detector configuration for '
                f'channel "{heater_channel}".'
            )

        value = float(config[key])
        if np.isnan(value):
            raise ValueError(
                f'ERROR: "{key}" is NaN for channel "{heater_channel}". The '
                f'acquisition did not record a valid drive setting, so this '
                f'series cannot be fit.'
            )

        return value

    params = dict()
    params['sgfreq_hz'] = read_required('signal_gen_frequency')
    params['sgamp_amps'] = read_required('signal_gen_current')

    # the shunt resistance is stored under either of two names depending on
    # the acquisition version
    if 'shunt_resistance' in config:
        params['rshunt_ohms'] = float(config['shunt_resistance'])
    elif 'rshunt' in config:
        params['rshunt_ohms'] = float(config['rshunt'])
    else:
        raise ValueError(
            f'ERROR: no shunt resistance found for channel '
            f'"{heater_channel}".'
        )

    if np.isnan(params['rshunt_ohms']):
        raise ValueError(
            f'ERROR: shunt resistance is NaN for channel "{heater_channel}".'
        )

    params['duty_cycle'] = float(duty_cycle)

    return params


def validate_drive_period_coverage(fs, nb_samples, sgfreq_hz,
                                   series_name=None):
    """
    Check that a trace is long enough to hold at least one whole drive period.

    The dIdV deconvolution trims every trace to a whole number of square wave
    periods before transforming it, so a trace holding less than one period is
    trimmed to nothing. qetpy then fails deep inside _deconvolvedidv with an
    IndexError that names neither the series nor the drive frequency, and only
    after the pile-up cut has already spent minutes on traces that were never
    usable. This check runs on metadata alone, before any trace is read.

    Parameters
    ----------
    fs : float
        Sample rate, in Hz.
    nb_samples : int
        Number of samples in one trace.
    sgfreq_hz : float
        Square wave drive frequency, in Hz.
    series_name : str, optional
        Series name, included in the error message when given.

    Return
    ------
    None
    """

    fs = float(fs)
    nb_samples = int(nb_samples)
    sgfreq_hz = float(sgfreq_hz)

    label = 'this series'
    if series_name is not None:
        label = f'series {series_name}'

    if not np.isfinite(sgfreq_hz) or (sgfreq_hz <= 0.0):
        raise ValueError(
            f'ERROR: {label} records a square wave drive frequency of '
            f'{sgfreq_hz} Hz. The drive frequency must be a positive number '
            f'for the trace to contain any periods at all.'
        )

    duration_s = nb_samples / fs
    periods_per_trace = duration_s * sgfreq_hz
    samples_per_period = int(np.ceil(fs / sgfreq_hz))

    if periods_per_trace >= 1.0:
        return

    raise ValueError(
        f'ERROR: {label} holds only {periods_per_trace:.2f} periods of its '
        f'{sgfreq_hz:g} Hz square wave drive, but at least one whole period '
        f'is required.\n'
        f'  trace        : {nb_samples} samples at {fs:g} Hz '
        f'= {duration_s:g} s\n'
        f'  drive period : {1.0 / sgfreq_hz:g} s '
        f'= {samples_per_period} samples\n'
        f'The deconvolution trims each trace to a whole number of drive '
        f'periods, so this trace would be trimmed to zero samples. Retake '
        f'the data with at least {samples_per_period} samples per trace, or '
        f'drive the heater faster than {1.0 / duration_s:.3f} Hz.'
    )


def get_driven_bin_mask(freq, sgfreq_hz, duty_cycle=0.5):
    """
    Build a mask selecting the frequency bins actually driven by the square wave.

    A square wave with a 50 percent duty cycle has power only in odd harmonics
    of its fundamental. An asymmetric duty cycle also drives even harmonics.
    The zero-frequency bin is always excluded.

    The undriven bins are exactly the bins that qetpy assigns an error of 1e20,
    so restricting the fit to the driven bins is equivalent to fitting all of
    them, while being roughly 100 times faster.

    Parameters
    ----------
    freq : ndarray
        Frequency array, in Hz, as produced by the qetpy dIdV processing.
    sgfreq_hz : float
        Square wave fundamental frequency, in Hz.
    duty_cycle : float, optional
        Square wave duty cycle. Default is 0.5.

    Return
    ------
    mask : ndarray of bool
        True where the bin is driven by the square wave.
    """

    # the tolerance mirrors qetpy, which allows for float error of order 1e-10
    # when taking the modulus of large numbers
    ratio = np.abs(np.asarray(freq) / sgfreq_hz)
    odd_bins = (np.abs(np.mod(ratio, 2.0) - 1.0) < 1e-8)

    if duty_cycle == 0.5:
        mask = odd_bins
    else:
        even_bins = (np.abs(np.mod(ratio + 1.0, 2.0) - 1.0) < 1e-8)
        mask = np.logical_or(odd_bins, even_bins)

    mask = np.array(mask, dtype=bool)
    mask[np.asarray(freq) == 0.0] = False

    return mask


def compute_snr_diagnostic(freq, didv_mean, didv_std, driven_mask,
                           snr_threshold=3.0):
    """
    Summarise where the measurement still carries signal.

    This is advisory only. It never changes the fit or the cutoff. Its purpose
    is to make visible how far up in frequency the data actually constrains the
    model, so a user choosing a cutoff has the information available.

    Parameters
    ----------
    freq : ndarray
        Frequency array, in Hz.
    didv_mean : ndarray of complex
        Ensemble mean transfer function.
    didv_std : ndarray of complex
        Ensemble standard deviation of the transfer function.
    driven_mask : ndarray of bool
        Mask selecting the driven frequency bins.
    snr_threshold : float, optional
        Threshold defining a bin as carrying signal. Default is 3.0.

    Return
    ------
    diagnostic : dict
        Dictionary with keys f_max_snr3_hz, the highest positive driven
        frequency whose SNR exceeds the threshold, and n_freq_bins_snr3, how
        many positive driven bins exceed it. f_max_snr3_hz is NaN when no bin
        does.
    """

    freq = np.asarray(freq)
    selection = np.logical_and(driven_mask, freq > 0.0)

    diagnostic = {'f_max_snr3_hz': float('nan'),
                  'n_freq_bins_snr3': 0}

    if not selection.any():
        return diagnostic

    magnitude = np.abs(np.asarray(didv_mean)[selection])
    sigma = np.abs(np.asarray(didv_std)[selection])

    # bins that qetpy neutralised have an enormous sigma, so they fall below
    # the threshold naturally and need no special handling here
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(sigma > 0.0, magnitude / sigma, 0.0)

    above = (snr > snr_threshold)
    diagnostic['n_freq_bins_snr3'] = int(np.sum(above))

    if above.any():
        diagnostic['f_max_snr3_hz'] = float(np.max(freq[selection][above]))

    return diagnostic


class DIDQAnalysis(FilterData):
    """
    Class to manage dIdQ calculations using QETpy.

    A dIdQ measurement injects a large-amplitude square wave into one TES on a
    chip, the heater, and measures the heat response in another TES on the same
    chip, the thermometer. The thermometer response is fitted in the frequency
    domain with the same two-pole and three-pole models used for dIdV.

    Only the raw fit output is reported. Small signal parameters are never
    calculated, and the fitted poles are not related back to dIdV quantities.
    """

    def __init__(self, verbose=True,
                 auto_save_hdf5=False,
                 file_path_name=None,
                 filter_data=None):
        """
        Initialize a dIdQ analysis.

        Parameters
        ----------
        verbose : bool, optional
            Print progress information. Default is True.
        auto_save_hdf5 : bool, optional
            Automatically save results to HDF5. Default is False.
        file_path_name : str, optional
            Output file or directory for saved results.
        filter_data : dict, optional
            Pre-existing filter data to attach.

        Return
        ------
        None
        """

        super().__init__(verbose=verbose, filter_data=filter_data)

        self._didq_data = dict()
        self._verbose = verbose
        self._save_hdf5 = auto_save_hdf5
        self._save_path = None
        self._file_name = 'didq_analysis.hdf5'

        if file_path_name is not None:
            if os.path.isfile(file_path_name):
                self._save_path = os.path.dirname(file_path_name)
                self._file_name = os.path.basename(file_path_name)
            elif os.path.isdir(file_path_name):
                self._save_path = file_path_name
            else:
                raise ValueError('ERROR: "file_path_name" should be a '
                                 'file or path!')

    def get_series_names(self):
        """
        List the series that have been loaded.

        Parameters
        ----------
        None

        Return
        ------
        names : list of str
            Series names, in insertion order.
        """

        return list(self._didq_data.keys())

    def get_didq_data(self, series_name):
        """
        Return the stored data for one series.

        Parameters
        ----------
        series_name : str
            Series name.

        Return
        ------
        data : dict
            Stored data for the series.
        """

        if series_name not in self._didq_data:
            raise ValueError(
                f'ERROR: no dIdQ data found for series "{series_name}".'
            )

        return self._didq_data[series_name]

    def set_traces(self, traces, fs, drive_params,
                   series_name='synthetic', metadata=None):
        """
        Load a trace ensemble directly, bypassing raw file reading.

        Used by tests and by notebook workflows that already hold traces in
        memory. The ensemble is averaged immediately, exactly as dIdV does.

        Parameters
        ----------
        traces : ndarray
            Array of shape (n_traces, n_samples), in amps.
        fs : float
            Sample rate, in Hz.
        drive_params : dict
            Drive parameters as returned by resolve_drive_parameters.
        series_name : str, optional
            Name to store the data under. Default is "synthetic".
        metadata : dict, optional
            Extra metadata to record alongside the results.

        Return
        ------
        None
        """

        traces = np.asarray(traces)
        if traces.ndim != 2:
            raise ValueError(
                f'ERROR: expecting traces of shape (n_traces, n_samples), '
                f'got shape {traces.shape}.'
            )

        # guards the in-memory entry point used by notebooks and by the
        # parallel workers, so a short trace cannot reach the deconvolution
        validate_drive_period_coverage(
            fs=fs,
            nb_samples=traces.shape[1],
            sgfreq_hz=drive_params['sgfreq_hz'],
            series_name=series_name,
        )

        didvobj = qp.DIDV(
            traces,
            fs,
            drive_params['sgfreq_hz'],
            drive_params['sgamp_amps'],
            drive_params['rshunt_ohms'],
            dutycycle=drive_params['duty_cycle'],
            add180phase=False,
        )
        didvobj.processtraces()

        driven_mask = get_driven_bin_mask(
            freq=didvobj._freq,
            sgfreq_hz=drive_params['sgfreq_hz'],
            duty_cycle=drive_params['duty_cycle'],
        )

        diagnostic = compute_snr_diagnostic(
            freq=didvobj._freq,
            didv_mean=didvobj._didvmean,
            didv_std=didvobj._didvstd,
            driven_mask=driven_mask,
        )

        if metadata is None:
            metadata = dict()

        self._didq_data[series_name] = {
            'didvobj': didvobj,
            'drive_params': dict(drive_params),
            'driven_mask': driven_mask,
            'snr_diagnostic': diagnostic,
            'n_freq_bins_total': int(didvobj._freq.size),
            'n_traces_used': int(didvobj._ntraces),
            'fs': float(fs),
            'nb_samples': int(traces.shape[1]),
            'metadata': dict(metadata),
            'fit_results': dict(),
        }

    def pool_series(self, series_names=None, pooled_name=None,
                    drop_series=True):
        """
        Merge the trace ensembles of several series into a single ensemble.

        The series of a dIdQ group are repeat measurements at one bias point,
        so their traces describe the same thing and gain nothing from being
        averaged apart. Pooling concatenates the traces that survived the cuts
        in every series and averages them together. That is not the same as
        averaging the per-series mean traces: the ensemble standard deviation
        that weights the fit is rebuilt from the pooled ensemble rather than
        approximated from the per-series spreads.

        The series must share a bias point and a drive, or their traces are
        measurements of different things and averaging them is meaningless.
        That is checked here rather than left to the fit.

        Parameters
        ----------
        series_names : list of str, optional
            Series to pool. Default is every loaded series.
        pooled_name : str, optional
            Name to store the pooled ensemble under. Default is derived from
            the group name.
        drop_series : bool, optional
            Remove the individual series once they are pooled, so only the
            pooled ensemble can be fitted or saved. Default is True.

        Return
        ------
        pooled_name : str
            Name the pooled ensemble is stored under.
        """

        if series_names is None:
            series_names = self.get_series_names()

        if not series_names:
            raise ValueError('ERROR: there are no series to pool.')

        self.validate_series_are_poolable(series_names=series_names)

        reference = self.get_didq_data(series_names[0])

        if pooled_name is None:
            pooled_name = build_pooled_series_name(
                group_name=reference['metadata'].get('group_name', 'pooled')
            )

        pooled_metadata = self._build_pooled_metadata(
            series_names=series_names, pooled_name=pooled_name,
        )

        # the traces that survived the cuts live on the qetpy object of each
        # series, in amps and at full length, so the pooled ensemble is just
        # their concatenation
        trace_blocks = list()
        for series_name in series_names:
            trace_blocks.append(
                self.get_didq_data(series_name)['didvobj']._rawtraces
            )

        pooled_traces = np.concatenate(trace_blocks, axis=0)

        if self._verbose:
            print(f'INFO: pooling {len(series_names)} series into '
                  f'{pooled_traces.shape[0]} traces, stored as {pooled_name}')

        # the blocks reference the per-series objects, so they are released
        # before the pooled ensemble is averaged. Otherwise every trace is
        # held twice through the averaging, which is the memory high water
        # mark of the whole run
        del trace_blocks

        if drop_series:
            for series_name in series_names:
                del self._didq_data[series_name]

        self.set_traces(
            traces=pooled_traces,
            fs=reference['fs'],
            drive_params=reference['drive_params'],
            series_name=pooled_name,
            metadata=pooled_metadata,
        )

        return pooled_name

    def validate_series_are_poolable(self, series_names):
        """
        Check that several loaded series may be averaged into one ensemble.

        Parameters
        ----------
        series_names : list of str
            Series to check.

        Return
        ------
        None
        """

        validate_series_are_poolable(
            series_names=series_names,
            series_data=[
                self.get_didq_data(series_name)
                for series_name in series_names
            ],
        )

    def _build_pooled_metadata(self, series_names, pooled_name):
        """
        Build the metadata recorded alongside a pooled ensemble.

        Parameters
        ----------
        series_names : list of str
            Series being pooled, in order.
        pooled_name : str
            Name the pooled ensemble is stored under.

        Return
        ------
        metadata : dict
            Metadata for the pooled ensemble.
        """

        series_metadata = [
            self.get_didq_data(series_name)['metadata']
            for series_name in series_names
        ]

        return build_pooled_metadata(
            series_metadata=series_metadata,
            series_names=series_names,
            pooled_name=pooled_name,
        )

    def dofit(self, list_of_poles=(2, 3), fcutoff_hz=np.inf,
              max_nfev=5000, use_driven_bins=True,
              series_names=None,
              guess_params_2poles=None,
              guess_params_3poles=None,
              progress_callback=None):
        """
        Fit the loaded series with the requested pole models.

        The fit is performed on the ensemble mean, weighted by the ensemble
        standard deviation, exactly as dIdV does. Small signal parameters are
        never calculated.

        Parameters
        ----------
        list_of_poles : tuple or list of int, optional
            Pole models to fit. Default is (2, 3).
        fcutoff_hz : float, optional
            Upper frequency bound for the fit, in Hz. Default is infinity,
            matching dIdV.
        max_nfev : int, optional
            Maximum number of fit iterations. Default is 5000.
        use_driven_bins : bool, optional
            Restrict the fit to bins actually driven by the square wave. This
            is equivalent to fitting all bins, because the others carry zero
            weight, but roughly 100 times faster. Default is True.
        series_names : list of str, optional
            Series to fit. Default is all loaded series.
        guess_params_2poles : tuple, optional
            Starting guess for the two-pole fit, as
            (A, B, tau1, tau2, dt). No element may be zero.
        guess_params_3poles : tuple, optional
            Starting guess for the three-pole fit, as
            (A, B, C, tau1, tau2, tau3, dt). No element may be zero.

        Return
        ------
        None
        """

        if series_names is None:
            series_names = self.get_series_names()

        for series_name in series_names:

            data = self.get_didq_data(series_name)
            didvobj = data['didvobj']

            # snapshot the full frequency-domain arrays so the stored object
            # can be restored afterwards. Fitting the driven bins only would
            # otherwise leak a permanent truncation into the shared didvobj,
            # so a later call with use_driven_bins=False would silently keep
            # fitting the driven subset, and n_freq_bins_total would no
            # longer agree with didvobj._freq.size.
            original_freq = didvobj._freq
            original_didvmean = didvobj._didvmean
            original_didvstd = didvobj._didvstd

            if use_driven_bins:
                # replace only the frequency-domain arrays. _tmean and
                # _flatinds must keep their full length, because dofit uses
                # them to build its initial parameter guesses.
                mask = data['driven_mask']
                didvobj._freq = original_freq[mask]
                didvobj._didvmean = original_didvmean[mask]
                didvobj._didvstd = original_didvstd[mask]

            try:
                n_bins_fit = int(np.sum(np.abs(didvobj._freq) < fcutoff_hz))

                for poles in list_of_poles:

                    guess_params = None
                    if (poles == 2) and (guess_params_2poles is not None):
                        guess_params = guess_params_2poles
                    if (poles == 3) and (guess_params_3poles is not None):
                        guess_params = guess_params_3poles

                    if self._verbose:
                        print(f'INFO: fitting {poles}-pole model for series '
                              f'{series_name} using {n_bins_fit} frequency '
                              f'bins')

                    fit_success = True
                    try:
                        didvobj.dofit(
                            poles,
                            fcutoff=fcutoff_hz,
                            max_nfev=max_nfev,
                            guess_params=guess_params,
                        )
                        results = didvobj.fitresult(poles)
                    except Exception as error:
                        print(f'WARNING: {poles}-pole fit failed for series '
                              f'{series_name}: {error}')
                        data['fit_results'][poles] = None
                        # a failed fit is still a completed unit of work, so
                        # report it or the progress total is never reached
                        if progress_callback is not None:
                            progress_callback(
                                series_name, 'fit ' + str(poles) + '-pole'
                            )
                        continue

                    if not results:
                        data['fit_results'][poles] = None
                        if progress_callback is not None:
                            progress_callback(
                                series_name, 'fit ' + str(poles) + '-pole'
                            )
                        continue

                    # a fit that did not converge still returns numbers, so
                    # record whether they are trustworthy rather than
                    # discarding them
                    if not np.all(np.isfinite(results['params_array'])):
                        fit_success = False

                    results['fit_success'] = fit_success
                    results['n_freq_bins_fit'] = n_bins_fit
                    # qetpy has already divided cost by this same quantity,
                    # so the reported cost and ndof use one consistent
                    # convention
                    results['ndof'] = max(
                        n_bins_fit - len(results['params_array']), 1
                    )
                    results['fcutoff_hz'] = float(fcutoff_hz)
                    results.update(data['snr_diagnostic'])

                    data['fit_results'][poles] = results

                    if progress_callback is not None:
                        progress_callback(
                            series_name, 'fit ' + str(poles) + '-pole'
                        )

            finally:
                didvobj._freq = original_freq
                didvobj._didvmean = original_didvmean
                didvobj._didvstd = original_didvstd

    def get_fit_results(self, series_name, poles):
        """
        Return the fit results for one series and pole model.

        Parameters
        ----------
        series_name : str
            Series name.
        poles : int
            Pole model, 2 or 3.

        Return
        ------
        results : dict
            Fit results, or an empty dictionary when the fit was not run or
            did not produce output.
        """

        data = self.get_didq_data(series_name)
        results = data['fit_results'].get(poles, None)

        if results is None:
            return dict()

        return results

    def get_series_names_from_path(self, raw_path, series=None):
        """
        List the series available under a raw path without loading them.

        Thin wrapper over the private grouping helper, so callers such as
        parallel processing scripts do not need to reach into a private
        method to discover series names before dispatching work.

        Parameters
        ----------
        raw_path : str
            Raw data group directory, series directory, or a single HDF5
            file.
        series : str or list of str, optional
            Restrict to these series names.

        Return
        ------
        series_names : list of str
            Sorted series names found under raw_path.
        """

        series_dict, _, _ = self._get_series_file_dict(
            raw_path=raw_path, series=series
        )

        return sorted(series_dict.keys())

    def get_series_files_from_path(self, raw_path, series=None):
        """
        Map each available series to its sorted list of dump files.

        Discovering the series of a group means reading the header of every
        file in it, so a caller that dispatches one worker per series should
        scan once here and hand each worker its own file list, rather than
        letting every worker rescan the whole group.

        Parameters
        ----------
        raw_path : str
            Raw data group directory, series directory, or a single HDF5
            file.
        series : str or list of str, optional
            Restrict to these series names.

        Return
        ------
        series_files : dict
            Mapping from series name to sorted list of dump file paths, in
            series name order.
        """

        series_dict, _, _ = self._get_series_file_dict(
            raw_path=raw_path, series=series
        )

        return {name: series_dict[name] for name in sorted(series_dict)}

    def validate_drive_configuration(self, series_files, heater_channel=None,
                                     thermometer_channel=None,
                                     duty_cycle=0.5):
        """
        Check every series can be deconvolved, from file headers alone.

        Reads no traces, so a group can be rejected in seconds rather than
        after the pile-up cut has run. Every series is checked before raising,
        so a group with several misconfigured series reports them together
        rather than one run at a time.

        Parameters
        ----------
        series_files : dict
            Mapping from series name to its list of dump files, as returned by
            get_series_files_from_path.
        heater_channel : str, optional
            Override heater auto-detection.
        thermometer_channel : str, optional
            Override thermometer auto-detection.
        duty_cycle : float, optional
            Square wave duty cycle. Default is 0.5.

        Return
        ------
        None
        """

        h5reader = h5io.H5Reader()
        problems = list()

        for series_name, file_list in series_files.items():

            detector_config = h5reader.get_detector_config(
                file_name=file_list[0]
            )

            heater, _ = identify_didq_channels(
                detector_config=detector_config,
                heater_channel=heater_channel,
                thermometer_channel=thermometer_channel,
            )

            drive_params = resolve_drive_parameters(
                detector_config=detector_config,
                heater_channel=heater,
                duty_cycle=duty_cycle,
            )

            fs, nb_samples, _ = self._get_series_trace_shape(
                h5reader=h5reader, file_list=file_list,
                series_name=series_name,
            )

            try:
                validate_drive_period_coverage(
                    fs=fs,
                    nb_samples=nb_samples,
                    sgfreq_hz=drive_params['sgfreq_hz'],
                    series_name=series_name,
                )
            except ValueError as error:
                problems.append(str(error))

        if problems:
            raise ValueError('\n\n'.join(problems))

    def _get_series_file_dict(self, raw_path, series=None):
        """
        Group the raw files of a group directory by series.

        Parameters
        ----------
        raw_path : str
            Raw data group directory, or a single HDF5 file.
        series : str or list of str, optional
            Restrict to these series names.

        Return
        ------
        series_dict : dict
            Mapping from series name to sorted list of file paths.
        base_path : str
            Parent directory of the raw group.
        group_name : str
            Raw group name.
        """

        if isinstance(raw_path, str):
            raw_path = [raw_path]

        file_list = list()
        base_path = None
        group_name = None

        for a_path in raw_path:
            if os.path.isdir(a_path):
                if base_path is None:
                    base_path = str(Path(a_path).parent)
                    group_name = str(Path(a_path).name)
                file_list.extend(glob(a_path + '/*.hdf5'))
            elif os.path.isfile(a_path):
                if base_path is None:
                    base_path = str(Path(a_path).parents[1])
                    group_name = str(Path(Path(a_path).parent).name)
                file_list.append(a_path)
            else:
                raise ValueError(f'ERROR: "{a_path}" does not exist!')

        if not file_list:
            raise ValueError('ERROR: no raw input data found. Check arguments!')

        file_list.sort()

        h5reader = h5io.H5Reader()
        series_dict = dict()

        for file_name in file_list:
            metadata = h5reader.get_metadata(file_name)
            series_name = h5io.extract_series_name(
                int(metadata['series_num'])
            )

            if series is not None:
                requested = series
                if isinstance(requested, str):
                    requested = [requested]
                if series_name not in requested:
                    continue

            if series_name not in series_dict:
                series_dict[series_name] = list()
            series_dict[series_name].append(file_name)

        if not series_dict:
            raise ValueError(
                f'ERROR: no series matching {series} found in {raw_path}.'
            )

        for series_name in series_dict:
            series_dict[series_name].sort()

        return series_dict, base_path, group_name

    def process_raw_data(self, raw_path, series=None, nb_events=None,
                         heater_channel=None, thermometer_channel=None,
                         duty_cycle=0.5,
                         apply_autocuts=True, progress_callback=None):
        """
        Read a raw dIdQ group and prepare every series for fitting.

        Parameters
        ----------
        raw_path : str
            Raw data group directory, or a single HDF5 file.
        series : str or list of str, optional
            Restrict to these series names.
        nb_events : int, optional
            Cap the number of traces read per series. Default is all.
        heater_channel : str, optional
            Override heater auto-detection.
        thermometer_channel : str, optional
            Override thermometer auto-detection.
        duty_cycle : float, optional
            Square wave duty cycle. Default is 0.5.
        apply_autocuts : bool, optional
            Apply the same cuts dIdV applies. Default is True.
        progress_callback : callable, optional
            Called as progress_callback(series_name, stage) each time a stage
            of the per-series pipeline completes. The read emits one event per
            dump, named "read dump i/n"; the stages that follow are listed in
            PROCESS_RAW_DATA_STAGES. Stages are reported rather than a
            fraction because the pile-up cut is a single opaque call that
            dominates the runtime, so a percentage within it is not available.

        Return
        ------
        None
        """

        def report(series_name, stage):
            """
            Send one progress event, ignoring the absence of a callback.

            Parameters
            ----------
            series_name : str
                Series the stage belongs to.
            stage : str
                Stage that just completed.

            Return
            ------
            None
            """
            if progress_callback is not None:
                progress_callback(series_name, stage)

        series_dict, base_path, group_name = self._get_series_file_dict(
            raw_path=raw_path, series=series
        )

        h5reader = h5io.H5Reader()

        for series_name, file_list in series_dict.items():

            detector_config = h5reader.get_detector_config(
                file_name=file_list[0]
            )

            heater, thermometer = identify_didq_channels(
                detector_config=detector_config,
                heater_channel=heater_channel,
                thermometer_channel=thermometer_channel,
            )

            drive_params = resolve_drive_parameters(
                detector_config=detector_config,
                heater_channel=heater,
                duty_cycle=duty_cycle,
            )

            if self._verbose:
                print(f'INFO: reading series {series_name}, heater '
                      f'{heater}, thermometer {thermometer}')

            # the trace shape is recorded in the file header, so the drive
            # period can be checked before a single trace is read. A series
            # that cannot be deconvolved must fail here rather than after the
            # pile-up cut has spent minutes on it.
            fs, nb_samples, nb_events_per_dump = (
                self._get_series_trace_shape(
                    h5reader=h5reader, file_list=file_list,
                    series_name=series_name,
                )
            )

            validate_drive_period_coverage(
                fs=fs,
                nb_samples=nb_samples,
                sgfreq_hz=drive_params['sgfreq_hz'],
                series_name=series_name,
            )

            read_kwargs = {'detector_chans': [thermometer],
                           'output_format': 2,
                           'include_metadata': True,
                           'adctoamp': True}

            traces, info = self._read_series_traces(
                h5reader=h5reader,
                file_list=file_list,
                read_kwargs=read_kwargs,
                nb_events=nb_events,
                nb_samples=nb_samples,
                nb_events_per_dump=nb_events_per_dump,
                series_name=series_name,
                report=report,
            )
            nb_traces_read = int(traces.shape[0])

            # drop dead traces, then pile-up, exactly as dIdV does
            nonzero_cut = np.all(traces != 0, axis=1)
            traces = traces[nonzero_cut]
            report(series_name, 'zero cut')

            if apply_autocuts and traces.shape[0] > 1:
                autocut = qp.autocuts_didv(traces, fs=fs)
                traces = traces[autocut]
            report(series_name, 'pile-up cut')

            if traces.shape[0] == 0:
                print(f'WARNING: all traces cut for series {series_name}, '
                      f'skipping.')
                # a skipped series is still a completed unit of work, so
                # report it or the progress total is never reached
                report(series_name, 'averaging')
                continue

            metadata = self._build_series_metadata(
                info=info[0],
                detector_config=detector_config,
                heater=heater,
                thermometer=thermometer,
                base_path=base_path,
                group_name=group_name,
                file_list=file_list,
                nb_traces_read=nb_traces_read,
            )

            self.set_traces(
                traces=traces,
                fs=fs,
                drive_params=drive_params,
                series_name=series_name,
                metadata=metadata,
            )
            report(series_name, 'averaging')

    @staticmethod
    def _get_series_trace_shape(h5reader, file_list, series_name):
        """
        Read the sample rate and trace length of a series from file headers.

        Reads metadata only, so it costs a few milliseconds per dump and can
        run before any decision about whether the series is worth reading.

        Parameters
        ----------
        h5reader : H5Reader
            Open reader used for the metadata calls.
        file_list : list of str
            Dumps belonging to the series, in order.
        series_name : str
            Series name, used in error messages.

        Return
        ------
        fs : float
            Sample rate, in Hz.
        nb_samples : int
            Trace length, in samples.
        nb_events_per_dump : list of int
            Number of events recorded in each dump.
        """

        fs = None
        nb_samples_per_dump = list()
        nb_events_per_dump = list()

        for file_name in file_list:
            metadata = h5reader.get_metadata(file_name)
            adc_name = metadata['adc_list'][0]
            adc_metadata = metadata['groups'][adc_name]

            dump_fs = float(adc_metadata['sample_rate'])
            if (fs is not None) and (dump_fs != fs):
                raise ValueError(
                    f'ERROR: series {series_name} mixes sample rates '
                    f'({fs} Hz and {dump_fs} Hz across its dumps).'
                )
            fs = dump_fs

            nb_samples_per_dump.append(int(adc_metadata['nb_samples']))
            nb_events_per_dump.append(int(adc_metadata['nb_events']))

        if len(set(nb_samples_per_dump)) > 1:
            raise ValueError(
                f'ERROR: series {series_name} mixes trace lengths across its '
                f'dumps ({sorted(set(nb_samples_per_dump))} samples). The '
                f'ensemble cannot be averaged.'
            )

        return fs, nb_samples_per_dump[0], nb_events_per_dump

    @staticmethod
    def _read_series_traces(h5reader, file_list, read_kwargs, nb_events,
                            nb_samples, nb_events_per_dump, series_name,
                            report):
        """
        Read a series dump by dump, reporting progress after each one.

        Reading the whole series in one call leaves the progress bar idle for
        the entire read, which on a multi-dump series is minutes. The dump
        event counts come from the file headers, so the ensemble is filled
        into a single exactly sized array. Concatenating per-dump arrays
        instead would hold both the parts and the result at once, which on a
        full series is several gigabytes.

        Parameters
        ----------
        h5reader : H5Reader
            Open reader used for the read calls.
        file_list : list of str
            Dumps belonging to the series, in order.
        read_kwargs : dict
            Arguments passed to read_many_events, without filepath or nevents.
        nb_events : int or None
            Cap on the traces read for the whole series.
        nb_samples : int
            Trace length, in samples.
        nb_events_per_dump : list of int
            Number of events recorded in each dump.
        series_name : str
            Series being read.
        report : callable
            Called as report(series_name, stage) after each dump.

        Return
        ------
        traces : ndarray
            Array of shape (n_traces, n_samples), in amps.
        info : list of dict
            Event metadata from the first dump that returned traces.
        """

        nb_dumps = len(file_list)

        nb_to_read = int(sum(nb_events_per_dump))
        if nb_events is not None:
            nb_to_read = min(nb_to_read, int(nb_events))

        traces = None
        info = None
        nb_filled = 0

        for dump_index, file_name in enumerate(file_list):

            stage = f'read dump {dump_index + 1}/{nb_dumps}'

            # the cap applies to the whole series, so once it is reached the
            # remaining dumps are skipped. They are still reported, or the
            # progress bar never reaches its total.
            if nb_filled >= nb_to_read:
                report(series_name, stage)
                continue

            dump_kwargs = dict(read_kwargs)
            dump_kwargs['filepath'] = [file_name]
            dump_kwargs['nevents'] = nb_to_read - nb_filled

            dump_traces, dump_info = h5reader.read_many_events(**dump_kwargs)
            dump_traces = dump_traces[:, 0, :]

            if info is None:
                info = dump_info

            if traces is None:
                traces = np.zeros(
                    (nb_to_read, nb_samples), dtype=dump_traces.dtype
                )

            nb_new = min(dump_traces.shape[0], nb_to_read - nb_filled)
            traces[nb_filled:nb_filled + nb_new] = dump_traces[:nb_new]
            nb_filled = nb_filled + nb_new

            report(series_name, stage)

        if traces is None:
            raise ValueError(
                f'ERROR: no traces were read for series {series_name}.'
            )

        return traces[:nb_filled], info

    @staticmethod
    def _build_series_metadata(info, detector_config, heater, thermometer,
                               base_path, group_name, file_list,
                               nb_traces_read):
        """
        Collect the metadata recorded alongside the fit results.

        Parameters
        ----------
        info : dict
            Event metadata from the first event of the series.
        detector_config : dict
            Detector configuration keyed by channel name.
        heater : str
            Heater channel name.
        thermometer : str
            Thermometer channel name.
        base_path : str
            Parent directory of the raw group.
        group_name : str
            Raw group name.
        file_list : list of str
            Files belonging to this series.
        nb_traces_read : int
            Number of traces read before cuts.

        Return
        ------
        metadata : dict
            Flat dictionary of metadata values.
        """

        metadata = dict()
        metadata['series_number'] = int(info['series_num'])
        metadata['series_name'] = h5io.extract_series_name(
            int(info['series_num'])
        )
        metadata['group_name'] = str(group_name)
        metadata['base_path'] = str(base_path)
        metadata['nb_dumps'] = int(len(file_list))
        metadata['nb_traces_read'] = int(nb_traces_read)
        metadata['heater_channel'] = str(heater)
        metadata['thermometer_channel'] = str(thermometer)

        for key, name in (('series_start', 'series_start_time'),
                          ('group_start', 'group_start_time'),
                          ('fridge_run_start', 'fridge_run_start_time'),
                          ('fridge_run', 'fridge_run_number'),
                          ('data_purpose', 'data_type')):
            if key in info:
                metadata[name] = info[key]

        for channel, label in ((heater, 'heater'),
                               (thermometer, 'thermometer')):
            config = detector_config.get(channel, dict())
            if 'tes_bias' in config:
                metadata['tes_bias_' + label] = float(config['tes_bias'])

        thermo_config = detector_config.get(thermometer, dict())
        for key in ('close_loop_norm', 'output_gain', 'output_offset',
                    'temperature_cp', 'temperature_still', 'temperature_mc'):
            if key in thermo_config:
                metadata[key] = float(thermo_config[key])

        return metadata

    def save_didq_data(self, series_names=None, file_path_name=None,
                       save_hdf5=False):
        """
        Store the fit results in the FilterData structure and optionally save.

        Parameters
        ----------
        series_names : list of str, optional
            Series to save. Default is all loaded series.
        file_path_name : str, optional
            Output file or directory. Overrides the value given at
            construction.
        save_hdf5 : bool, optional
            Write the results to HDF5. Default is False.

        Return
        ------
        None
        """

        if series_names is None:
            series_names = self.get_series_names()

        save_data = False

        for series_name in series_names:

            data = self.get_didq_data(series_name)

            metadata = dict(data['metadata'])
            metadata.update(data['drive_params'])
            metadata['fs'] = data['fs']
            metadata['nb_samples'] = data['nb_samples']
            metadata['n_traces_used'] = data['n_traces_used']

            series_has_results = False
            for poles in data['fit_results'].keys():
                results = self.get_fit_results(series_name, poles)
                if results:
                    self.set_didq_results(
                        series_name, results, poles,
                        metadata=dict(metadata),
                    )
                    series_has_results = True

            # the mean trace, transfer function and its uncertainty are
            # per-series, not per-pole-model, and are saved once alongside
            # whichever pole models actually produced fit output, so a saved
            # series can always be re-plotted without re-reading raw data
            if series_has_results:
                didvobj = data['didvobj']
                self.set_didq_traces(
                    series_name,
                    tmean=didvobj._tmean,
                    didv_mean=didvobj._didvmean,
                    didv_std=didvobj._didvstd,
                    freq=didvobj._freq,
                    metadata=dict(metadata),
                )
                save_data = True

        if not save_data:
            return

        if save_hdf5:
            file_path = self._save_path
            if file_path is None:
                file_path = './'
            file_name = self._file_name

            if file_path_name is not None:
                if os.path.isdir(file_path_name):
                    file_path = file_path_name
                else:
                    file_path = os.path.dirname(file_path_name)
                    file_name = os.path.basename(file_path_name)

            full_file_name = os.path.join(file_path, file_name)
            self.save_hdf5(full_file_name, overwrite=True)


DIDQ_PARAM_NAMES = ('A', 'B', 'C', 'tau1', 'tau2', 'tau3', 'dt')

# metadata that has to agree across series before their traces may be pooled.
# The drive parameters are checked separately, since they are held apart from
# the metadata
POOLABLE_METADATA_KEYS = (
    'heater_channel',
    'thermometer_channel',
    'tes_bias_heater',
    'tes_bias_thermometer',
    'close_loop_norm',
    'output_gain',
)


def build_pooled_series_name(group_name):
    """
    Build the name a pooled ensemble is stored under.

    The name is derived from the raw group rather than from any one series,
    since no single series owns the pool. The acquisition prefix is dropped so
    the result reads as an identifier rather than as a directory name.

    Parameters
    ----------
    group_name : str
        Raw group name, for example exttrig_I2_D20260731_T121720.

    Return
    ------
    pooled_name : str
        Name of the form I2_D20260731_T121720_pooled.
    """

    stem = str(group_name)
    for prefix in ('exttrig_', 'cont_', 'rand_'):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break

    return stem + '_pooled'


def validate_series_are_poolable(series_names, series_data):
    """
    Check that several series may be averaged into one ensemble.

    Traces taken at a different bias point, with a different drive, or at a
    different sample rate are measurements of different things, so averaging
    them together produces a mean trace that describes nothing. A mismatch is
    an error rather than a warning.

    Kept at module level so the parallel path, which holds only what its
    workers returned, applies the same check as the in-memory path.

    Parameters
    ----------
    series_names : list of str
        Names of the series to check, in order.
    series_data : list of dict
        One dictionary per series, each holding the keys drive_params,
        metadata, fs and nb_samples.

    Return
    ------
    None
    """

    reference_name = series_names[0]
    reference = series_data[0]

    advice = ('They are not repeat measurements, so pooling them would '
              'average different measurements together. Process them '
              'separately with --per_series instead.')

    for series_name, data in zip(series_names[1:], series_data[1:]):

        for key in reference['drive_params']:
            if data['drive_params'][key] != reference['drive_params'][key]:
                raise ValueError(
                    f'ERROR: series {reference_name} and {series_name} '
                    f'disagree on drive parameter "{key}" '
                    f'({reference["drive_params"][key]} against '
                    f'{data["drive_params"][key]}). {advice}'
                )

        for key in POOLABLE_METADATA_KEYS:
            if key not in reference['metadata']:
                continue
            if data['metadata'].get(key) != reference['metadata'][key]:
                raise ValueError(
                    f'ERROR: series {reference_name} and {series_name} '
                    f'disagree on "{key}" '
                    f'({reference["metadata"][key]} against '
                    f'{data["metadata"].get(key)}). {advice}'
                )

        if data['fs'] != reference['fs']:
            raise ValueError(
                f'ERROR: series {reference_name} and {series_name} were '
                f'sampled at different rates ({reference["fs"]} Hz against '
                f'{data["fs"]} Hz) and cannot be pooled.'
            )

        if data['nb_samples'] != reference['nb_samples']:
            raise ValueError(
                f'ERROR: series {reference_name} and {series_name} hold '
                f'different trace lengths ({reference["nb_samples"]} against '
                f'{data["nb_samples"]} samples) and cannot be pooled.'
            )


def build_pooled_metadata(series_metadata, series_names, pooled_name):
    """
    Build the metadata recorded alongside a pooled ensemble.

    Fields describing the measurement are carried over from the first series,
    since pooling requires them to be identical across all of them. Fields
    that count traces or identify a single series are rebuilt to describe the
    pool.

    Kept at module level so the parallel path, which holds metadata returned
    by its workers rather than loaded series, builds the same record.

    Parameters
    ----------
    series_metadata : list of dict
        Metadata of each series being pooled, in order.
    series_names : list of str
        Names of those series, in the same order.
    pooled_name : str
        Name the pooled ensemble is stored under.

    Return
    ------
    metadata : dict
        Metadata for the pooled ensemble.
    """

    metadata = dict(series_metadata[0])

    nb_dumps = 0
    nb_traces_read = 0
    for entry in series_metadata:
        nb_dumps = nb_dumps + int(entry.get('nb_dumps', 0))
        nb_traces_read = nb_traces_read + int(entry.get('nb_traces_read', 0))

    metadata['series_name'] = pooled_name
    metadata['nb_dumps'] = nb_dumps
    metadata['nb_traces_read'] = nb_traces_read

    # a pool is not a series, so the start time of whichever series came first
    # would be misleading on its own. What went into the pool is recorded
    # alongside it instead
    metadata['pooled_series'] = ','.join(series_names)
    metadata['nb_series_pooled'] = len(series_names)
    metadata['is_pooled'] = True

    return metadata

# Stages reported by process_raw_data through its progress_callback, after the
# per-dump read events. Reading is reported once per dump rather than once per
# series, because a series can hold several dumps and take minutes to read.
#
# The pile-up cut is a single qetpy call and cannot be subdivided, so it stays
# one event. It also dominates the runtime: it is roughly O(trace length) in
# qetpy's per-frequency-bin inverse, which on 656250-sample dIdQ traces costs
# minutes per series against seconds to read. The bar is expected to sit on it.
PROCESS_RAW_DATA_STAGES = ('zero cut', 'pile-up cut', 'averaging')


def count_progress_stages(nb_dumps_per_series, list_of_poles):
    """
    Count the progress events a full run will emit.

    Reading is reported per dump, so a series contributes one event per dump
    plus the fixed per-series stages and one event per pole model fitted.

    Parameters
    ----------
    nb_dumps_per_series : list of int
        Number of dumps in each series to be processed.
    list_of_poles : tuple or list of int
        Pole models that will be fitted.

    Return
    ------
    total : int
        Number of progress events, for use as a progress bar total.
    """

    per_series = len(PROCESS_RAW_DATA_STAGES) + len(tuple(list_of_poles))

    total = 0
    for nb_dumps in nb_dumps_per_series:
        total = total + int(nb_dumps) + per_series

    return int(total)


def build_didq_row(analysis, series_name, processing_id,
                   list_of_poles=(2, 3), save_covariance=False):
    """
    Flatten one series of dIdQ results into a single dataframe row.

    Fall times are the reproducible output of the fit and are listed first.
    The raw parameters are included for completeness, but for the three-pole
    model they are start-point dependent and must not be compared across
    series. See the design document for measurements of this degeneracy.

    Parameters
    ----------
    analysis : DIDQAnalysis
        Analysis object holding the fitted series.
    series_name : str
        Series to flatten.
    processing_id : str
        Processing identifier recorded on the row.
    list_of_poles : tuple of int, optional
        Pole models to include. Default is (2, 3).
    save_covariance : bool, optional
        Include the flattened covariance matrix. Default is False.

    Return
    ------
    row : dict
        Mapping from column name to a scalar value.
    """

    data = analysis.get_didq_data(series_name)

    row = dict()

    for key, value in data['metadata'].items():
        row[key] = value

    row['series_name'] = str(series_name)
    row['processing_id'] = str(processing_id)

    for key, value in data['drive_params'].items():
        row[key] = float(value)

    row['fs_hz'] = float(data['fs'])
    row['nb_samples'] = int(data['nb_samples'])
    row['nb_traces_used'] = int(data['n_traces_used'])
    row['n_freq_bins_total'] = int(data['n_freq_bins_total'])
    row['f_max_snr3_hz'] = float(data['snr_diagnostic']['f_max_snr3_hz'])
    row['n_freq_bins_snr3'] = int(data['snr_diagnostic']['n_freq_bins_snr3'])

    for poles in list_of_poles:

        prefix = 'didq_' + str(poles) + 'pole_'
        results = analysis.get_fit_results(series_name, poles)

        if not results:
            for name in DIDQ_PARAM_NAMES:
                row[prefix + name] = float('nan')
                row[prefix + name + '_err'] = float('nan')
            for index in (1, 2, 3):
                row[prefix + 'falltime_' + str(index)] = float('nan')
            row[prefix + 'cost'] = float('nan')
            row[prefix + 'ndof'] = 0
            row[prefix + 'fit_success'] = False
            row[prefix + 'n_freq_bins_fit'] = 0
            row[prefix + 'fcutoff_hz'] = float('nan')
            row[prefix + 'offset'] = float('nan')
            row[prefix + 'offset_err'] = float('nan')
            if save_covariance:
                for i in range(len(DIDQ_PARAM_NAMES)):
                    for j in range(i, len(DIDQ_PARAM_NAMES)):
                        name = prefix + 'cov_' + str(i) + '_' + str(j)
                        row[name] = float('nan')
            continue

        params = results.get('params', dict())
        errors = results.get('errors', dict())

        for name in DIDQ_PARAM_NAMES:
            row[prefix + name] = float(params.get(name, float('nan')))
            row[prefix + name + '_err'] = float(
                errors.get(name, float('nan'))
            )

        # largest-magnitude fall time first, so the physically meaningful
        # pole leads. The sign is preserved: a negative fall time is the
        # loop-gain-greater-than-one branch and must not be mistaken for the
        # near-zero artifact that sorting by signed value would put first.
        falltimes = np.asarray(results.get('falltimes', []))
        falltimes = falltimes[np.argsort(np.abs(falltimes))[::-1]]
        for index in (1, 2, 3):
            if index <= falltimes.size:
                row[prefix + 'falltime_' + str(index)] = float(
                    falltimes[index - 1]
                )
            else:
                row[prefix + 'falltime_' + str(index)] = float('nan')

        row[prefix + 'cost'] = float(results.get('cost', float('nan')))
        row[prefix + 'ndof'] = int(results.get('ndof', 0))
        row[prefix + 'fit_success'] = bool(results.get('fit_success', False))
        row[prefix + 'n_freq_bins_fit'] = int(
            results.get('n_freq_bins_fit', 0)
        )
        row[prefix + 'fcutoff_hz'] = float(
            results.get('fcutoff_hz', float('nan'))
        )
        row[prefix + 'offset'] = float(results.get('offset', float('nan')))
        row[prefix + 'offset_err'] = float(
            results.get('offset_err', float('nan'))
        )

        if save_covariance:
            cov = np.asarray(results.get('cov', np.zeros((0, 0))))
            for i in range(len(DIDQ_PARAM_NAMES)):
                for j in range(i, len(DIDQ_PARAM_NAMES)):
                    name = prefix + 'cov_' + str(i) + '_' + str(j)
                    if (i < cov.shape[0]) and (j < cov.shape[1]):
                        row[name] = float(cov[i, j])
                    else:
                        row[name] = float('nan')

    return row


def rows_to_dataframe(rows):
    """
    Assemble per-series row dictionaries into a single vaex dataframe.

    build_didq_row returns the same key set for every row it produces, but
    the column union is still built defensively here, and any row missing a
    key is filled with NaN for that column. If a column mixes real string
    values in some rows with the NaN fallback in others, the column is built
    with an explicit object dtype. Without that, numpy infers a single dtype
    across the whole column and silently converts the float NaN into the
    literal string "nan", which is indistinguishable from real string data
    and would corrupt the column.

    Parameters
    ----------
    rows : list of dict
        One flat dictionary of scalars per series, as returned by
        build_didq_row.

    Return
    ------
    dataframe : vaex.dataframe.DataFrameLocal
        One row per input dict, one column per key seen across all rows.
    """

    # every row must share the same columns for vaex to build the dataframe
    all_columns = list()
    for row in rows:
        for column in row.keys():
            if column not in all_columns:
                all_columns.append(column)

    data_dict = dict()
    for column in all_columns:
        values = list()
        for row in rows:
            values.append(row.get(column, float('nan')))

        has_string_value = any(isinstance(value, str) for value in values)
        has_non_string_value = any(
            not isinstance(value, str) for value in values
        )

        if has_string_value and has_non_string_value:
            data_dict[column] = np.array(values, dtype=object)
        else:
            data_dict[column] = np.array(values)

    dataframe = vx.from_dict(data_dict)

    return dataframe
