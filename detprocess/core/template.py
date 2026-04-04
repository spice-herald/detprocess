import os
import copy
import pandas as pd
import numpy as np
from pprint import pprint
import qetpy as qp
from glob import glob
import vaex as vx
from pathlib import Path
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from detprocess.core.filterdata import FilterData
import pytesio as h5io
from qetpy.utils import (
    make_template_twopole,
    make_template_threepole,
    make_template_fourpole,
    lowpassfilter,
)
from qetpy.core.didv import stdcomplex
from scipy.fftpack import fft, ifft, fftfreq

class Template(FilterData):
    """
    Class to manage template calculations for multiple channels.

    This class now supports storing mean pulses in both current/power
    and time/frequency domains, as well as fitted analytic templates.
    All internal containers are keyed by channel name.
    """
    
    def __init__(self, verbose=True, filter_data=None):
        super().__init__(verbose=verbose, filter_data=filter_data)
        
        self._traces_i_t = dict()
        self._mean_i_t = dict()
        self._mean_i_f = dict()
        self._psd_i = dict()
        self._std_i_f = dict()

        self._mean_p_t = dict()
        self._mean_p_f = dict()
        self._psd_p = dict()
        self._std_p_f = dict()

        self._dpdi = dict()
        self._dpdi_err = dict()
        self._dpdi_freqs = dict()
        self._dpdi_metadata = dict()

        self._fit_models = dict()
        self._fit_vars = dict()
        self._fit_cov = dict()
        self._template_fit_p_t = dict()
        self._template_fit_p_f = dict()
        self._template_fit_i_t = dict()
        self._template_fit_i_f = dict()

        self._sample_rate = dict()
        self._time_axis = dict()
        self._freqs = dict()
        self._pretrigger_samples = dict()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_channels(self, channels):
        if isinstance(channels, str):
            return [channels]
        if isinstance(channels, (list, tuple, np.ndarray)):
            out = list(channels)
            if not out:
                raise ValueError('ERROR: "channels" cannot be empty!')
            return out
        raise ValueError('ERROR: "channels" should be a string or a list of strings!')

    def _check_channels_in_filter_data(self, channels):
        missing = [chan for chan in channels if chan not in self._filter_data]
        if missing:
            raise ValueError(
                'ERROR: channel(s) not available in internal filter data: '
                f'{missing}. Available channels: {list(self._filter_data.keys())}'
            )

    def _check_channels_in_container(self, channels, container, container_name):
        missing = [chan for chan in channels if chan not in container]
        if missing:
            raise ValueError(
                f'ERROR: channel(s) {missing} not available in {container_name}. '
                'Run the required calculation first.'
            )

    def _get_single_channel_value(self, value, channels, ichan):
        if isinstance(value, dict):
            if channels[ichan] not in value:
                raise ValueError(
                    f'ERROR: channel {channels[ichan]} missing from provided dictionary.'
                )
            return value[channels[ichan]]
        return value

    def _prepare_complex_array(self, arr, channels, array_name):
        if isinstance(arr, dict):
            return {
                chan: np.asarray(arr[chan], dtype=np.complex128)
                for chan in channels
            }

        arr = np.asarray(arr)
        if arr.ndim == 1:
            return {chan: np.asarray(arr, dtype=np.complex128).copy() for chan in channels}

        if arr.ndim == 2:
            if arr.shape[0] != len(channels):
                raise ValueError(
                    f'ERROR: {array_name} first dimension ({arr.shape[0]}) '
                    f'is inconsistent with number of channels ({len(channels)}).'
                )
            return {
                chan: np.asarray(arr[ichan], dtype=np.complex128).copy()
                for ichan, chan in enumerate(channels)
            }

        raise ValueError(
            f'ERROR: {array_name} should be 1D, 2D, or a dictionary keyed by channel.'
        )

    def _prepare_optional_complex_array(self, arr, channels, default_len, array_name):
        if arr is None:
            return {chan: np.zeros(default_len[chan], dtype=np.complex128) for chan in channels}
        return self._prepare_complex_array(arr, channels, array_name)

    def _get_trace_axis_info(self, metadata, trace_length_samples,
                             pretrigger_length_msec, pretrigger_length_samples):
        fs = float(metadata[0]['sample_rate'])
        nsamples = int(trace_length_samples) if trace_length_samples is not None else None
        if nsamples is None:
            nsamples = int(metadata[0].get('nb_samples', 0))
            if nsamples <= 0:
                raise ValueError('ERROR: unable to determine number of samples from metadata.')

        trig = nsamples // 2
        if pretrigger_length_msec is not None:
            trig = h5io.convert_length_msec_to_samples(pretrigger_length_msec, fs) - 1
        elif pretrigger_length_samples is not None:
            trig = int(pretrigger_length_samples) - 1

        t = np.arange(nsamples, dtype=np.float64) / fs
        freqs = fftfreq(nsamples, 1.0 / fs)
        return fs, nsamples, trig, t, freqs

    def _complex_std_error(self, values):
        values = np.asarray(values, dtype=np.complex128)
        if values.ndim != 1:
            raise ValueError('ERROR: _complex_std_error expects a 1D array.')
        return stdcomplex(values) / np.sqrt(len(values))

    def _get_twopole_t_template(self, amp1, fall_1, rise,
                                t_arr, start_time, fs):
        pulse = make_template_twopole(
            t_arr, A=amp1, tau_r=rise, tau_f=fall_1,
            t0=start_time, fs=fs, normalize=False
        )
        if np.isnan(pulse).any() or np.isinf(pulse).any():
            pulse = np.zeros(len(t_arr), dtype=np.float64)
        return pulse

    def _get_threepole_t_template(self, amp1, amp2, fall_1, fall_2, rise,
                                  t_arr, start_time, fs):
        pulse = make_template_threepole(
            t_arr, A=amp1, B=amp2, tau_r=rise,
            tau_f1=fall_1, tau_f2=fall_2,
            t0=start_time, fs=fs, normalize=False
        )
        if np.isnan(pulse).any() or np.isinf(pulse).any():
            pulse = np.zeros(len(t_arr), dtype=np.float64)
        return pulse

    def _get_fourpole_t_template(self, amp1, amp2, amp3, fall_1, fall_2, fall_3,
                                 rise, t_arr, start_time, fs):
        pulse = make_template_fourpole(
            t_arr, A=amp1, B=amp2, C=amp3, tau_r=rise,
            tau_f1=fall_1, tau_f2=fall_2, tau_f3=fall_3,
            t0=start_time, fs=fs, normalize=False
        )
        if np.isnan(pulse).any() or np.isinf(pulse).any():
            pulse = np.zeros(len(t_arr), dtype=np.float64)
        return pulse

    def _get_modeled_template_t(self, template_model, params, t_arr, start_time, fs):
        if template_model == 'twopole':
            amp_1, fall_1, rise = params
            return self._get_twopole_t_template(amp_1, fall_1, rise, t_arr, start_time, fs)
        if template_model == 'threepole':
            amp_1, amp_2, fall_1, fall_2, rise = params
            return self._get_threepole_t_template(amp_1, amp_2, fall_1, fall_2, rise,
                                                  t_arr, start_time, fs)
        if template_model == 'fourpole':
            amp_1, amp_2, amp_3, fall_1, fall_2, fall_3, rise = params
            return self._get_fourpole_t_template(amp_1, amp_2, amp_3, fall_1, fall_2,
                                                 fall_3, rise, t_arr, start_time, fs)
        raise ValueError(
            'ERROR: unsupported template_model. Supported models are '
            '"twopole", "threepole", and "fourpole".'
        )

    def _get_modeled_template_f(self, template_model, params, t_arr, start_time, fs):
        template_t = self._get_modeled_template_t(template_model, params, t_arr, start_time, fs)
        return fft(template_t) / np.sqrt(len(template_t) * fs)

    def _print_fit_summary(self, channel):
        popt = self._fit_vars[channel]
        pcov = self._fit_cov[channel]
        pstds = np.sqrt(np.diag(pcov))
        model = self._fit_models[channel]

        print(f'Channel: {channel}')
        print(f'Model: {model}')
        print('popt:')
        print(popt)
        print('')
        print('cov:')
        print(pcov)
        print('')

        if model == 'twopole':
            amp_1, fall_1, rise = popt
            amp_1_err, fall_1_err, rise_err = pstds
            print(f'Amplitude 1: {amp_1} +/- {amp_1_err}')
            print(f'Fall Time 1: {fall_1*1e6} +/- {fall_1_err*1e6} us')
            print(f'Rise Time: {rise*1e6} +/- {rise_err*1e6} us')
        elif model == 'threepole':
            amp_1, amp_2, fall_1, fall_2, rise = popt
            amp_1_err, amp_2_err, fall_1_err, fall_2_err, rise_err = pstds
            print(f'Amplitude 1: {amp_1} +/- {amp_1_err}')
            print(f'Amplitude 2: {amp_2} +/- {amp_2_err}')
            print(f'Fall Time 1: {fall_1*1e6} +/- {fall_1_err*1e6} us')
            print(f'Fall Time 2: {fall_2*1e6} +/- {fall_2_err*1e6} us')
            print(f'Rise Time: {rise*1e6} +/- {rise_err*1e6} us')
        elif model == 'fourpole':
            amp_1, amp_2, amp_3, fall_1, fall_2, fall_3, rise = popt
            amp_1_err, amp_2_err, amp_3_err, fall_1_err, fall_2_err, fall_3_err, rise_err = pstds
            print(f'Amplitude 1: {amp_1} +/- {amp_1_err}')
            print(f'Amplitude 2: {amp_2} +/- {amp_2_err}')
            print(f'Amplitude 3: {amp_3} +/- {amp_3_err}')
            print(f'Fall Time 1: {fall_1*1e6} +/- {fall_1_err*1e6} us')
            print(f'Fall Time 2: {fall_2*1e6} +/- {fall_2_err*1e6} us')
            print(f'Fall Time 3: {fall_3*1e6} +/- {fall_3_err*1e6} us')
            print(f'Rise Time: {rise*1e6} +/- {rise_err*1e6} us')

    def _get_temp_i_t_from_temp_p_f(self, temp_p_f, dpdi, fs):
        template_i_f = temp_p_f / dpdi
        template_i_t = -1.0 * ifft(template_i_f) * np.sqrt(len(temp_p_f) * fs)
        return np.real(template_i_t)

    def _get_temp_i_f_from_temp_p_f(self, temp_p_f, dpdi):
        return temp_p_f / dpdi

    # ------------------------------------------------------------------
    # New functionality copied/adapted from photon_calibration.py
    # ------------------------------------------------------------------
    def calc_average_pulses(self, channels, file_path, event_list,
                            trace_length_msec=None,
                            pretrigger_length_msec=None,
                            trace_length_samples=None,
                            pretrigger_length_samples=None,
                            nevents=2000,
                            lgc_plot=False,
                            lgc_filter_freq=True,
                            filter_freq=50e3,
                            time_lims=None):
        """
        Calculate and store average current-domain pulses in time domain.

        The internal containers are keyed by channel name. The raw traces are also
        stored for later frequency-domain and fit calculations.
        """
        channels = self._normalize_channels(channels)

        h5reader = h5io.H5Reader()
        traces, metadata = h5reader.read_many_events(
            filepath=file_path,
            nevents=nevents,
            output_format=2,
            detector_chans=channels,
            event_list=event_list,
            trace_length_msec=trace_length_msec,
            trace_length_samples=trace_length_samples,
            pretrigger_length_msec=pretrigger_length_msec,
            pretrigger_length_samples=pretrigger_length_samples,
            include_metadata=True,
            adctoamp=True,
            baselinesub=False,
        )

        traces = np.asarray(traces)
        if traces.ndim != 3:
            raise ValueError(
                'ERROR: expected traces array with shape [nevents, nchans, nsamples]. '
                f'Got shape {traces.shape}.'
            )
        if traces.shape[1] != len(channels):
            raise ValueError(
                'ERROR: traces shape is inconsistent with the requested channels. '
                f'Expected {len(channels)} channels, got {traces.shape[1]}.'
            )

        fs, nsamples, trigger_index, t, freqs = self._get_trace_axis_info(
            metadata,
            trace_length_samples=traces.shape[-1],
            pretrigger_length_msec=pretrigger_length_msec,
            pretrigger_length_samples=pretrigger_length_samples,
        )

        mean_traces = np.mean(traces, axis=0)
        baseline_stop = max(trigger_index - 100, 1)
        baseline = mean_traces[:, :baseline_stop].mean(axis=1, keepdims=True)
        mean_traces = mean_traces - baseline

        for ichan, chan in enumerate(channels):
            self._traces_i_t[chan] = np.asarray(traces[:, ichan, :], dtype=np.float64).copy()
            self._mean_i_t[chan] = np.asarray(mean_traces[ichan, :], dtype=np.float64).copy()
            self._sample_rate[chan] = fs
            self._time_axis[chan] = t.copy()
            self._freqs[chan] = freqs.copy()
            self._pretrigger_samples[chan] = trigger_index

        if lgc_plot:
            if time_lims is None:
                time_lims = [t[max(trigger_index - int(0.1 * fs * 1e-3), 0)],
                             t[min(trigger_index + int(5e-4 * fs), len(t)-1)]]

            for ichan, chan in enumerate(channels):
                mean_i_t = self._mean_i_t[chan]
                plt.plot(t * 1e3, mean_i_t,
                         label=f'Mean Trace {chan}', alpha=0.5, color=f'C{ichan % 10}')
                if lgc_filter_freq:
                    plt.plot(
                        t * 1e3,
                        lowpassfilter(mean_i_t, cut_off_freq=filter_freq, order=2, fs=fs),
                        label=f'Filtered Mean Trace {chan}, Fcut={filter_freq*1e-3:.1f} kHz',
                        color=f'C{ichan % 10}'
                    )
            plt.xlabel('Time (ms)')
            plt.ylabel('Average Pulse Height (A)')
            plt.legend()
            plt.xlim(time_lims[0] * 1e3, time_lims[1] * 1e3)
            plt.title('Average Current-Domain Pulses')
            plt.show()

            for ichan, chan in enumerate(channels):
                mean_i_t = self._mean_i_t[chan]
                norm = np.max(np.abs(mean_i_t[trigger_index-50:trigger_index+200]))
                print(f'channel {chan}, max = {norm}')
                if norm == 0:
                    norm = 1.0
                curve = mean_i_t / norm
                plt.plot(t * 1e3, curve,
                         label=f'Normalized Mean Trace {chan}',
                         alpha=0.5, color=f'C{ichan % 10}')
                if lgc_filter_freq:
                    plt.plot(
                        t * 1e3,
                        lowpassfilter(curve, cut_off_freq=filter_freq, order=2, fs=fs),
                        label=f'Filtered Normalized Trace {chan}, Fcut={filter_freq*1e-3:.1f} kHz',
                        color=f'C{ichan % 10}'
                    )
            plt.xlabel('Time (ms)')
            plt.ylabel('Normalized Pulse Height')
            plt.legend()
            plt.title('Normalized Average Current-Domain Pulses')
            plt.xlim(time_lims[0] * 1e3, time_lims[1] * 1e3)
            plt.show()

    def calc_frequency_template_power(self, channels, dpdi=None, dpdi_err=None,
                                      poles=None, dpdi_tag='default',
                                      lgc_plot=False, filter_freq=50e3,
                                      time_lims=None):
        """
        Calculate current/power mean pulses in frequency and time domains.

        Parameters
        ----------
        channels : str or list
            Channel(s) to process.
        dpdi : ndarray, dict, or None
            dPdI array(s). If None, attempt to use previously stored values,
            then fall back to FilterData.get_dpdi if poles is provided.
        dpdi_err : ndarray, dict, or None
            Optional dPdI uncertainties. If None, zero uncertainties are used
            unless already stored.
        poles : int, optional
            Number of dPdI poles, used only if loading from FilterData.
        """
        channels = self._normalize_channels(channels)
        self._check_channels_in_container(channels, self._traces_i_t, '_traces_i_t')

        default_len = {chan: self._traces_i_t[chan].shape[-1] for chan in channels}

        if dpdi is None:
            dpdi_dict = {}
            dpdi_err_dict = {}
            for chan in channels:
                if chan in self._dpdi:
                    dpdi_dict[chan] = self._dpdi[chan]
                else:
                    if poles is None:
                        raise ValueError(
                            f'ERROR: no stored dpdi for channel {chan}. '
                            'Provide dpdi explicitly or set "poles" so it can be loaded.'
                        )
                    dpdi_vals, dpdi_err, dpdi_freqs = self.get_dpdi(chan, poles=poles,
                                                                    return_dpdi_err=True,
                                                                    tag=dpdi_tag)
                    
                    dpdi_dict[chan] = np.asarray(dpdi_vals, dtype=np.complex128)
                    dpdi_err_dict[chan] = np.asarray(dpdi_err, dtype=np.complex128)
                    self._dpdi[chan] = dpdi_dict[chan].copy()
                    self._dpdi_err[chan] = dpdi_err_dict[chan].copy()
                    self._dpdi_freqs[chan] = np.asarray(dpdi_freqs).copy()
                    self._dpdi_metadata[chan] = {'poles': poles, 'tag': dpdi_tag}
                    print('Got from filter_data')
        else:
            raise ValueError(
                f'ERROR: no stored dpdi for channel {chan}. '
                'Provide dpdi explicitly or set "poles" so it can be loaded.'
            )
          
        if dpdi_err is None:
            print('dpdi error is None')
            dpdi_err_dict = {}
            for chan in channels:
                if chan in self._dpdi_err:
                    dpdi_err_dict[chan] = self._dpdi_err[chan]
                else:
                    dpdi_err_dict[chan] = np.zeros(default_len[chan], dtype=np.complex128)
        else:
            dpdi_err_dict = self._prepare_complex_array(dpdi_err, channels, 'dpdi_err')

        for chan in channels:
            traces_i_t = np.asarray(self._traces_i_t[chan], dtype=np.float64)
            fs = self._sample_rate[chan]
            t = self._time_axis[chan]
            freqs = self._freqs[chan]
            pretrigger_samples = self._pretrigger_samples[chan]
            dpdi_chan = np.asarray(dpdi_dict[chan], dtype=np.complex128)
            dpdi_err_chan = np.asarray(dpdi_err_dict[chan], dtype=np.complex128)

            if dpdi_chan.ndim != 1:
                raise ValueError(f'ERROR: dpdi for channel {chan} should be 1D.')
            if len(dpdi_chan) != traces_i_t.shape[-1]:
                raise ValueError(
                    f'ERROR: dpdi length for channel {chan} ({len(dpdi_chan)}) does not match '
                    f'trace length ({traces_i_t.shape[-1]}).'
                )
            if len(dpdi_err_chan) != len(dpdi_chan):
                raise ValueError(
                    f'ERROR: dpdi_err length for channel {chan} ({len(dpdi_err_chan)}) '
                    f'does not match dpdi length ({len(dpdi_chan)}).'
                )

            traces_i_f = fft(traces_i_t, axis=-1) / np.sqrt(traces_i_t.shape[-1] * fs)

            mean_i_f = np.mean(traces_i_f.real, axis=0) + 1.0j * np.mean(traces_i_f.imag, axis=0)
            std_i_f = np.asarray([
                self._complex_std_error(traces_i_f[:, ibin])
                for ibin in range(traces_i_f.shape[-1])
            ], dtype=np.complex128)
            psd_i = np.sqrt(np.mean(np.abs(fft(traces_i_t, axis=-1))**2.0, axis=0)) / np.sqrt(traces_i_t.shape[-1] * fs)

            mean_p_f = mean_i_f * dpdi_chan
            std_p_f_real = np.sqrt((mean_i_f.real * dpdi_err_chan.real) ** 2 + (std_i_f.real * np.abs(dpdi_chan)) ** 2)
            std_p_f_imag = np.sqrt((mean_i_f.imag * dpdi_err_chan.imag) ** 2 + (std_i_f.imag * np.abs(dpdi_chan)) ** 2)
            std_p_f = std_p_f_real + 1.0j * std_p_f_imag
            mean_p_t = ifft(mean_p_f) * np.sqrt(len(t) * fs)
            baseline_stop = max(int(0.5 * pretrigger_samples), 1)
            mean_p_t = mean_p_t - np.mean(mean_p_t[:baseline_stop])
            psd_p = dpdi_chan * np.abs(psd_i)

            self._mean_i_f[chan] = mean_i_f
            self._psd_i[chan] = psd_i
            self._std_i_f[chan] = std_i_f
            self._mean_p_t[chan] = np.real(-1.0 * mean_p_t)
            self._mean_p_f[chan] = -1.0 * mean_p_f
            self._psd_p[chan] = psd_p
            self._std_p_f[chan] = std_p_f
            self._dpdi[chan] = dpdi_chan.copy()
            self._dpdi_err[chan] = dpdi_err_chan.copy()
            if chan not in self._dpdi_freqs:
                self._dpdi_freqs[chan] = freqs.copy()
            if chan not in self._dpdi_metadata:
                self._dpdi_metadata[chan] = {'tag': dpdi_tag, 'poles': poles}

        if lgc_plot:
            chan0 = channels[0]
            freqs = self._freqs[chan0]
            t = self._time_axis[chan0]
            if time_lims is None:
                trig = self._pretrigger_samples[chan0]
                time_lims = [t[max(trig - int(0.1 * self._sample_rate[chan0] * 1e-3), 0)],
                             t[min(trig + int(5e-4 * self._sample_rate[chan0]), len(t)-1)]]

            for ichan, chan in enumerate(channels):
                plt.plot(freqs[:len(freqs)//2],
                         np.abs(self._mean_i_f[chan])[:len(freqs)//2],
                         alpha=0.5, color=f'C{ichan % 10}', label=chan)
                plt.fill_between(
                    freqs[:len(freqs)//2],
                    np.abs(np.abs(self._mean_i_f[chan])[:len(freqs)//2] - np.abs(self._std_i_f[chan])[:len(freqs)//2]),
                    np.abs(self._mean_i_f[chan])[:len(freqs)//2] + np.abs(self._std_i_f[chan])[:len(freqs)//2],
                    color=f'C{ichan % 10}', alpha=0.1
                )
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Mean Trace Current PSD (A/rt(Hz)), unfolded')
            plt.legend()
            plt.grid()
            plt.title('Current-Domain Calibration Pulse PSDs')
            plt.show()

            for ichan, chan in enumerate(channels):
                plt.plot(freqs[:len(freqs)//2],
                         np.abs(self._mean_p_f[chan])[:len(freqs)//2],
                         alpha=0.5, color=f'C{ichan % 10}', label=chan)
                plt.fill_between(
                    freqs[:len(freqs)//2],
                    np.abs(np.abs(self._mean_p_f[chan])[:len(freqs)//2] - np.abs(self._std_p_f[chan])[:len(freqs)//2]),
                    np.abs(self._mean_p_f[chan])[:len(freqs)//2] + np.abs(self._std_p_f[chan])[:len(freqs)//2],
                    color=f'C{ichan % 10}', alpha=0.1
                )
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Mean Trace Power PSD (W/rt(Hz)), unfolded')
            plt.legend()
            plt.grid()
            plt.title('Power-Domain Calibration Pulse PSDs')
            plt.show()

            for ichan, chan in enumerate(channels):
                lp_pulse = -1.0 * lowpassfilter(self._mean_p_t[chan], order=2, fs=self._sample_rate[chan], cut_off_freq=filter_freq)
                plt.plot(self._time_axis[chan] * 1e3, lp_pulse,
                         alpha=0.7, label=chan, color=f'C{ichan % 10}')
            plt.xlim(time_lims[0] * 1e3, time_lims[1] * 1e3)
            plt.xlabel('Time (ms)')
            plt.ylabel('Power (W)')
            plt.title(f'Power-Domain Templates, filtered with {filter_freq*1e-3:.1f} kHz')
            plt.legend()
            plt.show()

    def fit_templates(self, channels, template_model='twopole',
                      f_fit_cutoff=50e3, guess=None, bounds=None,
                      max_nfev=600, dt=0.0, lgc_diagnostics=True,
                      lgc_plot=True, filter_freq=50e3,
                      time_lims=None):
        """
        Fit analytic pulse models to the power-domain mean template.

        Parameters
        ----------
        channels : str or list
            Channel(s) to fit.
        template_model : str or dict
            One of "twopole", "threepole", or "fourpole". A channel-keyed
            dictionary is also accepted.
        guess : array-like or dict, optional
            Initial guess. May be a single array used for all channels or a
            dictionary keyed by channel.
        bounds : tuple/list or dict, optional
            Bounds passed to scipy.optimize.least_squares. May be channel keyed.
        dt : float
            deviation of the template from pretrigger time, in seconds

        """
        channels = self._normalize_channels(channels)
        self._check_channels_in_container(channels, self._mean_p_f, '_mean_p_f')
        self._check_channels_in_container(channels, self._std_p_f, '_std_p_f')

        for chan in channels:
            chan_model = self._get_single_channel_value(template_model, channels, channels.index(chan))
            if chan_model not in ('twopole', 'threepole', 'fourpole'):
                raise ValueError(
                    f'ERROR: unsupported template_model for channel {chan}: {chan_model}'
                )

            mean_p_t = self._mean_p_t[chan]
            mean_p_f = self._mean_p_f[chan]
            std_p_f = self._std_p_f[chan]
            fs = self._sample_rate[chan]
            t_arr = self._time_axis[chan]
            freqs = self._freqs[chan]
            start_time = self._pretrigger_samples[chan] / fs + dt

            chan_guess = self._get_single_channel_value(guess, channels, channels.index(chan)) if guess is not None else None
            chan_bounds = self._get_single_channel_value(bounds, channels, channels.index(chan)) if bounds is not None else None

            if chan_guess is None:
                peak_amp = float(np.max(np.abs(mean_p_t))) if np.max(np.abs(mean_p_t)) != 0 else 1.0
                if chan_model == 'twopole':
                    chan_guess = np.array([peak_amp, 100e-6, 20e-6], dtype=np.float64)
                elif chan_model == 'threepole':
                    chan_guess = np.array([0.7*peak_amp, 0.3*peak_amp, 100e-6, 300e-6, 20e-6], dtype=np.float64)
                else:
                    chan_guess = np.array([0.6*peak_amp, 0.3*peak_amp, 0.1*peak_amp,
                                           80e-6, 200e-6, 500e-6, 20e-6], dtype=np.float64)
            else:
                chan_guess = np.asarray(chan_guess, dtype=np.float64)

            def _get_resid_template(params):
                model_template = self._get_modeled_template_f(chan_model, params, t_arr, start_time, fs)
                difference = mean_p_f - model_template
                weights = 1.0 / (std_p_f.real) + 1.0j / (std_p_f.imag)
                weights[np.isnan(weights)] = 0.0 + 0.0j
                weights[np.isinf(weights)] = 0.0 + 0.0j
                weights[0] = 0.0 + 0.0j
                weights[np.abs(freqs) > f_fit_cutoff] = 0.0

                temp1d = np.zeros(freqs.size * 2, dtype=np.float64)
                temp1d[0:temp1d.size:2] = difference.real * weights.real
                temp1d[1:temp1d.size:2] = difference.imag * weights.imag
                return temp1d

            verbose_ = 2 if lgc_diagnostics else 0

            if lgc_plot:
                model_template_f_guess = self._get_modeled_template_f(chan_model, chan_guess, t_arr, start_time, fs)
                model_template_t_guess = self._get_modeled_template_t(chan_model, chan_guess, t_arr, start_time, fs)
                if time_lims is None:
                    trig = self._pretrigger_samples[chan]
                    time_lims_use = [t_arr[max(trig - int(0.1 * fs * 1e-3), 0)],
                                     t_arr[min(trig + int(5e-4 * fs), len(t_arr)-1)]]
                else:
                    time_lims_use = time_lims

                plt.plot(t_arr * 1e3, mean_p_t, label=f'Data {chan}', alpha=0.5, color='C0')
                lp_mean_p_t = lowpassfilter(mean_p_t, cut_off_freq=filter_freq, order=2, fs=fs)
                plt.plot(t_arr * 1e3, lp_mean_p_t, label='Filtered data', color='C0')
                plt.plot(t_arr * 1e3,
                         lowpassfilter(model_template_t_guess, cut_off_freq=filter_freq, order=2, fs=fs),
                         color='C1', label='Initial model')
                plt.legend()
                plt.ylabel('Power (W)')
                plt.xlabel('Time (ms)')
                plt.xlim(time_lims_use[0] * 1e3, time_lims_use[1] * 1e3)
                plt.ylim(-0.2*max(lp_mean_p_t[400:-400]), 1.2*max(lp_mean_p_t[400:-400]))
                plt.title(f'Initial Template Guess - {chan}')
                plt.show()

                plt.plot(freqs, np.abs(mean_p_f), label='Data')
                plt.plot(freqs, np.abs(model_template_f_guess), label='Initial model')
                plt.vlines(f_fit_cutoff, np.nanmin(np.abs(mean_p_f[np.abs(mean_p_f) > 0])),
                           np.nanmax(np.abs(mean_p_f)), label='Frequency cutoff', color='C3')
                plt.yscale('log')
                plt.xscale('log')
                plt.legend()
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power PSD absolute value (W/rt(Hz))')
                plt.title(f'Initial Template Guess, Frequency Domain - {chan}')
                plt.show()

            if chan_bounds is not None:
                result = least_squares(
                    _get_resid_template, chan_guess, bounds=chan_bounds,
                    xtol=1e-20, ftol=1e-6, max_nfev=max_nfev, verbose=verbose_
                )
            else:
                result = least_squares(
                    _get_resid_template, chan_guess,
                    xtol=1e-20, ftol=1e-6, max_nfev=max_nfev, verbose=verbose_
                )

            popt = np.asarray(result['x'], dtype=np.float64)
            jac = np.asarray(result['jac'], dtype=np.float64)
            pcovinv = np.dot(jac.transpose(), jac)
            pcov = np.linalg.pinv(pcovinv)

            self._fit_models[chan] = chan_model
            self._fit_vars[chan] = popt
            self._fit_cov[chan] = pcov

            model_template_p_t = self._get_modeled_template_t(chan_model, popt, t_arr, start_time, fs)
            model_template_p_f = self._get_modeled_template_f(chan_model, popt, t_arr, start_time, fs)
            model_template_i_f = self._get_temp_i_f_from_temp_p_f(model_template_p_f, self._dpdi[chan])
            model_template_i_t = self._get_temp_i_t_from_temp_p_f(model_template_p_f, self._dpdi[chan], fs=fs)

            self._template_fit_p_t[chan] = np.real(model_template_p_t)
            self._template_fit_p_f[chan] = model_template_p_f
            self._template_fit_i_t[chan] = np.real(model_template_i_t)
            self._template_fit_i_f[chan] = model_template_i_f

            if lgc_plot:
                model_template_t = self._template_fit_p_t[chan]
                model_template_f = self._template_fit_p_f[chan]

                plt.plot(t_arr * 1e3, mean_p_t, label='Data', alpha=0.5, color='C0')
                plt.plot(t_arr * 1e3,
                         lowpassfilter(mean_p_t, cut_off_freq=filter_freq, order=2, fs=fs),
                         label='Filtered data', color='C0')
                plt.plot(t_arr * 1e3,
                         lowpassfilter(model_template_t, cut_off_freq=filter_freq, order=2, fs=fs),
                         color='C1', label='Fit template')
                plt.legend()
                plt.ylabel('Power (W)')
                plt.xlabel('Time (ms)')
                plt.xlim(time_lims_use[0] * 1e3, time_lims_use[1] * 1e3)
                plt.ylim(-0.2*max(lp_mean_p_t[400:-400]), 1.2*max(lp_mean_p_t[400:-400]))
                plt.title(f'Fit Template, Time Domain - {chan}')
                plt.show()

                plt.plot(freqs, np.abs(mean_p_f), label='Data')
                plt.plot(freqs, np.abs(model_template_f), label='Model')
                plt.vlines(f_fit_cutoff,
                           np.nanmin(np.abs(mean_p_f[np.abs(mean_p_f) > 0])),
                           np.nanmax(np.abs(mean_p_f)), label='Frequency cutoff', color='C3')
                plt.yscale('log')
                plt.xscale('log')
                plt.legend()
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power PSD absolute value (W/rt(Hz))')
                plt.title(f'Fit Template, Frequency Domain - {chan}')
                plt.show()

                plt.plot(t_arr * 1e3, self._mean_i_t[chan], label='Current-domain pulse sum')
                plt.plot(t_arr * 1e3, self._template_fit_i_t[chan], label='Current-domain analytic template')
                plt.xlabel('Time (ms)')
                plt.ylabel('Current (A)')
                plt.xlim(time_lims_use[0] * 1e3, time_lims_use[1] * 1e3)
                plt.legend()
                plt.title(f'Current-Domain Template Comparison - {chan}')
                plt.show()

            if lgc_diagnostics:
                self._print_fit_summary(chan)

    def get_template_in_current(self, channels, domain='time', use_fit=True,
                                return_metadata=False):
        """
        Get current-domain template(s).

        By default, the fitted template is returned when available; otherwise
        the stored mean template is returned.
        """
        channels = self._normalize_channels(channels)
        container = self._template_fit_i_t if (use_fit and domain == 'time') else None
        if use_fit and domain == 'freq':
            container = self._template_fit_i_f
        if container is None or any(chan not in container for chan in channels):
            container = self._mean_i_t if domain == 'time' else self._mean_i_f

        self._check_channels_in_container(channels, container, f'current {domain} template container')

        arrays = [np.asarray(container[chan]).copy() for chan in channels]
        inds = [self._time_axis[chan].copy() if domain == 'time' else self._freqs[chan].copy()
                for chan in channels]

        if len(channels) == 1:
            out_vals = arrays[0]
            out_inds = inds[0]
            metadata = {
                'channel': channels[0],
                'sample_rate': self._sample_rate[channels[0]],
                'domain': domain,
                'type': 'current',
                'source': 'fit' if use_fit and channels[0] in self._template_fit_i_t else 'mean',
            }
        else:
            out_vals = np.asarray(arrays)
            out_inds = np.asarray(inds)
            metadata = {
                'channel': channels,
                'sample_rate': {chan: self._sample_rate[chan] for chan in channels},
                'domain': domain,
                'type': 'current',
                'source': 'fit_or_mean',
            }

        if return_metadata:
            return out_vals, out_inds, metadata
        return out_vals, out_inds

    def get_template_in_power(self, channels, dpdi=None, domain='time',
                              use_fit=True, return_metadata=False):
        """
        Get power-domain template(s).

        If ``dpdi`` is provided and a current-domain template is being returned
        from a fit/mean current template path, the power-domain template is
        recalculated from that ``dpdi``. Otherwise the stored power template is used.
        """
        channels = self._normalize_channels(channels)

        if dpdi is None:
            container = self._template_fit_p_t if (use_fit and domain == 'time') else None
            if use_fit and domain == 'freq':
                container = self._template_fit_p_f
            if container is None or any(chan not in container for chan in channels):
                container = self._mean_p_t if domain == 'time' else self._mean_p_f

            self._check_channels_in_container(channels, container, f'power {domain} template container')
            arrays = [np.asarray(container[chan]).copy() for chan in channels]
        else:
            dpdi_dict = self._prepare_complex_array(dpdi, channels, 'dpdi')
            current_container = self._template_fit_i_t if (use_fit and domain == 'time') else None
            if use_fit and domain == 'freq':
                current_container = self._template_fit_i_f
            if current_container is None or any(chan not in current_container for chan in channels):
                current_container = self._mean_i_t if domain == 'time' else self._mean_i_f
            self._check_channels_in_container(channels, current_container, f'current {domain} template container')

            arrays = []
            for chan in channels:
                current_vals = np.asarray(current_container[chan])
                dpdi_chan = np.asarray(dpdi_dict[chan], dtype=np.complex128)
                if domain == 'time':
                    current_f = fft(current_vals) / np.sqrt(len(current_vals) * self._sample_rate[chan])
                    power_f = current_f * dpdi_chan
                    power_t = ifft(power_f) * np.sqrt(len(current_vals) * self._sample_rate[chan])
                    arrays.append(np.real(-1.0 * power_t))
                else:
                    arrays.append(current_vals * dpdi_chan)

        inds = [self._time_axis[chan].copy() if domain == 'time' else self._freqs[chan].copy()
                for chan in channels]

        if len(channels) == 1:
            out_vals = arrays[0]
            out_inds = inds[0]
            metadata = {
                'channel': channels[0],
                'sample_rate': self._sample_rate[channels[0]],
                'domain': domain,
                'type': 'power',
                'source': 'fit' if use_fit and channels[0] in self._template_fit_p_t else 'mean',
            }
        else:
            out_vals = np.asarray(arrays)
            out_inds = np.asarray(inds)
            metadata = {
                'channel': channels,
                'sample_rate': {chan: self._sample_rate[chan] for chan in channels},
                'domain': domain,
                'type': 'power',
                'source': 'fit_or_mean',
            }

        if return_metadata:
            return out_vals, out_inds, metadata
        return out_vals, out_inds

    # ------------------------------------------------------------------
    # Template creation methods
    # ------------------------------------------------------------------
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

        if sample_rate is None:
            raise ValueError('ERROR: "sample_rate" argument required')

        if (trace_length_msec is None and trace_length_samples is None):
            raise ValueError(
                'ERROR: Trace length required ("trace_length_msec" or '
                '"trace_length_samples")!'
            )

        if (pretrigger_length_msec is None and pretrigger_length_samples is None):
            raise ValueError(
                'ERROR: Pretrigger length required ("pretrigger_length_msec" '
                'or "pretrigger_length_samples")!'
            )

        if tau_r is None:
            raise ValueError('ERROR: "tau_r" argument required')

        if (A is None and B is None and C is None):
            raise ValueError('ERROR: "A" and/or "B" and/or "C" argument(s) required!')

        if (tau_f1 is None and tau_f2 is None and tau_f3 is None):
            raise ValueError('ERROR: "tau_f1" and/or "tau_f2" and/or "tau_f3" argument(s) required!')

        if trace_length_samples is None:
            trace_length_samples = int(round(1e-3 * trace_length_msec * sample_rate))

        if pretrigger_length_msec is None:
            pretrigger_length_msec = 1e3 * pretrigger_length_samples / sample_rate
        else:
            pretrigger_length_samples = int(round(1e-3 * pretrigger_length_msec * sample_rate))

        dt = 1 / sample_rate
        t0 = pretrigger_length_msec * 1e-3
        t = np.asarray(list(range(trace_length_samples))) * dt

        template = None
        poles = None

        if (A is not None and B is not None and C is not None):
            if self._verbose:
                print(f'INFO: Creating 4-poles template (tag="{tag}")')
            poles = 4
            if (tau_f1 is None or tau_f2 is None or tau_f3 is None):
                raise ValueError('ERROR: 4-poles template requires 3 fall times: "tau_f1", "tau_f2" and "tau_f3"')
            template = qp.utils.make_template_fourpole(
                t, A, B, C, tau_r, tau_f1, tau_f2, tau_f3,
                t0=t0, fs=sample_rate, normalize=True
            )
        elif (A is not None and B is not None):
            if self._verbose:
                print(f'INFO: Creating 3-poles template (tag="{tag}")')
            poles = 3
            if (tau_f1 is None or tau_f2 is None):
                raise ValueError('ERROR: 3-poles template requires 2 fall times: "tau_f1" and "tau_f2"')
            template = qp.utils.make_template_threepole(
                t, A, B, tau_r, tau_f1, tau_f2,
                t0=t0, fs=sample_rate, normalize=True
            )
        elif A is not None:
            if self._verbose:
                print(f'INFO: Creating 2-poles template (tag="{tag}")')
            poles = 2
            if tau_f1 is None:
                raise ValueError('ERROR: 2-poles template requires 1 fall time: "tau_f1"')
            template = qp.utils.make_template_twopole(
                t, A, tau_r, tau_f1,
                t0=t0, fs=sample_rate, normalize=True
            )
        else:
            raise ValueError('ERROR: Unrecognize arguments. Unable to create template')

        template_name = 'template' + '_' + tag
        metadata = {
            'sample_rate': sample_rate,
            'nb_samples': trace_length_samples,
            'nb_pretrigger_samples': pretrigger_length_samples,
            'nb_poles': poles,
            'A': A, 'tau_r': tau_r, 'tau_f1': tau_f1,
        }
        if B is not None:
            metadata['B'] = B
            metadata['tau_f2'] = tau_f2
        if C is not None:
            metadata['C'] = C
            metadata['tau_f3'] = tau_f3

        channels = self._normalize_channels(channels)
        for chan in channels:
            if chan not in self._filter_data.keys():
                self._filter_data[chan] = dict()
            self._filter_data[chan][template_name] = pd.Series(template, t)
            chan_metadata = copy.deepcopy(metadata)
            chan_metadata['channel'] = chan
            self._filter_data[chan][template_name + '_metadata'] = chan_metadata

    def create_template_sum_twopoles(self, channels,
                                     amplitudes,
                                     rise_times,
                                     fall_times,
                                     sample_rate=None,
                                     trace_length_msec=None,
                                     trace_length_samples=None,
                                     pretrigger_length_msec=None,
                                     pretrigger_length_samples=None,
                                     tag='default'):
        """
        Create sum of two-poles functional forms.
        """
        if sample_rate is None:
            raise ValueError('ERROR: "sample_rate" argument required')

        if (trace_length_msec is None and trace_length_samples is None):
            raise ValueError(
                'ERROR: Trace length required ("trace_length_msec" or '
                '"trace_length_samples")!'
            )

        if (pretrigger_length_msec is None and pretrigger_length_samples is None):
            raise ValueError(
                'ERROR: Pretrigger length required ("pretrigger_length_msec" '
                'or "pretrigger_length_samples")!'
            )

        if trace_length_samples is None:
            trace_length_samples = int(round(1e-3 * trace_length_msec * sample_rate))

        if pretrigger_length_msec is None:
            pretrigger_length_msec = 1e3 * pretrigger_length_samples / sample_rate
        else:
            pretrigger_length_samples = int(round(1e-3 * pretrigger_length_msec * sample_rate))

        dt = 1 / sample_rate
        time_array = np.asarray(list(range(trace_length_samples))) * dt

        template = qp.utils.make_template_sum_twopoles(
            time_array, amplitudes, rise_times, fall_times, normalize=True
        )

        template_name = 'template' + '_' + tag
        metadata = {
            'sample_rate': sample_rate,
            'nb_samples': trace_length_samples,
            'nb_pretrigger_samples': pretrigger_length_samples,
            'nb_sum_twopoles': len(amplitudes),
        }

        channels = self._normalize_channels(channels)
        for chan in channels:
            if chan not in self._filter_data.keys():
                self._filter_data[chan] = dict()
            self._filter_data[chan][template_name] = pd.Series(template, time_array)
            chan_metadata = copy.deepcopy(metadata)
            chan_metadata['channel'] = chan
            self._filter_data[chan][template_name + '_metadata'] = chan_metadata
