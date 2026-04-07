import copy
import pandas as pd
import numpy as np
import qetpy as qp
import pytesio as h5io

from detprocess.core.filterdata import FilterData
from qetpy.core._template import Template as QETpyTemplate


class Template(FilterData):
    """
    detprocess wrapper around qetpy.core._template.Template.

    detprocess handles trace I/O and FilterData persistence.
    QETpy handles the array-based template calculations and fitting.
    """

    def __init__(self, verbose=True, filter_data=None):
        super().__init__(verbose=verbose)
        if filter_data is not None:
            if not isinstance(filter_data, dict):
                raise ValueError('ERROR: "filter_data" should be a dictionary or None!')
            self._filter_data = filter_data
        self._qp_template = QETpyTemplate(verbose=verbose)

    def clear(self, channels=None, clear_filter_data=False, tag=None):
        self._qp_template.clear(channels=channels)
        if clear_filter_data:
            self.clear_data(channels=channels, tag=tag)

    def _normalize_channels(self, channels):
        if isinstance(channels, str):
            channels = [channels]
        elif isinstance(channels, (list, tuple, np.ndarray)):
            channels = list(channels)
        else:
            raise ValueError('ERROR: "channels" should be a string or a list of strings!')
        if not channels:
            raise ValueError('ERROR: "channels" cannot be empty!')
        if not all(isinstance(chan, str) for chan in channels):
            raise ValueError('ERROR: all channel names should be strings!')
        return channels

    def _get_trace_axis_info(self, metadata, trace_length_samples=None,
                             pretrigger_length_msec=None,
                             pretrigger_length_samples=None):
        fs = float(metadata[0]['sample_rate'])
        nsamples = int(trace_length_samples) if trace_length_samples is not None else None
        if nsamples is None:
            if 'nb_samples' in metadata[0]:
                nsamples = int(metadata[0]['nb_samples'])
            else:
                raise ValueError('ERROR: unable to determine trace length in samples.')

        trigger_index = nsamples // 2
        if pretrigger_length_msec is not None:
            trigger_index = h5io.convert_length_msec_to_samples(pretrigger_length_msec, fs) - 1
        elif pretrigger_length_samples is not None:
            trigger_index = int(pretrigger_length_samples) - 1

        t = np.arange(nsamples, dtype=np.float64) / fs
        freqs = np.fft.fftfreq(nsamples, d=1.0 / fs)
        return fs, nsamples, trigger_index, t, freqs

    def _store_template(self, channels, template, sample_rate,
                        pretrigger_length_samples, tag='default', metadata=None):
        template = np.asarray(template)
        if template.ndim == 1:
            if len(channels) != 1:
                raise ValueError('ERROR: 1D template provided for multiple channels.')
            self.set_template(
                channels=channels[0],
                template=template,
                sample_rate=sample_rate,
                pretrigger_length_samples=pretrigger_length_samples,
                metadata=metadata,
                tag=tag,
            )
        elif template.ndim == 2:
            if template.shape[0] != len(channels):
                raise ValueError('ERROR: template channel dimension is inconsistent with channels.')
            for ichan, chan in enumerate(channels):
                chan_metadata = copy.deepcopy(metadata) if metadata is not None else {}
                self.set_template(
                    channels=chan,
                    template=template[ichan],
                    sample_rate=sample_rate,
                    pretrigger_length_samples=pretrigger_length_samples,
                    metadata=chan_metadata,
                    tag=tag,
                )
        else:
            raise ValueError('ERROR: template should be 1D or 2D [nchans, nsamples].')

    def _load_dpdi_from_store(self, channels, poles, dpdi_tag):
        if poles is None:
            raise ValueError(
                'ERROR: dpdi=None and no internal dpdi available. '
                'Need "poles" (and optionally "dpdi_tag") to load stored dpdi.'
            )

        dpdi_dict = {}
        dpdi_freqs_dict = {}
        metadata_dict = {}
        for chan in channels:
            dpdi_vals, dpdi_freqs = self.get_dpdi(chan, poles=poles, tag=dpdi_tag)
            dpdi_dict[chan] = np.asarray(dpdi_vals, dtype=np.complex128)
            dpdi_freqs_dict[chan] = np.asarray(dpdi_freqs, dtype=np.float64)
            metadata_dict[chan] = {'poles': poles, 'tag': dpdi_tag}
        return dpdi_dict, dpdi_freqs_dict, metadata_dict

    def calc_average_pulses(self, channels, file_path, event_list,
                            trace_length_msec=None,
                            pretrigger_length_msec=None,
                            trace_length_samples=None,
                            pretrigger_length_samples=None,
                            nevents=2000,
                            lgc_plot=False,
                            lgc_filter_freq=True,
                            filter_freq=50e3,
                            time_lims=None,
                            store_filterdata=True,
                            tag='default_mean_current'):
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

        fs, _, trigger_index, _, _ = self._get_trace_axis_info(
            metadata,
            trace_length_samples=traces.shape[-1],
            pretrigger_length_msec=pretrigger_length_msec,
            pretrigger_length_samples=pretrigger_length_samples,
        )

        self._qp_template.calc_average_pulses(
            traces=traces,
            channels=channels,
            sample_rate=fs,
            trigger_index=trigger_index,
            lgc_plot=lgc_plot,
            lgc_filter_freq=lgc_filter_freq,
            filter_freq=filter_freq,
            time_lims=time_lims,
        )

        if store_filterdata:
            mean_i_t, _ = self._qp_template.get_template_in_current(
                channels, use_fit=False, domain='time'
            )
            metadata_out = {
                'template_source': 'average_current_pulse',
                'template_domain': 'current',
                'template_kind': 'mean',
                'nb_events': traces.shape[0],
            }
            self._store_template(
                channels=channels,
                template=mean_i_t,
                sample_rate=fs,
                pretrigger_length_samples=trigger_index,
                tag=tag,
                metadata=metadata_out,
            )

    def calc_power_template(self, channels, dpdi=None, dpdi_err=None,
                            dpdi_freqs=None, poles=None, dpdi_tag='default',
                            dpdi_metadata=None, lgc_plot=False,
                            filter_freq=50e3, time_lims=None,
                            store_filterdata=True, tag='default_mean_power'):
        channels = self._normalize_channels(channels)

        if dpdi is None:
            missing = [chan for chan in channels if chan not in self._qp_template._dpdi]
            if missing:
                loaded_dpdi, loaded_freqs, loaded_meta = self._load_dpdi_from_store(channels, poles, dpdi_tag)
                dpdi_dict = loaded_dpdi
                dpdi_freqs_dict = loaded_freqs
                if dpdi_metadata is None:
                    dpdi_metadata = loaded_meta
            else:
                dpdi_dict = {chan: self._qp_template._dpdi[chan] for chan in channels}
                dpdi_freqs_dict = {chan: self._qp_template._dpdi_freqs.get(chan, self._qp_template._freqs[chan])
                                   for chan in channels}
        else:
            dpdi_dict = self._qp_template._prepare_channel_dict(dpdi, channels, dtype=np.complex128, name='dpdi')
            dpdi_freqs_dict = None if dpdi_freqs is None else self._qp_template._prepare_channel_dict(
                dpdi_freqs, channels, dtype=np.float64, name='dpdi_freqs'
            )

        if dpdi_err is None:
            dpdi_err_dict = {chan: self._qp_template._dpdi_err.get(
                chan, np.zeros_like(dpdi_dict[chan], dtype=np.complex128)
            ) for chan in channels}
        else:
            dpdi_err_dict = self._qp_template._prepare_channel_dict(dpdi_err, channels, dtype=np.complex128, name='dpdi_err')

        self._qp_template.calc_power_template(
            channels=channels,
            dpdi=dpdi_dict,
            dpdi_err=dpdi_err_dict,
            dpdi_freqs=dpdi_freqs_dict,
            dpdi_metadata=dpdi_metadata,
            lgc_plot=lgc_plot,
            filter_freq=filter_freq,
            time_lims=time_lims,
        )

        if store_filterdata:
            mean_p_t, _ = self._qp_template.get_template_in_power(
                channels, use_fit=False, domain='time'
            )
            fs = self._qp_template._sample_rate[channels[0]]
            pretrigger_samples = self._qp_template._pretrigger_samples[channels[0]]
            metadata_out = {
                'template_source': 'average_power_pulse',
                'template_domain': 'power',
                'template_kind': 'mean',
            }
            self._store_template(
                channels=channels,
                template=mean_p_t,
                sample_rate=fs,
                pretrigger_length_samples=pretrigger_samples,
                tag=tag,
                metadata=metadata_out,
            )

    def fit_templates(self, channels, template_model='twopole',
                      guess=None, bounds=None, f_fit_cutoff=50e3,
                      max_nfev=800, dt=0.0, lgc_diagnostics=False,
                      lgc_plot=True, filter_freq=50e3,
                      time_lims=None, store_filterdata=True,
                      tag='default'):
        channels = self._normalize_channels(channels)
        fit_result = self._qp_template.fit_templates(
            channels=channels,
            template_model=template_model,
            guess=guess,
            bounds=bounds,
            dt=dt,
            f_fit_cutoff=f_fit_cutoff,
            max_nfev=max_nfev,
            lgc_diagnostics=lgc_diagnostics,
            lgc_plot=lgc_plot,
            filter_freq=filter_freq,
            time_lims=time_lims,
        )

        if store_filterdata:
            fit_i_t, _ = self._qp_template.get_template_in_current(
                channels, use_fit=True, domain='time'
            )
            fs = self._qp_template._sample_rate[channels[0]]
            pretrigger_samples = self._qp_template._pretrigger_samples[channels[0]]
            metadata_out = {
                'template_source': 'fitted_current_template',
                'template_domain': 'current',
                'template_kind': 'fit',
                'template_model': template_model,
            }
            self._store_template(
                channels=channels,
                template=fit_i_t,
                sample_rate=fs,
                pretrigger_length_samples=pretrigger_samples,
                tag=tag,
                metadata=metadata_out,
            )

        return fit_result

    def get_template_in_current(self, channels, use_fit=True, domain='time', return_metadata=False):
        channels = self._normalize_channels(channels)
        return self._qp_template.get_template_in_current(
            channels=channels,
            use_fit=use_fit,
            domain=domain,
            return_metadata=return_metadata,
        )

    def get_template_in_power(self, channels, dpdi=None, dpdi_freqs=None,
                              use_fit=True, domain='time', return_metadata=False):
        channels = self._normalize_channels(channels)
        return self._qp_template.get_template_in_power(
            channels=channels,
            dpdi=dpdi,
            dpdi_freqs=dpdi_freqs,
            use_fit=use_fit,
            domain=domain,
            return_metadata=return_metadata
        )

    @property
    def qp_template(self):
        return self._qp_template

    # ------------------------------------------------------------------
    # Original template creation helpers retained
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
        if sample_rate is None:
            raise ValueError('ERROR: "sample_rate" argument required')
        if trace_length_msec is None and trace_length_samples is None:
            raise ValueError('ERROR: Trace length required ("trace_length_msec" or "trace_length_samples")!')
        if pretrigger_length_msec is None and pretrigger_length_samples is None:
            raise ValueError('ERROR: Pretrigger length required ("pretrigger_length_msec" or "pretrigger_length_samples")!')
        if tau_r is None:
            raise ValueError('ERROR: "tau_r" argument required')
        if A is None and B is None and C is None:
            raise ValueError('ERROR: "A" and/or "B" and/or "C" argument(s) required!')
        if tau_f1 is None and tau_f2 is None and tau_f3 is None:
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

        if A is not None and B is not None and C is not None:
            poles = 4
            template = qp.utils.make_template_fourpole(
                t, A, B, C, tau_r, tau_f1, tau_f2, tau_f3,
                t0=t0, fs=sample_rate, normalize=True
            )
        elif A is not None and B is not None:
            poles = 3
            template = qp.utils.make_template_threepole(
                t, A, B, tau_r, tau_f1, tau_f2,
                t0=t0, fs=sample_rate, normalize=True
            )
        else:
            poles = 2
            template = qp.utils.make_template_twopole(
                t, A, tau_r, tau_f1,
                t0=t0, fs=sample_rate, normalize=True
            )

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
            if chan not in self._filter_data:
                self._filter_data[chan] = {}
            self._filter_data[chan]['template_' + tag] = pd.Series(template, t)
            chan_metadata = copy.deepcopy(metadata)
            chan_metadata['channel'] = chan
            self._filter_data[chan]['template_' + tag + '_metadata'] = chan_metadata

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
        if sample_rate is None:
            raise ValueError('ERROR: "sample_rate" argument required')
        if trace_length_msec is None and trace_length_samples is None:
            raise ValueError('ERROR: Trace length required ("trace_length_msec" or "trace_length_samples")!')
        if pretrigger_length_msec is None and pretrigger_length_samples is None:
            raise ValueError('ERROR: Pretrigger length required ("pretrigger_length_msec" or "pretrigger_length_samples")!')

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

        metadata = {
            'sample_rate': sample_rate,
            'nb_samples': trace_length_samples,
            'nb_pretrigger_samples': pretrigger_length_samples,
            'nb_sum_twopoles': len(amplitudes),
        }

        channels = self._normalize_channels(channels)
        for chan in channels:
            if chan not in self._filter_data:
                self._filter_data[chan] = {}
            self._filter_data[chan]['template_' + tag] = pd.Series(template, time_array)
            chan_metadata = copy.deepcopy(metadata)
            chan_metadata['channel'] = chan
            self._filter_data[chan]['template_' + tag + '_metadata'] = chan_metadata
