"""Unit tests for dIdQ processing helpers. These tests do not read detector data."""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import qetpy as qp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from process_didq import build_output_directory, resolve_worker_count
from detprocess.core.didq import (
    build_didq_row,
    build_pooled_metadata,
    build_pooled_series_name,
    build_didq_key,
    build_3pole_seeds,
    three_pole_improves_on_two_pole,
    DIDQ_NB_FREE_PARAMS,
    count_progress_stages,
    DIDQAnalysis,
    compute_snr_diagnostic,
    get_driven_bin_mask,
    parse_thermometer_channels,
    resolve_drive_parameters,
    validate_channels_present,
    rows_to_dataframe,
    validate_drive_period_coverage,
    validate_series_are_poolable,
)
from detprocess.core.filterdata import FilterData


def make_detector_config():
    """
    Build a detector config dictionary shaped like the one returned by
    pytesio H5Reader.get_detector_config for a dIdQ dataset.

    Parameters
    ----------
    None

    Return
    ------
    detector_config : dict
        Dictionary keyed by detector channel name.
    """
    detector_config = {
        'ChanA': {
            'signal_gen_source': 'tes',
            'signal_gen_frequency': 4.0,
            'signal_gen_current': 9.960079840319362e-05,
            'signal_gen_offset': 0.5,
            'shunt_resistance': 0.004999999888241291,
            'tes_bias': '0.00012002442002442',
        },
        'ChanB': {
            'signal_gen_source': 'none',
            'signal_gen_frequency': 4.0,
            'signal_gen_current': 9.960079840319362e-05,
            'signal_gen_offset': 0.5,
            'shunt_resistance': 0.004999999888241291,
            'tes_bias': '1.6483516483516496e-05',
        },
    }

    return detector_config


def test_parse_thermometer_channels_splits_on_commas_and_spaces():
    assert parse_thermometer_channels('ChanA') == ['ChanA']
    assert parse_thermometer_channels('ChanA,ChanB') == ['ChanA', 'ChanB']
    assert parse_thermometer_channels('ChanA, ChanB') == ['ChanA', 'ChanB']
    assert parse_thermometer_channels('ChanA ChanB') == ['ChanA', 'ChanB']
    assert parse_thermometer_channels(
        ['ChanA,ChanB', 'ChanC']
    ) == ['ChanA', 'ChanB', 'ChanC']


def test_parse_thermometer_channels_drops_duplicates_keeping_order():
    assert parse_thermometer_channels(
        'ChanB,ChanA,ChanB'
    ) == ['ChanB', 'ChanA']


def test_parse_thermometer_channels_raises_when_empty():
    for value in (None, '', '  ', [], [',']):
        with pytest.raises(ValueError, match='no thermometer channel'):
            parse_thermometer_channels(value)


def test_validate_channels_present_accepts_recorded_channels():
    validate_channels_present(
        detector_config=make_detector_config(),
        channels=['ChanA', 'ChanB'],
    )


def test_validate_channels_present_raises_and_lists_the_alternatives():
    with pytest.raises(ValueError, match='NoSuchChannel') as error:
        validate_channels_present(
            detector_config=make_detector_config(),
            channels=['ChanA', 'NoSuchChannel'],
        )
    assert 'ChanB' in str(error.value)


def test_signal_gen_source_is_never_consulted():
    # every channel driven, or none, resolves the same as the default fixture
    for source in ('tes', 'none'):
        detector_config = make_detector_config()
        for config in detector_config.values():
            config['signal_gen_source'] = source

        validate_channels_present(
            detector_config=detector_config, channels=['ChanA', 'ChanB'],
        )
        params = resolve_drive_parameters(
            detector_config=detector_config, channel='ChanB',
        )
        assert params['sgfreq_hz'] == pytest.approx(4.0)


def test_build_didq_key_joins_the_series_and_channel():
    assert build_didq_key('I2_D20260731_T135414', 'ChanA') == (
        'I2_D20260731_T135414_ChanA'
    )


def test_resolve_drive_parameters_reads_the_channel_it_is_given():
    # the acquisition replicates the drive on every channel, so an undriven
    # thermometer resolves the same parameters as the driven channel
    for channel in ('ChanA', 'ChanB'):
        params = resolve_drive_parameters(
            detector_config=make_detector_config(), channel=channel,
        )
        assert params['sgfreq_hz'] == pytest.approx(4.0)
        assert params['sgamp_amps'] == pytest.approx(9.960079840319362e-05)


def test_resolve_drive_parameters_from_metadata():
    params = resolve_drive_parameters(
        detector_config=make_detector_config(),
        channel='ChanA',
    )
    assert params['sgfreq_hz'] == pytest.approx(4.0)
    assert params['sgamp_amps'] == pytest.approx(9.960079840319362e-05)
    assert params['rshunt_ohms'] == pytest.approx(0.005, rel=1e-3)
    assert params['duty_cycle'] == pytest.approx(0.5)


def test_resolve_drive_parameters_drops_recorded_offset():
    params = resolve_drive_parameters(
        detector_config=make_detector_config(),
        channel='ChanA',
    )
    assert 'sgoffset_v' not in params


def test_resolve_drive_parameters_takes_duty_cycle_only():
    params = resolve_drive_parameters(
        detector_config=make_detector_config(),
        channel='ChanA',
        duty_cycle=0.25,
    )
    assert params['duty_cycle'] == pytest.approx(0.25)

    # the recorded values stay untouched, since the duty cycle is the only
    # drive parameter the acquisition does not write
    assert params['sgfreq_hz'] == pytest.approx(4.0)
    assert params['sgamp_amps'] == pytest.approx(9.960079840319362e-05)


def test_resolve_drive_parameters_rejects_recorded_overrides():
    signature = inspect.signature(resolve_drive_parameters)
    accepted = set(signature.parameters)

    for name in ('sgfreq_hz', 'sgamp_ua', 'sgoffset_v', 'rshunt_mohm'):
        assert name not in accepted


def test_resolve_drive_parameters_raises_on_nan():
    detector_config = make_detector_config()
    detector_config['ChanA']['signal_gen_current'] = float('nan')

    with pytest.raises(ValueError, match='signal_gen_current'):
        resolve_drive_parameters(
            detector_config=detector_config,
            channel='ChanA',
        )


def test_resolve_drive_parameters_accepts_rshunt_alias():
    detector_config = make_detector_config()
    detector_config['ChanA'].pop('shunt_resistance')
    detector_config['ChanA']['rshunt'] = 0.006

    params = resolve_drive_parameters(
        detector_config=detector_config,
        channel='ChanA',
    )
    assert params['rshunt_ohms'] == pytest.approx(0.006)


def test_resolve_drive_parameters_raises_on_missing_key():
    detector_config = make_detector_config()
    detector_config['ChanA'].pop('signal_gen_frequency')

    with pytest.raises(ValueError, match='signal_gen_frequency'):
        resolve_drive_parameters(
            detector_config=detector_config,
            channel='ChanA',
        )


def test_output_directory_is_sibling_of_raw(tmp_path):
    group_dir = tmp_path / 'run74' / 'raw' / 'exttrig_I2_D20260719_T145253'
    group_dir.mkdir(parents=True)

    output_dir, group_name = build_output_directory(group_dir)

    assert group_name == 'exttrig_I2_D20260719_T145253'
    assert output_dir == (tmp_path / 'run74' / 'processed'
                          / 'exttrig_I2_D20260719_T145253')

    # the output must never be nested inside the raw tree
    assert 'raw' not in output_dir.parts


def test_output_directory_from_a_raw_file(tmp_path):
    group_dir = tmp_path / 'run74' / 'raw' / 'exttrig_I2_D20260719_T145253'
    group_dir.mkdir(parents=True)
    raw_file = group_dir / 'exttrig_I2_D20260719_T145304_F0001.hdf5'
    raw_file.touch()

    output_dir, group_name = build_output_directory(raw_file)

    assert group_name == 'exttrig_I2_D20260719_T145253'
    assert output_dir == (tmp_path / 'run74' / 'processed'
                          / 'exttrig_I2_D20260719_T145253')


def test_output_directory_matches_raw_as_whole_component(tmp_path):
    # a directory merely starting with "raw" is not the raw tree
    group_dir = tmp_path / 'rawdata' / 'somegroup'
    group_dir.mkdir(parents=True)

    output_dir, _ = build_output_directory(group_dir)

    assert output_dir == tmp_path / 'rawdata' / 'processed' / 'somegroup'


def test_driven_bin_mask_selects_odd_harmonics():
    sgfreq_hz = 4.0
    freq = np.arange(-40.0, 41.0, 4.0)

    mask = get_driven_bin_mask(freq=freq, sgfreq_hz=sgfreq_hz)

    selected = np.sort(np.abs(freq[mask]))
    expected = np.array([4.0, 4.0, 12.0, 12.0, 20.0, 20.0, 28.0, 28.0,
                         36.0, 36.0])
    assert np.allclose(selected, expected)


def test_driven_bin_mask_excludes_dc():
    freq = np.arange(-40.0, 41.0, 4.0)
    mask = get_driven_bin_mask(freq=freq, sgfreq_hz=4.0)
    assert not mask[freq == 0.0].any()


def test_driven_bin_mask_includes_even_harmonics_when_duty_cycle_not_half():
    freq = np.arange(-40.0, 41.0, 4.0)
    mask = get_driven_bin_mask(freq=freq, sgfreq_hz=4.0, duty_cycle=0.25)

    selected = np.sort(np.abs(freq[mask]))
    # every harmonic except DC is driven for an asymmetric duty cycle
    assert 8.0 in selected
    assert 4.0 in selected
    assert not mask[freq == 0.0].any()


def test_snr_diagnostic_finds_last_bin_above_threshold():
    freq = np.array([4.0, 12.0, 20.0, 28.0, 36.0])
    driven_mask = np.ones(freq.size, dtype=bool)
    # SNR of 10, 5, 4, 1, 1 -> last bin above 3.0 is at 20 Hz
    didv_mean = np.array([10.0, 5.0, 4.0, 1.0, 1.0], dtype=complex)
    didv_std = np.ones(freq.size, dtype=complex)

    diagnostic = compute_snr_diagnostic(
        freq=freq,
        didv_mean=didv_mean,
        didv_std=didv_std,
        driven_mask=driven_mask,
        snr_threshold=3.0,
    )

    assert diagnostic['f_max_snr3_hz'] == pytest.approx(20.0)
    assert diagnostic['n_freq_bins_snr3'] == 3


def test_snr_diagnostic_handles_no_bins_above_threshold():
    freq = np.array([4.0, 12.0])
    driven_mask = np.ones(freq.size, dtype=bool)
    didv_mean = np.array([1.0, 1.0], dtype=complex)
    didv_std = np.ones(freq.size, dtype=complex)

    diagnostic = compute_snr_diagnostic(
        freq=freq,
        didv_mean=didv_mean,
        didv_std=didv_std,
        driven_mask=driven_mask,
        snr_threshold=3.0,
    )

    assert np.isnan(diagnostic['f_max_snr3_hz'])
    assert diagnostic['n_freq_bins_snr3'] == 0


SYNTHETIC_FS = 12500.0
SYNTHETIC_SGFREQ = 4.0
SYNTHETIC_SGAMP = 1.0e-4
SYNTHETIC_RSHUNT = 5.0e-3
SYNTHETIC_DUTY = 0.5

# A + B must not be zero. The DC impedance of the model is A + B, so A = -B
# gives an infinite admittance, a trace full of NaN, and a confusing
# "Initial guess is outside of provided bounds" error out of the fit.
SYNTHETIC_TRUE_PARAMS = {'A': 1.0e3,
                         'B': -9.0e2,
                         'C': 0.0,
                         'tau1': -2.0e-2,
                         'tau2': 1.0e-5,
                         'tau3': 0.0}


def make_synthetic_traces(n_periods=4, n_traces=25, noise_fraction=1.0e-3,
                          seed=1234):
    """
    Build a set of synthetic dIdQ traces from a known two-pole model.

    Parameters
    ----------
    n_periods : int, optional
        Number of square wave periods in each trace.
    n_traces : int, optional
        Number of traces in the ensemble.
    noise_fraction : float, optional
        White noise amplitude as a fraction of the clean trace peak-to-peak.
    seed : int, optional
        Seed for the random number generator.

    Return
    ------
    traces : ndarray
        Array of shape (n_traces, n_samples).
    """
    n_samples = int(n_periods * SYNTHETIC_FS / SYNTHETIC_SGFREQ)
    times = np.arange(n_samples) / SYNTHETIC_FS

    clean = qp.squarewaveresponse(
        times,
        SYNTHETIC_SGAMP,
        SYNTHETIC_SGFREQ,
        SYNTHETIC_TRUE_PARAMS,
        dutycycle=SYNTHETIC_DUTY,
        rsh=SYNTHETIC_RSHUNT,
    )
    assert np.all(np.isfinite(clean)), 'synthetic trace must be finite'

    generator = np.random.default_rng(seed)
    noise = generator.normal(
        0.0,
        noise_fraction * np.ptp(clean),
        size=(n_traces, n_samples),
    )

    return clean[None, :] + noise


def make_synthetic_drive_params():
    """
    Build the drive parameter dictionary matching the synthetic traces.

    Parameters
    ----------
    None

    Return
    ------
    params : dict
        Drive parameters for the synthetic dataset.
    """
    return {'sgfreq_hz': SYNTHETIC_SGFREQ,
            'sgamp_amps': SYNTHETIC_SGAMP,
            'rshunt_ohms': SYNTHETIC_RSHUNT,
            'duty_cycle': SYNTHETIC_DUTY}


def test_two_pole_roundtrip_recovers_injected_parameters():
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(2,), fcutoff_hz=np.inf)

    results = analysis.get_fit_results(series_name='synthetic', poles=2)
    params = results['params']

    assert results['fit_success']
    for name in ('A', 'B', 'tau1'):
        assert params[name] == pytest.approx(
            SYNTHETIC_TRUE_PARAMS[name], rel=1.0e-3
        )
    assert params['tau2'] == pytest.approx(
        SYNTHETIC_TRUE_PARAMS['tau2'], rel=1.0e-2
    )


def test_driven_bin_selection_matches_fitting_all_bins():
    traces = make_synthetic_traces()
    drive_params = make_synthetic_drive_params()

    masked = DIDQAnalysis(verbose=False)
    masked.set_traces(traces=traces, fs=SYNTHETIC_FS,
                      drive_params=drive_params, series_name='synthetic')
    masked.dofit(list_of_poles=(2,), fcutoff_hz=np.inf, use_driven_bins=True)

    full = DIDQAnalysis(verbose=False)
    full.set_traces(traces=traces, fs=SYNTHETIC_FS,
                    drive_params=drive_params, series_name='synthetic')
    full.dofit(list_of_poles=(2,), fcutoff_hz=np.inf, use_driven_bins=False)

    masked_params = masked.get_fit_results('synthetic', 2)['params']
    full_params = full.get_fit_results('synthetic', 2)['params']

    for name in ('A', 'B', 'tau1'):
        assert masked_params[name] == pytest.approx(
            full_params[name], rel=1.0e-8
        )
    assert masked_params['tau2'] == pytest.approx(
        full_params['tau2'], rel=1.0e-6
    )


def test_driven_bin_selection_reduces_bin_count():
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(2,), fcutoff_hz=np.inf)

    results = analysis.get_fit_results('synthetic', 2)
    data = analysis.get_didq_data('synthetic')

    # only odd harmonics survive, so far fewer bins are fitted than exist
    assert results['n_freq_bins_fit'] < data['n_freq_bins_total'] / 4


def test_dofit_does_not_permanently_truncate_shared_didvobj():
    """dofit(use_driven_bins=True) used to replace the frequency-domain
    arrays on the stored qp.DIDV object in place. A later call with
    use_driven_bins=False then silently kept fitting the already-truncated
    driven subset, because the object was never restored. This pins that the
    object is restored after every dofit call, so a later full-bin fit
    genuinely uses the full bin set."""
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    data = analysis.get_didq_data('synthetic')
    n_freq_bins_total = data['n_freq_bins_total']

    analysis.dofit(list_of_poles=(2,), fcutoff_hz=np.inf,
                   use_driven_bins=True)

    # the stored object must come back full-length immediately after a
    # driven-bins-only fit
    didvobj = data['didvobj']
    assert didvobj._freq.size == n_freq_bins_total
    assert didvobj._didvmean.size == n_freq_bins_total
    assert didvobj._didvstd.size == n_freq_bins_total

    driven_results = analysis.get_fit_results('synthetic', 2)

    analysis.dofit(list_of_poles=(2,), fcutoff_hz=np.inf,
                   use_driven_bins=False)
    full_results = analysis.get_fit_results('synthetic', 2)

    assert driven_results['n_freq_bins_fit'] < n_freq_bins_total
    assert full_results['n_freq_bins_fit'] == n_freq_bins_total
    assert didvobj._freq.size == n_freq_bins_total


@pytest.mark.filterwarnings(
    'ignore:invalid value encountered in sqrt:RuntimeWarning'
)
def test_three_pole_falltimes_are_stable_across_starting_points():
    """The three-pole model is degenerate in its raw parameters but its real
    fall times are reproducible. This pins the reproducible property."""
    traces = make_synthetic_traces()
    drive_params = make_synthetic_drive_params()

    falltimes = list()
    for seed_scale in (1.0, 2.0, 5.0):
        analysis = DIDQAnalysis(verbose=False)
        analysis.set_traces(traces=traces, fs=SYNTHETIC_FS,
                            drive_params=drive_params,
                            series_name='synthetic')
        analysis.dofit(
            list_of_poles=(3,),
            fcutoff_hz=np.inf,
            max_nfev=20000,
            guess_params_3poles=(
                1.0e3 * seed_scale,
                -9.0e2 * seed_scale,
                -0.5,
                -2.0e-2,
                1.0e-5,
                1.0e-3,
                1.0e-6,
            ),
        )
        results = analysis.get_fit_results('synthetic', 3)
        falltimes.append(np.sort(results['falltimes']))

    falltimes = np.array(falltimes)

    # the longest fall time is the physically meaningful one and must agree
    longest = falltimes[:, 0]
    spread = (np.max(longest) - np.min(longest)) / np.abs(np.mean(longest))
    assert spread < 0.01


# a genuine slow thermal pole an order of magnitude above the electrical one,
# which is the case qetpy's own starting point cannot reach. Taken from a
# measured run 75 channel with every tau scaled down by ten, so the poles stay
# well inside a 4 Hz drive period
SYNTHETIC_3POLE_TRUE_PARAMS = {'A': 1582.18,
                               'B': -1590.16,
                               'C': -1.129e-3,
                               'tau1': -9.276e-6,
                               'tau2': 4.821e-6,
                               'tau3': 1.293e-2}


def make_synthetic_3pole_traces(n_periods=4, n_traces=25,
                                noise_fraction=1.0e-3, seed=1234):
    """
    Build synthetic dIdQ traces from a known three-pole model.

    Parameters
    ----------
    n_periods : int, optional
        Number of square wave periods in each trace.
    n_traces : int, optional
        Number of traces in the ensemble.
    noise_fraction : float, optional
        White noise amplitude as a fraction of the clean trace peak-to-peak.
    seed : int, optional
        Seed for the random number generator.

    Return
    ------
    traces : ndarray
        Array of shape (n_traces, n_samples).
    """
    n_samples = int(n_periods * SYNTHETIC_FS / SYNTHETIC_SGFREQ)
    times = np.arange(n_samples) / SYNTHETIC_FS

    clean = qp.squarewaveresponse(
        times,
        SYNTHETIC_SGAMP,
        SYNTHETIC_SGFREQ,
        SYNTHETIC_3POLE_TRUE_PARAMS,
        dutycycle=SYNTHETIC_DUTY,
        rsh=SYNTHETIC_RSHUNT,
    )
    assert np.all(np.isfinite(clean)), 'synthetic trace must be finite'

    generator = np.random.default_rng(seed)
    noise = generator.normal(
        0.0,
        noise_fraction * np.ptp(clean),
        size=(n_traces, n_samples),
    )

    return clean[None, :] + noise


def synthetic_3pole_true_falltimes():
    """
    Fall times of the synthetic three-pole model, descending by magnitude.

    Parameters
    ----------
    None

    Return
    ------
    falltimes : ndarray
        The three fall times, in seconds.
    """
    params_array = np.array(
        [SYNTHETIC_3POLE_TRUE_PARAMS[name]
         for name in ('A', 'B', 'C', 'tau1', 'tau2', 'tau3')] + [0.0]
    )
    falltimes = qp.DIDV._findpolefalltimes(params_array)

    return np.sort(falltimes)[::-1]


def test_build_3pole_seeds_carries_the_two_pole_scale():
    params_2poles = {'A': 1.0e3, 'B': 9.0e2, 'tau1': 2.0e-2,
                     'tau2': 1.0e-5, 'dt': -1.5e-3}

    seeds = build_3pole_seeds(params_2poles=params_2poles)

    assert len(seeds) == 4
    for seed in seeds:
        a, b, c, tau1, tau2, tau3, dt = seed
        assert a == 1.0e3
        # the fit wants the loop gain branch signs, whatever the fit returned
        assert b == -9.0e2
        assert tau1 == -2.0e-2
        assert tau2 == 1.0e-5
        assert dt == -1.5e-3
        # any small C reaches the slow pole, qetpy's own -0.05 does not
        assert abs(c) < 1.0e-2

    assert len(set(seed[5] for seed in seeds)) == len(seeds)


def test_build_3pole_seeds_floors_a_zero_starting_value():
    """No element of a starting guess may be zero, and a converged two-pole
    tau2 or dt can land on numerical zero."""
    params_2poles = {'A': 1.0e3, 'B': -9.0e2, 'tau1': -2.0e-2,
                     'tau2': 0.0, 'dt': 0.0}

    for seed in build_3pole_seeds(params_2poles=params_2poles):
        assert all(value != 0.0 for value in seed)


def test_three_pole_improves_on_two_pole_compares_chi_square_not_cost():
    """The models do not share a degree of freedom count, so a three-pole fit
    with the same chi square reports a slightly larger cost per degree of
    freedom. Comparing the costs directly would reject it."""
    n_bins_fit = 100
    chisq = 250.0

    cost_2poles = chisq / (n_bins_fit - DIDQ_NB_FREE_PARAMS[2])
    cost_3poles = chisq / (n_bins_fit - DIDQ_NB_FREE_PARAMS[3])

    assert cost_3poles > cost_2poles
    assert three_pole_improves_on_two_pole(
        cost_3poles=cost_3poles,
        cost_2poles=cost_2poles,
        n_bins_fit=n_bins_fit,
    )


def test_three_pole_improves_on_two_pole_rejects_a_stalled_fit():
    n_bins_fit = 100

    assert not three_pole_improves_on_two_pole(
        cost_3poles=50.0,
        cost_2poles=2.0,
        n_bins_fit=n_bins_fit,
    )


def test_three_pole_fit_recovers_a_slow_pole_without_a_hand_seed():
    """The whole point of seeding from the two-pole fit: qetpy's own starting
    point stalls on data with a slow thermal pole."""
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_3pole_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(2, 3), fcutoff_hz=np.inf, max_nfev=20000)

    results = analysis.get_fit_results('synthetic', 3)
    falltimes = np.sort(results['falltimes'])[::-1]
    truth = synthetic_3pole_true_falltimes()

    assert results['seeded_from_2pole']
    for index in (0, 1):
        assert falltimes[index] == pytest.approx(truth[index], rel=1.0e-2)


def test_three_pole_fit_never_costs_more_than_the_two_pole_fit():
    """Zeroing C and tau3 recovers the two-pole model, so a three-pole fit
    that costs more has stalled rather than found a worse model."""
    for traces in (make_synthetic_traces(), make_synthetic_3pole_traces()):
        analysis = DIDQAnalysis(verbose=False)
        analysis.set_traces(
            traces=traces,
            fs=SYNTHETIC_FS,
            drive_params=make_synthetic_drive_params(),
            series_name='synthetic',
        )
        analysis.dofit(list_of_poles=(2, 3), fcutoff_hz=np.inf,
                       max_nfev=20000)

        results_2poles = analysis.get_fit_results('synthetic', 2)
        results_3poles = analysis.get_fit_results('synthetic', 3)

        assert three_pole_improves_on_two_pole(
            cost_3poles=results_3poles['cost'],
            cost_2poles=results_2poles['cost'],
            n_bins_fit=results_3poles['n_freq_bins_fit'],
        )


def test_three_pole_fit_seeds_itself_when_only_three_poles_are_asked_for():
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_3pole_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(3,), fcutoff_hz=np.inf, max_nfev=20000)

    results = analysis.get_fit_results('synthetic', 3)
    truth = synthetic_3pole_true_falltimes()

    assert results['seeded_from_2pole']
    assert np.sort(results['falltimes'])[::-1][0] == pytest.approx(
        truth[0], rel=1.0e-2
    )

    # the two-pole fit was a means to a starting point, not a requested result
    assert not analysis.get_fit_results('synthetic', 2)


def test_explicit_three_pole_guess_is_used_as_given():
    """An explicit guess is the escape hatch, so it must not be quietly
    replaced by a seeded start."""
    guess = (1.0e3, -9.0e2, -0.5, -2.0e-2, 1.0e-5, 1.0e-3, 1.0e-6)

    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(3,), fcutoff_hz=np.inf, max_nfev=20000,
                   guess_params_3poles=guess)

    results = analysis.get_fit_results('synthetic', 3)

    assert 'seeded_from_2pole' not in results
    assert 'nb_starts_tried' not in results


def test_ndof_counts_the_free_parameters_of_each_model():
    """The stored parameter vector is seven long whatever the model, so it
    cannot be used to count degrees of freedom."""
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(2, 3), fcutoff_hz=np.inf)

    for poles in (2, 3):
        results = analysis.get_fit_results('synthetic', poles)
        assert len(results['params_array']) == 7
        assert results['ndof'] == (
            results['n_freq_bins_fit'] - DIDQ_NB_FREE_PARAMS[poles]
        )


def test_fit_results_never_contain_small_signal_parameters():
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(2,), fcutoff_hz=np.inf)

    results = analysis.get_fit_results('synthetic', 2)
    for forbidden in ('smallsignalparams', 'ssp_light', 'biasparams'):
        assert forbidden not in results


@pytest.mark.filterwarnings(
    'ignore::pandas.errors.PerformanceWarning'
)
def test_save_didq_data_roundtrips_through_filterdata(tmp_path):
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(2,), fcutoff_hz=np.inf)
    analysis.save_didq_data(
        file_path_name=str(tmp_path / 'didq_test.hdf5'),
        save_hdf5=True,
    )

    stored = analysis.get_didq_results(channel='synthetic', poles=2)

    assert 'params' in stored
    assert stored['params']['A'] == pytest.approx(
        SYNTHETIC_TRUE_PARAMS['A'], rel=1.0e-3
    )
    assert (tmp_path / 'didq_test.hdf5').exists()


@pytest.mark.filterwarnings(
    'ignore::pandas.errors.PerformanceWarning'
)
def test_save_didq_data_saves_traces_needed_to_replot(tmp_path):
    """The design document requires the saved didq_results object to carry
    the mean trace, the mean transfer function and its standard deviation,
    so a fit can be re-plotted without re-reading raw data. This roundtrips
    through an actual HDF5 file, via a fresh FilterData object, rather than
    reading back off the in-memory analysis instance."""
    analysis = DIDQAnalysis(verbose=False)
    traces = make_synthetic_traces()
    analysis.set_traces(
        traces=traces,
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    n_freq_bins_total = analysis.get_didq_data('synthetic')['n_freq_bins_total']
    nb_samples = traces.shape[1]

    analysis.dofit(list_of_poles=(2,), fcutoff_hz=np.inf)
    analysis.save_didq_data(
        file_path_name=str(tmp_path / 'didq_traces_test.hdf5'),
        save_hdf5=True,
    )

    reloaded = FilterData(verbose=False)
    reloaded.load_hdf5(str(tmp_path / 'didq_traces_test.hdf5'))

    stored_traces = reloaded.get_didq_traces(channel='synthetic')

    for key in ('tmean', 'didv_mean', 'didv_std', 'freq'):
        assert key in stored_traces

    assert stored_traces['tmean'].shape == (nb_samples,)
    assert stored_traces['didv_mean'].shape == (n_freq_bins_total,)
    assert stored_traces['didv_std'].shape == (n_freq_bins_total,)
    assert stored_traces['freq'].shape == (n_freq_bins_total,)


@pytest.mark.filterwarnings(
    'ignore:invalid value encountered in sqrt:RuntimeWarning'
)
def test_build_didq_row_has_expected_schema():
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(2, 3), fcutoff_hz=np.inf)

    row = build_didq_row(
        analysis=analysis,
        series_name='synthetic',
        processing_id='I2_D20260726_T120000',
    )

    for column in ('series_name', 'processing_id', 'sgfreq_hz',
                   'sgamp_amps', 'rshunt_ohms', 'duty_cycle', 'fs_hz',
                   'nb_samples', 'nb_traces_used', 'f_max_snr3_hz'):
        assert column in row

    for prefix in ('didq_2pole_', 'didq_3pole_'):
        for name in ('A', 'B', 'C', 'tau1', 'tau2', 'tau3', 'dt'):
            assert prefix + name in row
            assert prefix + name + '_err' in row
        for name in ('falltime_1', 'falltime_2', 'falltime_3', 'cost',
                     'ndof', 'fit_success', 'n_freq_bins_fit', 'fcutoff_hz'):
            assert prefix + name in row

    # every value must be a scalar, so vaex can build a column from it
    for key, value in row.items():
        assert np.isscalar(value) or isinstance(value, (str, bool)), (
            f'column {key} is not scalar: {type(value)}'
        )


def test_build_didq_row_orders_falltimes_by_descending_magnitude():
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(2,), fcutoff_hz=np.inf)

    row = build_didq_row(
        analysis=analysis,
        series_name='synthetic',
        processing_id='test',
    )

    magnitude_1 = abs(row['didq_2pole_falltime_1'])
    magnitude_2 = abs(row['didq_2pole_falltime_2'])
    magnitude_3 = abs(row['didq_2pole_falltime_3'])
    assert magnitude_1 >= magnitude_2
    assert magnitude_2 >= magnitude_3


def test_build_didq_row_falltime_1_is_negative_physical_pole_not_artifact():
    """The default synthetic fixture (SYNTHETIC_TRUE_PARAMS above) fits into
    the loop-gain-greater-than-one branch, where qetpy returns a negative
    fall time as the physically meaningful pole. A signed descending sort
    would put the near-zero artifact first instead; falltime_1 must be the
    largest-magnitude entry, sign intact, matching the raw qetpy output of
    approximately [-1.999e-01, 9.978e-06, 0.0]."""
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(2,), fcutoff_hz=np.inf)

    results = analysis.get_fit_results('synthetic', 2)
    raw_falltimes = np.asarray(results['falltimes'])

    # confirm the fixture still exercises the negative-falltime branch this
    # test is meant to catch
    assert np.any(raw_falltimes < 0.0)

    row = build_didq_row(
        analysis=analysis,
        series_name='synthetic',
        processing_id='test',
    )

    expected_falltime_1 = raw_falltimes[np.argmax(np.abs(raw_falltimes))]
    assert row['didq_2pole_falltime_1'] == pytest.approx(
        expected_falltime_1, rel=1.0e-6
    )
    assert row['didq_2pole_falltime_1'] < 0.0
    assert row['didq_2pole_falltime_1'] == pytest.approx(
        -1.99903375e-01, rel=1.0e-3
    )


def test_build_didq_row_handles_missing_fit():
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(2,), fcutoff_hz=np.inf)

    row = build_didq_row(
        analysis=analysis,
        series_name='synthetic',
        processing_id='test',
        list_of_poles=(2, 3),
    )

    # the three-pole fit was never run, so its columns exist but are NaN
    assert np.isnan(row['didq_3pole_A'])
    assert not row['didq_3pole_fit_success']


def test_build_didq_row_covariance_columns_match_across_fitted_and_unfitted():
    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='synthetic',
    )
    analysis.dofit(list_of_poles=(2,), fcutoff_hz=np.inf)

    row = build_didq_row(
        analysis=analysis,
        series_name='synthetic',
        processing_id='test',
        list_of_poles=(2, 3),
        save_covariance=True,
    )

    # Extract covariance key sets for both pole models
    cov_2pole_keys = {
        k.replace('didq_2pole_', '') for k in row.keys()
        if k.startswith('didq_2pole_cov_')
    }
    cov_3pole_keys = {
        k.replace('didq_3pole_', '') for k in row.keys()
        if k.startswith('didq_3pole_cov_')
    }

    # both pole models should have the same covariance key set
    assert cov_2pole_keys == cov_3pole_keys
    # each should have 28 keys (7x7 upper triangle)
    assert len(cov_2pole_keys) == 28
    assert len(cov_3pole_keys) == 28


def test_rows_to_dataframe_builds_rectangular_frame_with_missing_columns():
    """rows_to_dataframe must not silently corrupt a column that mixes real
    string values in some rows with the NaN fallback used for rows missing
    that key. Every real build_didq_row output shares the same key set, so
    this union/fallback branch is otherwise untested by both the
    fitted-series unit tests and the multi-minute smoke suite."""

    row_1 = {
        'series_name': 'series_one',
        'thermometer_channel': 'ChanA',
        'falltime': 1.0,
    }
    row_2 = {
        'series_name': 'series_two',
        # thermometer_channel intentionally omitted: mixes a string column with
        # the NaN fallback used for a missing key
        'falltime': 2.0,
    }
    row_3 = {
        'series_name': 'series_three',
        'thermometer_channel': 'ChanC',
        # falltime intentionally omitted: ordinary numeric NaN fallback
    }

    dataframe = rows_to_dataframe(rows=[row_1, row_2, row_3])

    # rectangular: every row-derived key becomes a column shared by all rows
    assert dataframe.shape[0] == 3
    for column in ('series_name', 'thermometer_channel', 'falltime'):
        assert column in dataframe.get_column_names()

    # the mixed string/NaN column must land on a genuine missing value at
    # the row that omitted it, not a coerced literal string "nan"
    thermometer_channel = dataframe['thermometer_channel']
    assert thermometer_channel.tolist() == ['ChanA', None, 'ChanC']
    assert thermometer_channel.isna().tolist() == [False, True, False]

    # an ordinary all-numeric column keeps a real NaN at the omitted row
    falltime = dataframe['falltime'].tolist()
    assert falltime[0] == pytest.approx(1.0)
    assert falltime[1] == pytest.approx(2.0)
    assert np.isnan(falltime[2])


# ---------------------------------------------------------------------------
# drive period validation
# ---------------------------------------------------------------------------

def test_validate_drive_period_accepts_a_trace_holding_several_periods():
    """The reference dIdQ datasets hold 2.1 periods per trace and must pass."""

    validate_drive_period_coverage(
        fs=1.25e6,
        nb_samples=656250,
        sgfreq_hz=4.0,
        series_name='I2_D20260719_T145304',
    )


def test_validate_drive_period_accepts_exactly_one_period():
    """One whole period is the minimum qetpy can deconvolve, so it must pass.

    processtraces trims to np.floor(duration * sgfreq) periods, which is 1
    here, so the trimmed trace is non-empty.
    """

    validate_drive_period_coverage(
        fs=1.25e6, nb_samples=1250000, sgfreq_hz=1.0
    )


def test_validate_drive_period_raises_below_one_period():
    """A trace shorter than one drive period must be rejected up front.

    This is the exttrig_I2_D20260728_T132444 configuration: a 1 Hz drive
    recorded in 0.525 s traces. Left to qetpy it trims the trace to zero
    samples and dies with an opaque IndexError inside _deconvolvedidv, but
    only after the multi-minute pile-up cut has already run.
    """

    with pytest.raises(ValueError) as excinfo:
        validate_drive_period_coverage(
            fs=1.25e6,
            nb_samples=656250,
            sgfreq_hz=1.0,
            series_name='I2_D20260728_T132455',
        )

    message = str(excinfo.value)

    # the message has to name the series and both sides of the comparison,
    # so the cause is actionable without reading the source
    assert 'I2_D20260728_T132455' in message
    assert '0.53' in message
    assert '1250000' in message
    assert '656250' in message


def test_validate_drive_period_raises_on_nonpositive_frequency():
    """A zero or negative drive frequency has no period to cover."""

    for bad_frequency in (0.0, -4.0):
        with pytest.raises(ValueError):
            validate_drive_period_coverage(
                fs=1.25e6, nb_samples=656250, sgfreq_hz=bad_frequency
            )


def test_set_traces_rejects_a_trace_shorter_than_one_period():
    """The in-memory entry point must reject a short trace as well.

    set_traces is the path used by notebooks and by the parallel workers, so
    the guard cannot live only in the raw file reader.
    """

    traces = make_synthetic_traces(n_periods=0.5, n_traces=5)
    analysis = DIDQAnalysis(verbose=False)

    with pytest.raises(ValueError) as excinfo:
        analysis.set_traces(
            traces=traces,
            fs=SYNTHETIC_FS,
            drive_params=make_synthetic_drive_params(),
            series_name='short_series',
        )

    assert 'short_series' in str(excinfo.value)


def test_set_traces_still_accepts_a_trace_holding_whole_periods():
    """The guard must not reject the ordinary synthetic ensemble."""

    analysis = DIDQAnalysis(verbose=False)
    analysis.set_traces(
        traces=make_synthetic_traces(n_periods=4, n_traces=5),
        fs=SYNTHETIC_FS,
        drive_params=make_synthetic_drive_params(),
        series_name='good_series',
    )

    assert analysis.get_series_names() == ['good_series']


# ---------------------------------------------------------------------------
# progress accounting
# ---------------------------------------------------------------------------

def test_count_progress_stages_counts_every_dump_separately():
    """Reading is reported per dump, so a series with more dumps costs more.

    A whole series can take minutes, so a per-series read tick leaves the bar
    motionless for most of a run.
    """

    one_dump = count_progress_stages(
        nb_dumps_per_series=[1], list_of_poles=(2, 3)
    )
    three_dumps = count_progress_stages(
        nb_dumps_per_series=[3], list_of_poles=(2, 3)
    )

    # zero cut, pile-up cut, averaging, plus one event per pole model
    assert one_dump == 1 + 3 + 2
    assert three_dumps == 3 + 3 + 2


def test_count_progress_stages_sums_over_series():
    """The total is the sum over series of that series' own dump count."""

    total = count_progress_stages(
        nb_dumps_per_series=[3, 2, 1], list_of_poles=(2, 3)
    )

    assert total == (3 + 5) + (2 + 5) + (1 + 5)


def test_count_progress_stages_tracks_the_pole_count():
    """One fit event per requested pole model."""

    assert count_progress_stages(
        nb_dumps_per_series=[1], list_of_poles=(2,)
    ) == 1 + 3 + 1


# ---------------------------------------------------------------------------
# worker allocation
# ---------------------------------------------------------------------------

def test_resolve_worker_count_caps_at_the_number_of_series(capsys):
    """One core processes one series, so extra cores are pure waste.

    Six cores against the single-series exttrig group spawned five processes
    that never received work.
    """

    nb_workers = resolve_worker_count(ncores=6, nb_series=1, verbose=True)

    assert nb_workers == 1
    assert 'WARNING' in capsys.readouterr().out


def test_resolve_worker_count_is_silent_when_cores_fit():
    """No warning when every worker has a series to process."""

    nb_workers = resolve_worker_count(ncores=3, nb_series=6, verbose=True)

    assert nb_workers == 3


def test_resolve_worker_count_floors_at_one():
    """A nonsensical core count still has to produce a usable pool size."""

    assert resolve_worker_count(ncores=0, nb_series=4, verbose=False) == 1
    assert resolve_worker_count(ncores=-2, nb_series=4, verbose=False) == 1


def make_loaded_analysis_with_series(series_specs, seed_base=1000):
    """
    Build an analysis object holding several synthetic series.

    Parameters
    ----------
    series_specs : list of dict
        One entry per series. Each may carry n_traces, plus drive_params and
        metadata overrides applied on top of the synthetic defaults.
    seed_base : int, optional
        Base seed, offset per series so the ensembles differ.

    Return
    ------
    analysis : DIDQAnalysis
        Analysis holding one entry per requested series.
    series_names : list of str
        Names the series were stored under, in order.
    """

    analysis = DIDQAnalysis(verbose=False)
    series_names = list()

    for index, spec in enumerate(series_specs):
        drive_params = make_synthetic_drive_params()
        drive_params.update(spec.get('drive_params', dict()))

        metadata = {'group_name': 'exttrig_I2_D20260731_T121720',
                    'nb_dumps': 2,
                    'nb_traces_read': 40,
                    'thermometer_channel': 'ChanA',
                    'tes_bias_thermometer': 1.9e-5}
        metadata.update(spec.get('metadata', dict()))

        series_name = 'series_{:d}'.format(index)
        analysis.set_traces(
            traces=make_synthetic_traces(
                n_traces=spec.get('n_traces', 10),
                seed=seed_base + index,
            ),
            fs=SYNTHETIC_FS,
            drive_params=drive_params,
            series_name=series_name,
            metadata=metadata,
        )
        series_names.append(series_name)

    return analysis, series_names


def test_build_pooled_series_name_strips_the_acquisition_prefix():
    """The pooled name reads as an identifier, not as a directory name."""

    assert build_pooled_series_name(
        'exttrig_I2_D20260731_T121720', 'ChanA'
    ) == 'I2_D20260731_T121720_ChanA_pooled'
    assert build_pooled_series_name(
        'cont_I2_D20260731_T121720', 'ChanA'
    ) == 'I2_D20260731_T121720_ChanA_pooled'
    assert build_pooled_series_name(
        'I2_D20260731', 'ChanA') == 'I2_D20260731_ChanA_pooled'


def test_build_pooled_series_name_separates_the_channels():
    """Two channels of one group must not be stored under one name."""

    assert build_pooled_series_name(
        'exttrig_I2_D20260731_T121720', 'ChanA'
    ) != build_pooled_series_name(
        'exttrig_I2_D20260731_T121720', 'ChanB'
    )


def test_build_pooled_metadata_sums_counts_and_records_the_pool():
    """Trace counts add up and what went into the pool is recorded."""

    metadata = build_pooled_metadata(
        series_metadata=[{'nb_dumps': 3, 'nb_traces_read': 440, 'fs': 1.0},
                         {'nb_dumps': 3, 'nb_traces_read': 441, 'fs': 1.0}],
        series_names=['series_a', 'series_b'],
        pooled_name='group_pooled',
    )

    assert metadata['nb_dumps'] == 6
    assert metadata['nb_traces_read'] == 881
    assert metadata['series_name'] == 'group_pooled'
    assert metadata['pooled_series'] == 'series_a,series_b'
    assert metadata['nb_series_pooled'] == 2
    assert metadata['is_pooled'] is True

    # fields shared by every series are carried through untouched
    assert metadata['fs'] == 1.0


def test_pool_series_concatenates_every_trace():
    """The pooled ensemble holds the traces of every series it was given."""

    analysis, series_names = make_loaded_analysis_with_series(
        [{'n_traces': 7}, {'n_traces': 5}, {'n_traces': 9}]
    )

    pooled_names = analysis.pool_series()
    assert len(pooled_names) == 1
    pooled_name = pooled_names[0]

    data = analysis.get_didq_data(pooled_name)
    assert data['n_traces_used'] == 7 + 5 + 9
    assert data['metadata']['nb_series_pooled'] == 3
    assert data['metadata']['pooled_series'] == ','.join(series_names)


def test_pool_series_drops_the_individual_series_by_default():
    """Only the pool can be fitted or saved, so a series cannot leak out."""

    analysis, series_names = make_loaded_analysis_with_series(
        [{'n_traces': 4}, {'n_traces': 4}]
    )

    pooled_names = analysis.pool_series()

    assert analysis.get_series_names() == pooled_names
    for series_name in series_names:
        with pytest.raises(ValueError, match='no dIdQ data'):
            analysis.get_didq_data(series_name)


def test_pool_series_can_keep_the_individual_series():
    """The series survive when the caller asks to keep them."""

    analysis, series_names = make_loaded_analysis_with_series(
        [{'n_traces': 4}, {'n_traces': 4}]
    )

    pooled_names = analysis.pool_series(drop_series=False)

    assert set(analysis.get_series_names()) == set(
        series_names + pooled_names
    )


def test_pool_series_averages_traces_not_per_series_means():
    """
    Pooling has to rebuild the ensemble spread from every trace at once.

    Averaging the per-series mean traces would give the same mean but a
    standard error built from a handful of means rather than from every
    trace, so the fit would be weighted by the wrong uncertainty.
    """

    specs = [{'n_traces': 6}, {'n_traces': 6}, {'n_traces': 6}]

    pooled_analysis, _ = make_loaded_analysis_with_series(specs)
    pooled_name = pooled_analysis.pool_series()[0]
    pooled = pooled_analysis.get_didq_data(pooled_name)

    single_analysis, single_names = make_loaded_analysis_with_series(
        [{'n_traces': 6}]
    )
    single = single_analysis.get_didq_data(single_names[0])

    # qetpy parks the undriven bins at a 1e20 sentinel, so only the bins the
    # square wave actually drives carry a meaningful uncertainty
    pooled_std = np.median(
        np.abs(pooled['didvobj']._didvstd[pooled['driven_mask']])
    )
    single_std = np.median(
        np.abs(single['didvobj']._didvstd[single['driven_mask']])
    )

    # three times the traces, so the standard error of the mean shrinks by
    # about sqrt(3). It cannot be unchanged, which is what averaging the
    # per-series means would have produced
    assert pooled_std == pytest.approx(single_std / np.sqrt(3.0), rel=0.35)
    assert pooled_std < single_std
    assert pooled['didvobj']._ntraces == 18


def test_pool_series_rejects_a_different_bias_point():
    """Series at different bias points are not repeat measurements."""

    analysis, _ = make_loaded_analysis_with_series([
        {'n_traces': 4},
        {'n_traces': 4, 'metadata': {'tes_bias_thermometer': 9.9e-4}},
    ])

    with pytest.raises(ValueError, match='tes_bias_thermometer'):
        analysis.pool_series()


def test_pool_series_rejects_a_different_drive():
    """Series driven differently cannot be averaged together."""

    analysis, _ = make_loaded_analysis_with_series([
        {'n_traces': 4},
        {'n_traces': 4, 'drive_params': {'sgamp_amps': 5.0e-4}},
    ])

    with pytest.raises(ValueError, match='sgamp_amps'):
        analysis.pool_series()


def test_pool_series_pools_each_thermometer_separately():
    """Two thermometers are two measurements, whatever the bias point."""

    analysis, _ = make_loaded_analysis_with_series([
        {'n_traces': 4},
        {'n_traces': 6, 'metadata': {'thermometer_channel': 'ChanB'}},
        {'n_traces': 5},
    ])

    pooled_names = analysis.pool_series()

    assert len(pooled_names) == 2

    pooled_by_channel = {
        analysis.get_didq_data(name)['metadata']['thermometer_channel']: name
        for name in pooled_names
    }
    assert set(pooled_by_channel) == {'ChanA', 'ChanB'}

    # ChanA pooled its two series, ChanB kept its own traces to itself
    assert analysis.get_didq_data(
        pooled_by_channel['ChanA'])['n_traces_used'] == 4 + 5
    assert analysis.get_didq_data(
        pooled_by_channel['ChanB'])['n_traces_used'] == 6


def test_validate_series_are_poolable_accepts_repeat_measurements():
    """Identical configuration passes, which is the common case."""

    series_data = [
        {'drive_params': make_synthetic_drive_params(),
         'metadata': {'tes_bias_thermometer': 1.2e-4},
         'fs': SYNTHETIC_FS,
         'nb_samples': 1000},
        {'drive_params': make_synthetic_drive_params(),
         'metadata': {'tes_bias_thermometer': 1.2e-4},
         'fs': SYNTHETIC_FS,
         'nb_samples': 1000},
    ]

    validate_series_are_poolable(
        series_names=['a', 'b'], series_data=series_data,
    )


def test_validate_series_are_poolable_rejects_mismatched_sample_rate():
    """A different sample rate makes the frequency axes incompatible."""

    series_data = [
        {'drive_params': make_synthetic_drive_params(),
         'metadata': dict(),
         'fs': SYNTHETIC_FS,
         'nb_samples': 1000},
        {'drive_params': make_synthetic_drive_params(),
         'metadata': dict(),
         'fs': 2.0 * SYNTHETIC_FS,
         'nb_samples': 1000},
    ]

    with pytest.raises(ValueError, match='sampled at different rates'):
        validate_series_are_poolable(
            series_names=['a', 'b'], series_data=series_data,
        )


def test_validate_series_are_poolable_rejects_mismatched_trace_length():
    """Traces of different length cannot be concatenated into one ensemble."""

    series_data = [
        {'drive_params': make_synthetic_drive_params(),
         'metadata': dict(),
         'fs': SYNTHETIC_FS,
         'nb_samples': 1000},
        {'drive_params': make_synthetic_drive_params(),
         'metadata': dict(),
         'fs': SYNTHETIC_FS,
         'nb_samples': 2000},
    ]

    with pytest.raises(ValueError, match='different trace lengths'):
        validate_series_are_poolable(
            series_names=['a', 'b'], series_data=series_data,
        )


def test_pooled_row_carries_the_pool_provenance():
    """A pooled row has to say what it was built from."""

    analysis, series_names = make_loaded_analysis_with_series(
        [{'n_traces': 5}, {'n_traces': 5}]
    )
    pooled_name = analysis.pool_series()[0]
    analysis.dofit(list_of_poles=(2,), series_names=[pooled_name])

    row = build_didq_row(
        analysis=analysis,
        series_name=pooled_name,
        processing_id='I2_D20260731_T142224',
        list_of_poles=(2,),
    )

    assert row['series_name'] == pooled_name
    assert row['nb_series_pooled'] == 2
    assert row['pooled_series'] == ','.join(series_names)
    assert row['nb_traces_used'] == 10


class StubH5Reader:
    """
    Stand-in for the pytesio reader, returning a fixed two-channel ensemble.

    Records every read_many_events call so a test can assert the raw files
    were passed over once however many channels were requested.
    """

    calls = list()
    traces_by_channel = dict()

    def get_metadata(self, file_name):
        return {'series_num': 220260731135414,
                'adc_list': ['adc1'],
                'groups': {'adc1': {'sample_rate': SYNTHETIC_FS,
                                    'nb_samples': STUB_NB_SAMPLES,
                                    'nb_events': STUB_NB_TRACES}}}

    def get_detector_config(self, file_name):
        return make_detector_config()

    def read_many_events(self, **kwargs):
        StubH5Reader.calls.append(dict(kwargs))

        channels = kwargs['detector_chans']
        traces = np.stack(
            [StubH5Reader.traces_by_channel[channel] for channel in channels],
            axis=1,
        )
        nevents = kwargs.get('nevents', traces.shape[0])

        return traces[:nevents], [{'series_num': 220260731135414}]


STUB_NB_TRACES = 4
STUB_NB_PERIODS = 2
STUB_NB_SAMPLES = int(STUB_NB_PERIODS * SYNTHETIC_FS / SYNTHETIC_SGFREQ)
STUB_SERIES = 'I2_D20260731_T135414'


@pytest.fixture
def stub_raw_group(tmp_path, monkeypatch):
    """
    Point the reader at a fake two-channel group and return its directory.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.
    monkeypatch : MonkeyPatch
        Pytest monkeypatch fixture.

    Return
    ------
    group_dir : Path
        Directory holding the placeholder dump files.
    """

    from detprocess.core import didq as didq_module

    base = make_synthetic_traces(
        n_periods=STUB_NB_PERIODS, n_traces=STUB_NB_TRACES, seed=7,
    )

    StubH5Reader.calls = list()
    StubH5Reader.traces_by_channel = {'ChanA': base, 'ChanB': 2.0 * base}

    group_dir = tmp_path / 'raw' / 'exttrig_I2_D20260731_T135403'
    group_dir.mkdir(parents=True)
    for index in (1, 2):
        (group_dir / f'exttrig_I2_D20260731_T135414_F{index:04d}.hdf5').touch()

    monkeypatch.setattr(didq_module.h5io, 'H5Reader', StubH5Reader)

    return group_dir


def test_process_raw_data_stores_one_entry_per_channel(stub_raw_group):
    """Each requested channel becomes its own entry, keyed by channel."""

    analysis = DIDQAnalysis(verbose=False)
    analysis.process_raw_data(
        raw_path=str(stub_raw_group),
        thermometer_channels='ChanA,ChanB',
        apply_autocuts=False,
    )

    assert analysis.get_series_names() == [
        build_didq_key(STUB_SERIES, 'ChanA'),
        build_didq_key(STUB_SERIES, 'ChanB'),
    ]


def test_process_raw_data_reads_the_dumps_once_for_every_channel(
        stub_raw_group):
    """Two channels must not cost two passes over the raw files."""

    analysis = DIDQAnalysis(verbose=False)
    analysis.process_raw_data(
        raw_path=str(stub_raw_group),
        thermometer_channels='ChanA,ChanB',
        apply_autocuts=False,
    )

    # two dumps, read once each, with both channels asked for together
    assert len(StubH5Reader.calls) == 2
    for call in StubH5Reader.calls:
        assert call['detector_chans'] == ['ChanA', 'ChanB']


def test_process_raw_data_gives_each_channel_its_own_traces(stub_raw_group):
    """A channel must be fitted on its own plane of the ensemble."""

    analysis = DIDQAnalysis(verbose=False)
    analysis.process_raw_data(
        raw_path=str(stub_raw_group),
        thermometer_channels='ChanA,ChanB',
        apply_autocuts=False,
    )

    for channel in ('ChanA', 'ChanB'):
        data = analysis.get_didq_data(build_didq_key(STUB_SERIES, channel))

        # the stub hands back its whole ensemble for each of the two dumps
        expected = np.concatenate(
            [StubH5Reader.traces_by_channel[channel]] * 2, axis=0
        )
        np.testing.assert_allclose(data['didvobj']._rawtraces, expected)
        assert data['metadata']['thermometer_channel'] == channel

    # the fixture scales ChanB, so a swapped plane would fail the check above
    assert not np.allclose(
        StubH5Reader.traces_by_channel['ChanA'],
        StubH5Reader.traces_by_channel['ChanB'],
        atol=0.0,
    )


def test_process_raw_data_records_the_channel_bias_not_the_other(
        stub_raw_group):
    """Metadata describes the channel it belongs to, and no other."""

    analysis = DIDQAnalysis(verbose=False)
    analysis.process_raw_data(
        raw_path=str(stub_raw_group),
        thermometer_channels='ChanA,ChanB',
        apply_autocuts=False,
    )

    detector_config = make_detector_config()

    for channel in ('ChanA', 'ChanB'):
        metadata = analysis.get_didq_data(
            build_didq_key(STUB_SERIES, channel)
        )['metadata']

        assert metadata['tes_bias_thermometer'] == pytest.approx(
            float(detector_config[channel]['tes_bias'])
        )
        assert 'heater_channel' not in metadata
        assert 'tes_bias_heater' not in metadata


def test_process_raw_data_fits_a_driven_channel_like_any_other(
        stub_raw_group):
    """The channel the square wave is injected into is not treated apart."""

    analysis = DIDQAnalysis(verbose=False)
    analysis.process_raw_data(
        raw_path=str(stub_raw_group),
        thermometer_channels='ChanA',
        apply_autocuts=False,
    )

    # ChanA is the fixture's signal_gen_source "tes" channel
    assert analysis.get_series_names() == [build_didq_key(STUB_SERIES, 'ChanA')]


def test_process_raw_data_raises_on_a_channel_that_was_not_recorded(
        stub_raw_group):
    """A misspelt channel is caught before any trace is read."""

    analysis = DIDQAnalysis(verbose=False)

    with pytest.raises(ValueError, match='NoSuchChannel'):
        analysis.process_raw_data(
            raw_path=str(stub_raw_group),
            thermometer_channels='ChanA,NoSuchChannel',
        )

    assert StubH5Reader.calls == []
