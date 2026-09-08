"""End-to-end smoke test for dIdQ processing.

Skipped automatically when the reference dataset is not mounted, so the test
suite still runs on machines without access to detector data.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import vaex as vx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

RAW_PATH = '/sdata1/runs/run74/raw/exttrig_I2_D20260719_T145253'

pytestmark = pytest.mark.skipif(
    not os.path.isdir(RAW_PATH),
    reason=f'reference dataset not available at {RAW_PATH}',
)


@pytest.mark.filterwarnings(
    'ignore:Conversion of an array with ndim > 0 to a scalar is deprecated:'
    'DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:visit_NameConstant is deprecated:DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:`product` is deprecated as of NumPy 1.25.0:DeprecationWarning'
)
@pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
@pytest.mark.filterwarnings(
    'ignore:invalid value encountered in sqrt:RuntimeWarning'
)
@pytest.mark.filterwarnings(
    'ignore:`np.bool8` is a deprecated alias:DeprecationWarning'
)
def test_process_didq_end_to_end(tmp_path):
    from process_didq import process_didq

    output_file = process_didq(
        raw_path=RAW_PATH,
        thermometer_channels='Mv6GaAs4pcBigFinsLeft',
        nb_events=20,
        fcutoff_hz=200.0,
        output_path=str(tmp_path),
        verbose=False,
    )

    assert output_file.exists()

    dataframe = vx.open(str(output_file))

    # the reference group holds six series
    assert dataframe.shape[0] == 6

    for column in ('series_name', 'thermometer_channel',
                   'sgfreq_hz', 'didq_2pole_falltime_1',
                   'didq_3pole_falltime_1'):
        assert column in dataframe.get_column_names()

    assert 'heater_channel' not in dataframe.get_column_names()
    assert (dataframe['thermometer_channel'].tolist()[0]
            == 'Mv6GaAs4pcBigFinsLeft')
    assert dataframe['sgfreq_hz'].tolist()[0] == pytest.approx(4.0)


@pytest.mark.filterwarnings(
    'ignore:Conversion of an array with ndim > 0 to a scalar is deprecated:'
    'DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:visit_NameConstant is deprecated:DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:`product` is deprecated as of NumPy 1.25.0:DeprecationWarning'
)
@pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
@pytest.mark.filterwarnings(
    'ignore:invalid value encountered in sqrt:RuntimeWarning'
)
@pytest.mark.filterwarnings(
    'ignore:`np.bool8` is a deprecated alias:DeprecationWarning'
)
def test_process_didq_fits_both_channels_of_a_chip(tmp_path):
    from process_didq import process_didq

    channels = ['Mv6GaAs4pcBigFinsLeft', 'Mv6GaAs4pcBigFinsRight']

    output_file = process_didq(
        raw_path=RAW_PATH,
        thermometer_channels=','.join(channels),
        series=['I2_D20260719_T145304'],
        nb_events=20,
        fcutoff_hz=200.0,
        output_path=str(tmp_path),
        verbose=False,
    )

    dataframe = vx.open(str(output_file))

    # one pooled row per channel, and the channel driving the square wave is
    # fitted like any other
    assert dataframe.shape[0] == 2
    assert sorted(dataframe['thermometer_channel'].tolist()) == sorted(channels)

    falltimes = dataframe['didq_2pole_falltime_1'].tolist()
    assert falltimes[0] != falltimes[1]


@pytest.mark.filterwarnings(
    'ignore:Conversion of an array with ndim > 0 to a scalar is deprecated:'
    'DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:visit_NameConstant is deprecated:DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:`product` is deprecated as of NumPy 1.25.0:DeprecationWarning'
)
@pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
def test_process_didq_single_series_matches_design_measurement(tmp_path):
    from process_didq import process_didq

    output_file = process_didq(
        raw_path=RAW_PATH,
        thermometer_channels='Mv6GaAs4pcBigFinsLeft',
        series=['I2_D20260719_T145304'],
        nb_events=100,
        fcutoff_hz=200.0,
        output_path=str(tmp_path),
        verbose=False,
    )

    dataframe = vx.open(str(output_file))
    assert dataframe.shape[0] == 1

    # design measurement: dominant two-pole fall time about 4.70e-02 s.
    # the tolerance is loose because autocuts may retain a different subset.
    falltime = dataframe['didq_2pole_falltime_1'].tolist()[0]
    assert falltime == pytest.approx(4.70e-02, rel=0.15)


@pytest.mark.filterwarnings(
    'ignore:Conversion of an array with ndim > 0 to a scalar is deprecated:'
    'DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:visit_NameConstant is deprecated:DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:`product` is deprecated as of NumPy 1.25.0:DeprecationWarning'
)
@pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning')
@pytest.mark.filterwarnings(
    'ignore:invalid value encountered in sqrt:RuntimeWarning'
)
def test_parallel_matches_serial(tmp_path):
    from process_didq import process_didq

    serial_file = process_didq(
        raw_path=RAW_PATH,
        thermometer_channels='Mv6GaAs4pcBigFinsLeft',
        nb_events=20,
        fcutoff_hz=200.0,
        output_path=str(tmp_path / 'serial'),
        ncores=1,
        verbose=False,
    )
    parallel_file = process_didq(
        raw_path=RAW_PATH,
        thermometer_channels='Mv6GaAs4pcBigFinsLeft',
        nb_events=20,
        fcutoff_hz=200.0,
        output_path=str(tmp_path / 'parallel'),
        ncores=3,
        verbose=False,
    )

    serial = vx.open(str(serial_file)).sort('series_name')
    parallel = vx.open(str(parallel_file)).sort('series_name')

    assert serial.shape[0] == parallel.shape[0]

    for column in ('didq_2pole_falltime_1', 'didq_3pole_falltime_1'):
        np.testing.assert_allclose(
            np.asarray(serial[column].tolist()),
            np.asarray(parallel[column].tolist()),
            rtol=1.0e-8,
        )


# The 2026-07-28 exttrig group was acquired with a 1 Hz drive in 0.525 s
# traces, so it holds only half a square wave period. It is kept as the
# regression case for the fail-fast check: before that check existed, this
# group spent minutes in the pile-up cut and then died inside qetpy with an
# IndexError that named neither the series nor the drive frequency.
SHORT_PERIOD_RAW_PATH = ('/sdata1/runs/run74/raw/'
                         'exttrig_I2_D20260728_T132444')


@pytest.mark.skipif(
    not os.path.isdir(SHORT_PERIOD_RAW_PATH),
    reason=f'short-period dataset not available at {SHORT_PERIOD_RAW_PATH}',
)
def test_process_didq_rejects_a_group_shorter_than_one_drive_period(tmp_path):
    import time

    from process_didq import process_didq

    output_dir = tmp_path / 'processed'

    start_time = time.time()

    with pytest.raises(ValueError) as excinfo:
        process_didq(
            raw_path=SHORT_PERIOD_RAW_PATH,
            thermometer_channels='Mv6GaAs4pcBigFinsLeft',
            output_path=str(output_dir),
            ncores=6,
            verbose=False,
        )

    elapsed_s = time.time() - start_time
    message = str(excinfo.value)

    # the point of the check is that it reads headers only, so it must fail
    # long before the pile-up cut could have run
    assert elapsed_s < 60.0

    assert 'I2_D20260728_T132455' in message
    assert '0.53' in message
    assert '1250000' in message

    # nothing may be written for a group that cannot be processed, not even
    # the output directory
    assert not output_dir.exists()
