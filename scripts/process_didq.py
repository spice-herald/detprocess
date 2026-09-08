"""
process_didq — Fits the thermal response of a dIdQ measurement.

A dIdQ measurement drives a large-amplitude square wave across a chip and
measures the heat response in a TES on that chip, the thermometer. This script
fits the thermometer response in the frequency domain using the same two-pole
and three-pole models used for dIdV, and reports the raw fitted parameters and
poles. No small signal parameters are calculated.

The channel the square wave is injected into plays no part in the analysis and
is never identified. The thermometer channels to fit are named with --channels
(or -c), which is required. Several may be given, separated by commas or
spaces, and each is read, cut and fitted on its own, giving one row apiece. All
of them come out of a single pass over the raw files.

The series of a dIdQ group are repeat measurements at one bias point, so by
default the traces of every series are pooled into a single ensemble per
channel, averaged together and fitted once. The output is one row per channel
describing the whole group. Pooling happens at the trace level rather than by
averaging the per-series mean traces, so the ensemble standard deviation that
weights the fit is built from every trace at once. Channels never pool
together, however well their bias points agree.

Series that disagree on their bias point, drive or sample rate are not repeat
measurements, and pooling them is rejected rather than silently averaged. Pass
--per_series to average and fit each series on its own instead, which restores
the older one-row-per-series output.

Example usage
-------------
Process every series in a raw group, fitting one channel:

    python scripts/process_didq.py
        --raw_path /sdata1/runs/run74/raw/exttrig_I2_D20260719_T145253
        --channels Mv6Si4pcBigFinsRight

Fit both channels of a chip in one pass over the data:

    python scripts/process_didq.py
        --raw_path /sdata1/runs/run74/raw/exttrig_I2_D20260719_T145253
        --channels Mv6Si4pcBigFinsRight,Mv6Si4pcBigFinsLeft

Restrict to one series and cap the number of traces, for a quick check:

    python scripts/process_didq.py
        --raw_path /sdata1/runs/run74/raw/exttrig_I2_D20260719_T145253
        --channels Mv6Si4pcBigFinsRight
        --series I2_D20260719_T145304
        --nb_events 100

Supply a fit cutoff. The default is infinity, matching dIdV. A cutoff below
about 5 kHz is generally too aggressive:

    python scripts/process_didq.py
        --raw_path /sdata1/runs/run74/raw/exttrig_I2_D20260719_T145253
        --fcutoff_hz 50000

The drive frequency, amplitude and shunt resistance are always read from the
thermometer channel's own acquisition metadata and cannot be overridden. The duty cycle is the exception,
since the acquisition does not record it:

    python scripts/process_didq.py
        --raw_path /sdata1/runs/run74/raw/exttrig_I2_D20260719_T145253
        --duty_cycle 0.25

The three-pole fit starts from the converged two-pole fit of the same channel,
over a short ladder of tau3 values, and keeps the first start that costs no
more than the two-pole fit. Left to itself qetpy starts C at -0.05 and tau3 at
1 ms whatever the data looks like, which leaves a channel whose slow thermal
pole is tens of ms stuck in a bad minimum, so no hand seeding should be needed.

A starting guess can still be given by hand, in the order A B C tau1 tau2 tau3
dt. It is then used exactly as given and nothing else is tried:

    python scripts/process_didq.py
        --raw_path /sdata1/runs/run74/raw/exttrig_I2_D20260731_T135403
        --guess_3poles 16482.45,-16485.82,-2.4044e-4,-1.653e-5,1.4975e-5,6.7015e-2,1.4207e-5

A series is the unit of parallel work, so --ncores is capped at the number of
series in the group and a larger request is warned about. A group holding one
series gains nothing from more than one core:

    python scripts/process_didq.py
        --raw_path /sdata1/runs/run74/raw/exttrig_I2_D20260719_T145253
        --ncores 6

The square wave is deconvolved over a whole number of drive periods, so each
trace must be at least one period long. A group that fails this is rejected
from its file headers alone, in seconds, before any trace is read.

Output
------
Two files are written to <run>/processed/<group_name>/, a sibling of the raw
tree rather than a directory inside it:

    didq_<processing_id>_F0001.hdf5          vaex dataframe, one row per fit
    didq_results_<processing_id>.hdf5        FilterData object with full results

A pooled run writes one row per channel, named after the group and the channel
with a "_pooled" suffix, and records which series went into it under
pooled_series. A --per_series run writes one row per series per channel, named
after each series, with the channel in the thermometer_channel column.

The dataframe leads with fall times. For the three-pole model the raw A, B, C
and tau parameters are start-point dependent and must not be compared across
runs; use the fall times instead.
"""

import argparse
import multiprocessing
import queue
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from detprocess.core.didq import (
    DIDQAnalysis,
    build_didq_row,
    build_pooled_metadata,
    build_pooled_series_name,
    count_progress_stages,
    parse_thermometer_channels,
    rows_to_dataframe,
    validate_series_are_poolable,
)


def get_processing_id(raw_group_name):
    """
    Build a processing id from the facility in the group name and the time now.

    Parameters
    ----------
    raw_group_name : str
        Raw group name, for example exttrig_I2_D20260719_T145253.

    Return
    ------
    processing_id : str
        Processing id of the form I<facility>_DYYYYMMDD_THHMMSS.
    """

    facility = '0'
    if ('_I' in raw_group_name) and ('_D' in raw_group_name):
        try:
            facility = raw_group_name.split('_I')[1].split('_D')[0]
        except IndexError:
            facility = '0'

    now = datetime.now()

    return f'I{facility}_D{now.strftime("%Y%m%d")}_T{now.strftime("%H%M%S")}'


def build_output_directory(raw_path):
    """
    Derive the processed output directory from the raw group directory.

    Processed data is a sibling of the raw tree rather than a child of it, so a
    group read from <run>/raw/<group> is written to <run>/processed/<group>.
    This matches where the feature, trigger and randoms processing write.

    Parameters
    ----------
    raw_path : Path
        Raw group directory, series directory, or a single raw data file.

    Return
    ------
    output_dir : Path
        Processed output directory.
    group_name : str
        Raw group name.
    """

    # a raw_path may be a group directory, a series directory, or a single
    # file within one, so the group directory is the path itself when it is
    # a directory, and its parent otherwise
    group_dir = raw_path if raw_path.is_dir() else raw_path.parent
    group_name = group_dir.name

    # step out of the raw tree when there is one, otherwise the output lands
    # in raw/processed instead of alongside raw. The name is compared as a
    # whole path component so that a directory such as "rawdata" is left alone.
    base_dir = group_dir.parent
    if base_dir.name == 'raw':
        base_dir = base_dir.parent

    output_dir = base_dir / 'processed' / group_name

    return output_dir, group_name


def resolve_worker_count(ncores, nb_series, verbose=True):
    """
    Cap the worker count at the number of series, warning when it was reduced.

    A series is the unit of parallel work, so a pool larger than the number of
    series spawns processes that never receive one. They still fork the parent
    and cost memory, and the "(N cores)" label on the progress bar would claim
    a parallelism the run does not have.

    Parameters
    ----------
    ncores : int
        Number of cores requested.
    nb_series : int
        Number of series available to process.
    verbose : bool, optional
        Print the warning when the count is reduced. Default is True.

    Return
    ------
    nb_workers : int
        Number of worker processes to start.
    """

    requested = int(ncores)
    if requested < 1:
        requested = 1

    nb_workers = min(requested, int(nb_series))

    if (requested > nb_series) and verbose:
        print(f'WARNING: {requested} cores requested but only {nb_series} '
              f'series available. One core processes one series, so '
              f'{nb_workers} will be used and the remaining '
              f'{requested - nb_workers} would sit idle.')

    return nb_workers


def print_series_summary(series_name, n_traces_used, snr_diagnostic,
                         fit_results, list_of_poles):
    """
    Print a short human-readable summary of one series' fit.

    Fall times are printed first, because they are the reproducible output.
    Takes explicit values rather than a DIDQAnalysis object, because the
    parallel branch registers results without keeping the loaded series
    data around on the parent process.

    Parameters
    ----------
    series_name : str
        Series being summarised.
    n_traces_used : int
        Number of traces retained after cuts.
    snr_diagnostic : dict
        Dictionary with keys f_max_snr3_hz and n_freq_bins_snr3.
    fit_results : dict
        Mapping from pole count to fit result dictionary. A pole model that
        was not fitted, or did not produce output, is simply absent or maps
        to an empty dictionary.
    list_of_poles : tuple of int
        Pole models that were requested.

    Return
    ------
    None
    """

    print(f'\n=== {series_name} ===')
    print(f'  traces used      : {n_traces_used}')
    print(f'  highest SNR>3 bin: {snr_diagnostic["f_max_snr3_hz"]} Hz '
          f'({snr_diagnostic["n_freq_bins_snr3"]} bins)')

    for poles in list_of_poles:
        results = fit_results.get(poles)
        if not results:
            print(f'  {poles}-pole         : no result')
            continue

        falltimes = np.asarray(results['falltimes'])
        falltimes = falltimes[np.argsort(np.abs(falltimes))[::-1]]
        formatted = ', '.join(f'{value:.4e}' for value in falltimes)
        flag = ''
        if not results['fit_success']:
            flag = '   [DID NOT CONVERGE]'
        print(f'  {poles}-pole falltimes: [{formatted}] s   '
              f'cost {results["cost"]:.4f}{flag}')


def drain_progress_queue(progress_queue, pending, total, nb_workers):
    """
    Show a stage-level progress bar while parallel workers run.

    With one worker per series, every series starts at once and none finishes
    until near the end, so a bar counting completed series stays at zero for
    almost the whole run. This drains per-stage events instead, so the bar
    advances several times per series.

    Parameters
    ----------
    progress_queue : multiprocessing.managers.BaseProxy
        Queue the workers push (series_name, stage) tuples onto.
    pending : multiprocessing.pool.MapResult
        Handle for the pool work, used to detect completion.
    total : int
        Total number of stage events expected.
    nb_workers : int
        Number of worker processes, shown in the bar description.

    Return
    ------
    outputs : list
        The list of worker return values.
    """

    active = dict()

    with tqdm(total=total, desc=f'dIdQ ({nb_workers} cores)') as bar:

        while True:
            drained_any = False

            # take everything currently queued before touching the bar, so a
            # burst of events costs one redraw rather than one each
            while True:
                try:
                    series_name, stage = progress_queue.get_nowait()
                except queue.Empty:
                    break
                active[series_name] = stage
                bar.update(1)
                drained_any = True

            if drained_any:
                in_flight = ', '.join(
                    f'{name}:{stage}' for name, stage in sorted(active.items())
                )
                bar.set_postfix_str(in_flight[:110])

            if pending.ready() and progress_queue.empty():
                break

            time.sleep(0.2)

        # the workers may have finished between the last drain and the check
        remaining = total - bar.n
        if remaining > 0:
            bar.update(remaining)

    return pending.get()


def read_series_worker(worker_args):
    """
    Read and cut one series, in an isolated process, without fitting it.

    Used by the pooled path, where the fit happens once in the parent over
    every series at once and so cannot be done here. The cut traces are the
    product, and they are far too large to pickle back through the pool, so
    they are spilled to a temporary file and only the path is returned.

    Parameters
    ----------
    worker_args : dict
        Dictionary with keys series_name, spill_dir, progress_queue and
        read_kwargs.

    Return
    ------
    spills : list of dict
        One entry per thermometer channel that survived the cuts, each with
        keys trace_file, series_name, metadata, drive_params, fs and
        nb_samples. Empty when every trace of the series was cut.
    """

    series_name = worker_args['series_name']
    progress_queue = worker_args.get('progress_queue', None)

    def report(stage_series_name, stage):
        """
        Push one progress event to the parent, if a queue was provided.

        Parameters
        ----------
        stage_series_name : str
            Series the stage belongs to.
        stage : str
            Stage that just completed.

        Return
        ------
        None
        """
        if progress_queue is not None:
            progress_queue.put((stage_series_name, stage))

    analysis = DIDQAnalysis(verbose=False)
    analysis.process_raw_data(
        series=[series_name],
        progress_callback=report,
        **worker_args['read_kwargs'],
    )

    spills = list()

    for key in analysis.get_series_names():

        data = analysis.get_didq_data(key)

        trace_file = Path(worker_args['spill_dir']) / f'{key}.npy'
        np.save(trace_file, data['didvobj']._rawtraces)

        spills.append({'trace_file': str(trace_file),
                       'series_name': key,
                       'metadata': dict(data['metadata']),
                       'drive_params': dict(data['drive_params']),
                       'fs': data['fs'],
                       'nb_samples': data['nb_samples']})

    return spills


def process_series_worker(worker_args):
    """
    Read, fit and flatten one series, in an isolated process.

    Only picklable results are returned. The qetpy DIDV object itself is
    deliberately not returned, because it holds the full trace ensemble. The
    small per-series arrays needed to re-plot the fit (mean trace, mean
    transfer function, its standard deviation, and the frequency array) are
    returned separately, since they are cheap to pickle.

    Parameters
    ----------
    worker_args : dict
        Dictionary with keys series_name, processing_id, list_of_poles,
        fcutoff_hz, max_nfev, guess_params_3poles, save_covariance and
        read_kwargs.

    Return
    ------
    outputs : list of dict
        One entry per thermometer channel that survived the cuts, each with
        keys key (the name the channel is stored under), row, fit_results,
        metadata and snr_diagnostic, plus traces holding tmean, didv_mean,
        didv_std and freq. Empty when every trace of the series was cut.
    """

    series_name = worker_args['series_name']
    list_of_poles = worker_args['list_of_poles']
    progress_queue = worker_args.get('progress_queue', None)

    def report(stage_series_name, stage):
        """
        Push one progress event to the parent, if a queue was provided.

        Parameters
        ----------
        stage_series_name : str
            Series the stage belongs to.
        stage : str
            Stage that just completed.

        Return
        ------
        None
        """
        if progress_queue is not None:
            progress_queue.put((stage_series_name, stage))

    analysis = DIDQAnalysis(verbose=False)
    analysis.process_raw_data(
        series=[series_name],
        progress_callback=report,
        **worker_args['read_kwargs'],
    )

    loaded = analysis.get_series_names()
    nb_channels = len(parse_thermometer_channels(
        worker_args['read_kwargs']['thermometer_channels']
    ))

    # a channel whose traces were all cut reported its read stages but will
    # never reach its fit stages, so account for them here or the parent's
    # total is never reached
    for _ in range(nb_channels - len(loaded)):
        for poles in list_of_poles:
            report(series_name, 'fit ' + str(poles) + '-pole')

    if not loaded:
        return list()

    analysis.dofit(
        list_of_poles=list_of_poles,
        progress_callback=report,
        fcutoff_hz=worker_args['fcutoff_hz'],
        max_nfev=worker_args['max_nfev'],
        guess_params_3poles=worker_args['guess_params_3poles'],
    )

    outputs = list()

    for key in loaded:

        row = build_didq_row(
            analysis=analysis,
            series_name=key,
            processing_id=worker_args['processing_id'],
            list_of_poles=list_of_poles,
            save_covariance=worker_args['save_covariance'],
        )

        fit_results = dict()
        for poles in list_of_poles:
            results = analysis.get_fit_results(key, poles)
            if results:
                fit_results[poles] = results

        data = analysis.get_didq_data(key)
        metadata = dict(data['metadata'])
        metadata.update(data['drive_params'])
        metadata['fs'] = data['fs']
        metadata['nb_samples'] = data['nb_samples']
        metadata['n_traces_used'] = data['n_traces_used']

        didvobj = data['didvobj']
        traces = {'tmean': didvobj._tmean,
                  'didv_mean': didvobj._didvmean,
                  'didv_std': didvobj._didvstd,
                  'freq': didvobj._freq}

        outputs.append({'key': key,
                        'row': row,
                        'fit_results': fit_results,
                        'metadata': metadata,
                        'traces': traces,
                        'snr_diagnostic': data['snr_diagnostic']})

    return outputs


def build_pooled_ensemble_in_parallel(analysis, group_name, worker_args_list,
                                      progress_total, ncores, verbose):
    """
    Read every series in parallel and pool the result, one ensemble per channel.

    The traces are the product of the read, and a full ensemble is far too
    large to pickle back from a worker, so each worker spills its cut traces
    to a temporary file and the parent concatenates them. The temporary files
    are removed before returning, whether or not the pooling succeeded.

    Parameters
    ----------
    analysis : DIDQAnalysis
        Analysis object the pooled ensemble is registered on.
    group_name : str
        Raw group name, used to name the pooled ensemble.
    worker_args_list : list of dict
        Per-series worker arguments, as built by process_didq.
    progress_total : int
        Number of read stage events expected across all workers.
    ncores : int
        Requested number of worker processes.
    verbose : bool
        Print progress.

    Return
    ------
    pooled_names : list of str
        Names the pooled ensembles are stored under, one per channel.
    """

    nb_workers = resolve_worker_count(
        ncores=ncores, nb_series=len(worker_args_list), verbose=verbose,
    )

    spill_dir = tempfile.mkdtemp(prefix='didq_pool_')

    try:
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        for worker_args in worker_args_list:
            worker_args['progress_queue'] = progress_queue
            worker_args['spill_dir'] = spill_dir

        with multiprocessing.Pool(processes=nb_workers) as pool:
            pending = pool.map_async(read_series_worker, worker_args_list)
            outputs = drain_progress_queue(
                progress_queue=progress_queue,
                pending=pending,
                total=progress_total,
                nb_workers=nb_workers,
            )

        spills = list()
        for entry in outputs:
            spills.extend(entry)

        if not spills:
            raise ValueError(
                'ERROR: every trace was cut, in every series. There is '
                'nothing to pool.'
            )

        # a channel is pooled only with itself, so the spills are grouped
        # before anything is concatenated
        spills_by_channel = dict()
        for entry in spills:
            channel = str(entry['metadata']['thermometer_channel'])
            spills_by_channel.setdefault(channel, list()).append(entry)

        pooled_names = list()

        for channel, channel_spills in spills_by_channel.items():

            pooled_series_names = [
                entry['series_name'] for entry in channel_spills
            ]

            validate_series_are_poolable(
                series_names=pooled_series_names, series_data=channel_spills,
            )

            pooled_name = build_pooled_series_name(
                group_name=group_name, channel=channel,
            )
            pooled_metadata = build_pooled_metadata(
                series_metadata=[
                    entry['metadata'] for entry in channel_spills
                ],
                series_names=pooled_series_names,
                pooled_name=pooled_name,
            )

            # memory mapped so the spilled traces are read straight into the
            # concatenated ensemble, rather than every series being held in
            # full alongside it
            pooled_traces = np.concatenate(
                [np.load(entry['trace_file'], mmap_mode='r')
                 for entry in channel_spills],
                axis=0,
            )

            if verbose:
                print(f'INFO: pooling {len(channel_spills)} series of channel '
                      f'{channel} into {pooled_traces.shape[0]} traces, '
                      f'stored as {pooled_name}')

            analysis.set_traces(
                traces=pooled_traces,
                fs=channel_spills[0]['fs'],
                drive_params=channel_spills[0]['drive_params'],
                series_name=pooled_name,
                metadata=pooled_metadata,
            )

            pooled_names.append(pooled_name)

    finally:
        shutil.rmtree(spill_dir, ignore_errors=True)

    return pooled_names


def finish_pooled_run(analysis, group_name, worker_args_list, processing_id,
                      poles, fcutoff_hz, max_nfev, guess_params_3poles,
                      save_covariance, ncores, nb_channels,
                      serial_bar, progress_total, output_file, results_file,
                      verbose):
    """
    Pool every series per channel, fit each once and write the output.

    Parameters
    ----------
    analysis : DIDQAnalysis
        Analysis object. In a serial run it already holds every series.
    group_name : str
        Raw group name, used to name the pooled ensemble.
    worker_args_list : list of dict
        Per-series worker arguments, used only by the parallel path.
    processing_id : str
        Processing identifier recorded on the output row.
    poles : tuple of int
        Pole models to fit.
    fcutoff_hz : float
        Fit cutoff, in Hz.
    max_nfev : int
        Maximum fit iterations.
    guess_params_3poles : tuple of float or None
        Starting guess for the three-pole fit, as
        (A, B, C, tau1, tau2, tau3, dt). None leaves the guess to qetpy.
    save_covariance : bool
        Write the flattened covariance matrix.
    ncores : int
        Requested number of worker processes.
    nb_channels : int
        Number of thermometer channels being fitted.
    serial_bar : tqdm or None
        Progress bar of the serial read, already advanced through it.
    progress_total : int
        Total number of stage events for the whole run.
    output_file : Path
        Dataframe file to write.
    results_file : Path
        Results file to write.
    verbose : bool
        Print progress.

    Return
    ------
    output_file : Path
        Path to the written dataframe file.
    """

    nb_fit_stages = len(poles) * nb_channels
    nb_read_stages = progress_total - nb_fit_stages

    if ncores == 1:
        # the serial path already read every series into the analysis object
        pooled_names = analysis.pool_series()
    else:
        pooled_names = build_pooled_ensemble_in_parallel(
            analysis=analysis,
            group_name=group_name,
            worker_args_list=worker_args_list,
            progress_total=nb_read_stages,
            ncores=ncores,
            verbose=verbose,
        )

    fit_bar = serial_bar
    if fit_bar is None:
        fit_bar = tqdm(total=nb_fit_stages, desc='dIdQ (pooled fit)')

    def report_fit(stage_series_name, stage):
        """
        Advance the fit progress bar by one stage.

        Parameters
        ----------
        stage_series_name : str
            Series the stage belongs to.
        stage : str
            Stage that just completed.

        Return
        ------
        None
        """
        fit_bar.set_postfix_str(f'{stage_series_name} {stage}')
        fit_bar.update(1)

    analysis.dofit(
        list_of_poles=poles,
        series_names=pooled_names,
        fcutoff_hz=fcutoff_hz,
        max_nfev=max_nfev,
        guess_params_3poles=guess_params_3poles,
        progress_callback=report_fit,
    )

    remaining_stages = fit_bar.total - fit_bar.n
    if remaining_stages > 0:
        fit_bar.update(remaining_stages)
    fit_bar.close()

    rows = list()

    for pooled_name in pooled_names:

        if verbose:
            fit_results = {
                fit_poles: analysis.get_fit_results(pooled_name, fit_poles)
                for fit_poles in poles
            }
            data = analysis.get_didq_data(pooled_name)
            print_series_summary(
                series_name=pooled_name,
                n_traces_used=data['n_traces_used'],
                snr_diagnostic=data['snr_diagnostic'],
                fit_results=fit_results,
                list_of_poles=poles,
            )

        rows.append(build_didq_row(
            analysis=analysis,
            series_name=pooled_name,
            processing_id=processing_id,
            list_of_poles=poles,
            save_covariance=save_covariance,
        ))

    dataframe = rows_to_dataframe(rows=rows)
    dataframe.export_hdf5(str(output_file), mode='w')

    # only the pooled ensembles are registered, so the individual series never
    # reach the output and cannot be read back as if they had been fitted
    analysis.save_didq_data(
        series_names=pooled_names,
        file_path_name=str(results_file),
        save_hdf5=True,
    )

    if verbose:
        print(f'\nSaved dataframe    : {output_file}')
        if results_file.exists():
            print(f'Saved didq_results : {results_file}')

    return output_file


def process_didq(raw_path, thermometer_channels, series=None, nb_events=None,
                 poles=(2, 3), fcutoff_hz=np.inf, duty_cycle=0.5,
                 max_nfev=5000, guess_params_3poles=None,
                 output_path=None, save_covariance=False,
                 apply_autocuts=True, ncores=1, pool_series=True,
                 verbose=True):
    """
    Run dIdQ processing over every series of a raw group.

    Parameters
    ----------
    raw_path : str
        Raw group directory, series directory, or a single raw data file.
    thermometer_channels : str or list of str
        Thermometer channels to fit, separated by commas or spaces when given
        as one string. Each is fitted on its own and produces its own rows.
    series : str or list of str, optional
        Restrict to these series names.
    nb_events : int, optional
        Cap the traces read per series.
    poles : tuple of int, optional
        Pole models to fit. Default is (2, 3).
    fcutoff_hz : float, optional
        Fit cutoff, in Hz. Default is infinity, matching dIdV.
    duty_cycle : float, optional
        Square wave duty cycle. Default is 0.5.
    max_nfev : int, optional
        Maximum fit iterations. Default is 5000.
    guess_params_3poles : tuple of float, optional
        Starting guess for the three-pole fit, as
        (A, B, C, tau1, tau2, tau3, dt). Default leaves the guess to qetpy.
    output_path : str, optional
        Override the output directory.
    save_covariance : bool, optional
        Write the flattened covariance matrix. Default is False.
    apply_autocuts : bool, optional
        Apply the same cuts dIdV applies. Default is True.
    ncores : int, optional
        Number of series read in parallel. Default is 1.
    pool_series : bool, optional
        Pool the traces of every series into one ensemble per channel and fit
        each once, giving one row per channel. Default is True. Set False to
        average and fit each series on its own, giving one row per series per
        channel.
    verbose : bool, optional
        Print progress. Default is True.

    Return
    ------
    output_file : Path
        Path to the written dataframe file.
    """

    channels = parse_thermometer_channels(thermometer_channels)

    raw_dir = Path(raw_path).expanduser().resolve()
    if not (raw_dir.is_dir() or raw_dir.is_file()):
        raise ValueError(f'ERROR: raw path does not exist: {raw_dir}')

    output_dir, group_name = build_output_directory(raw_dir)
    if output_path is not None:
        output_dir = Path(output_path).expanduser().resolve()

    processing_id = get_processing_id(group_name)
    output_file = output_dir / f'didq_{processing_id}_F0001.hdf5'
    results_file = output_dir / f'didq_results_{processing_id}.hdf5'

    analysis = DIDQAnalysis(verbose=verbose)

    # cheap: reads only file metadata headers, no trace data. Needed up front
    # in both paths so the progress bar knows its total before any slow work
    # begins. Scanned once here and handed to the workers, so they do not each
    # rescan the whole group to find their own series.
    series_files = analysis.get_series_files_from_path(
        raw_path=str(raw_dir), series=series
    )
    series_names = list(series_files.keys())

    if not series_names:
        raise ValueError('ERROR: no series were loaded.')

    nb_dumps_per_series = [len(series_files[name]) for name in series_names]

    # reject a group that cannot be deconvolved before any trace is read and
    # before the pool is started, so the failure is a plain error rather than
    # one wrapped in a worker traceback minutes into the run
    analysis.validate_drive_configuration(
        series_files=series_files,
        thermometer_channels=channels,
        duty_cycle=duty_cycle,
    )

    # created only once the group is known to be processable, so a rejected
    # run leaves no empty directory behind
    output_dir.mkdir(parents=True, exist_ok=True)

    # a pooled run fits once at the end rather than once per series, so the
    # per-series fit stages leave the progress total
    if pool_series:
        progress_total = count_progress_stages(
            nb_dumps_per_series=nb_dumps_per_series,
            list_of_poles=(),
            nb_channels=len(channels),
        ) + len(tuple(poles)) * len(channels)
    else:
        progress_total = count_progress_stages(
            nb_dumps_per_series=nb_dumps_per_series,
            list_of_poles=tuple(poles),
            nb_channels=len(channels),
        )

    serial_bar = None

    if ncores == 1:
        serial_bar = tqdm(
            total=progress_total,
            desc='dIdQ (serial)',
        )

        def report_serial(stage_series_name, stage):
            """
            Advance the serial progress bar by one stage.

            Parameters
            ----------
            stage_series_name : str
                Series the stage belongs to.
            stage : str
                Stage that just completed.

            Return
            ------
            None
            """
            serial_bar.set_postfix_str(f'{stage_series_name} {stage}')
            serial_bar.update(1)

        analysis.process_raw_data(
            raw_path=str(raw_dir),
            thermometer_channels=channels,
            series=series,
            nb_events=nb_events,
            duty_cycle=duty_cycle,
            apply_autocuts=apply_autocuts,
            progress_callback=report_serial,
        )
        # the loaded names are per-channel keys, not bare series names
        loaded_keys = analysis.get_series_names()

        if not loaded_keys:
            raise ValueError('ERROR: no series were loaded.')

    worker_args_list = list()
    for series_name in series_names:
        # each worker is given only its own dump files, so process_raw_data
        # does not reread the header of every file in the group
        read_kwargs = {'raw_path': list(series_files[series_name]),
                       'thermometer_channels': channels,
                       'nb_events': nb_events,
                       'duty_cycle': duty_cycle,
                       'apply_autocuts': apply_autocuts}

        worker_args_list.append({
            'series_name': series_name,
            'processing_id': processing_id,
            'list_of_poles': tuple(poles),
            'fcutoff_hz': fcutoff_hz,
            'max_nfev': max_nfev,
            'guess_params_3poles': guess_params_3poles,
            'save_covariance': save_covariance,
            'read_kwargs': read_kwargs,
        })

    rows = list()

    if pool_series:
        return finish_pooled_run(
            analysis=analysis,
            group_name=group_name,
            worker_args_list=worker_args_list,
            processing_id=processing_id,
            poles=tuple(poles),
            fcutoff_hz=fcutoff_hz,
            max_nfev=max_nfev,
            guess_params_3poles=guess_params_3poles,
            save_covariance=save_covariance,
            ncores=ncores,
            nb_channels=len(channels),
            serial_bar=serial_bar,
            progress_total=progress_total,
            output_file=output_file,
            results_file=results_file,
            verbose=verbose,
        )

    if ncores > 1:
        # each series is independent, so it can be read and fitted in its own
        # process. The parent keeps only the returned results.
        #
        # Progress is reported per stage rather than per series. With one
        # worker per series every series starts at once and none finishes
        # until near the end, so a per-series bar sits at 0 for the whole
        # run. Workers push stage events onto a shared queue and the parent
        # drains it here.
        nb_workers = resolve_worker_count(
            ncores=ncores, nb_series=len(series_names), verbose=verbose,
        )

        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        for worker_args in worker_args_list:
            worker_args['progress_queue'] = progress_queue

        with multiprocessing.Pool(processes=nb_workers) as pool:
            pending = pool.map_async(process_series_worker, worker_args_list)
            outputs = drain_progress_queue(
                progress_queue=progress_queue,
                pending=pending,
                total=progress_total,
                nb_workers=nb_workers,
            )

        for worker_output in outputs:
            for entry in worker_output:

                rows.append(entry['row'])
                fit_results = entry['fit_results']
                metadata = entry['metadata']

                # keyed by the per-channel name, or the two channels of one
                # series would overwrite each other in the results file
                for fit_poles, results in fit_results.items():
                    analysis.set_didq_results(
                        entry['key'], results, fit_poles,
                        metadata=dict(metadata),
                    )

                # traces are saved alongside the fit output they support, so a
                # series with no successful fit does not gain an orphaned entry
                if fit_results:
                    analysis.set_didq_traces(
                        entry['key'],
                        metadata=dict(metadata),
                        **entry['traces'],
                    )

                if verbose:
                    print_series_summary(
                        series_name=entry['key'],
                        n_traces_used=metadata['n_traces_used'],
                        snr_diagnostic=entry['snr_diagnostic'],
                        fit_results=fit_results,
                        list_of_poles=tuple(poles),
                    )
    else:
        # serial_bar already counted this run's read and cut stages while
        # process_raw_data ran; the fits continue on the same bar
        for series_name in loaded_keys:
            analysis.dofit(
                list_of_poles=tuple(poles),
                fcutoff_hz=fcutoff_hz,
                max_nfev=max_nfev,
                guess_params_3poles=guess_params_3poles,
                series_names=[series_name],
                progress_callback=report_serial,
            )

            if verbose:
                fit_results = {
                    fit_poles: analysis.get_fit_results(series_name, fit_poles)
                    for fit_poles in tuple(poles)
                }
                print_series_summary(
                    series_name=series_name,
                    n_traces_used=analysis.get_didq_data(
                        series_name
                    )['n_traces_used'],
                    snr_diagnostic=analysis.get_didq_data(
                        series_name
                    )['snr_diagnostic'],
                    fit_results=fit_results,
                    list_of_poles=tuple(poles),
                )

            rows.append(build_didq_row(
                analysis=analysis,
                series_name=series_name,
                processing_id=processing_id,
                list_of_poles=tuple(poles),
                save_covariance=save_covariance,
            ))

    if serial_bar is not None:
        # a series whose traces were all cut never reaches its fit stages, so
        # top the bar up rather than leaving it short of 100 percent
        remaining_stages = serial_bar.total - serial_bar.n
        if remaining_stages > 0:
            serial_bar.update(remaining_stages)
        serial_bar.close()

    if not rows:
        raise ValueError('ERROR: no series produced results.')

    dataframe = rows_to_dataframe(rows=rows)
    dataframe.export_hdf5(str(output_file), mode='w')

    if ncores > 1:
        # results were already registered from the worker output above
        analysis.save_hdf5(str(results_file), overwrite=True)
    else:
        analysis.save_didq_data(
            file_path_name=str(results_file),
            save_hdf5=True,
        )

    if verbose:
        print(f'\nSaved dataframe    : {output_file}')
        # save_didq_data (serial branch) writes nothing when no series
        # produced a saveable fit, so the message must match reality
        if results_file.exists():
            print(f'Saved didq_results : {results_file}')

    return output_file


def parse_guess_3poles(text):
    """
    Parse a three-pole starting guess given as comma separated values.

    Taken as one string rather than seven separate arguments because a
    negative value written in scientific notation, which these parameters
    routinely are, is read as an option name by argparse.

    Parameters
    ----------
    text : str
        Seven comma separated numbers, in the order A, B, C, tau1, tau2,
        tau3, dt.

    Return
    ------
    guess : tuple of float
        The seven values, in the order given.
    """

    fields = [field.strip() for field in text.split(',')]

    if len(fields) != 7:
        raise argparse.ArgumentTypeError(
            f'expecting 7 comma separated values in the order '
            f'A,B,C,tau1,tau2,tau3,dt, got {len(fields)}'
        )

    try:
        guess = tuple(float(field) for field in fields)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f'could not read every value as a number: {error}'
        )

    return guess


def parse_args():
    """
    Parse command-line arguments for dIdQ processing.

    Parameters
    ----------
    None

    Return
    ------
    args : argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description='Process dIdQ raw data into fitted thermal poles.'
    )
    parser.add_argument('--raw_path', type=str, required=True,
                        help='Raw group directory, series directory, or a '
                             'single raw data file.')
    parser.add_argument('-c', '--channels', type=str, nargs='+',
                        required=True,
                        help=('Thermometer channels to fit, separated by '
                              'commas or spaces. Each is read, cut and '
                              'fitted on its own and produces its own row. '
                              'The channel the square wave is injected into '
                              'is never named and plays no part in the '
                              'analysis.'))
    parser.add_argument('--series', type=str, nargs='+', default=None,
                        help='(optional) Restrict processing to these series '
                             'names.')
    parser.add_argument('--nb_events', type=int, default=None,
                        help='(optional) Cap the traces read per series.')
    parser.add_argument('--poles', type=int, nargs='+', default=[2, 3],
                        choices=[1, 2, 3],
                        help=('(optional) Pole models to fit. Default is 2 '
                              'and 3.'))
    parser.add_argument('--fcutoff_hz', type=float, default=np.inf,
                        help=('(optional) Fit cutoff in Hz. Default is '
                              'infinity, matching dIdV. A cutoff below about '
                              '5 kHz is generally too aggressive.'))
    parser.add_argument('--duty_cycle', type=float, default=0.5,
                        help=('(optional) Square wave duty cycle. Default is '
                              '0.5. This is the only drive parameter not '
                              'recorded by the acquisition.'))
    parser.add_argument('--max_nfev', type=int, default=5000,
                        help=('(optional) Maximum fit iterations. Default is '
                              '5000.'))
    parser.add_argument('--guess_3poles', type=parse_guess_3poles,
                        default=None, metavar='A,B,C,TAU1,TAU2,TAU3,DT',
                        help=('(optional) Starting guess for the three-pole '
                              'fit, given as seven comma separated values in '
                              'the order A,B,C,tau1,tau2,tau3,dt. Commas '
                              'rather than spaces, because a negative value '
                              'in scientific notation is otherwise read as an '
                              'option. Given one, the fit starts there and '
                              'nowhere else. The default is to start from the '
                              'converged two-pole fit of the same channel, '
                              'which reaches a slow thermal pole that qetpy\'s '
                              'own starting point strands.'))
    parser.add_argument('--output_path', type=str, default=None,
                        help='(optional) Override the output directory.')
    parser.add_argument('--save_covariance', action='store_true',
                        help=('(optional) Write the flattened covariance '
                              'matrix.'))
    parser.add_argument('--no_autocuts', action='store_true',
                        help='(optional) Skip the dIdV pile-up cuts.')
    parser.add_argument('--ncores', type=int, default=1,
                        help=('(optional) Number of series read in parallel. '
                              'One core reads one series, so a value '
                              'above the number of series is capped. '
                              'Default is 1.'))
    parser.add_argument('--per_series', action='store_true',
                        help=('(optional) Average and fit each series on its '
                              'own, giving one row per series per channel. '
                              'The default pools the traces of every series '
                              'into one ensemble per channel and fits each '
                              'once, giving one row per channel.'))

    return parser.parse_args()


def main():
    """
    Run dIdQ processing from the command line.

    Parameters
    ----------
    None

    Return
    ------
    None
    """

    args = parse_args()

    output_file = process_didq(
        raw_path=args.raw_path,
        series=args.series,
        nb_events=args.nb_events,
        poles=tuple(args.poles),
        fcutoff_hz=args.fcutoff_hz,
        thermometer_channels=args.channels,
        duty_cycle=args.duty_cycle,
        max_nfev=args.max_nfev,
        guess_params_3poles=args.guess_3poles,
        output_path=args.output_path,
        save_covariance=args.save_covariance,
        apply_autocuts=not args.no_autocuts,
        ncores=args.ncores,
        pool_series=not args.per_series,
    )

    print(f'Saved processed dIdQ data to: {output_file}')


if __name__ == '__main__':
    main()
