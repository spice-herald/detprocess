"""
process_transducer_sweep — Fits the complex sinusoidal amplitudes from raw transducer sweep data.

Each raw sweep directory contains one HDF5 file per frequency step.
This script reads each file, fits A*cos(wt) + B*sin(wt) to every trace in every channel,
and saves the complex amplitudes to a single processed HDF5 file (as Vaex DataFrame).

Expects as input a raw sweep group directory containing HDF5 files, acquired through detprocess run_transducer_sweep.py:

Example usage
-----
Basic (process all frequency steps, use native trace length):

    python process_transducer_sweep.py --raw_path /sdata2/runs/run69/raw/transducer_sweep_I2_D20260227_T163327

Override trace length (concatenate events, re-chop to 9000 ms segments. This overrides the native trace length defined in the run_transducer_sweep.py daq config):

    python process_transducer_sweep.py
        --raw_path /sdata2/runs/run69/raw/transducer_sweep_I2_D20260227_T163327
        --trace_length_override_msec 9000

Downsample in frequency (process every 10th frequency step recorded — 10x faster):

    python process_transducer_sweep.py
        --raw_path /sdata2/runs/run69/raw/transducer_sweep_I2_D20260227_T163327
        --freq_step 10

Downsample to a target number of frequency steps (e.g. ~60 steps total, evenly spaced across the recorded sweep steps):

    python scripts/process_transducer_sweep.py
        --raw_path /sdata2/runs/run69/raw/transducer_sweep_I2_D20260227_T163327
        --n_freq_steps 60

Override accelerometer gain (instead of reading from metadata. Useful if wrong gain was used during acquisition):

    python scripts/process_transducer_sweep.py
        --raw_path /sdata2/runs/run68/raw/accel_freq_sweep_I2_D20260206_T084123
        --accel_gain 100

Output
------
Processed file is written to:
    <raw_parent>/../processed/<group_name>/amplitude_<processing_id>_F0001.hdf5

The output DataFrame contains columns:
    frequency_hz, signal_gen_voltage, trace_index, file_name,
    sample_rate_hz, trace_length_msec, accel_gain,
    amp_real_<channel>, amp_imag_<channel>, amp_complex_<channel>

Each row represents a single frequency step.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import vaex as vx
from tqdm import tqdm

import pytesdaq.io as h5io


def _get_processing_id(raw_group_name: str) -> str:
    """
    Build a processing id using the facility from the raw group name and
    the current datetime.

    Parameters
    ----
    raw_group_name : str
        Raw group name, e.g. transducer_sweep_I2_D20260205_T162140.

    Return
    ---
    processing_id : str
        Processing id in the form I<facility>_DYYYYMMDD_THHMMSS.
    """
    facility = "0"
    if "_I" in raw_group_name and "_D" in raw_group_name:
        try:
            facility = raw_group_name.split("_I")[1].split("_D")[0]
        except IndexError:
            facility = "0"

    now = datetime.now()
    day = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    processing_id = f"I{facility}_D{day}_T{time_str}"

    return processing_id


def _build_output_directory(raw_path: Path) -> Tuple[Path, str]:
    """
    Build the processed output directory from the raw path.

    Parameters
    ----
    raw_path : Path
        Path to the raw sweep group directory.

    Return
    ---
    output_dir : Path
        Path to the processed output directory.
    group_name : str
        Raw group name used for output naming.
    """
    group_name = raw_path.name
    raw_parent = raw_path.parent
    output_root = raw_parent.parent / "processed"
    output_dir = output_root / group_name

    return output_dir, group_name


def _get_file_list(raw_path: Path) -> List[Path]:
    """
    Get a sorted list of raw HDF5 files in the sweep group directory.

    Parameters
    ----
    raw_path : Path
        Path to the raw sweep group directory.

    Return
    ---
    file_list : list[Path]
        Sorted list of HDF5 files.
    """
    file_list = sorted(raw_path.glob("*.hdf5"))
    if not file_list:
        raise ValueError(f"No HDF5 files found in {raw_path}")

    return file_list


def _downsample_file_list(
    file_list: List[Path],
    freq_step: Optional[int] = None,
    n_freq_steps: Optional[int] = None,
) -> List[Path]:
    """
    Downsamples frequencies (picks from frequencies provided in file_list).

    Exactly one of freq_step or n_freq_steps may be provided.
    If neither is provided, the full list is returned unchanged.

    Parameters
    ----
    file_list : list[Path]
        SORTED list of raw HDF5 files (one per frequency step).
    freq_step : int or None
        Keep every Nth file (e.g. freq_step=3 keeps files 0, 3, 6, ...).
    n_freq_steps : int or None
        Target number of frequency steps in the output.
        Picks evenly spaced indices (including first and last) to get
        close to this count.

    Return
    ---
    downsampled : list[Path]
        Subset of the input file list.
    """
    if freq_step is not None and n_freq_steps is not None:
        raise ValueError(
            "Cannot specify both freq_step and n_freq_steps. Pick one."
        )

    # No downsampling requested — return full list.
    if freq_step is None and n_freq_steps is None:
        return file_list

    n_total = len(file_list)

    if freq_step is not None:
        if freq_step < 1:
            raise ValueError("freq_step must be >= 1.")
        indices = list(range(0, n_total, freq_step))
    else:
        # n_freq_steps mode
        if n_freq_steps < 2:
            raise ValueError("n_freq_steps must be >= 2.")
        if n_freq_steps >= n_total:
            return file_list
        # Evenly spaced indices including first and last file.
        indices = np.round(
            np.linspace(0, n_total - 1, n_freq_steps)
        ).astype(int).tolist()

    downsampled = [file_list[i] for i in indices]
    return downsampled


def _get_sample_rate_hz(h5reader: h5io.H5Reader, file_path: str) -> float:
    """
    Extract sample rate from the raw file metadata.

    Parameters
    ----
    h5reader : H5Reader
        Pytesdaq HDF5 reader instance.
    file_path : str
        Full path to the raw file.

    Return
    ---
    sample_rate_hz : float
        Sample rate in Hz.
    """
    metadata = h5reader.get_metadata(file_name=file_path)
    adc_list = metadata.get("adc_list", [])
    if not adc_list:
        raise ValueError(f"Missing adc_list metadata in {file_path}")

    adc_name = adc_list[0]
    adc_metadata = metadata["groups"][adc_name]
    if "sample_rate" not in adc_metadata:
        raise ValueError(f"Missing sample_rate metadata in {file_path}")

    return float(adc_metadata["sample_rate"])


def _extract_signal_gen_metadata(detector_config: Dict) -> Tuple[float, np.ndarray]:
    """
    Extract signal generator frequency and voltage from detector config.

    Parameters
    ----
    detector_config : dict
        Detector config dictionary.

    Return
    ---
    frequency_hz : float
        Signal generator frequency used for the sweep step.
    signal_gen_voltage : np.ndarray
        Signal generator voltage per channel.
    """
    frequency_keys = ["signal_gen_frequency", "signal_gen_frequency_hz"]
    voltage_keys = ["signal_gen_voltage", "signal_gen_voltage_v"]

    frequency_value = None
    for key in frequency_keys:
        if key in detector_config:
            frequency_value = detector_config[key]
            break

    if frequency_value is None:
        raise ValueError("Signal generator frequency not found in detector config.")

    voltage_value = None
    for key in voltage_keys:
        if key in detector_config:
            voltage_value = detector_config[key]
            break

    if voltage_value is None:
        raise ValueError("Signal generator voltage not found in detector config.")

    frequency_hz = float(np.atleast_1d(frequency_value)[0])
    signal_gen_voltage = np.array(voltage_value, dtype=float, copy=False)

    return frequency_hz, signal_gen_voltage


def _extract_accel_gain(detector_config: Dict) -> float:
    """
    Extract accelerometer gain from detector config metadata.

    Parameters
    ----
    detector_config : dict
        Detector configuration dictionary.

    Return
    ---
    accel_gain : float
        Accelerometer gain value.
    """
    if "accel_gain" not in detector_config:
        return 100.0

    accel_gain_value = detector_config["accel_gain"]
    accel_gain_array = np.atleast_1d(accel_gain_value).astype(float)
    if accel_gain_array.size < 1:
        return 100.0

    return float(accel_gain_array[0])


def _iter_file_traces(
    h5reader: h5io.H5Reader,
    file_path: str,
    detector_channels: List[str],
    accel_gain: float,
    trace_length_override_samples: Optional[int] = None,
) -> Generator[Tuple[np.ndarray, int], None, None]:
    """
    Yields (trace_chunk, trace_idx) pairs for one file.

    Default mode (trace_length_override_samples is None):
        Each recorded event is yielded directly as its own trace.

    Override mode (trace_length_override_samples is set):
        Events are accumulated one at a time until enough samples are available to fill one override-length segment.
        The first `trace_length_override_samples` number of samples of the accumulated stream are yielded;
        any leftover samples in the last consumed event are discarded.
        (This guarantees that phase-locked reference ramains valid).

    The returned arrays are converted to g by dividing by accel_gain.

    Parameters
    ----
    h5reader : H5Reader
        Pytesdaq HDF5 reader instance.
    file_path : str
        Full path to the raw file.
    detector_channels : list[str]
        Channel list to read.
    accel_gain : float
        Accelerometer gain for ADC-to-g conversion; must be > 0.
    trace_length_override_samples : int or None
       Override trace length (read note about behavior above). When None, yield each event as-is.

    Yields
    ------
    trace_chunk : np.ndarray
        Shape (n_channels, n_samples), converted to g acceleration units.
    trace_idx : int
        Index of this trace within the file, starting from 0.
    """
    if accel_gain <= 0:
        raise ValueError("accel_gain must be > 0.")

    h5reader.set_files(filepaths=[file_path])

    if trace_length_override_samples is None:
        # Default: yield each event directly as its own trace.
        trace_idx = 0
        while True:
            traces, info = h5reader.read_next_event(
                detector_chans=detector_channels,
                adctovolt=False,
                adctoamp=True,
                include_metadata=True,
            )
            if info.get("read_status", 0) > 0:
                break
            if traces is None or traces.size == 0:
                continue
            yield traces / float(accel_gain), trace_idx
            trace_idx += 1
    else:
        # trace length override mode
        trace_idx = 0
        accumulated: List[np.ndarray] = []
        accumulated_samples = 0

        while True:
            traces, info = h5reader.read_next_event(
                detector_chans=detector_channels,
                adctovolt=False,
                adctoamp=True,
                include_metadata=True,
            )
            if info.get("read_status", 0) > 0:
                break
            if traces is None or traces.size == 0:
                continue

            accumulated.append(traces)
            accumulated_samples = accumulated_samples + traces.shape[1]

            # If enough samples are buffered, yield one segment and
            # drop any leftover samples from the last consumed event.
            if accumulated_samples >= trace_length_override_samples:
                combined = np.concatenate(accumulated, axis=1)
                chunk = combined[:, :trace_length_override_samples]
                yield chunk / float(accel_gain), trace_idx
                trace_idx = trace_idx + 1
                accumulated = []
                accumulated_samples = 0


def _fit_trace_sinusoid(trace: np.ndarray,
                        time_s: np.ndarray) -> Tuple[float, float]:
    """
    Fit a single trace to A cos(wt) + B sin(wt) using least squares linear fit.

    Parameters
    ----
    trace : np.ndarray
        1D array of trace samples.
    time_s : np.ndarray
        1D array of time values in seconds.

    Return
    ---
    amp_cos : float
        Cosine coefficient A.
    amp_sin : float
        Sine coefficient B.
    """
    if trace.size != time_s.size:
        raise ValueError("Trace length and time vector length mismatch.")

    cos_term = np.cos(time_s)
    sin_term = np.sin(time_s)
    design_matrix = np.column_stack((cos_term, sin_term))
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, trace, rcond=None)

    amp_cos = float(coefficients[0])
    amp_sin = float(coefficients[1])

    return amp_cos, amp_sin


def process_transducer_sweep(
    raw_path: str,
    trace_length_override_msec: Optional[float] = None,
    accel_gain: float = None,
    freq_step: Optional[int] = None,
    n_freq_steps: Optional[int] = None,
) -> Path:
    """
    Process transducer_sweep raw data and save fit amplitudes.

    Parameters
    ----
    raw_path : str
        Path to the raw sweep group directory.
    trace_length_override_msec : float or None
        When provided, all events in each frequency-step file are
        concatenated into a continuous stream and re-chopped into segments
        of this length. When None (default), each recorded event/dump is
        used as its own trace.
    accel_gain : float or None
        Accelerometer gain used to convert ADC to g. If None, the value
        is read from raw data metadata (fallback to 100 if missing).
    freq_step : int or None
        Process every Nth frequency step (e.g. 3 keeps every 3rd file).
        Cannot be used together with n_freq_steps.
    n_freq_steps : int or None
        Target number of frequency steps to process. Evenly spaced files
        (including first and last) are selected to approximate this count.
        Cannot be used together with freq_step.

    Return
    ---
    output_file : Path
        Path to the processed HDF5 output file.
    """
    raw_dir = Path(raw_path).expanduser().resolve()
    if not raw_dir.is_dir():
        raise ValueError(f"Raw path does not exist: {raw_dir}")

    output_dir, group_name = _build_output_directory(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processing_id = _get_processing_id(group_name)
    output_file = output_dir / f"amplitude_{processing_id}_F0001.hdf5"

    file_list = _get_file_list(raw_dir)
    file_list = _downsample_file_list(
        file_list=file_list,
        freq_step=freq_step,
        n_freq_steps=n_freq_steps,
    )
    h5reader = h5io.H5Reader()

    channel_names: List[str] = []
    amp_real_by_chan: Dict[str, List[float]] = {}
    amp_imag_by_chan: Dict[str, List[float]] = {}
    amp_complex_by_chan: Dict[str, List[str]] = {}
    signal_gen_voltage_list: List[float] = []
    accel_gain_list: List[float] = []
    frequency_list: List[float] = []
    trace_index_list: List[int] = []
    file_name_list: List[str] = []
    sample_rate_list: List[float] = []
    trace_length_list: List[float] = []

    # Loop over files to process each sweep step.
    for file_path in tqdm(file_list, desc="Processing files"):
        detector_config = h5reader.get_detector_config(
            file_name=str(file_path),
            use_chan_dict=False
        )
        detector_channels = list(detector_config.get("detector_chans", []))
        if not detector_channels:
            raise ValueError(f"No detector channels found in {file_path}")

        if not channel_names:
            channel_names = detector_channels
            # Initialize per-channel columns once channel order is known.
            for chan in channel_names:
                amp_real_by_chan[chan] = []
                amp_imag_by_chan[chan] = []
                amp_complex_by_chan[chan] = []
        else:
            if detector_channels != channel_names:
                raise ValueError(
                    "Detector channel order changed across files. "
                    "Processing expects a consistent channel list."
                )

        sample_rate_hz = _get_sample_rate_hz(
            h5reader=h5reader,
            file_path=str(file_path)
        )
        frequency_hz, signal_gen_voltage = _extract_signal_gen_metadata(
            detector_config=detector_config
        )
        if signal_gen_voltage.size == 1 and len(detector_channels) > 1:
            signal_gen_voltage = np.full(
                len(detector_channels),
                float(signal_gen_voltage[0])
            )

        accel_gain_file = _extract_accel_gain(detector_config=detector_config)
        if accel_gain is not None:
            accel_gain_file = float(accel_gain)

        trace_length_override_samples = None
        if trace_length_override_msec is not None:
            trace_length_override_samples = int(
                round(sample_rate_hz * trace_length_override_msec / 1000.0)
            )
            if trace_length_override_samples <= 0:
                raise ValueError(
                    "trace_length_override_msec must produce at least 1 sample."
                )

        angular_frequency = 2.0 * np.pi * frequency_hz

        for trace_chunk, trace_idx in _iter_file_traces(
            h5reader=h5reader,
            file_path=str(file_path),
            detector_channels=detector_channels,
            accel_gain=accel_gain_file,
            trace_length_override_samples=trace_length_override_samples,
        ):
            n_samples = trace_chunk.shape[1]
            time_s = np.arange(n_samples) / sample_rate_hz
            time_arg = angular_frequency * time_s
            actual_trace_length_msec = n_samples / sample_rate_hz * 1000.0

            amp_real = np.zeros(len(detector_channels), dtype=float)
            amp_imag = np.zeros(len(detector_channels), dtype=float)
            amp_complex = np.zeros(len(detector_channels), dtype=np.complex128)

            # Fit A cos(wt) + B sin(wt) per channel.
            for chan_index in range(len(detector_channels)):
                trace = trace_chunk[chan_index, :]
                amp_cos, amp_sin = _fit_trace_sinusoid(
                    trace=trace,
                    time_s=time_arg
                )
                amp_real[chan_index] = amp_cos
                amp_imag[chan_index] = amp_sin
                amp_complex[chan_index] = complex(amp_cos, amp_sin)

            # Populate per-channel columns.
            for chan_index, chan_name in enumerate(channel_names):
                amp_real_by_chan[chan_name].append(float(amp_real[chan_index]))
                amp_imag_by_chan[chan_name].append(float(amp_imag[chan_index]))
                amp_complex_by_chan[chan_name].append(
                    f"{amp_complex[chan_index].real}+{amp_complex[chan_index].imag}j"
                )
            frequency_list.append(frequency_hz)
            signal_gen_voltage_list.append(float(signal_gen_voltage[0]))
            accel_gain_list.append(float(accel_gain_file))
            trace_index_list.append(trace_idx)
            file_name_list.append(file_path.name)
            sample_rate_list.append(sample_rate_hz)
            trace_length_list.append(actual_trace_length_msec)

    if not frequency_list:
        raise ValueError("No traces were processed. Check data or trace_length_override_msec.")

    data_dict = {
        "channel_names": [",".join(channel_names)] * len(frequency_list),
        "frequency_hz": frequency_list,
        "signal_gen_voltage": signal_gen_voltage_list,
        "trace_index": trace_index_list,
        "file_name": file_name_list,
        "sample_rate_hz": sample_rate_list,
        "trace_length_msec": trace_length_list,
        "accel_gain": accel_gain_list,
    }

    # Add per-channel amplitude columns.
    for chan_name in channel_names:
        data_dict[f"amp_real_{chan_name}"] = amp_real_by_chan[chan_name]
        data_dict[f"amp_imag_{chan_name}"] = amp_imag_by_chan[chan_name]
        data_dict[f"amp_complex_{chan_name}"] = amp_complex_by_chan[chan_name]

    processed_df = vx.from_dict(data_dict)
    processed_df.export_hdf5(str(output_file), mode="w")

    return output_file


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for transducer sweep processing.

    Parameters
    ----
    None

    Return
    ---
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process transducer_sweep raw data into fitted amplitudes."
    )
    parser.add_argument(
        "--raw_path",
        type=str,
        required=True,
        help="Path to the raw transducer_sweep group directory.",
    )
    parser.add_argument(
        "--trace_length_override_msec",
        "--trace_length_msec",
        dest="trace_length_override_msec",
        type=float,
        default=None,
        help=(
            "Override trace length in milliseconds. When provided, all events "
            "in each frequency-step file are concatenated into a continuous "
            "stream and re-chopped into segments of this length. When omitted "
            "(default), each recorded event/dump is used as its own trace."
        ),
    )
    parser.add_argument(
        "--accel_gain",
        "--accel-gain",
        type=float,
        default=None,
        help=(
            "Accelerometer gain for ADC-to-g conversion. "
            "If omitted, use the raw data metadata."
        ),
    )

    # Frequency downsampling (mutually exclusive).
    freq_group = parser.add_mutually_exclusive_group()
    freq_group.add_argument(
        "--freq_step",
        "--freq-step",
        type=int,
        default=None,
        help=(
            "Process every Nth frequency step file (e.g. --freq_step 5 "
            "keeps every 5th file). Reduces frequency resolution but "
            "speeds up processing proportionally."
        ),
    )
    freq_group.add_argument(
        "--n_freq_steps",
        "--n-freq-steps",
        type=int,
        default=None,
        help=(
            "Target number of frequency steps to process. Evenly spaced "
            "files (including first and last) are selected automatically. "
            "E.g. --n_freq_steps 60 on a 600-file sweep processes ~60 files."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Run transducer_sweep processing from the command line.

    Parameters
    ----
    None

    Return
    ---
    None
    """
    args = _parse_args()
    output_file = process_transducer_sweep(
        raw_path=args.raw_path,
        trace_length_override_msec=args.trace_length_override_msec,
        accel_gain=args.accel_gain,
        freq_step=args.freq_step,
        n_freq_steps=args.n_freq_steps,
    )
    print(f"Saved processed sweep data to: {output_file}")


if __name__ == "__main__":
    main()
