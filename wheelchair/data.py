"""Load and preprocess left/right motor-imagery trials from EDF files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from .config import EEGConfig


@dataclass(frozen=True)
class MotorImageryDataset:
    X: np.ndarray
    y: np.ndarray
    session_ids: np.ndarray
    channel_names: tuple[str, ...]
    sample_rate: float
    files_found: int
    files_used: int


def canonicalize_channel_name(name: str) -> str:
    cleaned = str(name).strip().replace("EEG ", "").replace("EEG_", "")
    return cleaned.replace("-Pz", "").replace("-PZ", "").replace(" ", "").upper()


def resolve_channel_order(
    available_names: list[str] | tuple[str, ...],
    desired_names: list[str] | tuple[str, ...],
) -> tuple[list[str], list[str]]:
    actual_by_name = {canonicalize_channel_name(name): str(name) for name in available_names}
    resolved = []
    missing = []
    for desired in desired_names:
        actual = actual_by_name.get(canonicalize_channel_name(desired))
        if actual is None:
            missing.append(str(desired))
        else:
            resolved.append(actual)
    return resolved, missing


def _standardize_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    if "EEG LE-Pz" in raw.ch_names:
        raw.set_eeg_reference(ref_channels=["EEG LE-Pz"], verbose="ERROR")

    rename_map = {}
    for old_name in raw.ch_names:
        new_name = old_name.replace("-Pz", "").replace("-PZ", "")
        if new_name == "Pz":
            new_name = "EEG Pz"
        if new_name != old_name:
            rename_map[old_name] = new_name
    if rename_map:
        raw.rename_channels(rename_map)
    return raw


def _find_stim_channel(raw: mne.io.BaseRaw) -> str:
    for candidate in ("Trigger", "TRG", "TRIGGER", "trigger", "STI", "stim", "Status", "status"):
        if candidate in raw.ch_names:
            return candidate
    raise RuntimeError(f"No trigger channel found. Available channels: {raw.ch_names}")


def _filter_session(data: np.ndarray, config: EEGConfig) -> np.ndarray:
    nyquist = float(config.sample_rate) / 2.0
    if not 0.0 < float(config.low_frequency) < float(config.high_frequency) < nyquist:
        raise ValueError("Bandpass frequencies must satisfy 0 < low < high < sample_rate / 2.")

    stages = []
    if config.notch_frequency is not None:
        b, a = iirnotch(
            w0=float(config.notch_frequency),
            Q=30.0,
            fs=float(config.sample_rate),
        )
        stages.append(tf2sos(b, a))
    stages.append(
        butter(
            4,
            [float(config.low_frequency), float(config.high_frequency)],
            btype="bandpass",
            output="sos",
            fs=float(config.sample_rate),
        )
    )
    sos = np.vstack(stages).astype(np.float64, copy=False)
    zi_template = sosfilt_zi(sos)
    output = np.empty_like(data, dtype=np.float64)
    for channel_index, channel in enumerate(np.asarray(data, dtype=np.float64)):
        filtered, _ = sosfilt(sos, channel, zi=zi_template * float(channel[0]))
        output[channel_index] = filtered
    return output.astype(np.float32, copy=False)


def _split_windows(
    epoch: np.ndarray,
    sample_rate: float,
    window_duration_s: float,
    window_step_s: float,
) -> np.ndarray:
    window_samples = int(round(float(window_duration_s) * float(sample_rate)))
    step_samples = int(round(float(window_step_s) * float(sample_rate)))
    if window_samples < 1 or step_samples < 1:
        raise ValueError("Window duration and step must produce at least one sample.")
    if epoch.shape[1] < window_samples:
        return np.empty((0, epoch.shape[0], window_samples), dtype=np.float32)
    starts = range(0, epoch.shape[1] - window_samples + 1, step_samples)
    return np.stack([epoch[:, start:start + window_samples] for start in starts]).astype(
        np.float32,
        copy=False,
    )


def load_edf_dataset(
    data_dir: str | Path,
    config: EEGConfig = EEGConfig(),
    edf_glob: str = "*.edf",
    participant: str | None = None,
) -> MotorImageryDataset:
    """Create CNN windows shaped ``(trials, channels, samples)`` from EDFs."""

    root = Path(data_dir).expanduser()
    paths = sorted(root.rglob(edf_glob))
    if participant:
        paths = [path for path in paths if participant in str(path)]
    if not paths:
        suffix = f" for participant {participant!r}" if participant else ""
        raise FileNotFoundError(f"No EDF files matched {edf_glob!r} under {root}{suffix}.")

    all_windows = []
    all_labels = []
    all_session_ids = []
    files_used = 0
    epoch_samples = int(round(float(config.epoch_duration_s) * float(config.sample_rate)))

    for session_id, path in enumerate(paths):
        raw = _standardize_raw(mne.io.read_raw_edf(path, preload=True, verbose="ERROR"))
        stim_channel = _find_stim_channel(raw)
        original_rate = float(raw.info["sfreq"])
        events = mne.find_events(raw, stim_channel=stim_channel, min_duration=0.0, verbose=False)
        event_times_s = events[:, 0].astype(np.float64) / original_rate
        if abs(original_rate - float(config.sample_rate)) > 1e-6:
            raw.resample(float(config.sample_rate), npad="auto", verbose="ERROR")

        picks, missing = resolve_channel_order(raw.ch_names, config.channels)
        if missing:
            raise RuntimeError(f"{path} is missing required channels {missing}.")
        continuous = raw.get_data(picks=picks).astype(np.float32, copy=False)
        continuous *= float(config.edf_scale_to_uv)
        continuous = _filter_session(continuous, config)
        used = False

        for event_time_s, event_code in zip(event_times_s, events[:, 2]):
            event_code = int(event_code)
            if event_code not in (int(config.left_code), int(config.right_code)):
                continue
            start = int(round(float(event_time_s) * float(config.sample_rate)))
            stop = start + epoch_samples
            if start < 0 or stop > continuous.shape[1]:
                continue
            windows = _split_windows(
                continuous[:, start:stop],
                config.sample_rate,
                config.window_duration_s,
                config.window_step_s,
            )
            if config.reject_peak_to_peak_uv is not None and len(windows):
                peak_to_peak = np.ptp(windows, axis=-1).max(axis=1)
                windows = windows[peak_to_peak <= float(config.reject_peak_to_peak_uv)]
            if not len(windows):
                continue
            all_windows.append(windows)
            all_labels.append(np.full(len(windows), event_code, dtype=np.int64))
            all_session_ids.append(np.full(len(windows), session_id, dtype=np.int64))
            used = True
        files_used += int(used)

    if not all_windows:
        raise RuntimeError("No usable LEFT/RIGHT windows were extracted from the EDF files.")

    return MotorImageryDataset(
        X=np.concatenate(all_windows).astype(np.float32, copy=False),
        y=np.concatenate(all_labels),
        session_ids=np.concatenate(all_session_ids),
        channel_names=tuple(config.channels),
        sample_rate=float(config.sample_rate),
        files_found=len(paths),
        files_used=files_used,
    )
