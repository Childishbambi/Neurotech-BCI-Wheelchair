"""Minimal configuration for the standalone wheelchair classifier."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EEGConfig:
    """EDF preprocessing settings copied from the ErrP motor-imagery path."""

    channels: tuple[str, ...] = ("Pz", "F4", "C4", "P4", "P3", "C3", "F3")
    sample_rate: float = 300.0
    low_frequency: float = 1.0
    high_frequency: float = 45.0
    notch_frequency: float | None = None
    reject_peak_to_peak_uv: float | None = 150.0
    epoch_duration_s: float = 3.0
    window_duration_s: float = 3.0
    window_step_s: float = 0.5
    edf_scale_to_uv: float = 1e6
    left_code: int = 1
    right_code: int = 2


@dataclass(frozen=True)
class TrainingConfig:
    """Compact EEGNet defaults for the seven-channel DSI-7 setup."""

    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.5
    f1: int = 8
    depth_multiplier: int = 2
    f2: int = 16
    temporal_kernel: int = 64
    separable_kernel: int = 16
    validation_fraction: float = 0.2
    patience: int = 7
    random_state: int = 42
    device: str = "auto"
    verbose: bool = True
