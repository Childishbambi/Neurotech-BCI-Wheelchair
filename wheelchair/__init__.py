"""Standalone motor-imagery CNN for wheelchair control experiments."""

from .config import EEGConfig, TrainingConfig
from .eegnet_classifier import EEGNetClassifier

__all__ = ["EEGConfig", "TrainingConfig", "EEGNetClassifier"]
