"""Predict LEFT/RIGHT from preprocessed EEG windows stored in a NumPy file."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load_model(path: str | Path) -> dict:
    # Pickle can execute code while loading; only open models you created or trust.
    with Path(path).expanduser().open("rb") as file_handle:
        payload = pickle.load(file_handle)
    if "classifier" not in payload:
        raise ValueError("Model bundle does not contain a classifier.")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify preprocessed motor-imagery windows.")
    parser.add_argument("--model", required=True, help="Model bundle produced by wheelchair.train.")
    parser.add_argument(
        "--window",
        required=True,
        help=".npy array shaped (channels, samples) or (trials, channels, samples).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_model(args.model)
    windows = np.load(Path(args.window).expanduser(), allow_pickle=False).astype(np.float32)
    if windows.ndim == 2:
        windows = windows[None, :, :]
    if windows.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D EEG array, received {windows.shape}.")

    expected_shape = (len(payload["channel_names"]), int(payload["window_samples"]))
    if tuple(windows.shape[1:]) != expected_shape:
        raise ValueError(f"Expected window shape {expected_shape}, received {windows.shape[1:]}.")

    classifier = payload["classifier"]
    predictions = classifier.predict(windows)
    probabilities = classifier.predict_proba(windows)
    rows = []
    for prediction, probability in zip(predictions, probabilities):
        rows.append(
            {
                "class": int(prediction),
                "command": "LEFT" if int(prediction) == 1 else "RIGHT" if int(prediction) == 2 else str(prediction),
                "probabilities": {
                    str(int(label)): float(value)
                    for label, value in zip(classifier.classes_, probability)
                },
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
