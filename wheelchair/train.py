"""Command-line training entry point for the standalone EEGNet classifier."""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from .config import EEGConfig, TrainingConfig
from .data import MotorImageryDataset, load_edf_dataset
from .eegnet_classifier import EEGNetClassifier


def build_classifier(config: TrainingConfig) -> EEGNetClassifier:
    return EEGNetClassifier(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        dropout=config.dropout,
        f1=config.f1,
        depth_multiplier=config.depth_multiplier,
        f2=config.f2,
        temporal_kernel=config.temporal_kernel,
        separable_kernel=config.separable_kernel,
        validation_fraction=config.validation_fraction,
        patience=config.patience,
        random_state=config.random_state,
        device=config.device,
        verbose=config.verbose,
    )


def evaluate_loso(dataset: MotorImageryDataset, config: TrainingConfig) -> dict[int, float]:
    """Leave one EDF session out at a time and return accuracy by session."""

    scores = {}
    for session_id in np.unique(dataset.session_ids):
        test_mask = dataset.session_ids == session_id
        train_mask = ~test_mask
        if len(np.unique(dataset.y[train_mask])) < 2 or len(np.unique(dataset.y[test_mask])) < 2:
            continue
        classifier = build_classifier(replace(config, verbose=False))
        classifier.fit(dataset.X[train_mask], dataset.y[train_mask])
        scores[int(session_id)] = float(
            accuracy_score(dataset.y[test_mask], classifier.predict(dataset.X[test_mask]))
        )
    return scores


def save_model(
    path: str | Path,
    classifier: EEGNetClassifier,
    dataset: MotorImageryDataset,
    eeg_config: EEGConfig,
    training_config: TrainingConfig,
) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classifier": classifier,
        "classes": [int(value) for value in classifier.classes_],
        "channel_names": list(dataset.channel_names),
        "sample_rate": dataset.sample_rate,
        "window_samples": int(dataset.X.shape[-1]),
        "eeg_config": asdict(eeg_config),
        "training_config": asdict(training_config),
    }
    with output.open("wb") as file_handle:
        pickle.dump(payload, file_handle)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LEFT/RIGHT EEGNet from motor-imagery EDFs.")
    parser.add_argument("--data-dir", required=True, help="Folder containing EDF recordings.")
    parser.add_argument("--output", default="wheelchair_model.pkl", help="Output model bundle.")
    parser.add_argument("--edf-glob", default="*.edf", help="Recursive EDF filename pattern.")
    parser.add_argument("--participant", help="Only use EDF paths containing this text.")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--device", default=TrainingConfig.device, help="auto, cpu, cuda, or mps")
    parser.add_argument("--evaluate-loso", action="store_true", help="Run slower leave-one-session-out evaluation.")
    parser.add_argument("--quiet", action="store_true", help="Hide per-epoch loss output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eeg_config = EEGConfig()
    training_config = replace(
        TrainingConfig(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        verbose=not args.quiet,
    )
    dataset = load_edf_dataset(
        args.data_dir,
        config=eeg_config,
        edf_glob=args.edf_glob,
        participant=args.participant,
    )
    print(
        f"Loaded {len(dataset.X)} windows from {dataset.files_used}/{dataset.files_found} EDF files; "
        f"shape={dataset.X.shape}, classes={dict(Counter(int(v) for v in dataset.y))}"
    )

    if args.evaluate_loso:
        scores = evaluate_loso(dataset, training_config)
        if scores:
            print(f"LOSO accuracy by session: {scores}")
            print(f"LOSO mean accuracy: {np.mean(list(scores.values())):.3f}")
        else:
            print("LOSO was skipped because no session had both classes in train and test sets.")

    classifier = build_classifier(training_config)
    classifier.fit(dataset.X, dataset.y)
    output = save_model(args.output, classifier, dataset, eeg_config, training_config)
    print(f"Saved model to {output.resolve()}")


if __name__ == "__main__":
    main()
