"""Scikit-learn compatible EEGNet classifier for motor-imagery EEG.

The BCI tasks in this repository expect a classifier with ``fit``, ``predict``
and ``predict_proba`` methods.  This module keeps that contract while training
an EEGNet-style convolutional neural network directly on windows shaped
``(trials, channels, samples)``.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _EEGNet(nn.Module):
    """Compact EEGNet implementation for short, low-channel-count windows."""

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        f1: int,
        depth_multiplier: int,
        f2: int,
        temporal_kernel: int,
        separable_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        temporal_kernel = max(3, min(int(temporal_kernel), int(n_samples)))
        separable_kernel = max(3, min(int(separable_kernel), int(n_samples)))
        f1 = int(f1)
        depth_multiplier = int(depth_multiplier)
        f2 = int(f2)
        spatial_filters = f1 * depth_multiplier

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, temporal_kernel),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                spatial_filters,
                kernel_size=(int(n_channels), 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(float(dropout)),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, separable_kernel),
                padding="same",
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(float(dropout)),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, int(n_channels), int(n_samples))
            feature_count = int(self.features(dummy).numel())
        self.classifier = nn.Linear(feature_count, int(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.flatten(start_dim=1))


class EEGNetClassifier(ClassifierMixin, BaseEstimator):
    """Train EEGNet and expose a scikit-learn classifier interface.

    Parameters are intentionally small for the seven-channel DSI-7 data used
    by this project.  Channel-wise normalization is learned from the training
    fold only, which keeps leave-one-session-out evaluation leakage-free.
    """

    def __init__(
        self,
        epochs: int = 40,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.5,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        temporal_kernel: int = 64,
        separable_kernel: int = 16,
        validation_fraction: float = 0.2,
        patience: int = 7,
        random_state: int = 42,
        device: str = "auto",
        verbose: bool = False,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.f1 = f1
        self.depth_multiplier = depth_multiplier
        self.f2 = f2
        self.temporal_kernel = temporal_kernel
        self.separable_kernel = separable_kernel
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _resolve_device(self) -> torch.device:
        requested = str(self.device).lower()
        if requested != "auto":
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                "EEGNet expects X shaped (trials, channels, samples); "
                f"received {X.shape}."
            )
        if X.shape[0] == 0 or X.shape[1] < 1 or X.shape[2] < 32:
            raise ValueError(f"EEG window array is too small for EEGNet: {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("EEGNet input contains NaN or infinite values.")
        return X

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.channel_mean_[None, :, None]) / self.channel_std_[None, :, None]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetClassifier":
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must contain one label for every EEG window.")

        self.classes_, y_indices = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("EEGNet requires at least two classes.")

        seed = int(self.random_state)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.n_channels_in_ = int(X.shape[1])
        self.n_samples_in_ = int(X.shape[2])
        self.channel_mean_ = X.mean(axis=(0, 2)).astype(np.float32)
        self.channel_std_ = X.std(axis=(0, 2)).astype(np.float32)
        self.channel_std_[self.channel_std_ < 1e-6] = 1.0
        X_norm = self._normalize(X).astype(np.float32, copy=False)

        indices = np.arange(len(X_norm))
        train_idx = indices
        val_idx = np.empty(0, dtype=int)
        class_counts = np.bincount(y_indices)
        can_validate = (
            0.0 < float(self.validation_fraction) < 0.5
            and len(X_norm) >= 10
            and int(class_counts.min()) >= 2
        )
        if can_validate:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=float(self.validation_fraction),
                random_state=seed,
                stratify=y_indices,
            )

        train_x = torch.from_numpy(X_norm[train_idx, None, :, :])
        train_y = torch.from_numpy(y_indices[train_idx].astype(np.int64))
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=max(1, int(self.batch_size)),
            shuffle=True,
            generator=generator,
        )

        device = self._resolve_device()
        model = _EEGNet(
            n_channels=self.n_channels_in_,
            n_samples=self.n_samples_in_,
            n_classes=len(self.classes_),
            f1=int(self.f1),
            depth_multiplier=int(self.depth_multiplier),
            f2=int(self.f2),
            temporal_kernel=int(self.temporal_kernel),
            separable_kernel=int(self.separable_kernel),
            dropout=float(self.dropout),
        ).to(device)

        train_counts = np.bincount(y_indices[train_idx], minlength=len(self.classes_)).astype(np.float32)
        class_weights = len(train_idx) / (len(self.classes_) * np.maximum(train_counts, 1.0))
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )

        val_x = None
        val_y = None
        if len(val_idx):
            val_x = torch.from_numpy(X_norm[val_idx, None, :, :]).to(device)
            val_y = torch.from_numpy(y_indices[val_idx].astype(np.int64)).to(device)

        best_state = copy.deepcopy(model.state_dict())
        best_loss = float("inf")
        stale_epochs = 0
        self.training_history_ = []

        for epoch in range(max(1, int(self.epochs))):
            model.train()
            running_loss = 0.0
            seen = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().cpu()) * len(batch_y)
                seen += len(batch_y)

            train_loss = running_loss / max(1, seen)
            monitored_loss = train_loss
            if val_x is not None and val_y is not None:
                model.eval()
                with torch.no_grad():
                    monitored_loss = float(criterion(model(val_x), val_y).detach().cpu())

            self.training_history_.append(
                {"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": monitored_loss}
            )
            if bool(self.verbose):
                print(
                    f"[EEGNet] epoch={epoch + 1:03d} "
                    f"train_loss={train_loss:.5f} val_loss={monitored_loss:.5f}"
                )

            if monitored_loss < best_loss - 1e-6:
                best_loss = monitored_loss
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if val_x is not None and stale_epochs >= max(1, int(self.patience)):
                    break

        model.load_state_dict(best_state)
        model.to(torch.device("cpu"))
        model.eval()
        self.model_ = model
        self.n_iter_ = len(self.training_history_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("EEGNetClassifier must be fitted before prediction.")
        X = self._validate_X(X)
        if X.shape[1:] != (self.n_channels_in_, self.n_samples_in_):
            raise ValueError(
                "EEGNet input shape changed after fitting: expected "
                f"(*, {self.n_channels_in_}, {self.n_samples_in_}), got {X.shape}."
            )
        X_norm = self._normalize(X).astype(np.float32, copy=False)
        tensor = torch.from_numpy(X_norm[:, None, :, :])
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
