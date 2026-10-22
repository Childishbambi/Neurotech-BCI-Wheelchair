# Standalone Wheelchair EEG Classifier

This folder contains only the code needed to train and run the LEFT/RIGHT
motor-imagery CNN. It is independent of the experiment interfaces in
`ErrP/bci`.

The model is a compact EEGNet network. It learns temporal EEG filters, then a
depthwise spatial filter across the seven DSI-7 channels:

```text
Pz, F4, C4, P4, P3, C3, F3
```

Trigger code `1` is LEFT and trigger code `2` is RIGHT. EDF recordings are
resampled to 300 Hz, converted from volts to microvolts, filtered from 1–45 Hz,
and divided into 3-second windows before training.

## Files

```text
wheelchair/
├── __init__.py
├── config.py              # EEG and CNN settings
├── data.py                # EDF loading, filtering, and windowing
├── eegnet_classifier.py   # sklearn-compatible PyTorch EEGNet
├── train.py               # training command
├── predict.py             # prediction command
└── requirements.txt
```

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r wheelchair/requirements.txt
```

## Train

Point `--data-dir` at the folder containing the motor-imagery EDF recordings:

```bash
python -m wheelchair.train \
  --data-dir /path/to/edf-data \
  --output wheelchair_model.pkl
```

Useful options:

```bash
# Train only on paths containing a participant/session name
python -m wheelchair.train --data-dir /path/to/data --participant mi

# Also run the slower leave-one-session-out evaluation
python -m wheelchair.train --data-dir /path/to/data --evaluate-loso

# Force CPU training
python -m wheelchair.train --data-dir /path/to/data --device cpu
```

The saved bundle contains the trained classifier, class labels, channel order,
sample rate, window size, and preprocessing/training configuration.

## Predict

Save one preprocessed window as a NumPy array shaped `(7, 900)`, or several
windows shaped `(trials, 7, 900)`. The order must be
`Pz, F4, C4, P4, P3, C3, F3`.

```bash
python -m wheelchair.predict \
  --model wheelchair_model.pkl \
  --window window.npy
```

The command prints the LEFT/RIGHT class and probability for each window. You
can also import `EEGNetClassifier` directly:

```python
from wheelchair import EEGNetClassifier

classifier = EEGNetClassifier()
classifier.fit(train_windows, train_labels)
probabilities = classifier.predict_proba(live_windows)
commands = classifier.predict(live_windows)
```

This folder stops at classification. Connect its LEFT/RIGHT output to a tested
wheelchair controller with an emergency stop, command timeout, speed limiting,
and an independent human override before operating physical hardware.
