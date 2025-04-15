# EEG Preprocessing Pipeline

This repository provides a modular EEG preprocessing pipeline for experiments using raw EEG data (e.g., in `.set` format from EEGLAB). It performs filtering, bad channel detection, ICA artifact correction, and epoching, with flexible options for AutoReject.

---

## Structure

```
eeg-preprocessing-pipeline/
├── preprocess_all.py              # Main script to run the pipeline
├── requirements.txt               # Required Python packages
├── README.md                      # Project documentation
├── scripts/
│   ├── filter.py                  # Filtering and CAR
│   ├── remove_bad_channels.py     # Bad channel detection
│   ├── to_fif.py                  # ICA training and saving
│   ├── apply_ica.py               # ICA application with flat-channel fix
│   ├── epochs_reject.py           # Epoching with fallback strategies
│   └── segmentation_autoreject.py # (Optional) AutoReject-based segmentation
├── data/
│   └── README.md                  # Instructions for dataset placement
```

---

## How It Works

### 1. Filter and CAR
```python
from scripts.filter import filter
```
- Bandpass filtering [0.1, 30 Hz]
- Standard 10-20 montage
- Iterative CAR with noisy channel interpolation

### 2. Identify Bad Channels
Handled internally in `filter.py` using `remove_bad_channels.py`

### 3. ICA
```python
from scripts.to_fif import eeg_to_fif
```
- Trains ICA on fixed-length epochs
- Detects EOG-related components (Fp1, Fp2, F7, F8)
- Saves `.fif` ICA object

### 4. Epoching & Artifact Rejection
```python
from scripts.epochs_reject import epoch_and_reject
```
- Event-based epoching
- Robust fallback: fixed-length epochs if no events or high rejection rate
- ICA applied with channel matching logic
- (Optional) AutoReject version in `segmentation_autoreject.py`

### 5. Main Runner
```bash
python preprocess_all.py
```
- Iterates through subject/session/trial folders
- Skips steps if intermediate files already exist

---

## Requirements
Install with:
```bash
pip install -r requirements.txt
```

---

## Dataset Format
- Raw `.set` files should be in:
  ```
  sub<subject_id>_res/trial<trial_id>/<session>/<session>.set
  ```

- Example:
  ```
  sub4_res/trial1/zeroBACK/zeroBACK.set
  ```

---

## Optional Features
- AutoReject-based cleaning (`segmentation_autoreject.py`)
- Drop log plotting for manual inspection

---

## Author
Adapted for modular usage by Lydia Exarchou. Based on EEG data cleaning routines using MNE-Python.

---


