# EEG Preprocessing Pipeline

This repository provides a modular EEG preprocessing and classification pipeline. It includes preprocessing of raw EEG files (e.g. EEGLAB `.set`), feature extraction, and cognitive load classification using machine learning models.

---

## Project Structure

```
eeg-preprocessing-pipeline/
├── preprocess_all.py              # Main runner script
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── data/
│   └── README.md                  # Data structure instructions
├── scripts/
│   ├── filter.py                  # Filtering and CAR
│   ├── remove_bad_channels.py     # Bad channel detection
│   ├── to_fif.py                  # ICA computation and saving
│   ├── apply_ica.py               # ICA application function
│   ├── epochs_reject.py           # Epoching with ICA and fallbacks
│   └── feature_extraction/
│       ├── extract_bandpower.py          # Bandpower features from frontal EEG
│       └── extract_eeglib_features.py    # Nonlinear features using eeglib
├── classification/
│   ├── svm_bandpower_train.py     # Train SVM on frontal bandpower features
│   ├── svm_bandpower_evaluate.py  # Test classifier on a new participant
│   └── README.md                  # Classification notes and structure
```

---

## Project Description
This project supports EEG-based cognitive load estimation using classical ML models, with a focus on robustness and generalization.

A detailed description of the methods, dataset, and results can be found in the associated thesis, available at:

📄 **Thesis link**: https://hdl.handle.net/10889/29100

> **Thesis Abstract:**
> Cognitive load estimation is crucial in understanding human cognitive processes, with applications in neuroscience, education, and human-computer interaction. One of the most prominent non-invasive techniques for assessing cognitive load through neural activity is Electroencephalography (EEG). In this thesis, EEG-based cognitive load estimation is investigated with an SVM classifier while focusing on how well it generalizes across different experimental conditions. A publicly available EEG dataset, which includes an N-back working memory task, has been preprocessed through filtering, ICA, and artifact rejection. Another EEG experiment is performed with an arithmetic task to create additional variation in cognitive load measurements. Different training sets were created from data in both the public dataset and the experiment to evaluate how well the classifier generalizes to new tasks and recording conditions. Cross-task and cross-subject validation were performed to evaluate its robustness. Results indicate that while the SVM classifier achieves high accuracy in subject-dependent settings, it can perform well in cross-subject and cross-task settings, but only with appropriate training data; otherwise, accuracy will suffer. Inter-individual variability and differences between tasks will affect classification performance. The importance of accomplishing feature selection and reduction of channels to boost classifier performance while simplifying computational effort has been studied. These results illustrate the difficulties that exist in attempting to build EEG-based cognitive load classifiers and suggest ways in which real-world usage can be enhanced.

---

##  Preprocessing Workflow
1. **Filtering** + **Montage assignment** + **CAR referencing**
2. **Bad channel detection & interpolation**
3. **ICA** computation (on short epochs)
4. **Epoching** (event-based with fallbacks)
5. **ICA application** (handling channel mismatches)

---

##  Feature Extraction
- **Bandpower features** (frontal): `extract_bandpower.py`
- **Nonlinear features** (PFD, LZC, entropy, DFA): `extract_eeglib_features.py`

Saved in `.npy` format per participant/session.

---

##  Classification
- SVM classifier trained on frontal EEG bandpower.
- See `classification/README.md` for details.

---

##  Requirements
```txt
mne>=1.4.2
numpy>=1.22
scipy>=1.10
matplotlib>=3.7
joblib>=1.2
scikit-learn>=1.3
imblearn>=0.11
eeglib>=2.2.1
```
Install with:
```bash
pip install -r requirements.txt
```

