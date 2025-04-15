# Classification Scripts

This folder contains scripts for training and evaluating an SVM classifier on EEG bandpower features extracted from preprocessed EEG data.

---

##  Files

### `svm_bandpower_train.py`
- Trains a multi-class SVM classifier to predict cognitive load levels (0-back, 1-back, 2-back, Rest).
- Uses EEG **bandpower features** from **frontal channels only**.
- Includes preprocessing (scaling, class balancing with SMOTE).
- Performs **hyperparameter tuning** for optimal SVM performance, which is NOT included in this script.
- Saves the trained model to a `.pkl` file using `joblib`.

### `svm_bandpower_evaluate.py`
- Loads a trained classifier.
- Evaluates it on **unseen data** from a different participant.
- Generates a classification report and confusion matrix.

---

##  Labels
- `0`: Resting state (e.g. RS_Beg_EC, RS_Beg_EO)
- `1`: 0-back task
- `2`: 1-back task
- `3`: 2-back task

---

##  Feature Format
- Bandpower features are extracted from `.fif` files saved after ICA and epoching.
- Each epoch is represented by flattened features of the form:
  
  `(n_channels * n_bands,)`
  
- Features are saved as `.npy` files:
  ```
  npy_features/bandpower_all/
  └── participant_<id>_<session>_trial_<trial>_features_frontpbp_all.npy
  ```

---

##  Notes
- Only the final and most interpretable version is uploaded here.
- Additional classifiers (e.g., **LDA**, **Random Forest**), **2-class models**, and models with different **channel selections** or **feature types** were tested and are available upon request.

---

##  Citation / Thesis
This classifier and feature extraction pipeline are part of the work presented in the corresponding master's thesis. If you're interested in deeper details, please check the thesis or contact the author.
