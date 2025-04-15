import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def load_new_data(participant, trial, session, input_dir='npy_features/bandpower_all', expected_feature_count=315):
    features_file = os.path.join(input_dir, f'participant_{participant}_{session}_trial_{trial}_features_frontpbp_all.npy')
    labels_file   = os.path.join(input_dir, f'participant_{participant}_{session}_trial_{trial}_labels_frontpbp_all.npy')

    if not os.path.exists(features_file) or not os.path.exists(labels_file):
        print(f"Missing: {features_file} or {labels_file}")
        return None, None

    features = np.load(features_file)
    labels = np.load(labels_file)

    if features.ndim == 3:
        features = features.reshape(features.shape[0], -1)

    if features.shape[1] < expected_feature_count:
        padding = np.zeros((features.shape[0], expected_feature_count - features.shape[1]))
        features = np.hstack((features, padding))
    elif features.shape[1] > expected_feature_count:
        features = features[:, :expected_feature_count]

    return features, labels


participant = 10 #participant that was not used in training
trial = 2 #you can use all trials all one by one for better result inspection
sessions = ["zeroBACK", "oneBACK", "twoBACK", "RS_Beg_EC", "RS_Beg_EO"]
expected_feature_count = 315

pipeline = joblib.load("my_svm_no10_nback_pipeline_frontpbpall.pkl")

all_features, all_labels = [], []
session_to_label = {"zeroBACK": 1, "oneBACK": 2, "twoBACK": 3}

for session in sessions:
    features, _ = load_new_data(participant, trial, session)
    if features is None:
        continue

    label = session_to_label.get(session, 0)
    all_features.append(features)
    all_labels.append(np.full(features.shape[0], label))

if not all_features:
    print("No data to evaluate.")
    exit()

X_new = np.vstack(all_features)
y_true = np.hstack(all_labels)
y_pred = pipeline.predict(X_new)

label_map = {0: "Rest", 1: "0-back", 2: "1-back", 3: "2-back"}
sorted_labels = sorted(np.unique(y_true))
sorted_names = [label_map[lbl] for lbl in sorted_labels]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=sorted_labels, target_names=sorted_names))

cm = confusion_matrix(y_true, y_pred, labels=sorted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted_names)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - Participant {participant}")
plt.show()
