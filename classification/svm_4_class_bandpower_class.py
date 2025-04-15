import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Define session names
session_folders = ["oneBACK", "zeroBACK", "twoBACK", "RS_Beg_EC", "RS_Beg_EO"]
expected_feature_count = 315


def load_features(participant, trial, session, input_dir='npy_features/bandpower'):
    features_path = os.path.join(input_dir, f'participant_{participant}_{session}_trial_{trial}_features_bandpower.npy')
    labels_path = os.path.join(input_dir, f'participant_{participant}_{session}_trial_{trial}_labels_bandpower.npy')

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"Missing files for participant {participant}, trial {trial}, session {session}")
        return None, None

    features = np.load(features_path)
    labels = np.load(labels_path)

    if features.ndim == 3:
        features = features.reshape(features.shape[0], -1)

    if features.shape[1] < expected_feature_count:
        padding = np.zeros((features.shape[0], expected_feature_count - features.shape[1]))
        features = np.hstack((features, padding))
    elif features.shape[1] > expected_feature_count:
        features = features[:, :expected_feature_count]

    return features, labels


all_data = []
for participant in [1, 3, 5, 6, 7, 8, 9, 12, 13, 16, 17, 18, 22]: #based on the data loss that occured after preprocessing
    for trial in range(1, 4):
        for session in session_folders:
            features, labels = load_features(participant, trial, session)
            if features is None or labels is None:
                continue
            if features.shape[0] != labels.shape[0]:
                continue

            if "zeroBACK" in session:
                label = 1
            elif "oneBACK" in session:
                label = 2
            elif "twoBACK" in session:
                label = 3
            else:
                label = 0

            load_levels = np.full(features.shape[0], label)
            participant_col = np.full((features.shape[0], 1), participant)
            trial_col = np.full((features.shape[0], 1), trial)
            combined = np.hstack((participant_col, trial_col, features, load_levels.reshape(-1, 1)))
            all_data.append(combined)

if not all_data:
    print("No data found.")
    exit()

final_data = np.vstack(all_data)
X = final_data[:, 2:-1]
y = final_data[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipeline = Pipeline([                  #based on hyperparameter tuning
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', SVC(
        kernel='poly',
        C=1.0,
        degree=3,
        gamma='auto',
        probability=True,
        class_weight='balanced',
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluation
label_map = {0: "Rest", 1: "0-back", 2: "1-back", 3: "2-back"}
sorted_labels = sorted(np.unique(y_train))
sorted_names = [label_map[lbl] for lbl in sorted_labels]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, labels=sorted_labels, target_names=sorted_names))

cm = confusion_matrix(y_test, y_pred, labels=sorted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Save model
model_file = "my_svm_bandpower_classifier.pkl"
joblib.dump(pipeline, model_file)
print(f"Model saved to {model_file}")
