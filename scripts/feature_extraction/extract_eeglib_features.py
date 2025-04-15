import numpy as np
import mne
import os
import eeglib.features


def extract_eeglib_features(file_path, participant, session, trial):
    save_dir = "npy_features"
    os.makedirs(save_dir, exist_ok=True)

    epochs = mne.read_epochs(file_path, preload=True)
    data = epochs.get_data()
    labels = epochs.events[:, -1]

    correct_label_id = {"zeroBACK": 4, "oneBACK": 4, "twoBACK": 4}.get(session)
    if correct_label_id:
        idx = np.where(labels == correct_label_id)[0]
        if len(idx) == 0:
            print(f"No correct responses for participant {participant}, session {session}, trial {trial}.")
            return
        epochs = epochs[idx]
        labels = labels[idx]

    features_all = []
    for epoch in epochs:
        epoch_features = []
        for channel in epoch:
            try:
                epoch_features.append(eeglib.features.PFD(channel))
                epoch_features.append(eeglib.features.LZC(channel))
                epoch_features.append(eeglib.features.sampEn(channel))
                epoch_features.append(eeglib.features.DFA(channel))
            except ValueError as e:
                print(f"Channel error: {e}")
        features_all.append(epoch_features)

    features_all = np.array(features_all)
    assert features_all.shape[0] == labels.shape[0], "Mismatch between features and labels."

    np.save(os.path.join(save_dir, f"participant_{participant}_{session}_trial_{trial}_features_corr.npy"), features_all)
    np.save(os.path.join(save_dir, f"participant_{participant}_{session}_trial_{trial}_labels_corr.npy"), labels)
    print(f"Saved features and labels for participant {participant}, session {session}, trial {trial}")


if __name__ == "__main__":
    session_folders = [
        "oneBACK", "zeroBACK", "twoBACK",
        "RS_Beg_EC", "RS_Beg_EO", "RS_End_EC", "RS_End_EO"
    ]
    participant = 10

    for trial in range(1, 4):
        base_dir = os.path.join(f"sub{participant}_res", f"trial{trial}")
        for session in session_folders:
            file_name = f"{session}-epo.fif"
            file_path = os.path.join(base_dir, session, file_name)
            if os.path.exists(file_path):
                extract_eeglib_features(file_path, participant, session, trial)
            else:
                print(f"Missing: {file_path}")
