import numpy as np
import mne
import os
from scipy.signal import welch


def pick_frontal_channels(epochs):
    """Select frontal and central EEG channels for feature extraction. These were the best channels based on this research."""
    selected = [
        'Fp1', 'Fp2', 'AF3', 'AF4', 'Fz', 'F3', 'F4',
        'FC1', 'FC2', 'C3', 'C4', 'Pz', 'P3', 'P4',
        'T7', 'T8', 'Oz', 'O1', 'O2'
    ]
    return epochs.pick_channels(selected)


def extract_bandpower_features(epoch, sfreq):
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    features = []
    for channel in epoch:
        f, psd = welch(channel, sfreq)
        for (low, high) in bands.values():
            power = np.sum(psd[(f >= low) & (f <= high)])
            features.append(power)
    return features


def extract_bandpower_from_file(file_path, participant, session_name):
    save_dir = "npy_features/bandpower_all"
    os.makedirs(save_dir, exist_ok=True)

    epochs = mne.read_epochs(file_path, preload=True)
    epochs = pick_frontal_channels(epochs)

    sfreq = int(epochs.info['sfreq'])
    labels = epochs.events[:, -1]

    correct_label_id = {"zeroBACK": 4, "oneBACK": 4, "twoBACK": 4}.get(session_name)
    if correct_label_id:
        idx = np.where(labels == correct_label_id)[0]
        if len(idx) == 0:
            print(f"No correct responses for participant {participant}, session {session_name}. Skipping.")
            return
        epochs = epochs[idx]
        labels = labels[idx]

    data = epochs.get_data()
    features = [extract_bandpower_features(epoch, sfreq) for epoch in data]

    features_file = os.path.join(save_dir, f'participant_{participant}_{session_name}_features_bandpower.npy')
    labels_file = os.path.join(save_dir, f'participant_{participant}_{session_name}_labels_bandpower.npy')
    np.save(features_file, np.array(features))
    np.save(labels_file, labels)
    print(f"Saved: {features_file} and {labels_file}")


if __name__ == "__main__":
    session_folders = ["RS_Beg_EO", "RS_Beg_EC"]
    participants = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 20] #should be adjusted based on the experiment
    trial = 2

    for participant in participants:
        base_dir = os.path.join(f"sub{participant}_res", f"trial{trial}")
        for session in session_folders:
            file_name = f"{session}_noauto-epo_car_bad.fif"
            file_path = os.path.join(base_dir, session, file_name)

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            extract_bandpower_from_file(file_path, participant, session)
