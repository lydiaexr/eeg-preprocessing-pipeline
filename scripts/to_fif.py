import mne
import numpy as np

def eeg_to_fif(path_set, path_ica):
    """
    Loads raw EEG data, performs ICA to remove EOG components, and saves the ICA object.

    Parameters:
        path_set (Path): Path to the .set file
        path_ica (Path): Output path for the ICA .fif file
    """
    raw = mne.io.read_raw_eeglab(path_set, preload=True)

    ica = mne.preprocessing.ICA(n_components=0.99, random_state=42)
    events = mne.make_fixed_length_events(raw, duration=1.0)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=1, baseline=None, preload=True)

    ica.fit(epochs)

    # Detect EOG artifacts using common frontal channels
    eog_indices, _ = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2', 'F7', 'F8'])
    ica.exclude = list(eog_indices)
    print(f"Excluding EOG components: {ica.exclude}")

    ica.save(path_ica, overwrite=True)
