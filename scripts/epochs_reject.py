import numpy as np
import mne
from mne.io import RawArray

def apply_ica_consistently(epochs, ica):
    """
    Applies a trained ICA to epochs. If channels from the ICA are missing in the epochs,
    they are reintroduced as flat channels.

    Parameters:
        epochs (mne.Epochs): EEG epochs
        ica (mne.preprocessing.ICA): Trained ICA object

    Returns:
        mne.Epochs: ICA-corrected epochs
    """
    try:
        missing_channels = list(set(ica.info['ch_names']) - set(epochs.info['ch_names']))
        if missing_channels:
            print(f"Reintroducing missing channels as flat: {missing_channels}")
            for ch_name in missing_channels:
                zero_data = np.zeros((1, epochs.get_data().shape[2]))
                zero_info = mne.create_info([ch_name], sfreq=epochs.info['sfreq'], ch_types='eeg')
                flat_channel = RawArray(zero_data, zero_info)
                epochs.add_channels([flat_channel])

        return ica.apply(epochs)

    except Exception as e:
        print(f"Error applying ICA: {e}")
        raise
