import mne
import numpy as np
from scripts.remove_bad_channels import identify_bad_channels

def filter(path_set, path_filtered):
    """
    Load raw EEG (.set), apply bandpass filtering, detect and interpolate bad channels,
    and apply Common Average Reference (CAR).
    """
    raw = mne.io.read_raw_eeglab(path_set, preload=True)

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')
    print(f"Data shape: {raw.get_data().shape}")
    print(f"Channel types: {raw.get_channel_types()}")

    raw.filter(0.1, 30)
    raw = identify_bad_channels(raw)

    car_raw = iterative_car_with_interpolation(raw)
    car_raw.save(path_filtered, overwrite=True)

def iterative_car_with_interpolation(raw, max_iterations=10, noise_threshold=5.0):
    """
    Perform CAR with iterative detection and interpolation of noisy channels.
    """
    raw = raw.copy()
    iteration = 0
    noisy_channels = set()

    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}")

        raw.set_eeg_reference(ref_channels='average', verbose=False)

        data = raw.get_data()
        channel_std = np.std(data, axis=1)
        z_scores = (channel_std - np.mean(channel_std)) / np.std(channel_std)

        current_noisy_channels = set(
            raw.ch_names[i] for i in np.where(z_scores > noise_threshold)[0]
        )
        print(f"Detected noisy channels: {current_noisy_channels}")

        if not current_noisy_channels.difference(noisy_channels):
            break

        noisy_channels.update(current_noisy_channels)
        raw.info['bads'] = list(noisy_channels)
        raw.interpolate_bads(reset_bads=False, verbose=False)

    raw.set_eeg_reference(ref_channels='average', verbose=False)
    print(f"Final interpolated channels: {noisy_channels}")
    return raw
