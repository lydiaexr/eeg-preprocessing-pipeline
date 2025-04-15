import mne
import numpy as np

def identify_bad_channels(raw):
    """
    Detect bad EEG channels using variance, NaN/Inf detection, and flat signals.

    Parameters:
        raw (mne.io.Raw): Raw EEG data

    Returns:
        mne.io.Raw: Raw data with bad channels marked in raw.info['bads']
    """
    data = raw.get_data(picks='eeg')
    bad_channels = []

    print(f"Inf count: {np.isinf(data).sum()}, NaN count: {np.isnan(data).sum()}")

    # High variance detection
    channel_variances = np.var(data, axis=1)
    threshold = np.mean(channel_variances) + 3 * np.std(channel_variances)
    bad_channels.extend(
        raw.ch_names[i] for i, var in enumerate(channel_variances) if var > threshold
    )

    # NaNs/Infs per channel
    for i, channel_data in enumerate(data):
        if np.any(np.isnan(channel_data)) or np.any(np.isinf(channel_data)):
            bad_channels.append(raw.ch_names[i])

    # Flat signal detection
    flat_channels = np.ptp(data, axis=1) < 1e-6
    bad_channels.extend(
        raw.ch_names[i] for i, is_flat in enumerate(flat_channels) if is_flat
    )

    raw.info['bads'] = list(set(bad_channels))
    print(f"Identified bad channels: {raw.info['bads']}")
    return raw
