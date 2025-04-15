from scripts import filter as filt
from scripts import to_fif
from scripts import epochs_reject as seg
from pathlib import Path

# Define session names
sessions = [
    "zeroBACK", "oneBACK", "twoBACK",
    "RS_beg_EC", "RS_beg_EO", "RS_End_EC", "RS_End_EO"
]

def preprocess_eeg():
    for subject in range(4, 28):
        subject = str(subject)
        for trial in range(1, 4):
            for session in sessions:
                test_file = f"trial{trial}/{session}/"
                base_path = f"sub{subject}_res/{test_file}"

                path_set = Path(base_path + f"{session}.set")
                path_filtered = Path(base_path + f"{session}_noauto_filt_car_bad.fif")
                path_ica = Path(base_path + f"{session}_noauto-ica_car_bad.fif")
                path_clean_epochs = Path(base_path + f"{session}_noauto-epo_car_bad.fif")

                try:
                    if not path_filtered.is_file():
                        print(f"Filtering: {path_set}")
                        filt.filter(path_set, path_filtered)

                    if not path_ica.is_file():
                        print(f"Performing ICA: {path_set}")
                        to_fif.eeg_to_fif(path_set, path_ica)

                    if not path_clean_epochs.is_file():
                        print(f"Segmenting: {path_filtered}")
                        seg.epoch_and_reject(path_filtered, path_clean_epochs, path_ica)

                except Exception as e:
                    print(f"Error processing {path_set}: {e}")

if __name__ == "__main__":
    preprocess_eeg()
