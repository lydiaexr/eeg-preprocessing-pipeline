# EEG Data Folder Structure

To run the pipeline, organize your raw EEG `.set` files in the following folder structure:

```
data/
└── sub4_res/
    ├── trial1/
    │   ├── zeroBACK/
    │   │   └── zeroBACK.set
    │   ├── oneBACK/
    │   │   └── oneBACK.set
    │   └── ...
    ├── trial2/
    │   └── ...
    └── trial3/
        └── ...
```

Each `.set` file must be named the same as its containing folder.

## Example File Path
```
data/sub4_res/trial1/zeroBACK/zeroBACK.set
```

The script will output intermediate and final `.fif` files in the same folders.

---

Ensure all `.set` files are valid EEGLAB format and contain appropriate metadata (e.g., channel locations, event annotations).

