"""Preprocesses data."""

import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_handle_files import load_file
from coh_preprocess_data import preprocessing


FOLDERPATH_DATA = "Path_to\\Data"
FOLDERPATH_PREPROCESSING = "Path_to\\Project\\Preprocessing"
PRESET = sys.argv[2]  # var
ANALYSIS = sys.argv[3]  # var
SETTINGS = sys.argv[4]  # var


if __name__ == "__main__":
    preset_fpath = os.path.join(
        FOLDERPATH_PREPROCESSING,
        "Settings",
        "Generic",
        "Data File Presets",
        f"{PRESET}.json",
    )
    recordings = load_file(preset_fpath)

    recording = recordings[int(sys.argv[1])]

    DATASET = recording["cohort"]
    SUBJECT = recording["sub"]
    SESSION = recording["ses"]
    TASK = recording["task"]
    ACQUISITION = recording["acq"]
    RUN = recording["run"]

    preprocessing(
        folderpath_data=FOLDERPATH_DATA,
        folderpath_preprocessing=FOLDERPATH_PREPROCESSING,
        dataset=DATASET,
        analysis=ANALYSIS,
        settings=SETTINGS,
        subject=SUBJECT,
        session=SESSION,
        task=TASK,
        acquisition=ACQUISITION,
        run=RUN,
        save=True,
    )
