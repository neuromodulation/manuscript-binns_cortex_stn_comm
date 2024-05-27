"""Preprocesses data."""

import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_handle_files import load_file
from coh_preprocess_data import preprocessing


FOLDERPATH_DATA = os.path.join(cd_path, "DEMO\\data")
FOLDERPATH_PREPROCESSING = os.path.join(cd_path, "Project\\Preprocessing")
PRESET = "Demo_OffOn"
ANALYSIS = "for_demo"
SETTINGS = "demo_analysis"


if __name__ == "__main__":
    preset_fpath = (
        f"{FOLDERPATH_PREPROCESSING}\\Settings\\Generic\\Data File Presets\\"
        f"{PRESET}.json"
    )
    settings = load_file(preset_fpath)

    for recording_settings in settings:
        DATASET = recording_settings["cohort"]
        SUBJECT = recording_settings["sub"]
        SESSION = recording_settings["ses"]
        TASK = recording_settings["task"]
        ACQUISITION = recording_settings["acq"]
        RUN = recording_settings["run"]

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
