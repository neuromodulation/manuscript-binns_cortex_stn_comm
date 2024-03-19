"""Compute univariate power."""

import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_handle_files import load_file
from coh_power_processing import standard_power_analysis
from coh_loading import load_preprocessed_dict


FOLDERPATH_PREPROCESSING = "Path_to\\Project\\Preprocessing"
FOLDERPATH_PROCESSING = "Path_to\\Project\\Processing"
PRESET = sys.argv[2]  # var
PREPROCESSING = f"preprocessed-{sys.argv[3]}"  # var
ANALYSIS = sys.argv[4]  # var


if __name__ == "__main__":
    preset_fpath = os.path.join(
        FOLDERPATH_PROCESSING,
        "Settings",
        "Generic",
        "Data File Presets",
        f"{PRESET}.json",
    )
    recordings = load_file(preset_fpath)

    preprocessing_analysis = PREPROCESSING[
        PREPROCESSING.index("-")
        + 1 : len(PREPROCESSING)
        - PREPROCESSING[::-1].index("-")
        - 1
    ]

    recording = recordings[int(sys.argv[1])]

    DATASET = recording["cohort"]
    SUBJECT = recording["sub"]
    SESSION = recording["ses"]
    TASK = recording["task"]
    ACQUISITION = recording["acq"]
    RUN = recording["run"]

    preprocessed = load_preprocessed_dict(
        folderpath_preprocessing=FOLDERPATH_PREPROCESSING,
        dataset=DATASET,
        preprocessing=PREPROCESSING,
        subject=SUBJECT,
        session=SESSION,
        task=TASK,
        acquisition=ACQUISITION,
        run=RUN,
        ftype=".pkl",
    )

    power = standard_power_analysis(
        preprocessed,
        FOLDERPATH_PROCESSING,
        DATASET,
        preprocessing_analysis,
        ANALYSIS,
        SUBJECT,
        SESSION,
        TASK,
        ACQUISITION,
        RUN,
        save=True,
        ask_before_overwrite=False,
    )
