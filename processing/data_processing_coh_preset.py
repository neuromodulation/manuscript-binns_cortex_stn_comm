"""Compute bivariate coherency-based metrics."""

import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_connectivity_processing import coherence_processing
from coh_handle_files import load_file
from coh_loading import load_preprocessed_dict


FOLDERPATH_PREPROCESSING = "Path_to\\Project\\Preprocessing"
FOLDERPATH_PROCESSING = "Path_to\\Project\\Processing"
PRESET = "Berlin_MedOffOn"  # var
PREPROCESSING = "preprocessed-med_analysis-for_connectivity"  # var
ANALYSIS = "con_imcoh"  # var


if __name__ == "__main__":
    preset_fpath = (
        f"{FOLDERPATH_PROCESSING}\\Settings\\Generic\\Data File Presets\\"
        f"{PRESET}.json"
    )
    recordings = load_file(preset_fpath)

    preprocessing_analysis = PREPROCESSING[
        PREPROCESSING.index("-")
        + 1 : len(PREPROCESSING)
        - PREPROCESSING[::-1].index("-")
        - 1
    ]

    for recording in recordings:
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

        coherence_processing(
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
            verbose=True,
            save=True,
        )
