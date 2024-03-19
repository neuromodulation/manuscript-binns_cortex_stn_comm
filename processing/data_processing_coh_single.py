"""Compute bivariate coherency-based metrics."""

import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_connectivity_processing import coherence_processing
from coh_loading import load_preprocessed_dict


FOLDERPATH_PREPROCESSING = "Path_to\\Project\\Preprocessing"
FOLDERPATH_PROCESSING = "Path_to\\Project\\Processing"
DATASET = "BIDS_01_Berlin_Neurophys"  # var
PREPROCESSING = "preprocessed-med_analysis-for_connectivity"  # var
ANALYSIS = "con_imcoh"  # var
SUBJECT = "EL006"  # var
SESSION = "EcogLfpMedOff01"  # var
TASK = "Rest"  # var
ACQUISITION = "StimOff"  # var
RUN = "1"  # var


if __name__ == "__main__":
    preprocessing_analysis = PREPROCESSING[
        PREPROCESSING.index("-")
        + 1 : len(PREPROCESSING)
        - PREPROCESSING[::-1].index("-")
        - 1
    ]

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
