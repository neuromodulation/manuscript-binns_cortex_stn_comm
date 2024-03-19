"""Preprocesses data."""

import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_preprocess_data import preprocessing


FOLDERPATH_DATA = "Path_to\\Data"
FOLDERPATH_PREPROCESSING = "Path_to\\Project\\Preprocessing"
DATASET = "BIDS_01_Berlin_Neurophys"  # var
ANALYSIS = "for_power"  # var
SETTINGS = "med_analysis"  # var
SUBJECT = "EL006"  # var
SESSION = "EcogLfpMedOff01"  # var
TASK = "Rest"  # var
ACQUISITION = "StimOff"  # var
RUN = "1"  # var


if __name__ == "__main__":
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
