"""Annotate artefacts in data."""

import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_preprocess_data import preprocessing
from coh_view_data import annotate_data


FOLDERPATH_DATA = "Path_to\\Data"
FOLDERPATH_PREPROCESSING = "Path_to\\Project\\Preprocessing"
DATASET = "BIDS_01_Berlin_Neurophys"  # var
ANALYSIS = "for_annotations"
SETTINGS = "annotation"
SUBJECT = "EL006"  # var
SESSION = "EcogLfpMedOff01"  # var
TASK = "Rest"  # var
ACQUISITION = "StimOff"  # var
RUN = "1"  # var
SHOW_ANNOTATIONS = True


if __name__ == "__main__":
    preprocessed = preprocessing(
        FOLDERPATH_DATA,
        FOLDERPATH_PREPROCESSING,
        DATASET,
        ANALYSIS,
        SETTINGS,
        SUBJECT,
        SESSION,
        TASK,
        ACQUISITION,
        RUN,
    )

    annotate_data(
        preprocessed,
        FOLDERPATH_PREPROCESSING,
        DATASET,
        SUBJECT,
        SESSION,
        TASK,
        ACQUISITION,
        RUN,
        load_annotations=SHOW_ANNOTATIONS,
    )
