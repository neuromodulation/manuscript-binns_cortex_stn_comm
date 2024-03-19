"""Analyse processed data."""

import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_results_analysis import analyse


FOLDERPATH_PROCESSING = "Path_to\\Project\\Processing"
FOLDERPATH_ANALYSIS = "Path_to\\Project\\Analysis"
ANALYSIS = "con_mic_whole-MedOffOn_multi_sub"  # var
TO_ANALYSE = "Berlin_MedOffOn"  # var


if __name__ == "__main__":
    analysed = analyse(
        folderpath_processing=FOLDERPATH_PROCESSING,
        folderpath_analysis=FOLDERPATH_ANALYSIS,
        analysis=ANALYSIS,
        to_analyse=TO_ANALYSE,
        to_analyse_ftype=".pkl",
        save=True,
    )
