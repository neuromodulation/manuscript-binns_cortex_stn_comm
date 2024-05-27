"""Analyse processed data."""

import os
import sys
from pathlib import Path

cd_path = Path(__file__).absolute().parent.parent
sys.path.append(os.path.join(cd_path, "coherence"))
from coh_results_analysis import analyse


FOLDERPATH_PROCESSING = os.path.join(cd_path, "Project\\Processing")
FOLDERPATH_ANALYSIS = os.path.join(cd_path, "Project\\Analysis")
ANALYSIS = "con_demo-DemoOffOn_multi_sub"
TO_ANALYSE = "Demo_OffOn"


if __name__ == "__main__":
    analysed = analyse(
        folderpath_processing=FOLDERPATH_PROCESSING,
        folderpath_analysis=FOLDERPATH_ANALYSIS,
        analysis=ANALYSIS,
        to_analyse=TO_ANALYSE,
        to_analyse_ftype=".pkl",
        save=True,
    )
