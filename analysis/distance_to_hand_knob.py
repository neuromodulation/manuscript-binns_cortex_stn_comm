"""Determines the distance from ECoG contacts to the hand knob area."""

import csv
import pandas as pd
import numpy as np
from scipy.stats import sem

FILEPATH_COORDS = "Path_to\\Contact_coordinates"

# Load coordinates of ECoG channels
with open(FILEPATH_COORDS, encoding="utf-8-sig") as file:
    coords = list(csv.reader(file, delimiter=","))
coords = pd.read_csv(FILEPATH_COORDS, index_col="Var1")
coords = coords.loc[coords["ch"].str.contains("ECOG")].reset_index(drop=True)
coords["x"] = np.abs(coords["x"])
coords["xyz"] = (np.array([coords["x"], coords["y"], coords["z"]]).T).tolist()

# Compute distance of ECoG channels to the hand knob area (note that this is a
# very conservative estimate of a single MNI coordinate, whereas the hand-knob
# area is actually a larger, subject-specific region)
hand_knob_mni = np.array([36, -19, 73])
distances = np.linalg.norm(
    np.array(coords["xyz"].tolist()) - hand_knob_mni, axis=1
)

print(
    f"Mean distance = {np.mean(distances):.1f} mm  "
    f"(SEM +/- {sem(distances):.1f})\n"
)

# (also min. distances for each subject)
subjects = np.unique(coords["sub"].tolist())
sub_distances = []
for sub in subjects:
    sub_coords = np.array(coords["xyz"].loc[coords["sub"] == sub].to_list())
    sub_distances.append(
        np.linalg.norm(sub_coords - hand_knob_mni, axis=1).min()
    )
sub_distances = np.array(sub_distances)
print(
    f"Mean minimum distance = {np.mean(sub_distances):.1f} mm  "
    f"(SEM +/- {sem(sub_distances):.1f})"
)
