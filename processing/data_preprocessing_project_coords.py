"""Project ECoG coordinates to brain surface."""

import os
import sys
from pathlib import Path

import mne
import trimesh
import numpy as np
import pandas as pd

cd_path = Path(os.getcwd()).absolute().parent
sys.path.append(os.path.join(cd_path, "coherence"))


FOLDERPATH_PREPROCESSING = "Path_to\\Project\\Preprocessing"


def project_to_mesh(coords: np.ndarray) -> np.ndarray:
    mesh_name = "mni_icbm152_nlin_asym_09b"
    sample_path = mne.datasets.sample.data_path()
    subjects_dir = sample_path / "subjects"

    # transform coords into proper space for projection
    mri_mni_trans = mne.read_talxfm(mesh_name, subjects_dir)
    mri_mni_inv = np.linalg.inv(mri_mni_trans["trans"])
    coords = mne.transforms.apply_trans(mri_mni_inv, coords)

    path_mesh = f"{subjects_dir}\\{mesh_name}\\surf\\{mesh_name}.glb"
    with open(path_mesh, mode="rb") as f:
        scene = trimesh.exchange.gltf.load_glb(f)
    mesh: trimesh.Trimesh = trimesh.Trimesh(**scene["geometry"]["geometry_0"])
    coords = mesh.nearest.on_surface(coords)[0]
    coords *= 1.05
    # transforms coords back into MNI space
    return mne.transforms.apply_trans(mri_mni_trans, coords)


ecog_lfp_coords = pd.read_csv(
    os.path.join(FOLDERPATH_PREPROCESSING, "ECoG_LFP_coords.csv")
)

# project to right hemisphere
ecog_lfp_coords["x"] = np.abs(ecog_lfp_coords["x"])

# for ECoG coordinates...
ecog_idcs = ecog_lfp_coords["ch"].str.contains("ECOG")
# convert from mm to m before projecting
for key in ["x", "y", "z"]:
    ecog_lfp_coords.loc[ecog_idcs, key] = (
        ecog_lfp_coords.loc[ecog_idcs, key] / 1000
    )
# project to mesh
projected_coords = project_to_mesh(
    ecog_lfp_coords.loc[ecog_idcs, ["x", "y", "z"]].to_numpy()
)
# convert back to mm
for key, value in zip(["x", "y", "z"], projected_coords.T):
    ecog_lfp_coords.loc[ecog_idcs, key] = value * 1000

ecog_lfp_coords.to_csv(
    os.path.join(FOLDERPATH_PREPROCESSING, "ECoG_LFP_coords_projected.csv"),
    index=False,
)
