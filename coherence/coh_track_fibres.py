"""Class for performing fibre tracking."""

import os
from typing import Callable

import numpy as np
import pandas as pd
import scipy as sp


class TrackFibres:
    """Class for performing fibre tracking between seed and target regions.

    Parameters
    ----------
    atlas_path : str
        Filepath to the fibre atlas.

    Methods
    -------
    find_within_radius
        Find the fibres within the radius of seeds (and targets).

    find_closest
        Find the closest fibres and their distance to seeds.

    Notes
    -----
    All units (e.g. coordinates, radii) are in those of the atlas (often mm).

    Distance is computed as the Euclidean distance using the
    scipy.spatial.distance.cdist function.
    """

    def __init__(self, atlas_path: str):  # noqa: D107
        self._load_atlas(atlas_path)

    def _load_atlas(self, atlas_path: str):
        """Load the fibre atlas from a file."""
        if not isinstance(atlas_path, str):
            raise TypeError("`atlas_path` must be a str.")
        self.atlas_path = atlas_path

        if not os.path.exists(self.atlas_path):
            raise ValueError(
                "The path to the atlas file does not exist:\n"
                f"{self.atlas_path}"
            )

        if self.atlas_path.endswith(".mat"):
            self.atlas = sp.io.loadmat(self.atlas_path)["fibers"]
        else:
            raise NotImplementedError(
                "Currently only atlases saved as .mat files are supported."
            )

        if not isinstance(self.atlas, np.ndarray):
            raise TypeError("The fibre atlas must be a NumPy array.")
        if self.atlas.ndim != 2:
            raise ValueError("The fibre atlas must be a 2D array.")
        if self.atlas.shape[1] != 4:
            raise ValueError("The fibre atlas must have the shape (X, 4).")

        self.fibres = self.atlas[:, :3]
        self.ids = self.atlas[:, 3].astype(int)

    def _pad_ragged_coords(self, coords: np.ndarray):
        """Pad potentially ragged arrays with np.nan to a full 3D array.

        Parameters
        ----------
        coords : numpy ndarray, shape of (N, 3) or (N, M, 3)
            Channel coordinates, where N is the number of connection and M the
            number of channels per connection. If the array is ragged, M varies
            across connections. If has shape (N, 3), will be expanded to shape
            (N, 1, 3). If has shape (N, M, 3) and M is variable across
            connections, M is padded with np.nan such that it is consistent
            across connections.

        Returns
        -------
        padded_coords : numpy ndarray, shape of (N, M, 3)
            The padded channel coordinates.
        """
        coords = coords.copy()
        if coords.ndim == 2:
            padded_coords = coords[:, np.newaxis, :]
        else:
            max_n_chans = max([group.shape[0] for group in coords])
            padded_coords = np.full((coords.shape[0], max_n_chans, 3), np.nan)
            for group_i, group in enumerate(coords):
                padded_coords[group_i, : group.shape[0]] = group

        return padded_coords

    def find_within_radius(
        self,
        seed_coords: np.ndarray,
        seed_sphere_radius: int | float,
        target_coords: np.ndarray | None = None,
        target_sphere_radius: int | float | None = None,
        allow_bypassing_fibres: bool = True,
    ) -> tuple[list[list[int]], list[int]]:
        """Find the fibres within the radius of seeds (and targets).

        Parameters
        ----------
        seed_coords : numpy ndarray, shape (N, (M, ) 3)
            Coordinates of the seed regions. Can be an array of shape (N, 3) or
            (N, M, 3), where N is the number of connections, and M is the
            number of channels per connection, and 3 corresponds to the x-, y-,
            and z-axis atlas coordinates. If multiple channels are given per
            connection, the fibre tracking results will be aggregated across
            these channels.

        seed_sphere_radius : int | float
            Radius of the spheres around the seed channels in mm in which
            fibres should be found.

        target_coords : numpy ndarray, shape (N, (K, ) 3) | None (default None)
            Coordinates of the target regions. Can be an array of shape (N, 3)
            or (N, K, 3), where N is the number of connections, and K is the
            number of channels per connection, and 3 corresponds to the x-, y-,
            and z-axis atlas coordinates. If multiple channels are given per
            connection, the fibre tracking results will be aggregated across
            these channels. If ``None``, all fibres within the radii of the
            seeds will be taken.

        target_sphere_radius : int | float | None (default None)
            Radius of the spheres around the target channels in mm in which
            fibres should be found. Can only be ``None`` if `target_coords` is
            also ``None``.

        allow_bypassing_fibres : bool (default True)
            Whether or not to allow the identified fibres to pass through the
            sphere radii. If ``False``, only fibres terminating in the spheres
            will be considered.

        Returns
        -------
        fibre_ids : list of list of int, length N
            Atlas IDs of the found fibres for each connection.

        n_fibres : list of int, length N
            The number of found fibres for each connection.
        """
        self._check_inputs_find_within_radius(
            seed_coords,
            target_coords,
            seed_sphere_radius,
            target_sphere_radius,
            allow_bypassing_fibres,
        )

        fibre_ids = []
        con_i = 0
        for seed_channels, target_channels in zip(
            self._seed_coords, self._target_coords
        ):
            fibre_ids.append([])
            for seed_channel in seed_channels:
                for target_channel in target_channels:
                    fibre_ids[con_i].extend(
                        self._identify_close_fibres(
                            seed_channel, target_channel
                        )
                    )
            fibre_ids[con_i] = np.unique(fibre_ids[con_i]).tolist()
            con_i += 1

        return fibre_ids, [len(fibres) for fibres in fibre_ids]

    def _check_inputs_find_within_radius(
        self,
        seed_coords: np.ndarray,
        target_coords: np.ndarray | None,
        seed_sphere_radius: int | float,
        target_sphere_radius: int | float | None,
        allow_bypassing_fibres: bool,
    ):
        """Check inputs of `find_within_radius`."""
        if not isinstance(seed_coords, np.ndarray):
            raise TypeError("`seed_coords` must be a NumPy array.")
        if target_coords is not None and not isinstance(
            target_coords, np.ndarray
        ):
            raise TypeError("`target_coords` must be a NumPy array.")
        if not isinstance(seed_sphere_radius, (int, float)):
            raise TypeError("`seed_sphere_radius` must be an int or float.")
        if not isinstance(target_sphere_radius, (int, float)):
            if target_coords is not None:
                raise TypeError(
                    "`target_sphere_radius` must be an int or float if "
                    "`target_coords` is not `None`."
                )

        seed_coords = self._pad_ragged_coords(seed_coords)
        if seed_coords.shape[2] != 3:
            raise ValueError(
                "The last axis of `seed_coords` must have a length of 3."
            )
        if target_coords is not None:
            if seed_coords.shape[0] != target_coords.shape[0]:
                raise ValueError(
                    "`seed_coords` and `target_coords` must have the same "
                    "number of entries along axis 0."
                )
            target_coords = self._pad_ragged_coords(target_coords)
            if target_coords.shape[2] != 3:
                raise ValueError(
                    "The last axis of `target_coords` must have a length of 3."
                )
            if target_sphere_radius <= 0:
                raise ValueError("`target_sphere_radius` must be > 0.")
        else:
            target_coords = [[None] for _ in range(seed_coords.shape[0])]

        if seed_sphere_radius <= 0:
            raise ValueError("`seed_sphere_radius` must be > 0.")

        if not isinstance(allow_bypassing_fibres, bool):
            raise TypeError("`allow_bypassing_fibres` must be a bool.")

        self._seed_coords = seed_coords
        self._target_coords = target_coords
        self._seed_sphere_radius = seed_sphere_radius
        self._target_sphere_radius = target_sphere_radius
        self._allow_bypassing_fibres = allow_bypassing_fibres

    def _identify_close_fibres(
        self, seed_coords: np.ndarray, target_coords: np.ndarray | None
    ):
        """Identify the fibres that pass close to one or two channels.

        Parameters
        ----------
        seed_coords : numpy ndarray, shape of (3, )
            The coordinates of the seed channel.

        target_coords : numpy ndarray, shape of (3, ) | None
            The coordinates of the target channel. If ``None``, all fibres
            within the radius of the seed are counted.

        Returns
        -------
        shared_fibres : list of int
            The IDs of the fibres.
        """
        if np.all(np.isnan(seed_coords)):
            if target_coords is not None and np.all(np.isnan(target_coords)):
                return []

        valid_fibres = self.fibres
        valid_ids = self.ids
        if not self._allow_bypassing_fibres:
            fibre_mask = self._filter_bypassing_fibres()
            valid_fibres = self.fibres[fibre_mask]
            valid_ids = self.ids[fibre_mask]

        distances = sp.spatial.distance.cdist(
            valid_fibres, seed_coords[np.newaxis, :]
        )
        close_entries = np.where(distances <= self._seed_sphere_radius)[0]
        close_ids = np.unique(valid_ids[close_entries])
        close_fibres = valid_fibres[np.isin(valid_ids, close_ids)]

        if target_coords is not None and close_ids != []:
            distances = sp.spatial.distance.cdist(
                close_fibres, target_coords[np.newaxis, :]
            )
            close_entries = np.where(distances <= self._target_sphere_radius)[
                0
            ]
            close_ids = np.unique(
                valid_ids[np.isin(valid_ids, close_ids)][close_entries]
            )

        return close_ids.tolist()

    def _filter_bypassing_fibres(self) -> np.ndarray:
        """Create a mask to remove the non-terminating parts of fibres.

        Returns
        -------
        fibre_mask : numpy ndarray
            Boolean mask to filter the fibres.
        """
        # initialize masks
        start_mask = np.ones(len(self.ids), dtype=bool)
        stop_mask = np.ones(len(self.ids), dtype=bool)
        middle_mask = np.zeros(len(self.ids), dtype=bool)

        # detect changing fiber indices for each consecutive row
        changes = np.diff(self.ids).astype(bool)
        start_mask[1:] = changes  # set mask for all fibers except first
        stop_mask[:-1] = changes  # set mask for all fibers except last

        # detect duplicated MNI coordinates for each consecutive row
        changes = np.diff(self.fibres, axis=0).astype(bool)
        changes = ~np.all(changes, axis=1)
        middle_mask[1:] = changes

        # remove duplicated MNI coords for each fiber to eliminate fork coords
        fibers = self.fibres.copy()
        fibers[~middle_mask] = np.nan  # remove irrelevant coords
        middle_mask = np.array(~pd.DataFrame(fibers).duplicated(keep=False))

        return start_mask | stop_mask | middle_mask

    def find_closest(
        self,
        seed_coords: np.ndarray,
        normalise_distance: Callable | None = None,
    ) -> tuple[list[int], list[float]]:
        """Find the closest fibres and their distance to seeds.

        Parameters
        ----------
        seed_coords : numpy ndarray, shape of (N, 3)
            Coordinates of the seed regions with shape (N, 3), where N is the
            number of connections and 3 corresponds to the x-, y-, and z-axis
            atlas coordinates.

        normalise_distance : lambda function | None = None
            Normalisation to apply to the distances. E.g. to take the inverse
            of the distance squared, one can use `lambda x: (1 / x) ** 2`. If
            ``None``, no normalisation is applied.

        Returns
        -------
        fibre_ids : list of int, length N
            Atlas IDs of the closest fibre for each connection.

        distances : list of float, length N
            The distance between seed and closest fibre for each connection.
        """
        self._check_inputs_find_closest(seed_coords, normalise_distance)

        distances = sp.spatial.distance.cdist(self.fibres, seed_coords)
        closest_fibre_ids = self.ids[np.argmin(distances, axis=0)]
        smallest_distances = np.min(distances, axis=0)
        if normalise_distance is not None:
            smallest_distances = normalise_distance(smallest_distances)

        return closest_fibre_ids.tolist(), smallest_distances.tolist()

    def _check_inputs_find_closest(
        self, seed_coords: np.ndarray, normalise_distance: Callable | None
    ):
        """Check inputs of `find_closest`."""
        if not isinstance(seed_coords, np.ndarray):
            raise TypeError("`seed_coords` must be a NumPy array.")
        if seed_coords.ndim != 2 or seed_coords.shape[1] != 3:
            raise ValueError("`seed_coords` must be an (N, 3) array.")

        if normalise_distance is not None and not callable(normalise_distance):
            raise TypeError(
                "`normalise_distance` must be a lambda function or `None`."
            )

        self._seed_coords = seed_coords
        self._normalise_distance = normalise_distance
