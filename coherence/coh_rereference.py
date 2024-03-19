"""Classes for rereferencing data in an MNE Raw object.

CLASSES
-------
Reref : abstract base class
-   Abstract class for rereferencing data in an MNE Raw object.

RerefBipolar : subclass of Reref
-   Bipolar rereferences data in an MNE Raw object.

RerefCommonAverage : subclass of Reref
-   Common-average rereferences data in an MNE Raw object.

RerefPseudo: subclass of Reref
-   Pseudo rereferences data in an MNE Raw object.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Union
import mne
import numpy as np
from coh_handle_entries import (
    check_lengths_list_equals_n,
    check_lengths_list_identical,
)


class Reref(ABC):
    """Abstract class for rereferencing data in an MNE Raw object.

    PARAMETERS
    ----------
    raw : MNE Raw
    -   The MNE Raw object containing the data to be rereferenced.

    ch_names_old : list[str]
    -   The names of the channels in the MNE Raw object to rereference.

    ch_names_new : list[str | None] | None; default None
    -   The names of the newly rereferenced channels, corresponding to the
        channels used for rerefrencing in ch_names_old.
    -   'None' values can be set based on 'ch_names_old'.

    ch_types_new : list[str | None] | None; default None
    -   The types of the newly rereferenced channels as recognised by MNE,
        corresponding to the channels in 'ch_names_new'.
    -   'None' values can be set based on the types of channels in
        'ch_names_old'.

    ch_reref_types : list[str | None] | None; default None
    -   The rereferencing type applied to the channels, corresponding to the
        channels in 'ch_names_new'.
    -   If some or all entries are 'None', they can be set as the rereferencing
        type being applied.

    ch_coords_new : list[list[int | float] | None] | None; default None
    -   The coordinates of the newly rereferenced channels, corresponding to
        the channels in 'ch_names_new'. The list should consist of sublists
        containing the x, y, and z coordinates of each channel.
    -   If the input is '[]' or some inputs are '[]', coordinates for the
        channels being rereferenced can be used.

    ch_regions_new : list[str | None] | None; default None
    -   The regions of the newly rereferenced channels, corresponding to the
        channels in 'ch_names_new'.
    -   If 'None', the regions can be determined based on the regions of the
        channels being rereferenced. If some entries are 'None', they are left
        as-is.

    ch_subregions_new : list[str | None] | None; default None
    -   The subregions of the newly rereferenced channels, corresponding to the
        channels in 'ch_names_new'.
    -   If 'None', the subregions can be determined based on the subregions of
        the channels being rereferenced. If some entries are 'None', they are
        left as-is.

    ch_hemispheres_new : list[str | None] | None; default None
    -   The hemispheres of the newly rereferenced channels, corresponding to the
        channels in 'ch_names_new'.
    -   If 'None' or if some entries are 'None', the hemispheres can be
        determined based on the subregions of  the channels being rereferenced.


    METHODS
    -------
    rereference (abstract)
    -   Rereferences the data in an MNE Raw object.
    """

    def __init__(
        self,
        raw: mne.io.Raw,
        extra_info: dict,
        ch_names_old: list[str],
        ch_names_new: Union[list[Union[str, None]], None] = None,
        ch_types_new: Union[list[Union[str, None]], None] = None,
        ch_reref_types: Union[list[Union[str, None]], None] = None,
        ch_coords_new: Union[
            list[Union[list[Union[int, float]], None]], None
        ] = None,
        ch_regions_new: Union[list[Union[str, None]], None] = None,
        ch_subregions_new: Union[list[Union[str, None]], None] = None,
        ch_hemispheres_new: Union[list[Union[str, None]], None] = None,
    ) -> None:
        # Initialises aspects of the Reref object that will be filled with
        # information as the data is processed.
        self._new_data = None
        self._new_data_info = None
        self._new_ch_coords = None
        self._n_channels = None
        self.ch_reref_types = None
        self.ch_regions = None
        self.ch_subregions = None
        self.ch_hemispheres = None
        self.epoch_orders = None

        # Initialises inputs of the Reref object.
        self.raw = raw
        (
            self._data,
            self._data_info,
            self._ch_names,
            self._ch_coords,
        ) = self._data_from_raw(self.raw)
        self._ch_names_old = ch_names_old
        self._ch_index = self._index_old_channels(
            ch_names=self._ch_names, reref_ch_names=self._ch_names_old
        )
        self._ch_names_new = ch_names_new
        self._ch_types_new = ch_types_new
        self._ch_coords_new = ch_coords_new
        self._ch_regions_new = ch_regions_new
        self._ch_subregions_new = ch_subregions_new
        self._ch_hemispheres_new = ch_hemispheres_new
        self._ch_reref_types = ch_reref_types
        self.extra_info = deepcopy(extra_info)
        self._sort_inputs()

    @abstractmethod
    def _sort_inputs(self) -> None:
        """Checks that rereferencing settings are compatible and discards
        rereferencing-irrelevant channels from the data."""

    @abstractmethod
    def _set_data(self) -> None:
        """Rereferences the data."""

    @abstractmethod
    def _set_coordinates(self) -> None:
        """Sets the coordinates of the new, rereferenced channels."""

    def _index_old_channels(
        self, ch_names: list[str], reref_ch_names: list[str]
    ) -> None:
        """Creates an index of channels that are being rereferenced.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   Names of the channels in the data.

        reref_ch_names : list[str]
        -   Names of the new channels being rereferenced.

        RETURNS
        -------
        list[int]
        -   Indices of the channels being rereferenced.
        """
        return [ch_names.index(name) for name in reref_ch_names]

    def _check_input_lengths(self) -> None:
        """Checks that the lengths of the entries (representing the features of
        channels that will be rereferenced, e.g. channel names, coordinates,
        etc...) within a list are of the same length.
        -   This length corresponds to the number of channels in the
            rereferenced data.

        RAISES
        ------
        ValueError
        -   Raised if the lengths of the list's entries are nonidentical.
        """
        equal_lengths, self._n_channels = check_lengths_list_identical(
            to_check=[
                self._ch_names_old,
                self._ch_names_new,
                self._ch_types_new,
                self._ch_reref_types,
                self._ch_coords_new,
                self._ch_regions_new,
                self._ch_subregions_new,
                self._ch_hemispheres_new,
            ],
            ignore_values=[None],
        )

        if not equal_lengths:
            raise ValueError(
                "Error when reading rereferencing settings:\nThe length of "
                "entries within the settings dictionary are not identical:\n"
                f"{self._n_channels}"
            )

    def _sort_raw(self, chs_to_analyse: list[str]) -> None:
        """Drops channels irrelevant to the rereferencing from an MNE Raw
        object.

        PARAMETERS
        ----------
        chs_to_analyse : list[str]
        -   List containing the names of the channels in MNE Raw to retain.
        """
        self.raw.drop_channels(
            [
                name
                for name in self.raw.info["ch_names"]
                if name not in chs_to_analyse
            ]
        )
        self.raw.reorder_channels(chs_to_analyse)

    def _sort_feature(self, feature: str, replacement: list[Any]) -> None:
        """Resolves any missing entries for channel features (e.g. names, types,
        etc...), taking the features from the channels being rereferenced.

        PARAMETERS
        ----------
        features : str
        -   Name of the feature being sorted. Recognised inputs are:
            "ch_names_new"; "ch_types_new"; "ch_reref_types"; "ch_coords_new"

        RAISES
        ------
        NotImplementedError
        -   Raised if the feature is not supported.
        ValueError
        -   Raised if channel coordinates are being sorted and any of the
            channel coordinates do not contain 3 entries, i.e. x, y, and z
            coordinates.
        """
        supported_features = [
            "ch_names_new",
            "ch_types_new",
            "ch_reref_types",
            "ch_coords_new",
            "ch_regions_new",
            "ch_subregions_new",
            "ch_hemispheres_new",
        ]
        if feature not in supported_features:
            raise NotImplementedError(
                f"The feature '{feature}' is not recognised. Supported "
                f"features are {supported_features}."
            )

        feature = f"_{feature}"
        attribute = getattr(self, feature)
        if attribute is None:
            setattr(self, feature, replacement)
        elif any(item is None for item in attribute):
            for ch_i, ch_attr in enumerate(attribute):
                if ch_attr is None:
                    setattr(self, feature, replacement[ch_i])

        if feature == "_ch_coords_new":
            if not check_lengths_list_equals_n(
                to_check=self._ch_coords_new, n=3
            ):
                raise ValueError(
                    "Error when setting coordinates for the rereferenced "
                    "data:\nThree, and only three coordinates (x, y, and z) "
                    "must be present, but the rereferencing settings specify "
                    "otherwise."
                )

    def _data_from_raw(
        self, raw: mne.io.Raw
    ) -> tuple[np.ndarray, mne.Info, list[str], list[list[Union[int, float]]]]:
        """Extracts components of an MNE Raw object and returns them.

        PARAMETERS
        ----------
        raw : MNE Raw
        -   The MNE Raw object whose data and information should be
            extracted.

        RETURNS
        -------
        numpy array
        -   Array of the data with shape [n_channels, n_timepoints].

        mne.Info
        -   Information taken from the MNE Raw object.

        list[str]
        -   List of channel names taken from the MNE Raw object corresponding
            to the channels in the data array.

        list[list[int | float]]
        -   List of channel coordinates taken from the MNE Raw object, with
            each channel's coordinates given in a sublist containing the x, y,
            and z coordinates.
        """
        return (
            raw.get_data(reject_by_annotation=None).copy(),
            raw.info.copy(),
            raw.info["ch_names"].copy(),
            raw._get_channel_positions(picks=raw.ch_names).copy().tolist(),
        )

    def _raw_from_data(self) -> None:
        """Generates an MNE Raw object based on the rereferenced data and its
        associated information."""
        self.raw = mne.io.RawArray(self._new_data, self._new_data_info)
        if self._new_ch_coords:
            self.raw._set_channel_positions(
                self._new_ch_coords, self._new_data_info["ch_names"]
            )

    def _store_feature(self, feature: str) -> None:
        """Generates a dictionary where the keys are channel names and the
        values correspond to a specified feature of the channels.

        PARAMETERS
        ----------
        feature : str
        -   The name of a feature of the channels. Recognised values are:
            "ch_reref_types"; "ch_regions"; "ch_subregions", and
            "ch_hemispheres". The values are stored in the object under these
            names.

        RAISES
        ------
        NotImplementedError
        -   Raised if the feature to be stored is not supported.
        """
        supported_features = [
            "ch_reref_types",
            "ch_regions",
            "ch_subregions",
            "ch_hemispheres",
        ]
        if feature not in supported_features:
            raise NotImplementedError(
                f"The feature '{feature}' is not recognised. Only the features "
                f"{supported_features} are supported."
            )

        if feature == "ch_reref_types":
            get_name = f"_{feature}"
        else:
            get_name = f"_{feature}_new"

        feature_dict = {
            self._ch_names_new[i]: getattr(self, get_name)[i]
            for i in range(len(self._ch_names_new))
        }
        setattr(self, feature, feature_dict)

    def _set_data_info(self) -> None:
        """Creates an mne.Info object containing information about the newly
        rereferenced data."""
        self._new_data_info = mne.create_info(
            self._ch_names_new, self._data_info["sfreq"], self._ch_types_new
        )
        add_info = ["experimenter", "line_freq", "description", "subject_info"]
        for key in add_info:
            self._new_data_info[key] = self._data_info[key]

    def rereference(
        self,
    ) -> tuple[mne.io.Raw, list[str], dict[str], dict[str]]:
        """Rereferences the data in an MNE Raw object.

        RETURNS
        -------
        MNE Raw
        -   The MNE Raw object containing the bipolar rereferenced data.

        list[str]
        -   Names of the channels that were produced by the rereferencing.

        dict[str]
        -   Dictionary containing information about the type of rereferencing
            applied to generate each new channel, with key:value pairs of
            channel name : rereference type.

        dict[str]
        -   Dictionary containing information about the regions of each new
            channel, with key:value pairs of channel name : region.

        dict[str]
        -   Dictionary containing information about the subregions of each new
            channel, with key:value pairs of channel name : subregion.

        dict[str]
        -   Dictionary containing information about the hemispheres of each new
            channel, with key:value pairs of channel name : hemisphere.
        """
        self._set_data()
        self._set_coordinates()
        self._set_data_info()
        self._raw_from_data()
        self._store_feature(feature="ch_reref_types")
        self._store_feature(feature="ch_regions")
        self._store_feature(feature="ch_subregions")
        self._store_feature(feature="ch_hemispheres")

        return (
            self.raw,
            self._ch_names_new,
            self.ch_reref_types,
            self.ch_regions,
            self.ch_subregions,
            self.ch_hemispheres,
        )


class RerefBipolar(Reref):
    """Bipolar rereferences data in an MNE Raw object.

    PARAMETERS
    ----------
    raw : MNE Raw
    -   The MNE Raw object containing the data to be rereferenced.

    ch_names_old : list[str]
    -   The names of the channels in the MNE Raw object to rereference.

    ch_names_new : list[str | None] | None; default None
    -   The names of the newly rereferenced channels, corresponding to the
        channels used for rerefrencing in ch_names_old.
    -   Missing values (None) will be set based on 'ch_names_old'.

    ch_types_new : list[str | None] | None; default None
    -   The types of the newly rereferenced channels as recognised by MNE,
        corresponding to the channels in 'ch_names_new'.
    -   Missing values (None) will be set based on the types of channels in
        'ch_names_old'.

    ch_reref_types : list[str | None] | None; default None
    -   The rereferencing type applied to the channels, corresponding to the
        channels in 'ch_names_new'.
    -   If some or all entries are None, they will be set as 'bipolar'.

    ch_coords_new : list[list[int | float] | None] | None; default None
    -   The coordinates of the newly rereferenced channels, corresponding to
        the channels in 'ch_names_new'. The list should consist of sublists
        containing the x, y, and z coordinates of each channel.
    -   If the input is '[]', the coordinates of the channels in
        'ch_names_old' in the MNE Raw object are used.
    -   If some sublists are '[]', those channels for which coordinates are
        given are used, whilst those channels for which the coordinates are
        missing have their coordinates taken from the MNE Raw object
        according to the corresponding channel in 'ch_names_old'.

    ch_regions_new : list[str | None] | None; default None
    -   The regions of the newly rereferenced channels, corresponding to the
        channels in 'ch_names_new'.
    -   If 'None', the regions can be determined based on the regions of the
        channels being rereferenced. If some entries are 'None', they are left
        as-is.

    ch_subregions_new : list[str | None] | None; default None
    -   The subregions of the newly rereferenced channels, corresponding to the
        channels in 'ch_names_new'.
    -   If 'None', the subregions can be determined based on the subregions of
        the channels being rereferenced. If some entries are 'None', they are
        left as-is.

    ch_hemispheres_new : list[str | None] | None; default None
    -   The hemispheres of the newly rereferenced channels, corresponding to the
        channels in 'ch_names_new'.
    -   If 'None' or if some entries are 'None', the hemispheres can be
        determined based on the subregions of  the channels being rereferenced.

    METHODS
    -------
    rereference
    -   Rereferences the data in an MNE Raw object.
    """

    def _check_ch_names_old(self) -> None:
        """Checks that two channel names (i.e. an anode and a cathode) are given
        for each new, rereferenced channels.

        RAISES
        ------
        ValueError
        -   Raised if two channel names (i.e. an anode and a cathode) in the
            original data are not provided for each new, bipolar-rereferenced
            channel being produced.
        """
        if not check_lengths_list_equals_n(to_check=self._ch_names_old, n=2):
            raise ValueError(
                "Error when bipolar rereferencing data:\nThis must involve "
                "two, and only two channels of data, but the rereferencing "
                "settings specify otherwise."
            )

    def _sort_ch_names_new(self) -> None:
        """Resolves any missing entries for the names of the new, rereferenced
        channels, taking names from the channels being rereferenced."""
        if self._ch_names_new is None:
            for ch_names in self._ch_names_old:
                self._ch_names_new.append("-".join(name for name in ch_names))
        elif any(item is None for item in self._ch_names_new):
            for i, ch_name in enumerate(self._ch_names_new):
                if ch_name is None:
                    self._ch_names_new[i] = "-".join(
                        name for name in self._ch_names_old[i]
                    )

    def _sort_feature(
        self, feature: str, replacement: list[list[Any]]
    ) -> None:
        """Resolves any missing entries from the channel features of the new
        channels.

        PARAMETERS
        ----------
        feature : str
        -   The feature of the channels to sort. Supported inputs are:
            "ch_types_new"; "ch_regions_new"; "ch_subregions_new";
            "ch_hemispheres_new". Features are stored under these names,
            preceded by an underscore (e.g. "_ch_types_new") in the object.

        replacement : list[Any]
        -   The values to replace any missing entries with. Each new channel
            should have its own sublist in which the features of the channels it
            is being rereferenced from is contained.

        RAISES
        ------
        NotImplementedError
        -   Raised if the feature being sorted is not recognised.
        ValueError
        -   Raised if the feature of the a new channel is not specified and the
            channels it is being rereferenced from have different features.
        """
        supported_features = [
            "ch_types_new",
            "ch_regions_new",
            "ch_subregions_new",
            "ch_hemispheres_new",
        ]
        if feature not in supported_features:
            raise NotImplementedError(
                f"The feature '{feature}' is not recognised. Supported "
                f"features are {supported_features}."
            )

        feature = f"_{feature}"
        attr = getattr(self, feature)
        new_attr = []
        if attr is None:
            for group_features in replacement:
                if len(np.unique(group_features)) == 1:
                    new_attr.append(group_features[0])
                else:
                    raise ValueError(
                        f"The channel feature '{feature[1:]}' has not been "
                        "specified for the rereferenced channels, but they "
                        "cannot be generated automatically based on the "
                        "channels being rereferenced as this feature of the "
                        f"channels is different ({np.unique(group_features)})."
                    )
        elif any(item is None for item in attr):
            for ch_i, ch_feature in enumerate(attr):
                if ch_feature is None:
                    if len(np.unique(replacement[ch_i])) == 1:
                        new_attr.append(replacement[ch_i][0])
                    else:
                        raise ValueError(
                            f"The channel feature '{feature[1:]}' has not been "
                            "specified for the rereferenced channel "
                            f"'{self._ch_names_new[ch_i]}', but this cannot be "
                            "generated automatically based on the channels "
                            "being rereferenced as this feature of the "
                            "channels is different "
                            f"({np.unique(replacement[ch_i])})."
                        )
                else:
                    new_attr.append(ch_feature)
        else:
            new_attr = attr
        setattr(self, feature, new_attr)

    def _sort_ch_coords_new(self) -> None:
        """Resolves any missing entries for the channel coordinates of the new,
        rereferenced channels by taking the value of the coordinates of the
        channels being rereferenced.

        RAISES
        ------
        ValueError
        -   Raised if any of the channel coordinates do not contain 3 entries,
            i.e. x, y, and z coordinates.
        """
        ch_coords_old = [
            [
                self._ch_coords[self._ch_index[i][0]],
                self._ch_coords[self._ch_index[i][1]],
            ]
            for i in range(self._n_channels)
        ]

        if self._ch_coords_new is None:
            self._ch_coords_new = [
                np.around(np.mean(ch_coords_old[i], axis=0), 5)
                for i in range(self._n_channels)
            ]
        elif any(item is None for item in self._ch_coords_new):
            for i, ch_coords in enumerate(self._ch_coords_new):
                if ch_coords is None:
                    self._ch_coords_new[i] = np.around(
                        np.mean(ch_coords_old[i], axis=0), 5
                    )

        if not check_lengths_list_equals_n(to_check=self._ch_coords_new, n=3):
            raise ValueError(
                "Error when setting coordinates for the rereferenced data:\n"
                "Three, and only three coordinates (x, y, and z) must be "
                "present, but the rereferencing settings specify otherwise."
            )

    def _sort_inputs(self) -> None:
        """Checks that rereferencing settings are compatible and discards
        rereferencing-irrelevant channels from the data."""
        ch_types_old = [
            list(np.unique(self.raw.get_channel_types(ch_names)))
            for ch_names in self._ch_names_old
        ]
        features = ["ch_hemispheres", "ch_regions", "ch_subregions"]
        ch_features = {}
        for feature in features:
            ch_features[feature] = []
            if feature in self.extra_info.keys():
                attr = self.extra_info[feature]
            else:
                attr = ["none" for _ in self._ch_index]
            for ch_index in self._ch_index:
                ch_features[feature].append(
                    np.unique(
                        [attr[self._ch_names[ch_i]] for ch_i in ch_index]
                    ).tolist()
                )

        self._check_input_lengths()
        self._sort_raw(
            chs_to_analyse=np.unique(
                [name for names in self._ch_names_old for name in names]
            ).tolist(),
        )
        self._sort_ch_names_new()
        self._sort_feature("ch_types_new", ch_types_old)
        super()._sort_feature(
            feature="ch_reref_types",
            replacement=["bipolar"] * self._n_channels,
        )
        self._sort_ch_coords_new()
        self._sort_feature("ch_regions_new", ch_features["ch_regions"])
        self._sort_feature("ch_subregions_new", ch_features["ch_subregions"])
        self._sort_feature("ch_hemispheres_new", ch_features["ch_hemispheres"])

    def _index_old_channels(
        self, ch_names: list[str], reref_ch_names: list[str]
    ) -> None:
        """Creates an index of channels that are being rereferenced.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   Names of the channels in the data.

        reref_ch_names : list[str]
        -   Names of the new channels being rereferenced.

        RETURNS
        -------
        list[int]
        -   Indices of the channels being rereferenced.
        """
        ch_index = deepcopy(reref_ch_names)
        for sublist_i, sublist in enumerate(reref_ch_names):
            for name_i, name in enumerate(sublist):
                ch_index[sublist_i][name_i] = ch_names.index(name)

        return ch_index

    def _set_data(self) -> None:
        """Bipolar rereferences the data, subtracting one channel's data from
        another channel's data."""
        self._new_data = [
            self._data[self._ch_index[ch_i][0]]
            - self._data[self._ch_index[ch_i][1]]
            for ch_i in range(self._n_channels)
        ]

    def _set_coordinates(self) -> None:
        """Sets the coordinates of the new, rereferenced channels.
        -   If no coordinates are provided, the coordinates are calculated by
            taking the average coordinates of the anode and the cathode involved
            in each new, rereferenced channel.
        -   If some coordinates are provided, this calculation is performed for
            only those missing coordinates.

        RAISES
        ------
        ValueError
        -   Raised if the provided list of coordinates for a channel does not
            have a length of 3 (i.e. an x, y, and z coordinate).
        """
        self._new_ch_coords = []
        for ch_i in range(self._n_channels):
            coords_set = False
            if self._ch_coords_new != []:
                if self._ch_coords_new[ch_i] != []:
                    if not check_lengths_list_equals_n(
                        to_check=self._ch_coords_new, n=3, ignore_values=[[]]
                    ):
                        raise ValueError(
                            "Error when setting coordinates for the "
                            "rereferenced data:\nThree, and only three "
                            "coordinates (x, y, and z) must be present, but "
                            "the rereferencing settings specify otherwise."
                        )
                    self._new_ch_coords.append(self._ch_coords_new[ch_i])
                    coords_set = True
            if coords_set is False:
                self._new_ch_coords.append(
                    np.around(
                        np.mean(
                            [
                                self._ch_coords[self._ch_index[ch_i][0]],
                                self._ch_coords[self._ch_index[ch_i][1]],
                            ],
                            axis=0,
                        ),
                        5,
                    )
                )


class RerefCommonAverage(Reref):
    """Common-average rereferences data in an MNE Raw object.

    PARAMETERS
    ----------
    raw : MNE Raw
    -   The MNE Raw object containing the data to be rereferenced.

    ch_names_old : list[str]
    -   The names of the channels in the MNE Raw object to rereference.

    ch_names_new : list[str | None] | None; default None
    -   The names of the newly rereferenced channels, corresponding to the
        channels used for rerefrencing in ch_names_old.
    -   Missing values (None) will be set based on 'ch_names_old'.

    ch_types_new : list[str | None] | None; default None
    -   The types of the newly rereferenced channels as recognised by MNE,
        corresponding to the channels in 'ch_names_new'.
    -   Missing values (None) will be set based on the types of channels in
        'ch_names_old'.

    ch_reref_types : list[str | None] | None; default None
    -   The rereferencing type applied to the channels, corresponding to the
        channels in 'ch_names_new'.
    -   If some or all entries are None, they will be set as 'common_average'.

    ch_coords_new : list[list[int | float] | None] | None; default None
    -   The coordinates of the newly rereferenced channels, corresponding to
        the channels in 'ch_names_new'. The list should consist of sublists
        containing the x, y, and z coordinates of each channel.
    -   If the input is '[]', the coordinates of the channels in
        'ch_names_old' in the MNE Raw object are used.
    -   If some sublists are '[]', those channels for which coordinates are
        given are used, whilst those channels for which the coordinates are
        missing have their coordinates taken from the MNE Raw object
        according to the corresponding channel in 'ch_names_old'.

    ch_regions_new : list[str | None] | None; default None
    -   The regions of the newly rereferenced channels, corresponding to the
        channels in 'ch_names_new'.
    -   If 'None', the regions can be determined based on the regions of the
        channels being rereferenced. If some entries are None, they are left
        as-is.

    ch_subregions_new : list[str | None] | None; default None
    -   The subregions of the newly rereferenced channels, corresponding to the
        channels in 'ch_names_new'.
    -   If 'None', the subregions can be determined based on the subregions of
        the channels being rereferenced. If some entries are 'None', they are
        left as-is.

    ch_hemispheres_new : list[str | None] | None; default None
    -   The hemispheres of the newly rereferenced channels, corresponding to the
        channels in 'ch_names_new'.
    -   If 'None' or if some entries are 'None', the hemispheres can be
        determined based on the subregions of  the channels being rereferenced.

    METHODS
    -------
    rereference
    -   Rereferences the data in an MNE Raw object.
    """

    def _sort_inputs(self) -> None:
        """Checks that rereferencing settings are compatible and discards
        rereferencing-irrelevant channels from the data."""
        self._check_input_lengths()
        self._sort_raw(
            chs_to_analyse=np.unique(list(self._ch_names_old)).tolist(),
        )
        self._sort_feature("ch_names_new", self._ch_names_old)
        self._sort_feature(
            "ch_types_new", self.raw.get_channel_types(self._ch_names_old)
        )
        self._sort_feature(
            "ch_reref_types",
            ["common_average"] * self._n_channels,
        )
        self._sort_feature(
            "ch_coords_new",
            [
                self._ch_coords[self._ch_index[i]]
                for i in range(self._n_channels)
            ],
        )
        self._sort_feature(
            "ch_regions_new",
            [
                self.extra_info["ch_regions"][ch_name]
                for ch_name in self._ch_names_old
            ],
        )
        self._sort_feature(
            "ch_subregions_new",
            [
                self.extra_info["ch_subregions"][ch_name]
                for ch_name in self._ch_names_old
            ],
        )
        self._sort_feature(
            "ch_hemispheres_new",
            [
                self.extra_info["ch_hemispheres"][ch_name]
                for ch_name in self._ch_names_old
            ],
        )

    def _set_data(self) -> None:
        """Common-average rereferences the data, subtracting the average of all
        channels' data from each individual channel."""
        avg_data = self._data[self._ch_index].mean(axis=0)
        self._new_data = [
            self._data[self._ch_index[ch_i]] - avg_data
            for ch_i in range(self._n_channels)
        ]

    def _set_coordinates(self) -> None:
        """Sets the coordinates of the new, rereferenced channels.
        -   If no coordinates are provided, the coordinates are calculated by
            taking the coordinates from the original channel involved in each
            new, rereferenced channel.
        -   If some coordinates are provided, this calculation is performed for
            only those missing coordinates.

        RAISES
        ------
        ValueError
        -   Raised if the provided list of coordinates for a channel does not
            have a length of 3 (i.e. an x, y, and z coordinate).
        """
        self._new_ch_coords = []
        for ch_i in range(self._n_channels):
            coords_set = False
            if self._ch_coords_new != []:
                if self._ch_coords_new[ch_i] != []:
                    if not check_lengths_list_equals_n(
                        to_check=self._ch_coords_new, n=3, ignore_values=[[]]
                    ):
                        raise Exception(
                            "Error when setting coordinates for the "
                            "rereferenced data.\nThree, and only three "
                            "coordinates (x, y, and z) must be present, but "
                            "the rereferencing settings specify otherwise."
                        )
                    self._new_ch_coords.append(self._ch_coords_new[ch_i])
                    coords_set = True
            if coords_set is False:
                self._new_ch_coords.append(
                    self._ch_coords[self._ch_index[ch_i]]
                )


class RerefPseudo(Reref):
    """Pseudo rereferences data in an MNE Raw object.
    -   This allows e.g. rereferencing types to be assigned to the channels,
        channel coordinates to be set, etc... without any rereferencing
        occuring.
    -   This is useful if e.g. the channels were already hardware rereferenced.

    PARAMETERS
    ----------
    raw : MNE Raw
    -   The MNE Raw object containing the data to be rereferenced.

    ch_names_old : list[str]
    -   The names of the channels in the MNE Raw object to rereference.

    ch_names_new : list[str | None] | None; default None
    -   The names of the newly rereferenced channels, corresponding to the
        channels used for rerefrencing in ch_names_old.
    -   Missing values (None) will be set based on 'ch_names_old'.

    ch_types_new : list[str | None] | None; default None
    -   The types of the newly rereferenced channels as recognised by MNE,
        corresponding to the channels in 'ch_names_new'.
    -   Missing values (None) will be set based on the types of channels in
        'ch_names_old'.

    ch_reref_types : list[str]
    -   The rereferencing type applied to the channels, corresponding to the
        channels in 'ch_names_new'.
    -   No missing values (None) can be given, as the rereferencing type cannot
        be determined dynamically from this arbitrary rereferencing method.

    ch_coords_new : list[list[int | float] | None] | None; default None
    -   The coordinates of the newly rereferenced channels, corresponding to
        the channels in 'ch_names_new'. The list should consist of sublists
        containing the x, y, and z coordinates of each channel.
    -   If the input is '[]', the coordinates of the channels in
        'ch_names_old' in the MNE Raw object are used.
    -   If some sublists are '[]', those channels for which coordinates are
        given are used, whilst those channels for which the coordinates are
        missing have their coordinates taken from the MNE Raw object
        according to the corresponding channel in 'ch_names_old'.

    ch_regions_new : list[str | None] | None; default None
    -   The regions of the newly rereferenced channels, corresponding to the
        channels in 'ch_names_new'.
    -   If 'None', the regions can be determined based on the regions of the
        channels being rereferenced.
    -   If some entries are None, these are left as-is.

    ch_subregions_new : list[str | None] | None; default None
    -   The subregions of the newly rereferenced channels, corresponding to the
        channels in 'ch_names_new'.
    -   If 'None', the subregions can be determined based on the subregions of
        the channels being rereferenced. If some entries are 'None', they are
        left as-is.

    ch_hemispheres_new : list[str | None] | None; default None
    -   The hemispheres of the newly rereferenced channels, corresponding to the
        channels in 'ch_names_new'.
    -   If 'None' or if some entries are 'None', the hemispheres can be
        determined based on the subregions of  the channels being rereferenced.

    METHODS
    -------
    rereference
    -   Rereferences the data in an MNE Raw object.
    """

    def _sort_ch_reref_types(self) -> None:
        """Checks that all rereferencing types have been specified for the new
        channels, as these cannot be derived from the arbitrary method of pseudo
        rereferencing.

        RAISES
        ------
        TypeError
        -   Raised if the rereferencing type variable or any of its entries are
            of type None.
        """
        if None in self._ch_reref_types:
            raise TypeError(
                "Error when pseudo rereferencing:\nRereferencing types of each "
                "new channel must be specified, there can be no missing entries"
                "."
            )

    def _sort_inputs(self) -> None:
        """Checks that rereferencing settings are compatible and discards
        rereferencing-irrelevant channels from the data."""
        self._check_input_lengths()
        self._sort_raw(
            chs_to_analyse=np.unique(list(self._ch_names_old)).tolist(),
        )
        self._sort_feature("ch_names_new", self._ch_names_old)
        self._sort_feature(
            "ch_types_new", self.raw.get_channel_types(self._ch_names_old)
        )
        self._sort_ch_reref_types()
        self._sort_feature(
            "ch_coords_new",
            [
                self._ch_coords[self._ch_index[i]]
                for i in range(self._n_channels)
            ],
        )
        self._sort_feature(
            "ch_regions_new",
            [
                self.extra_info["ch_regions"][ch_name]
                for ch_name in self._ch_names_old
            ],
        )
        self._sort_feature(
            "ch_subregions_new",
            [
                self.extra_info["ch_subregions"][ch_name]
                for ch_name in self._ch_names_old
            ],
        )
        self._sort_feature(
            "ch_hemispheres_new",
            [
                self.extra_info["ch_hemispheres"][ch_name]
                for ch_name in self._ch_names_old
            ],
        )

    def _set_data(self) -> None:
        """Pseudo rereferences the data, setting the data for each new,
        rereferenced channel equal to that of the corresponding channel in the
        original data."""
        self._new_data = [
            self._data[self._ch_index[ch_i]]
            for ch_i in range(self._n_channels)
        ]

    def _set_coordinates(self) -> None:
        """Sets the coordinates of the new, rereferenced channels.
        -   If no coordinates are provided, the coordinates are calculated by
            taking the coordinates from the original channel involved in each
            new, rereferenced channel.
        -   If some coordinates are provided, this calculation is performed for
            only those missing coordinates.

        RAISES
        ------
        ValueError
        -   Raised if the provided list of coordinates for a channel does not
            have a length of 3 (i.e. an x, y, and z coordinate).
        """
        self._new_ch_coords = []
        for ch_i in range(self._n_channels):
            coords_set = False
            if self._ch_coords_new != []:
                if self._ch_coords_new[ch_i] != []:
                    if not check_lengths_list_equals_n(
                        to_check=self._ch_coords_new, n=3, ignore_values=[[]]
                    ):
                        raise Exception(
                            "Error when setting coordinates for the "
                            "rereferenced data:\nThree, and only three "
                            "coordinates (x, y, and z) must be present, but "
                            "the rereferencing settings specify otherwise."
                        )
                    self._new_ch_coords.append(self._ch_coords_new[ch_i])
                    coords_set = True
            if coords_set is False:
                self._new_ch_coords.append(
                    np.around(self._ch_coords[self._ch_index[ch_i]], 5)
                )
