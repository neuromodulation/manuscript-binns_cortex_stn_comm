"""A class for loading, preprocessing, and epoching an mne.io.Raw object.

CLASSES
-------
Signal
-   Class for loading, preprocessing, and epoching an mne.io.Raw object.
"""


from copy import deepcopy
from typing import Any, Optional, Union
from numpy.typing import NDArray
from mne import concatenate_epochs
import mne
import mne_bids
import numpy as np
import pandas as pd
from pyparrm import PARRM
from coh_exceptions import (
    ChannelAttributeError,
    EntryLengthError,
    ProcessingOrderError,
    UnavailableProcessingError,
)
from coh_rereference import (
    Reref,
    RerefBipolar,
    RerefCommonAverage,
    RerefPseudo,
)
from coh_handle_entries import (
    check_lengths_list_identical,
    check_repeated_vals,
    ordered_dict_keys_from_list,
    ordered_list_from_dict,
    rearrange_axes,
    get_eligible_idcs_lists,
    get_group_names_idcs,
)
from coh_handle_files import (
    check_annots_empty,
    check_annots_orig_time,
    check_ftype_present,
    identify_ftype,
)
from coh_handle_objects import (
    create_extra_info,
    create_mne_data_object,
    extra_info_keys,
)
from coh_saving import check_before_overwrite, save_as_json, save_as_pkl


class Signal:
    """Class for loading, preprocessing, and epoching an mne.io.Raw object.

    PARAMETERS
    ----------
    verbose : bool; default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    order_channels
    -   Orders channels in the mne.io.Raw or mne.Epochs object based on a
        given order.

    get_coordinates
    -   Extracts coordinates of the channels from the mne.io.Raw or mne.Epochs
        object.

    set_coordinates
    -   Assigns coordinates to the channels in the mne.io.Raw or mne.Epochs
        object.

    get_data
    -   Extracts the data array from the mne.io.Raw or mne.Epochs object,
        excluding data based on the annotations.

    load_raw
    -   Loads an mne.io.Raw object, loads it into memory, and sets it as the
        data, also assigning rereferencing types in 'extra_info' for the
        channels present in the mne.io.Raw object to 'none'.

    load_annotations
    -   Loads annotations corresponding to the mne.io.Raw object.

    pick_channels
    -   Retains only certain channels in the mne.io.Raw or mne.Epochs object,
        also retaining only entries for these channels from the 'extra_info'.

    bandpass_filter
    -   Bandpass filters the mne.io.Raw or mne.Epochs object.

    notch_filter
    -   Notch filters the mne.io.Raw or mne.Epochs object.

    resample
    -   Resamples the mne.io.Raw or mne.Epochs object.

    combine_channels
    -   Combines the data of multiple channels in the mne.io.Raw object through
        addition and adds this combined data as a new channel.

    drop_unrereferenced_channels
    -   Drops channels that have not been rereferenced from the mne.io.Raw or
        mne.Epochs object, also discarding entries for these channels from
        'extra_info'.

    rereference_bipolar
    -   Bipolar rereferences channels in the mne.io.Raw object.

    rereference_common_average
    -   Common-average rereferences channels in the mne.io.Raw object.

    rereference_pseudo
    -   Pseudo rereferences channels in the mne.io.Raw object.
    -   This allows e.g. rereferencing types, channel coordinates, etc... to be
        assigned to the channels without any rereferencing occuring.
    -   This is useful if e.g. the channels were already hardware rereferenced.

    epoch
    -   Divides the mne.io.Raw object into epochs of a specified duration.

    save_object
    -   Saves the Signal object as a .pkl file.

    save_signals
    -   Saves the time-series data and additional information as a file.
    """

    def __init__(self, verbose: bool = True) -> None:
        # Initialises aspects of the Signal object that will be filled with
        # information as the data is processed.
        self.processing_steps = {"preprocessing": {}}
        self._processing_step_number = 1
        self.extra_info = {}
        self.data = [None]
        self._path_raw = None
        self._data_dimensions = None

        # Initialises inputs of the Signal object.
        self._verbose = verbose

        # Initialises aspects of the Signal object that indicate which methods
        # have been called (starting as 'False'), which can later be updated.
        self._data_loaded = False
        self._annotations_loaded = False
        self._channels_picked = False
        self._coordinates_set = False
        self._regions_set = False
        self._subregions_set = False
        self._hemispheres_set = False
        self._bandpass_filtered = False
        self._notch_filtered = False
        self._resampled = False
        self._rereferenced = False
        self._rereferenced_bipolar = False
        self._rereferenced_common_average = False
        self._rereferenced_pseudo = False
        self._z_scored = False
        self._windowed = False
        self._pseudo_windowed = False
        self._epoched = False
        self._bootstrapped = False
        self._shuffled = False

    def _update_processing_steps(
        self, step_name: str, step_value: Any
    ) -> None:
        """Updates the 'preprocessing' entry of the 'processing_steps'
        dictionary of the Signal object with new information consisting of a
        key:value pair in which the key is numbered based on the applied steps.

        PARAMETERS
        ----------
        step_name : str
        -   The name of the processing step.

        step_value : Any
        -   A value representing what processing has taken place.
        """
        step_name = f"{self._processing_step_number}.{step_name}"
        self.processing_steps["preprocessing"][step_name] = step_value
        self._processing_step_number += 1

    def add_metadata(self, metadata: dict) -> None:
        """Adds information about the data being preprocessed to the extra_info
        aspect.

        PARAMETERS
        ----------
        metadata : dict
        -   Information about the data being preprocessed.
        """
        self.extra_info["metadata"] = metadata

    def _order_extra_info(self, order: list[str]) -> None:
        """Order channels in 'extra_info'.

        PARAMETERS
        ----------
        order : list[str]
        -   The order in which the channels should appear in the attributes of
            the 'extra_info' dictionary.
        """
        to_order = [
            "ch_reref_types",
            "ch_regions",
            "ch_subregions",
            "ch_hemispheres",
            "ch_epoch_orders",
        ]
        for key in to_order:
            if key in self.extra_info.keys():
                self.extra_info[key] = ordered_dict_keys_from_list(
                    dict_to_order=self.extra_info[key], keys_order=order
                )

    def order_channels(self, ch_names: list[str]) -> None:
        """Orders channels in the mne.io.Raw or mne.Epochs objects, as well as
        the 'extra_info' dictionary, based on a given order.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   A list of channel names in the mne.io.Raw or mne.Epochs object in
            the order that you want the channels to be ordered.
        """
        # get names actually in data
        ch_names = [name for name in ch_names if name in self.data[0].ch_names]
        # get names in data missing from order
        missing_names = [
            name for name in self.data[0].ch_names if name not in ch_names
        ]

        if self._verbose:
            print("Reordering the channels in the following order:")
            [print(name) for name in ch_names]
            print(f"Removing the unlisted channels: {missing_names}\n")

        for data in self.data:
            data.drop_channels(missing_names)
            data.reorder_channels(ch_names)
        if missing_names != []:
            self._drop_extra_info(missing_names)
        self._order_extra_info(order=ch_names)

    def get_coordinates(
        self, picks: Union[str, list, slice, None] = None
    ) -> list[list[Union[int, float]]]:
        """Extracts coordinates of the channels from the mne.io.Raw or
        mne.Epochs objects.

        PARAMETERS
        ----------
        picks : str | list | slice | None; default None
        -   Selects which channels' coordinates should be returned.
        -   If 'None', returns coordinates for all good channels.

        RETURNS
        -------
        list[list[int or float]]
        -   List of the channel coordinates, with each list entry containing the
            x, y, and z coordinates of each channel.
        """
        picks = mne.io.pick._picks_to_idx(self.data[0].info, picks)
        chs = self.data[0].info["chs"]
        coords = np.array([chs[k]["loc"][:3] for k in picks])
        # for ch_i, ch_coords in enumerate(coords):
        #    if np.all(ch_coords == 0):
        #        coords[ch_i] = np.full((3,), np.nan)

        return coords.copy().tolist()

    def _discard_missing_coordinates(
        self, ch_names: list[str], ch_coords: list[list[Union[int, float]]]
    ) -> tuple[list, list]:
        """Removes empty sublists from a parent list of channel coordinates
        (also removes them from the corresponding entries of channel names)
        before applying the coordinates to the mne.io.Raw or mne.Epochs objects.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   Names of the channels corresponding to the coordinates in
            'ch_coords'.

        ch_coords : list[empty list | list[int | float]]
        -   Coordinates of the channels, with each entry consiting of a sublist
            containing the x, y, and z coordinates of the corresponding channel
            specified in 'ch_names', or being empty.

        RETURNS
        -------
        empty list | list[str]
        -   Names of the channels corresponding to the coordinates in
            'ch_coords', with those names corresponding to empty sublists (i.e
            missing coordinates) in 'ch_coords' having been removed.

        empty list  |list[list[int | float]]
        -   Coordinates of the channels corresponding the the channel names in
            'ch_names', with the empty sublists (i.e missing coordinates) having
            been removed.
        """
        keep_i = [i for i, coords in enumerate(ch_coords) if coords != []]
        return (
            [name for i, name in enumerate(ch_names) if i in keep_i],
            [coords for i, coords in enumerate(ch_coords) if i in keep_i],
        )

    def set_coordinates(
        self, ch_names: list[str], ch_coords: list[list[Union[int, float]]]
    ) -> None:
        """Assigns coordinates to the channels in the mne.io.Raw or mne.Epochs
        object.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels corresponding to the coordinates in
            'ch_coords'.

        ch_coords : list[empty list | list[int | float]]
        -   Coordinates of the channels, with each entry consiting of a sublist
            containing the x, y, and z coordinates of the corresponding channel
            specified in 'ch_names'.
        """
        ch_names, ch_coords = self._discard_missing_coordinates(
            ch_names, ch_coords
        )
        for data in self.data:
            data._set_channel_positions(ch_coords, ch_names)

        self._coordinates_set = True
        if self._verbose:
            print("Setting channel coordinates to:")
            [
                print(f"{ch_names[i]}: {ch_coords[i]}")
                for i in range(len(ch_names))
            ]

    def set_regions(self, ch_names: list[str], ch_regions: list[str]) -> None:
        """Adds channel regions (e.g. prefrontal, parietal) to the extra_info
        dictionary.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   Names of the channels.

        ch_regions : list[str]
        -   Regions of the channels.
        """
        if len(ch_names) != len(ch_regions):
            raise EntryLengthError(
                "The channel names and regions do not have the same length "
                f"({len(ch_names)} and {len(ch_regions)}, respectively)."
            )

        for i, ch_name in enumerate(ch_names):
            self.extra_info["ch_regions"][ch_name] = ch_regions[i]

        self._regions_set = True
        if self._verbose:
            print("Setting channel regions to:")
            [
                print(f"{ch_names[i]}: {ch_regions[i]}")
                for i in range(len(ch_names))
            ]

    def set_subregions(
        self, ch_names: list[str], ch_subregions: list[str]
    ) -> None:
        """Adds channel subregions (e.g. prefrontal, parietal) to the extra_info
        dictionary.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   Names of the channels.

        ch_subregions : list[str]
        -   Subregions of the channels.
        """
        if len(ch_names) != len(ch_subregions):
            raise EntryLengthError(
                "The channel names and subregions do not have the same length "
                f"({len(ch_names)} and {len(ch_subregions)}, respectively)."
            )

        for i, ch_name in enumerate(ch_names):
            self.extra_info["ch_subregions"][ch_name] = ch_subregions[i]

        self._subregions_set = True
        if self._verbose:
            print("Setting channel subregions to:")
            [
                print(f"{ch_names[i]}: {ch_subregions[i]}")
                for i in range(len(ch_names))
            ]

    def set_hemispheres(
        self, ch_names: list[str], ch_hemispheres: list[str]
    ) -> None:
        """Adds channel hemispheres to the extra_info dictionary.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   Names of the channels.

        ch_hemispheres : list[str]
        -   Hemispheres of the channels.
        """
        if len(ch_names) != len(ch_hemispheres):
            raise EntryLengthError(
                "The channel names and hemispheres do not have the same length "
                f"({len(ch_names)} and {len(ch_hemispheres)}, respectively)."
            )

        for i, ch_name in enumerate(ch_names):
            self.extra_info["ch_hemispheres"][ch_name] = ch_hemispheres[i]

        self._hemispheres_set = True
        if self._verbose:
            print("Setting channel hemispheres to:")
            [
                print(f"{ch_names[i]}: {ch_hemispheres[i]}")
                for i in range(len(ch_names))
            ]

    @property
    def data_dimensions(self) -> list[str]:
        """Returns the dimensions of the data you would get if you called
        'get_data'.

        RETURNS
        -------
        dims : list[str]
        -   Dimensions of the data.
        """
        if self._windowed:
            return deepcopy(self._data_dimensions)
        return deepcopy(self._data_dimensions[1:])

    def get_data(self) -> np.array:
        """Extracts the data array from the mne.io.Raw or mne.Epochs objects,
        excluding data based on the annotations if the data is an MNE Raw
        object.

        RETURNS
        -------
        data_arr : numpy array
        -   The data in array form.
        """
        data_arr = np.empty(len(self.data))
        for i, data in enumerate(self.data):
            if isinstance(data, mne.io.Raw):
                data_arr[i] = data.get_data(reject_by_annotation="omit").copy()
            else:
                data_arr[i] = data.get_data().copy()

        if self._windowed:
            return data_arr[0, :, :]
        return data_arr

    def _initialise_additional_info(self) -> None:
        """Fills the extra_info dictionary with placeholder information. This
        should only be called when the data is initially loaded."""
        info_to_set = [
            "ch_reref_types",
            "ch_regions",
            "ch_subregions",
            "ch_hemispheres",
        ]
        for info in info_to_set:
            self.extra_info[info] = {
                ch_name: None for ch_name in self.data[0].info["ch_names"]
            }
        self._data_dimensions = ["windows", "channels", "timepoints"]

    def _fix_coords(self) -> None:
        """Fixes the units of the channel coordinates in the data by multiplying
        them by 1,000."""
        raise NotImplementedError(
            "Fixing the coordinates with this method does not currently "
            "function correctly, and so should not be called!"
        )
        ch_coords = self.get_coordinates()
        for ch_i, coords in enumerate(ch_coords):
            ch_coords[ch_i] = [coord * 1000 for coord in coords]
        self.set_coordinates(
            ch_names=self.data[0].ch_names, ch_coords=ch_coords
        )

    def raw_from_fpath(self, path_raw: mne_bids.BIDSPath) -> None:
        """Loads an mne.io.Raw object from a filepath, loads it into memory, and
        sets it as the data, also assigning additional information in
        'extra_info' for the channels present in the mne.io.Raw object to
        'none'.

        PARAMETERS
        ----------
        path_raw : mne_bids.BIDSPath
        -   The path of the raw data to be loaded.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to load data if data has already been
            loaded.
        """
        if self._data_loaded:
            raise ProcessingOrderError(
                "Error when trying to load raw data: data has already been "
                "loaded."
            )

        self._path_raw = path_raw
        self.data[0] = mne_bids.read_raw_bids(
            bids_path=self._path_raw, verbose=False
        )
        self.data[0].load_data()
        self._initialise_additional_info()
        # self._fix_coords()

        self._data_loaded = True
        if self._verbose:
            print(f"Loading the data from the filepath:\n{path_raw}.\n")

    def data_from_objects(
        self,
        data: Union[
            mne.io.Raw, mne.Epochs, list[Union[mne.io.Raw, mne.Epochs]]
        ],
        processing_steps: Union[dict, None] = None,
        extra_info: Union[dict, None] = None,
    ) -> None:
        """Sets the data, its dimensions, processing steps, and additional
        information from their respective objects.

        PARAMETERS
        ----------
        data : MNE Raw | MNE Epochs | list of MNE Raw or MNE Epochs
        -   Data to load, stored in MNE objects. If the data is a list of MNE
            objects, the data is assumed to be windowed. Data must be of the
            same type (i.e. all Raw or all Epochs objects).

        processing_steps : dict | None; default None
        -   Information about the processing that has been applied to the data.

        extra_info : dict | None; default None
        -   Additional information about the data not included in the MNE
            objects.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if data has already been loaded.

        TypeError
        -   Raised if data of multiple types is being supplied.
        -   Raised if the data is not stored in MNE Raw or MNE Epochs objects.
        """
        if self._data_loaded:
            raise ProcessingOrderError(
                "Error when trying to load data:\nData has already been loaded "
                "into the object."
            )

        data_windowed = False
        data_epoched = False
        if isinstance(data, list):
            if not all(isinstance(window, type(data[0])) for window in data):
                raise TypeError(
                    "Only one form of MNE data can be loaded into a single "
                    "Signal object, but data of multiple types are being "
                    "loaded."
                )
            data_type = type(data[0])
            data_windowed = True
        else:
            data_type = type(data)
            data = [data]
        if not isinstance(data[0], mne.io.BaseRaw) and not isinstance(
            data[0], mne.BaseEpochs
        ):
            raise TypeError(
                "The data to load must be an MNE Raw or Epochs object, but it "
                f"is of type '{data_type}'."
            )
        if isinstance(data[0], mne.BaseEpochs):
            data_epoched = True

        [window_data.load_data() for window_data in data]

        data_dimensions = ["channels", "timepoints"]
        if data_epoched:
            data_dimensions = ["epochs", *data_dimensions]
            self._epoched = True
        if data_windowed:
            self._windowed = True
        data_dimensions = [
            "windows",
            *data_dimensions,
        ]  # leading "windows" tag
        # will be ignored when self._windowed == False

        self.data = data
        self._data_dimensions = data_dimensions
        self._instantiate_processing_steps(processing_steps)
        self._instantiate_extra_info(extra_info)
        self._set_method_bools()

        self._data_loaded = True
        if self._verbose:
            print(f"Loading the {data_type} data from MNE objects.\n")

    def _instantiate_processing_steps(
        self, processing_steps: Union[dict, None]
    ) -> None:
        """Instantiates the processing steps when preprocessed data is being
        loaded into the Signal object.

        PARAMETERS
        ----------
        processing_steps : dict | None
        -   The processing steps that have been performed on the data.
        """
        if processing_steps is not None:
            if not isinstance(processing_steps, dict):
                raise TypeError("The processing steps must be a dict.")
            self.processing_steps = processing_steps
        else:
            self.processing_steps = {}
        if "preprocessing" not in self.processing_steps.keys():
            self.processing_steps = {"preprocessing": {}}

    def _instantiate_extra_info(self, extra_info: Union[dict, None]) -> None:
        """Instantiates extra information about the data not stored in he data
        itself when preprocessed data is being loaded into the Signal object.

        PARAMETERS
        ----------
        extra_info : dict | None
        -   The extra information about the data.
        """
        if extra_info is not None:
            self.extra_info = extra_info
            for key in extra_info_keys:
                if key not in extra_info.keys():
                    self.extra_info[key] = None
        else:
            self.extra_info = {key: None for key in extra_info_keys}

    def _set_method_bools(self) -> None:
        """Sets the markers for methods that have been called on the data based
        on the processing steps and extra information dictionaries."""
        for steps in self.processing_steps.values():
            if steps is not None:
                for step in steps:
                    if "rereferencing_common_average" in step:
                        self._rereferenced_common_average = True
                        self._rereferenced = True
                    elif "rereferencing_bipolar" in step:
                        self._rereferenced_bipolar = True
                        self._rereferenced = True
                    elif "rereferencing_pseudo" in step:
                        self._rereferenced_pseudo = True
                        self._rereferenced = True
                    elif "annotations_loaded" in step:
                        self._annotations_loaded = True
                    elif "bandpass_filter" in step:
                        self._bandpass_filtered = True
                    elif "notch_filter" in step:
                        self._notch_filtered = True
                    elif "resample" in step:
                        self._resampled = True
                    elif "shuffle_data" in step:
                        self._shuffled = True

        if self.extra_info["ch_regions"] is not None:
            self._regions_set = True
        if self.extra_info["ch_subregions"] is not None:
            self._subregions_set = True
        if self.extra_info["ch_hemispheres"] is not None:
            self._hemispheres_set = True

    def _remove_bad_annotations(
        self, labels: list[str] | None = None
    ) -> tuple[mne.annotations.Annotations, mne.annotations.Annotations]:
        """Removes segments annotated as 'bad' from the Annotations object.

        Paramaters
        ----------
        labels : list of str | None (default None)
            Labels of the bad segments to remove.

        RETURNS
        -------
        bad_annotations
        -   The bad annotations which should be removed.

        good_annotations
        -   The good annotations which should be retained.
        """
        annotations = deepcopy(self.data[0].annotations)
        bad_annot_idcs = []
        for annot_i, annot_name in enumerate(annotations.description):
            if labels is None:
                if annot_name[:3] == "BAD":
                    bad_annot_idcs.append(annot_i)
            elif set([annot_name]).issubset(labels):
                bad_annot_idcs.append(annot_i)

        bad_annotations = annotations.copy()
        bad_annotations.delete(
            [i for i in range(len(bad_annotations)) if i not in bad_annot_idcs]
        )

        good_annotations = annotations.copy()
        good_annotations.delete(bad_annot_idcs)

        return bad_annotations, good_annotations

    def remove_bad_segments(self, labels: list[str] | None = None) -> None:
        """Removes segments annotated as 'bad' from the Raw object.

        Paramaters
        ----------
        labels : list of str | None (default None)
            Labels of the bad segments to remove.
        """
        if self._epoched:
            raise ProcessingOrderError(
                "Error when removing bad segments from the data:\nBad segments "
                "should be removed from the raw data, however the data in this "
                "class has been epoched."
            )

        if not isinstance(labels, list) and labels is not None:
            raise TypeError("`labels` must be a list of str or None.")

        bad_annotations, good_annotations = self._remove_bad_annotations(
            labels
        )
        self.data[0].set_annotations(bad_annotations)
        new_data = self.data[0].get_data(reject_by_annotation="omit")
        self.data[0] = mne.io.RawArray(data=new_data, info=self.data[0].info)

        for bad_annot_i in range(len(bad_annotations)):
            for good_annot_i in range(len(good_annotations)):
                if (
                    good_annotations.onset[good_annot_i]
                    > bad_annotations.onset[bad_annot_i]
                ):
                    good_annotations.onset[
                        good_annot_i
                    ] -= bad_annotations.duration[bad_annot_i]
        good_annotations = mne.Annotations(
            good_annotations.onset,
            good_annotations.duration,
            good_annotations.description,
        )

        self.data[0].set_annotations(annotations=good_annotations)

        if self._verbose:
            print(
                f"Removing {len(bad_annotations)} bad segment(s) from the "
                "data."
            )

    def load_annotations(self, fpath: str) -> None:
        """Loads annotations corresponding to the mne.io.Raw object.

        PARAMETERS
        ----------
        fpath : str
        -   The filepath of the annotations to load.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to load annotations into the data after
            it has been windowed.
        -   Raised if the user attempts to load annotations into the data after
            it has been epoched.
        -   Annotations should be loaded before epoching has occured, when the
            data is in the form of an mne.io.Raw object rather than an
            mne.Epochs object.

        Notes
        -----
        "BAD_recording_start" and "BAD_recording_end" annotations are removed
        when loaded. Other BAD annotations must be removed by calling
        `remove_bad_segments`.
        """
        if self._windowed:
            raise ProcessingOrderError(
                "Error when adding annotations to the data:\nAnnotations "
                "should be added to the raw data, however the data in this "
                "class has been windowed."
            )
        if self._epoched:
            raise ProcessingOrderError(
                "Error when adding annotations to the data:\nAnnotations "
                "should be added to the raw data, however the data in this "
                "class has been epoched."
            )

        if self._verbose:
            print(
                "Applying annotations to the data from the filepath:\n"
                f"{fpath}."
            )

        if check_annots_empty(fpath):
            print("There are no events to read from the annotations file.")
        else:
            self.data[0].set_annotations(
                check_annots_orig_time(mne.read_annotations(fpath))
            )
            self.remove_bad_segments(
                labels=["BAD_recording_start", "BAD_recording_end"]
            )

        self._annotations_loaded = True
        self._update_processing_steps(
            step_name="annotations_loaded", step_value=True
        )

    def _pick_extra_info(self, ch_names: list[str]) -> None:
        """Retains entries for selected channels in 'extra_info', discarding
        those for the remaining channels.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels whose entries should be retained.
        """
        for key in self.extra_info.keys():
            new_entry = {
                ch_name: self.extra_info[key][ch_name] for ch_name in ch_names
            }
            self.extra_info[key] = new_entry

    def _drop_extra_info(self, ch_names: list[str]) -> None:
        """Removes entries for selected channels in 'extra_info', retaining
        those for the remaining channels.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels whose entries should be discarded.
        """
        for key in self.extra_info.keys():
            [self.extra_info[key].pop(name) for name in ch_names]

    def _drop_channels(self, ch_names: list[str]) -> None:
        """Removes channels from the mne.io.Raw or mne.Epochs objects, as well
        as from entries in 'extra_info'.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels that should be discarded.
        """
        self._drop_extra_info(ch_names)
        extra_info_keys = ["ch_regions", "ch_subregions", "ch_hemispheres"]
        for i, data in enumerate(self.data):
            self.data[i] = data.drop_channels(ch_names)
            for key in extra_info_keys:
                setattr(self.data[i], key, self.extra_info[key])

    def drop_channels(self, eligible_entries: dict, conditions: dict) -> None:
        """Drop channels from the object based on criteria."""
        features = self._features_to_df()

        possible_channels = []
        for key, value in eligible_entries.items():
            feature = features[key].tolist()
            possible_channels.extend(
                [i for i, val in enumerate(feature) if val in value]
            )
        possible_channels = np.unique(possible_channels).tolist()

        bad_channels = []
        for key, value in conditions.items():
            feature = features[key].tolist()
            bad_channels.extend(
                [
                    i
                    for i, val in zip(
                        possible_channels,
                        [feature[ch_i] for ch_i in possible_channels],
                    )
                    if val in value
                ]
            )
        bad_channels = np.unique(bad_channels).tolist()

        self._drop_channels([self.data[0].ch_names[i] for i in bad_channels])

    def pick_channels(self, ch_names: list[str]) -> None:
        """Retains only certain channels in the mne.io.Raw or mne.Epochs
        objects, also retaining only entries for these channels from the
        'extra_info'.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   The names of the channels that should be retained.
        """
        self._pick_extra_info(ch_names)
        for data in self.data:
            data.pick(ch_names)

        self._channels_picked = True
        self._update_processing_steps("channel_picks", ch_names)
        if self._verbose:
            print(
                "Picking specified channels from the data.\nChannels: "
                f"{ch_names}."
            )

    def bandpass_filter(
        self,
        highpass_freq: Union[int, float],
        lowpass_freq: Union[int, float],
        picks: Union[list, None] = None,
    ) -> None:
        """Bandpass filters the mne.io.Raw or mne.Epochs objects.

        PARAMETERS
        ----------
        highpass_freq : int | float
        -   The frequency (Hz) at which to highpass filter the data.

        lowpass_freq : int | float
        -   The frequency (Hz) at which to lowpass filter the data.

        picks : list | None
        -   The channels to filter (including bad channels). Can either consist
            of channel names, or channel types. If None, all channels are
            used.
        """
        ch_names = self.data[0].ch_names
        if not picks:
            ch_picks = ch_names
        else:
            if not set(picks).issubset(ch_names):
                ch_types = self.data[0].get_channel_types(picks=ch_names)
                ch_picks = [
                    name
                    for ch_i, name in enumerate(ch_names)
                    if ch_types[ch_i] in picks
                ]
            else:
                ch_picks = picks
        picked_types = np.unique(
            self.data[0].get_channel_types(picks=ch_picks)
        )

        for data in self.data:
            data.filter(highpass_freq, lowpass_freq, picks=ch_picks)

        self._bandpass_filtered = True
        step_name = (
            "bandpass_filter-"
            + "".join(f"{ch_type}_" for ch_type in picked_types)[:-1]
        )
        self._update_processing_steps(step_name, [highpass_freq, lowpass_freq])
        if self._verbose:
            print(
                f"Bandpass filtering the data:\n- Channels: {ch_picks}\n- "
                f"Frequencies: {highpass_freq} - {lowpass_freq} Hz."
            )

    def notch_filter(
        self,
        base_freq: Union[int, float],
        notch_width: Union[int, float, None] = None,
    ) -> None:
        """Notch filters the mne.io.Raw or mne.Epochs object.

        PARAMETERS
        ----------
        base_freq : int | float
        -   The base frequency (Hz) for which the notch filter, including
            the harmonics, is produced.

        notch_width : int | float | None
        -   Width of the stop band in Hz. If 'None', the frequencies at which
            the notch filter is applied divided by 200 is used.
        """
        freqs = np.arange(
            base_freq,
            self.data[0].info["lowpass"],
            base_freq,
            dtype=int,
        ).tolist()
        for data in self.data:
            data.notch_filter(freqs, notch_widths=notch_width)

        self._notch_filtered = True
        self._update_processing_steps("notch_filter", freqs)
        if self._verbose:
            print(
                f"Notch filtering the data with base frequency {base_freq} Hz "
                f"at the following frequencies (Hz): {freqs}."
            )

    def parrm(
        self,
        stim_freq: int | float,
        filter_half_width: int | None = None,
        omit_n_samples: int = 0,
        filter_direction: str = "both",
        period_half_width: int | float | None = None,
        grouping: list[str] | None = None,
        eligible_entries: list[str] | None = None,
        group_names: list[str] | None = None,
        explore_params: bool = False,
        n_jobs: int = 1,
    ) -> None:
        """Apply PARRM to the data to remove stimulation artefacts."""
        if self._windowed or self._epoched:
            raise ProcessingOrderError(
                "PARRM can only be performed on non-windowed/non-epoched data."
            )

        ch_names = self.data[0].ch_names
        original_data = self.data[0].get_data(picks=ch_names)

        features = self._features_to_df()
        eligible_idcs = get_eligible_idcs_lists(features, eligible_entries)
        group_idcs = get_group_names_idcs(
            features,
            grouping,
            eligible_idcs=eligible_idcs,
            replacement_idcs=eligible_idcs,
        )
        if group_names is not None:
            if len(group_idcs.keys()) != len(group_names):
                raise ValueError(
                    "`group_names` must contain the same number of entries as "
                    "there are groups produced from the grouping criteria."
                )
            if not set(group_names).issubset(group_idcs.keys()):
                raise ValueError(
                    "Not all `group_names` are in the groups produced from "
                    "the grouping criteria."
                )

        picks = [
            ch_name
            for ch_name in ch_names
            if ch_name not in self.data[0].info["bads"]
            and ch_name in [ch_names[idx] for idx in eligible_idcs]
        ]

        if self._verbose:
            print(
                "Applying PARRM to the following channels:\n- Grouping "
                f"{grouping}\n- Eligible entries {eligible_entries}\n- Picks "
                f"{picks}\nPARRM has not been applied to the following bad "
                f"channels: {self.data[0].info['bads']}\n"
            )

        if not isinstance(filter_half_width, list):
            filter_half_width = [
                filter_half_width for _ in range(len(group_idcs))
            ]
        if not isinstance(omit_n_samples, list):
            omit_n_samples = [omit_n_samples for _ in range(len(group_idcs))]
        if not isinstance(filter_direction, list):
            filter_direction = [
                filter_direction for _ in range(len(group_idcs))
            ]
        if not isinstance(period_half_width, list):
            period_half_width = [
                period_half_width for _ in range(len(group_idcs))
            ]

        for param in [
            filter_half_width,
            omit_n_samples,
            filter_direction,
            period_half_width,
        ]:
            if len(param) != len(group_idcs):
                raise ValueError(
                    "The length of the PARRM parameters must match the number "
                    "of groups on which PARRM is being applied."
                )

        groups_filtered_data = []
        for group_name, ch_idcs in group_idcs.items():
            if explore_params:
                parrm = PARRM(
                    data=original_data[ch_idcs],
                    sampling_freq=self.data[0].info["sfreq"],
                    artefact_freq=stim_freq,
                    verbose=True,
                )
                parrm.find_period(n_jobs=n_jobs)
                parrm.explore_filter_params(n_jobs=n_jobs)
            else:
                group_i = group_names.index(group_name)
                parrm = PARRM(
                    data=original_data[ch_idcs],
                    sampling_freq=self.data[0].info["sfreq"],
                    artefact_freq=stim_freq,
                    verbose=True,
                )
                parrm.find_period(n_jobs=n_jobs)
                parrm.create_filter(
                    filter_half_width=filter_half_width[group_i],
                    omit_n_samples=omit_n_samples[group_i],
                    filter_direction=filter_direction[group_i],
                    period_half_width=period_half_width[group_i],
                )
                groups_filtered_data.append(
                    create_mne_data_object(
                        data=parrm.filter_data(),
                        data_dimensions=["channels", "timepoints"],
                        ch_names=[ch_names[idx] for idx in ch_idcs],
                        ch_types=self.data[0].get_channel_types(
                            picks=[ch_names[idx] for idx in ch_idcs]
                        ),
                        sfreq=self.data[0].info["sfreq"],
                        ch_coords=self.get_coordinates(
                            picks=[ch_names[idx] for idx in ch_idcs]
                        ),
                        annotations=self.data[0].annotations,
                        meas_date=self.data[0].info["meas_date"],
                        subject_info=self.data[0].info["subject_info"],
                        verbose=False,
                    )
                )

        if explore_params:
            return

        filtered_data = groups_filtered_data[0][0]
        if len(groups_filtered_data) > 1:
            filtered_data.add_channels(
                [data[0] for data in groups_filtered_data[1:]]
            )
        if len(self.data[0].info["bads"]) != 0:
            filtered_data.add_channels(
                create_mne_data_object(
                    data=self.data[0].get_data(
                        picks=[
                            idx
                            for idx, name in enumerate(ch_names)
                            if name in self.data[0].info["bads"]
                        ]
                    ),
                    data_dimensions=["channels", "timepoints"],
                    ch_names=ch_names,
                    ch_types=self.data[0].get_channel_types(
                        picks=self.data[0].info["bads"]
                    ),
                    sfreq=self.data[0].info["sfreq"],
                    ch_coords=self.get_coordinates(
                        picks=self.data[0].info["bads"]
                    ),
                    annotations=self.data[0].annotations,
                    meas_date=self.data[0].info["meas_date"],
                    subject_info=self.data[0].info["subject_info"],
                    verbose=self._verbose,
                )[0]
            )

        self.data[0] = filtered_data.reorder_channels(ch_names)

        self._parrm = True
        self._update_processing_steps("parrm", picks)

    def _features_to_df(self) -> pd.DataFrame:
        """Collates features of channels (e.g. names, types, regions, etc...)
        into a pandas DataFrame so that which channels belong to which groups
        can be easily checked.

        RETURNS
        -------
        pandas DataFrame
        -   DataFrame containing the features of each channel.
        """
        ch_names = self.data[0].ch_names
        feature_dict = {
            "ch_names": ch_names,
            "ch_types": self.data[0].get_channel_types(picks=ch_names),
        }
        for extra_info_key, extra_info_value in self.extra_info.items():
            if extra_info_key != "metadata":
                if extra_info_value is not None:
                    feature_dict[extra_info_key] = ordered_list_from_dict(
                        ch_names, extra_info_value
                    )

        return pd.DataFrame(feature_dict)

    def trim_start_end(
        self, trim_start: int | float, trim_end: int | float
    ) -> None:
        """Trim a certain amount of time from the start and end of the data.

        Parameters
        ----------
        trim_start : int | float
            Amount of data to trim from the start, in seconds.

        trim_end : int | float
            Amount of data to trim from the end, in seconds.
        """
        if self._epoched:
            raise ProcessingOrderError(
                "Trimming can only be performed on non-epoched data."
            )

        if self._verbose:
            print(
                f"Trimming {trim_start} and {trim_end} second(s) from the "
                "start and end of the data, respectively.\n"
            )

        for window_i in range(len(self.data)):
            self.data[window_i].crop(
                trim_start,
                self.data[window_i].times[-1] - trim_end,
                verbose=False,
            )

    def z_score(self, grouping: str = "ch_type") -> None:
        """Z-scores the mne.io.Raw object. Bad channels are not included in the
        Z-scoring, but are retained in the data.

        PARAMETERS
        ----------
        grouping : str; default "ch_type"
        -   How to group channels when Z-scoring. If "ch_type", channels of the
            same type are grouped together when calculating the mean and
            standard deviation. If "ch_units", channels with the same units are
            grouped together.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the data has already been windowed and/or epoched.
        """
        if self._windowed or self._epoched:
            raise ProcessingOrderError(
                "Z-scoring can only be performed on non-windowed/non-epoched "
                "data."
            )

        supported_grouping = ["ch_type", "ch_units"]
        if grouping not in supported_grouping:
            raise NotImplementedError(
                f"The requested grouping {grouping} is not supported for "
                f"Z-scoring. Supported groupings are {supported_grouping}."
            )

        ch_names = self.data[0].ch_names
        picks = [
            ch_name
            for ch_name in ch_names
            if ch_name not in self.data[0].info["bads"]
        ]
        pick_idcs = [self.data[0].ch_names.index(pick) for pick in picks]

        if grouping == "ch_units":
            pick_grouping = [
                self.data[0].info["chs"][idx]["unit"] for idx in pick_idcs
            ]
        else:
            pick_grouping = self.data[0].get_channel_types(picks=picks)

        data = self.data[0].get_data(picks=ch_names).copy()
        z_score_features = {
            group_type: {"mean": None, "std": None}
            for group_type in np.unique(pick_grouping)
        }
        for group_type in z_score_features.keys():
            group_idcs = [
                i for i in range(len(picks)) if pick_grouping[i] == group_type
            ]
            group_data = self.data[0].get_data(picks=group_idcs)
            z_score_features[group_type]["mean"] = np.mean(group_data)
            z_score_features[group_type]["std"] = np.std(group_data)

        for group_type, ch_idx in zip(pick_grouping, pick_idcs):
            data[ch_idx, :] = (
                data[ch_idx, :] - z_score_features[group_type]["mean"]
            ) / z_score_features[group_type]["std"]

        self.data[0] = create_mne_data_object(
            data=data,
            data_dimensions=["channels", "timepoints"],
            ch_names=ch_names,
            ch_types=self.data[0].get_channel_types(picks=ch_names),
            sfreq=self.data[0].info["sfreq"],
            ch_coords=self.get_coordinates(picks=ch_names),
            annotations=self.data[0].annotations,
            meas_date=self.data[0].info["meas_date"],
            subject_info=self.data[0].info["subject_info"],
            verbose=self._verbose,
        )[0]

        self._z_scored = True
        self._update_processing_steps(f"z_score-{grouping}", picks)
        if self._verbose:
            print(
                f"Z-scoring the following channels, grouping by {grouping}:\n"
                f"{picks}\nThe following bad channels have not been Z-scored:\n"
                f"{self.data[0].info['bads']}\n"
            )

    def resample(self, resample_freq: int) -> None:
        """Resamples the mne.io.Raw or mne.Epochs object.

        PARAMETERS
        ----------
        resample_freq : int
        -   The frequency, in Hz, at which to resample the data.
        """
        for data in self.data:
            data.resample(resample_freq)

        self._resampled = True
        self._update_processing_steps("resample", resample_freq)
        if self._verbose:
            print(f"Resampling the data at {resample_freq} Hz.")

    def _check_combination_input_lengths(
        self,
        ch_names_old: list[list[str]],
        ch_names_new: list[str],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_coords_new: Union[list[Union[list[Union[int, float]], None]], None],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_subregions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
    ) -> None:
        """Checks that the input for combining channels are all of the same
        length.

        PARAMETERS
        ----------
        ch_names_old : list[list[str]]
        -   A list containing sublists where the entries are the names of the
            channels to combine together.

        ch_names_new : list[str]
        -   The names of the combined channels.

        ch_types_new : list[str | None] | None
        -   The types of the new channels.
        -   If None or if some entries are None, the type is determined based on
            the channels being combined, in which case they must be of the same
            type.

        ch_coords_new : list[list[int | float] | None] | None
        -   The coordinates of the combined channels.
        -   If None or if some entries are None, the coordinates are determined
            based on the channels being combined.

        ch_regions_new : list[str | None] | None
        -   The regions (e.g. cortex, STN) of the new channels.
        -   If None or if some entries are None, the region is determined based
            on the channels being combined, in which case they must be from the
            same region.

        ch_subregions_new : list[str | None] | None
        -   The subregions (e.g. prefrontal, parietal) of the new channels.
        -   If None or if some entries are None, the subregion is determined
            based on the channels being combined, in which case they must be
            from the same region.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the new channels.
        -   If None or if some entries are None, the hemisphere is determined
            based on the channels being combined, in which case they must be
            from the same hemisphere.
        """
        identical, lengths = check_lengths_list_identical(
            to_check=[
                ch_names_old,
                ch_names_new,
                ch_types_new,
                ch_coords_new,
                ch_regions_new,
                ch_subregions_new,
                ch_hemispheres_new,
            ],
            ignore_values=[None],
        )
        if not identical:
            raise EntryLengthError(
                "Error when trying to combine data across channels:\nThe "
                "lengths of the inputs do not match: 'ch_names_old' "
                f"({lengths[0]}); 'ch_names_new' ({lengths[1]}); "
                f"'ch_types_new' ({lengths[2]}); "
                f"'ch_coords_new' ({lengths[3]}); "
                f"'ch_regions_new' ({lengths[4]}); "
                f"'ch_subregions_new' ({lengths[5]}); "
                f"'ch_hemispheres_new' ({lengths[6]});"
            )

    def _sort_combination_inputs_strings(
        self,
        ch_names_old: list[list[str]],
        inputs: Union[list[Union[str, None]], None],
        input_type: str,
    ) -> list[str]:
        """Sorts the inputs for combining channels that consist of a list of
        strings.

        PARAMETERS
        ----------
        ch_names_old : list[list[str]]
        -   A list containing sublists where the entries are the names of the
            channels to combine together.

        inputs : list[str | None] | None
        -   Features of the new, combined channels.
        -   If not None and no entries are None, no changes are made.
        -   If some entries are None, the inputs are determined based on the
            channels being combined. For this, the features of the channels
            being combined should be identical.
        -   If None, all entries are automatically determined.

        input_type : str
        -   The type of input being sorted.
        -   Supported values are: 'ch_types'; 'ch_hemispheres'; and
            'ch_regions'.

        RETURNS
        -------
        inputs : list[str]
        -   The sorted features of the new, combined channels.
        """
        supported_input_types = [
            "ch_types",
            "ch_hemispheres",
            "ch_regions",
            "ch_subregions",
        ]
        if input_type not in supported_input_types:
            raise UnavailableProcessingError(
                "Error when trying to combine data over channels:\n"
                f"The 'input_type' '{input_type}' is not recognised. "
                f"Supported values are {supported_input_types}."
            )

        if inputs is None:
            inputs = [None] * len(ch_names_old)

        for i, value in enumerate(inputs):
            if value is None:
                if input_type == "ch_types":
                    existing_values = np.unique(
                        self.data[0].get_channel_types(picks=ch_names_old[i])
                    )
                else:
                    existing_values = np.unique(
                        [
                            self.extra_info[input_type][channel]
                            for channel in ch_names_old[i]
                        ]
                    )
                if len(existing_values) > 1:
                    raise ChannelAttributeError(
                        "Error when trying to combine data over channels:\n"
                        f"The '{input_type}' for the combination of channels "
                        f"{ch_names_old[i]} is not specified, but cannot be "
                        "automatically generated as the data is being combined "
                        f"over channels with different '{input_type}' features "
                        f"({existing_values})."
                    )
                else:
                    inputs[i] = existing_values[0]

        return inputs

    def _sort_combination_inputs_numbers(
        self,
        ch_names_old: list[list[str]],
        inputs: Union[list[Union[list[Union[int, float]], None]], None],
        input_type: str,
    ) -> list[str]:
        """Sorts the inputs for combining channels that consist of a list of
        lists of numbers.

        PARAMETERS
        ----------
        ch_names_old : list[list[str]]
        -   A list containing sublists where the entries are the names of the
            channels to combine together.

        inputs : list[list[int | float] | None] | None
        -   Features of the new, combined channels.
        -   If not None and no entries are None, no changes are made.
        -   If some entries are None, the inputs are determined based on the
            channels being combined. For this, the features of the channels
            being combined should be identical.
        -   If None, all entries are automatically determined.

        input_type : str
        -   The type of input being sorted.
        -   Supported values are: 'ch_coords'.

        RETURNS
        -------
        ch_types_new : list[str]
        -   The sorted features of the new, combined channels.
        """
        supported_input_types = ["ch_coords"]
        if input_type not in supported_input_types:
            raise UnavailableProcessingError(
                "Error when trying to combine data over channels:\n"
                f"The 'input_type' '{input_type}' is not recognised. "
                f"Supported values are {supported_input_types}."
            )

        if inputs is None:
            inputs = [None] * len(ch_names_old)

        for i, value in enumerate(inputs):
            if value is None:
                if input_type == "ch_coords":
                    new_value = np.mean(
                        [
                            self.get_coordinates(channel)[0]
                            for channel in ch_names_old[i]
                        ],
                        axis=0,
                    ).tolist()
                inputs[i] = new_value

        return inputs

    def _sort_combination_inputs(
        self,
        ch_names_old: list[list[str]],
        ch_names_new: list[str],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_coords_new: Union[list[Union[list[Union[int, float]], None]], None],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_subregions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
    ) -> tuple[list[str], list[Union[int, float]], list[str]]:
        """Sorts the inputs for combining data over channels.

        PARAMETERS
        ----------
        ch_names_old : list[list[str]]
        -   A list containing sublists where the entries are the names of the
            channels to combine together.

        ch_names_new : list[str]
        -   The names of the new, combined channels, corresponding to the
            channel names in 'ch_names_old'.

        ch_types_new : list[str | None] | None
        -   The types of the new, combined channels.
        -   If an entry is None, the type is determined based on the types of
            the channels being combined. This only works if all channels being
            combined are of the same type.
        -   If None, all types are determined automatically.

        ch_coords_new : list[list[int | float] | None] | None
        -   The coordinates of the new, combined channels.
        -   If an entry is None, the coordinates are determined by averaging
            across the coordinates of the channels being combined.
        -   If None, the coordinates are automatically determined for all
            channels.

        ch_regions_new : list[str | None] | None; default None
        -   The regions (e.g. cortex, STN) of the new, combined channels.
        -   If an entry is None, the region is determined based on the regions
            of the channels being combined. This only works if all channels
            being combined are from the same region.
        -   If None, all regions are determined automatically.

        ch_subregions_new : list[str | None] | None; default None
        -   The regions (e.g. prefrontal, parietal) of the new, combined
            channels.
        -   If an entry is None, the subregion is determined based on the
            subregions of the channels being combined. This only works if all
            channels being combined are from the same region.
        -   If None, all subregions are determined automatically.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the new, combined channels.
        -   If an entry is None, the hemisphere is determined based on the
            hemispheres of the channels being combined. This only works if all
            channels being combined are from the same hemisphere.
        -   If None, all hemispheres are determined automatically.
        """
        self._check_combination_input_lengths(
            ch_names_old=ch_names_old,
            ch_names_new=ch_names_new,
            ch_types_new=ch_types_new,
            ch_coords_new=ch_coords_new,
            ch_regions_new=ch_regions_new,
            ch_subregions_new=ch_subregions_new,
            ch_hemispheres_new=ch_hemispheres_new,
        )

        ch_types_new = self._sort_combination_inputs_strings(
            ch_names_old=ch_names_old,
            inputs=ch_types_new,
            input_type="ch_types",
        )
        ch_regions_new = self._sort_combination_inputs_strings(
            ch_names_old=ch_names_old,
            inputs=ch_regions_new,
            input_type="ch_regions",
        )
        ch_subregions_new = self._sort_combination_inputs_strings(
            ch_names_old=ch_names_old,
            inputs=ch_subregions_new,
            input_type="ch_subregions",
        )
        ch_hemispheres_new = self._sort_combination_inputs_strings(
            ch_names_old=ch_names_old,
            inputs=ch_hemispheres_new,
            input_type="ch_hemispheres",
        )
        ch_coords_new = self._sort_combination_inputs_numbers(
            ch_names_old=ch_names_old,
            inputs=ch_coords_new,
            input_type="ch_coords",
        )

        return (
            ch_types_new,
            ch_coords_new,
            ch_regions_new,
            ch_subregions_new,
            ch_hemispheres_new,
        )

    def _combine_channel_data(self, to_combine: list[list[str]]) -> NDArray:
        """Combines the data of channels through addition.

        PARAMETERS
        ----------
        to_combine : list[list[str]]
        -   A list containing sublists where the entries are the names of the
            channels to combine together.

        RETURNS
        -------
        combined_data : list[numpy array]
        -   The combined data of the channels in each sublist in 'to_combine'.
        """
        combined_data = []
        for data in self.data:
            for channels in to_combine:
                data_arr = np.sum(
                    deepcopy(data.get_data(picks=channels)), axis=0
                )
                combined_data.append(data_arr)

        return combined_data

    def add_channels(
        self,
        data: list[list[NDArray]],
        data_dimensions: list[str],
        ch_names: list[str],
        ch_types: list[str],
        ch_coords: list[list[Union[int, float]]],
        ch_reref_types: list[str],
        ch_regions: list[str],
        ch_subregions: list[str],
        ch_hemispheres: list[str],
        ch_epoch_orders: Union[list[str], None] = None,
    ) -> None:
        """Adds channels to the Signal object.
        -   Data for the new channels should have the same sampling frequency as
            the data of channels already in the object.
        -   Each new channel must contain data for each window of the data in
            the signal object.

        PARAMETERS
        ----------
        data : list[list[numpy array]]
        -   List containing the data for the new channels, with each entry
            consisting of list corresponding to a data window, with each entry
            being a numpy array for the data of a channel.

        data_dimensions : list[str]
        -   Names of the dimensions in the data.

        ch_names : list[str]
        -   The names of the new channels.

        ch_types : list[str]
        -   The types of the new channels.

        ch_coords : list[list[int | float]]
        -   The coordinates of the new channels, with each entry in the list
            being a sublist containing the x-, y-, and z-axis coordinates of the
            channel.

        ch_reref_types : list[str]
        -   Rereferencing types of the new channels.

        ch_regions : list[str]
        -   The regions (e.g. cortex, STN) of the new channels.

        ch_subregions : list[str]
        -   The subregions (e.g. prefrontal, parietal) of the new channels.

        ch_hemispheres : list[str]
        -   The hemispheres of the new channels.

        ch_epoch_orders : list[str] | None; default None
        -   Epoch orders of the new channels. Entries should be "original" or
            "shuffled". If no epoching of the data has yet been performed, the
            value should be None (default).

        RAISES
        ------
        ValueError
        -   Raised if the number of windows in the original data and the data
            being added do not match.
        """
        if len(data) != len(self.data):
            raise ValueError(
                "Error when adding channels to the Signal object:\nThe number "
                f"of windows in the original data ({len(self.data)}) and in "
                f"the data being added ({len(data)}) do not match. If data is "
                "being added, data must be given for each window in the "
                "original data."
            )

        data = rearrange_axes(
            obj=data,
            old_order=data_dimensions,
            new_order=self._data_dimensions,
        )

        for i, data_window in enumerate(data):
            new_channels, _ = create_mne_data_object(
                data=data_window,
                data_dimensions=self._data_dimensions[1:],
                ch_names=ch_names,
                ch_types=ch_types,
                sfreq=self.data[0].info["sfreq"],
                ch_coords=ch_coords,
                verbose=self._verbose,
            )
            self.data[i].add_channels([new_channels], force_update_info=True)

        for i, name in enumerate(ch_names):
            self.extra_info["ch_reref_types"][name] = ch_reref_types[i]
            self.extra_info["ch_regions"][name] = ch_regions[i]
            self.extra_info["ch_subregions"][name] = ch_subregions[i]
            self.extra_info["ch_hemispheres"][name] = ch_hemispheres[i]
            if ch_epoch_orders is not None:
                self.extra_info["ch_epoch_orders"][name] = ch_epoch_orders[i]

    def combine_channels(
        self,
        ch_names_old: list[list[str]],
        ch_names_new: list[str],
        ch_types_new: Optional[list[Union[str, None]]] = None,
        ch_coords_new: Optional[
            list[Union[list[Union[int, float]], None]]
        ] = None,
        ch_regions_new: Optional[list[Union[str, None]]] = None,
        ch_subregions_new: Optional[list[Union[str, None]]] = None,
        ch_hemispheres_new: Optional[list[Union[str, None]]] = None,
    ) -> None:
        """Combines the data of multiple channels in the mne.io.Raw object through
        addition and adds this combined data as a new channel.

        PARAMETERS
        ----------
        ch_names_old : list[list[str]]
        -   A list containing sublists where the entries are the names of the
            channels to combine together.

        ch_names_new : list[str]
        -   The names of the new, combined channels, corresponding to the
            channel names in 'ch_names_old'.

        ch_types_new : list[str | None] | None; default None
        -   The types of the new, comined channels.
        -   If an entry is None, the type is determined based on the types of
            the channels being combined. This only works if all channels being
            combined are of the same type.
        -   If None, all types are determined automatically.

        ch_coords_new : list[list[int | float] | None] | None; default None
        -   The coordinates of the new, combined channels.
        -   If an entry is None, the coordinates are determined by averaging
            across the coordinates of the channels being combined.
        -   If None, the coordinates are automatically determined for all
            channels.

        ch_regions_new : list[str | None] | None; default None
        -   The regions (e.g. cortex, STN) of the new, combined channels.
        -   If an entry is None, the region is determined based on the regions
            of the channels being combined. This only works if all channels
            being combined are from the same region.
        -   If None, all regions are determined automatically.

        ch_subregions_new : list[str | None] | None; default None
        -   The regions (e.g. prefrontal, parietal) of the new, combined
            channels.
        -   If an entry is None, the subregion is determined based on the
            subregions of the channels being combined. This only works if all
            channels being combined are from the same region.
        -   If None, all subregions are determined automatically.

        ch_hemispheres_new : list[str | None] | None; default None
        -   The hemispheres of the new, combined channels.
        -   If an entry is None, the hemisphere is determined based on the
            hemispheres of the channels being combined. This only works if all
            channels being combined are from the same hemisphere.
        -   If None, all hemispheres are determined automatically.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the data has been windowed or epoched.
        """
        if self._windowed:
            raise ProcessingOrderError(
                "Error when attempting to combine data across channels:\nThis "
                "is only supported for non-windowed data, however the data has "
                "been windowed."
            )
        if self._epoched:
            raise ProcessingOrderError(
                "Error when attempting to combine data across channels:\nThis "
                "is only supported for non-epoched data, however the data has "
                "been epoched."
            )

        (
            ch_types_new,
            ch_coords_new,
            ch_regions_new,
            ch_subregions_new,
            ch_hemispheres_new,
        ) = self._sort_combination_inputs(
            ch_names_old=ch_names_old,
            ch_names_new=ch_names_new,
            ch_types_new=ch_types_new,
            ch_coords_new=ch_coords_new,
            ch_regions_new=ch_regions_new,
            ch_subregions_new=ch_subregions_new,
            ch_hemispheres_new=ch_hemispheres_new,
        )

        combined_data = self._combine_channel_data(
            to_combine=ch_names_old,
        )

        self.add_channels(
            data=combined_data,
            data_dimensions=["channels", "timepoints"],
            ch_names=ch_names_new,
            ch_types=ch_types_new,
            ch_coords=ch_coords_new,
            ch_reref_types=[None] * len(ch_names_new),
            ch_regions=ch_regions_new,
            ch_subregions=ch_subregions_new,
            ch_hemispheres=ch_hemispheres_new,
            ch_epoch_orders=None,
        )

        if self._verbose:
            print(
                "Creating new channels of data by combining the data of "
                "pre-existing channels:"
            )
            [
                print(f"{ch_names_old[i]} -> {ch_names_new[i]}")
                for i in range(len(ch_names_old))
            ]
            print("\n")

    def drop_unrereferenced_channels(self) -> None:
        """Drops channels that have not been rereferenced from the mne.io.Raw or
        mne.Epochs object, also discarding entries for these channels from
        'extra_info'.
        """
        self._drop_channels(
            [
                ch_name
                for ch_name in self.extra_info["ch_reref_types"].keys()
                if self.extra_info["ch_reref_types"][ch_name] == "none"
            ]
        )

    def _apply_rereference(
        self,
        reref_method: Reref,
        ch_names_old: list[Union[str, list[str]]],
        ch_names_new: Union[list[Union[str, None]], None],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_reref_types: Union[list[Union[str, None]], None],
        ch_coords_new: Union[list[Union[list[Union[int, float]], None]], None],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_subregions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
    ) -> tuple[mne.io.Raw, list[str], dict[str], dict[str]]:
        """Applies a rereferencing method to the mne.io.Raw object.

        PARAMETERS
        ----------
        reref_method : Reref
        -   The rereferencing method to apply.

        ch_names_old : list[str | list[str]]
        -   The names of the channels in the mne.io.Raw object to rereference.
        -   If bipolar rereferencing, each entry of the list should be a list of
            two channel names (i.e. a cathode and an anode).

        ch_names_new : list[str | None] | None
        -   The names of the newly rereferenced channels, corresponding to the
            channels used for rerefrencing in ch_names_old.
        -   If some or all entries are None, names of the new channels are
            determined based on those they are referenced from.

        ch_types_new : list[str | None] | None
        -   The types of the newly rereferenced channels as recognised by MNE,
            corresponding to the channels in 'ch_names_new'.
        -   If some or all entries are None, types of the new channels are
            determined based on those they are referenced from.

        reref_types : list[str | None] | None
        -   The rereferencing type applied to the channels, corresponding to the
            channels in 'ch_names_new'.
        -   If some or all entries are None, types of the new channels are
            determined based on those they are referenced from.

        ch_coords_new : list[list[int | float] | None] | None
        -   The coordinates of the newly rereferenced channels, corresponding to
            the channels in 'ch_names_new'. The list should consist of sublists
            containing the x, y, and z coordinates of each channel.
        -   If the input is None, the coordinates of the channels in
            'ch_names_old' in the mne.io.Raw object are used.
        -   If some entries are None, those channels for which coordinates are
            given are used, whilst those channels for which the coordinates are
            missing have their coordinates taken from the mne.io.Raw object
            according to the corresponding channel in 'ch_names_old'.

        ch_regions_new : list[str | None] | None
        -   The regions of the rereferenced channels (e.g. cortex, STN),
            corresponding to the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_subregions_new : list[str | None] | None
        -   The subregions of the rereferenced channels (e.g. prefrontal,
            parietal), corresponding to the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the rereferenced channels channels, corresponding
            to the channels in 'ch_names_new'.
        -   If some or all entries are None, hemispheres of the new channels are
            determined based on those they are referenced from.

        RETURNS
        -------
        MNE Raw
        -   The rereferenced data in an mne.io.Raw object.

        list[str]
        -   Names of the channels that were produced by the rereferencing.

        dict[str]
        -   Dictionary showing the rereferencing types applied to the channels,
            in which the key:value pairs are channel name : rereference type.

        dict[str]
        -   Dictionary showing the regions of the rereferenced channels, in
            which the key:value pairs are channel name : region.
        """
        return reref_method(
            self.data[0].copy(),
            self.extra_info,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            ch_reref_types,
            ch_coords_new,
            ch_regions_new,
            ch_subregions_new,
            ch_hemispheres_new,
        ).rereference()

    def _remove_conflicting_channels(self, ch_names: list[str]) -> None:
        """Removes channels from the self mne.io.Raw or mne.Epochs object.
        -   Designed for use alongside '_append_rereferenced_raw'.
        -   Useful to perform before appending an external mne.io.Raw or
            mne.Epochs object.

        PARAMETERS
        ----------
        ch_names : list[str]
        -   Names of channels to remove from the mne.io.Raw or mne.Epochs
            object.
        """
        self._drop_channels(ch_names)
        print(
            "Warning when rereferencing data:\nThe following rereferenced "
            f"channels {ch_names} are already present in the raw data.\n"
            "Removing the channels from the raw data.\n"
        )

    def _append_rereferenced_raw(self, rerefed_raw: mne.io.Raw) -> None:
        """Appends a rereferenced mne.io.Raw object to the self mne.io.Raw
        object, first discarding channels in the self mne.io.Raw object which
        have the same names as those in the mne.io.Raw object to append.

        PARAMETERS
        ----------
        rerefed_raw : mne.io.Raw
        -   An mne.io.Raw object that has been rereferenced which will be
            appended.
        """
        _, repeated_chs = check_repeated_vals(
            to_check=[
                *self.data[0].info["ch_names"],
                *rerefed_raw.info["ch_names"],
            ]
        )
        if repeated_chs is not None:
            self._remove_conflicting_channels(repeated_chs)

        self.data[0].add_channels([rerefed_raw])

    def _add_rereferencing_info(self, info_to_add: dict) -> None:
        """Adds channel rereferencing information to 'extra_info'.

        PARAETERS
        ---------
        info_to_add : dict
        -   A dictionary used for updating 'extra_info'.
        -   The dictionary's keys are the names of the entries that will be
            updated in 'extra_info' with the corresponding values.
        """
        [
            self.extra_info[key].update(info_to_add[key])
            for key in info_to_add.keys()
        ]

    def _get_channel_rereferencing_pairs(
        self,
        ch_names_old: list[Union[str, list[str]]],
        ch_names_new: list[Union[str, list[str]]],
    ) -> list[str]:
        """Collects the names of the channels that were referenced and the newly
        generated channels together.

        PARAMETERS
        ----------
        ch_names_old : list[str | list[str]]
        -   Names of the channels that were rereferenced.
        -   If bipolar rereferencing, each entry of the list should be a list of
            two channel names (i.e. a cathode and an anode).

        ch_names_new : list[str]
        -   Names of the channels that were produced by the rereferencing.

        RETURNS
        -------
        list[str | list[str]]
        -   List of sublists, in which each sublist contains the name(s) of the
            channel(s) that was(were) rereferenced, and the name of the channel
            that was produced.
        """
        return [
            [ch_names_old[i], ch_names_new[i]]
            for i in range(len(ch_names_old))
        ]

    def _rereference(
        self,
        reref_method: Reref,
        ch_names_old: list[Union[str, list[str]]],
        ch_names_new: Union[list[Union[str, None]], None],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_reref_types: Union[list[Union[str, None]], None],
        ch_coords_new: Union[list[Union[list[Union[int, float]], None]], None],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_subregions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
    ) -> list[str]:
        """Parent method for calling on other methods to rereference the data,
        add it to the self mne.io.Raw object, and add the rereferecing
        information to 'extra_info'.

        PARAMETERS
        ----------
        RerefMethod : Reref
        -   The rereferencing method to apply.

        ch_names_old : list[str]
        -   The names of the channels in the mne.io.Raw object to rereference.

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
        -   Missing values (None) will be set as 'common_average'.

        ch_coords_new : list[list[int | float] | None] | None; default None
        -   The coordinates of the newly rereferenced channels, corresponding to
            the channels in 'ch_names_new'. The list should consist of sublists
            containing the x, y, and z coordinates of each channel.
        -   If the input is '[]', the coordinates of the channels in
            'ch_names_old' in the mne.io.Raw object are used.
        -   If some sublists are '[]', those channels for which coordinates are
            given are used, whilst those channels for which the coordinates are
            missing have their coordinates taken from the mne.io.Raw object
            according to the corresponding channel in 'ch_names_old'.

        ch_regions_new : list[str | None] | None
        -   The regions of the rereferenced channels (e.g. cortex, STN),
            corresponding to the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_subregions_new : list[str | None] | None
        -   The subregions of the rereferenced channels (e.g. prefrontal,
            parietal), corresponding to the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the rereferenced channels channels, corresponding
            to the channels in 'ch_names_new'.
        -   If some or all entries are None, hemispheres of the new channels are
            determined based on those they are referenced from.

        RETURNS
        -------
        ch_names_new : list[str]
        -   List containing the names of the new, rereferenced channels.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to rereference the data after it has
            already been windowed or epoched.
        """
        if self._windowed:
            raise ProcessingOrderError(
                "Error when rereferencing the data:\nThe data to rereference "
                "has been windowed."
            )
        if self._epoched:
            raise ProcessingOrderError(
                "Error when rereferencing the data:\nThe data to rereference "
                "has been epoched."
            )

        (
            rerefed_raw,
            ch_names_new,
            ch_reref_types_dict,
            ch_regions_dict,
            ch_subregions_dict,
            ch_hemispheres_dict,
        ) = self._apply_rereference(
            reref_method,
            ch_names_old,
            ch_names_new,
            ch_types_new,
            ch_reref_types,
            ch_coords_new,
            ch_regions_new,
            ch_subregions_new,
            ch_hemispheres_new,
        )
        self._append_rereferenced_raw(rerefed_raw)
        self._add_rereferencing_info(
            info_to_add={
                "ch_reref_types": ch_reref_types_dict,
                "ch_regions": ch_regions_dict,
                "ch_subregions": ch_subregions_dict,
                "ch_hemispheres": ch_hemispheres_dict,
            }
        )

        self._rereferenced = True

        return ch_names_new

    def rereference_bipolar(
        self,
        ch_names_old: list[list[str]],
        ch_names_new: Union[list[Union[str, None]], None],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_reref_types: Union[list[Union[str, None]], None],
        ch_coords_new: Union[list[Union[list[Union[int, float]], None]], None],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_subregions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
        eligible_entries: dict | None = None,
    ) -> None:
        """Bipolar rereferences channels in the mne.io.Raw object.

        PARAMETERS
        ----------
        ch_names_old : list[list[str]]
        -   The names of the channels in the mne.io.Raw object to rereference.

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
        -   Missing values (None) will be set as 'common_average'.

        ch_coords_new : list[list[int | float] | None] or None; default None
        -   The coordinates of the newly rereferenced channels, corresponding to
            the channels in 'ch_names_new'. The list should consist of sublists
            containing the x, y, and z coordinates of each channel.
        -   If the input is '[]', the coordinates of the channels in
            'ch_names_old' in the mne.io.Raw object are used.
        -   If some sublists are '[]', those channels for which coordinates are
            given are used, whilst those channels for which the coordinates are
            missing have their coordinates taken from the mne.io.Raw object
            according to the corresponding channel in 'ch_names_old'.

        ch_regions_new : list[str | None] | None
        -   The regions of the rereferenced channels (e.g. cortex, STN),
            corresponding to the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_subregions_new : list[str | None] | None
        -   The subregions of the rereferenced channels (e.g. prefrontal,
            parietal), corresponding to the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the rereferenced channels channels, corresponding
            to the channels in 'ch_names_new'.
        -   If some or all entries are None, hemispheres of the new channels are
            determined based on those they are referenced from.

        eligible_entries : dict | None (default None)
            Eligible channels in the data to rereference. Keys of the dict
            should correspond to features of the channels.
        """
        kwargs = {
            "ch_names_old": ch_names_old,
            "ch_names_new": ch_names_new,
            "ch_types_new": ch_types_new,
            "ch_reref_types": ch_reref_types,
            "ch_coords_new": ch_coords_new,
            "ch_regions_new": ch_regions_new,
            "ch_subregions_new": ch_subregions_new,
            "ch_hemispheres_new": ch_hemispheres_new,
        }

        if eligible_entries is not None:
            kwargs = self._get_eligible_reref_entries(kwargs, eligible_entries)

        ch_names_new = self._rereference(RerefBipolar, **kwargs)

        self._rereferenced_bipolar = True
        ch_reref_pairs = self._get_channel_rereferencing_pairs(
            kwargs["ch_names_old"], kwargs["ch_names_new"]
        )
        self._update_processing_steps("rereferencing_bipolar", ch_reref_pairs)
        if self._verbose:
            print("The following channels have been bipolar rereferenced:")
            [
                print(f"{old[0]} - {old[1]} -> {new}")
                for [old, new] in ch_reref_pairs
            ]
            print("\n")

    def rereference_common_average(
        self,
        ch_names_old: list[str],
        ch_names_new: Union[list[Union[str, None]], None],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_reref_types: Union[list[Union[str, None]], None],
        ch_coords_new: Union[list[Union[list[Union[int, float]], None]], None],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_subregions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
        eligible_entries: dict | None = None,
    ) -> None:
        """Common-average rereferences channels in the mne.io.Raw object.

        PARAMETERS
        ----------
        ch_names_old : list[str]
        -   The names of the channels in the mne.io.Raw object to rereference.

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
        -   Missing values (None) will be set as 'common_average'.

        ch_coords_new : list[list[int | float] | None] | None; default None
        -   The coordinates of the newly rereferenced channels, corresponding to
            the channels in 'ch_names_new'. The list should consist of sublists
            containing the x, y, and z coordinates of each channel.
        -   If the input is '[]', the coordinates of the channels in
            'ch_names_old' in the mne.io.Raw object are used.
        -   If some sublists are '[]', those channels for which coordinates are
            given are used, whilst those channels for which the coordinates are
            missing have their coordinates taken from the mne.io.Raw object
            according to the corresponding channel in 'ch_names_old'.

        ch_regions_new : list[str | None] | None
        -   The regions of the rereferenced channels (e.g. cortex, STN),
            corresponding to the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_subregions_new : list[str | None] | None
        -   The subregions of the rereferenced channels (e.g. prefrontal,
            parietal), corresponding to the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the rereferenced channels channels, corresponding
            to the channels in 'ch_names_new'.
        -   If some or all entries are None, hemispheres of the new channels are
            determined based on those they are referenced from.

        eligible_entries : dict | None (default None)
            Eligible channels in the data to rereference. Keys of the dict
            should correspond to features of the channels.
        """
        kwargs = {
            "ch_names_old": ch_names_old,
            "ch_names_new": ch_names_new,
            "ch_types_new": ch_types_new,
            "ch_reref_types": ch_reref_types,
            "ch_coords_new": ch_coords_new,
            "ch_regions_new": ch_regions_new,
            "ch_subregions_new": ch_subregions_new,
            "ch_hemispheres_new": ch_hemispheres_new,
        }

        if eligible_entries is not None:
            kwargs = self._get_eligible_reref_entries(kwargs, eligible_entries)

        ch_names_new = self._rereference(RerefCommonAverage, **kwargs)

        self._rereferenced_common_average = True
        ch_reref_pairs = self._get_channel_rereferencing_pairs(
            kwargs["ch_names_old"], kwargs["ch_names_new"]
        )
        self._update_processing_steps(
            "rereferencing_common_average", ch_reref_pairs
        )
        if self._verbose:
            print(
                "The following channels have been common-average rereferenced:"
            )
            [print(f"{old} -> {new}") for [old, new] in ch_reref_pairs]
            print("\n")

    def rereference_pseudo(
        self,
        ch_names_old: list[str],
        ch_names_new: Union[list[Union[str, None]], None],
        ch_types_new: Union[list[Union[str, None]], None],
        ch_reref_types: list[str],
        ch_coords_new: Optional[list[Optional[list[Union[int, float]]]]],
        ch_regions_new: Union[list[Union[str, None]], None],
        ch_subregions_new: Union[list[Union[str, None]], None],
        ch_hemispheres_new: Union[list[Union[str, None]], None],
        eligible_entries: dict | None = None,
    ) -> None:
        """Pseudo rereferences channels in the mne.io.Raw object.
        -   This allows e.g. rereferencing types, channel coordinates, etc... to
            be assigned to the channels without any rereferencing occuring.
        -   This is useful if e.g. the channels were already hardware
            rereferenced.

        PARAMETERS
        ----------
        ch_names_old : list[str]
        -   The names of the channels in the mne.io.Raw object to rereference.

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
        -   No missing values (None) can be given, as the rereferencing type
            cannot be determined dynamically from this arbitrary rereferencing
            method.

        ch_coords_new : list[list[int | float] | None] | None; default None
        -   The coordinates of the newly rereferenced channels, corresponding to
            the channels in 'ch_names_new'. The list should consist of sublists
            containing the x, y, and z coordinates of each channel.
        -   If the input is '[]', the coordinates of the channels in
            'ch_names_old' in the mne.io.Raw object are used.
        -   If some sublists are '[]', those channels for which coordinates are
            given are used, whilst those channels for which the coordinates are
            missing have their coordinates taken from the mne.io.Raw object
            according to the corresponding channel in 'ch_names_old'.

        ch_regions_new : list[str | None] | None
        -   The regions of the rereferenced channels (e.g. cortex, STN),
            corresponding to the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_subregions_new : list[str | None] | None
        -   The subregions of the rereferenced channels (e.g. prefrontal,
            parietal), corresponding to the channels in 'ch_names_new'.
        -   If some or all entries are None, regions of the new channels are
            determined based on those they are referenced from.

        ch_hemispheres_new : list[str | None] | None
        -   The hemispheres of the rereferenced channels channels, corresponding
            to the channels in 'ch_names_new'.
        -   If some or all entries are None, hemispheres of the new channels are
            determined based on those they are referenced from.

        eligible_entries : dict | None (default None)
            Eligible channels in the data to rereference. Keys of the dict
            should correspond to features of the channels.
        """
        kwargs = {
            "ch_names_old": ch_names_old,
            "ch_names_new": ch_names_new,
            "ch_types_new": ch_types_new,
            "ch_reref_types": ch_reref_types,
            "ch_coords_new": ch_coords_new,
            "ch_regions_new": ch_regions_new,
            "ch_subregions_new": ch_subregions_new,
            "ch_hemispheres_new": ch_hemispheres_new,
        }

        if eligible_entries is not None:
            kwargs = self._get_eligible_reref_entries(kwargs, eligible_entries)

        ch_names_new = self._rereference(RerefPseudo, **kwargs)

        self._rereferenced_pseudo = True
        ch_reref_pairs = self._get_channel_rereferencing_pairs(
            kwargs["ch_names_old"], kwargs["ch_names_new"]
        )
        self._update_processing_steps("rereferencing_pseudo", ch_reref_pairs)
        if self._verbose:
            print("The following channels have been pseudo rereferenced:")
            [print(f"{old} -> {new}") for [old, new] in ch_reref_pairs]

    def _get_eligible_reref_entries(
        self, kwargs: dict, eligible_entries: dict
    ) -> dict:
        """Get the eligible channels for rereferencing."""
        drop_ch_idcs = []
        features = self._features_to_df()
        for key, value in eligible_entries.items():
            for ch_i, ch_name in enumerate(kwargs["ch_names_old"]):
                if not isinstance(ch_name, list):
                    ch_name = [ch_name]
                ch_feature = [
                    features.loc[features["ch_names"] == name, key].item()
                    for name in ch_name
                ]
                if not set(ch_feature).issubset(value):
                    drop_ch_idcs.append(ch_i)

        eligible_ch_idcs = [
            i
            for i in range(len(kwargs["ch_names_old"]))
            if i not in np.unique(drop_ch_idcs)
        ]
        for key, value in kwargs.items():
            if value is not None:
                kwargs[key] = [value[i] for i in eligible_ch_idcs]

        return kwargs

    def window(self, window_length: Union[int, float]) -> None:
        """Converts an MNE Raw object into multiple MNE Raw objects, each with a
        specified duration.

        PARAMETERS
        ----------
        window_length : int | float
        -   The duration of the windows (seconds) to divide the data into.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to divide the data into windows once it
            has already been windowed or epoched.
        """
        if self._windowed:
            raise ProcessingOrderError(
                "Error when windowing the data:\nThe data has already been "
                "windowed."
            )
        if self._epoched:
            raise ProcessingOrderError(
                "Error when windowing the data:\nThe data has been epoched. "
                "Windowing can only be performed on non-epoched data."
            )

        windows = mne.make_fixed_length_epochs(self.data[0], window_length)
        windowed_data = []
        for window in windows:
            data, _ = create_mne_data_object(
                data=window,
                data_dimensions=["channels", "timepoints"],
                ch_names=windows.ch_names,
                ch_types=windows.get_channel_types(),
                sfreq=windows.info["sfreq"],
                ch_coords=self.get_coordinates(),
                subject_info=windows.info["subject_info"],
            )
            windowed_data.append(data)
        self.data = windowed_data

        self._windowed = True
        self._update_processing_steps("window_data", window_length)
        if self._verbose:
            print(
                f"Windowing the data with window lengths of {window_length} "
                "seconds.\n"
            )

    def pseudo_window(self, window_length: Union[int, float]) -> None:
        """Shortens the length of the data in the MNE raw object as if the data
        was being windowed without altering the dimensionality of the data.

        E.g. if data of length 80 s was being divided into 30 s windows, two
        new 30 s windows of data would be generated from the first 60 s of data,
        and the final 20 s of data discarded. When pseudo-windowing data, the
        first 60 s of data would be retained, however each 30 s period of data
        would not actually be separated into certain windows.

        PARAMETERS
        ----------
        window_length : int | float
        -   The duration of the windows (seconds) to pseudo-divide the data into.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to pseudo-divide the data into windows
            once it has already been windowed or epoched.
        """
        if self._windowed:
            raise ProcessingOrderError(
                "Error when pseudo-windowing the data:\nThe data has already "
                "been windowed."
            )
        if self._epoched:
            raise ProcessingOrderError(
                "Error when pseudo-windowing the data:\nThe data has been "
                "epoched. Windowing can only be performed on non-epoched data."
            )

        n_windows = int(self.data[0].times[-1] // window_length)
        pseudo_window_length = window_length * n_windows
        pseudo_window = mne.make_fixed_length_epochs(
            self.data[0], pseudo_window_length
        )
        data, _ = create_mne_data_object(
            data=pseudo_window.get_data()[0, :, :],
            data_dimensions=["channels", "timepoints"],
            ch_names=pseudo_window.ch_names,
            ch_types=pseudo_window.get_channel_types(),
            sfreq=pseudo_window.info["sfreq"],
            ch_coords=self.get_coordinates(),
            subject_info=pseudo_window.info["subject_info"],
        )
        self.data = [data]

        self._pseudo_windowed = True
        self._update_processing_steps(
            "pseudo_window_data", pseudo_window_length
        )
        if self._verbose:
            print(
                f"Pseudo-windowing the data into {n_windows} windows with "
                f"lengths of {window_length} seconds ({pseudo_window_length} "
                "seconds total).\n"
            )

    def epoch(
        self, length: int | float, sd_outlier: int | float | None
    ) -> None:
        """Converts the data in one or many MNE Raw object(s) into one or many
        MNE Epochs object(s) containing epochs of a specified duration and
        drops bad epochs.

        PARAMETERS
        ----------
        length : int | float
            The duration of the epochs (seconds) to divide the data into.

        sd_outlier : int | float | None
            Standard deviation multiplier to treat as outliers. If not None,
            S.D. of each epoch is compared to the average across all epochs and
            if S.D. >= mean S.D. * `sd_outlier` for any channel, the epoch is
            removed.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the user attempts to epoch the data once it has already
            been epoched.
        -   This method can only be called if the data is stored as an MNE Raw
            object, not as an MNE Epochs object.
        """
        if self._epoched or self._windowed:
            raise ProcessingOrderError(
                "Error when epoching the data:\nThe data has already been "
                "epoched and/or windowed."
            )

        epochs = mne.make_fixed_length_epochs(
            self.data[0], duration=length, reject_by_annotation=True
        )
        epochs.load_data()
        epochs.apply_baseline((None, None))
        self.data[0] = epochs

        outlier_idcs = []
        if sd_outlier is not None:
            data = self.data[0].get_data()
            mean_sd = np.std(data, axis=-1).mean(axis=0)
            for epoch_i, epoch in enumerate(data):
                if any(np.std(epoch, axis=-1) >= mean_sd * sd_outlier):
                    outlier_idcs.append(epoch_i)
            if outlier_idcs != []:
                self.data[0].drop(outlier_idcs, verbose=False)

        self.data[0].load_data()

        self._epoched = True
        self._update_processing_steps("epoch_data", length)
        self._data_dimensions = ["windows", "epochs", "channels", "timepoints"]
        self.extra_info["ch_epoch_orders"] = {
            name: "original" for name in self.data[0].ch_names
        }
        if self._verbose:
            print(
                f"Epoching the data with epoch lengths of {length} seconds.\n"
                f"Removing the following artefact epochs: {outlier_idcs}"
            )

    def bootstrap(
        self,
        n_bootstraps: int,
        n_epochs_per_bootstrap: int,
        random_seed: int = 44,
    ):
        """Bootstraps epochs into windows. Epochs are sampled with replacement,
        using a uniform distribution.

        If data has already been windowed, bootstrapping is applied to each
        window separately (i.e. the data is not recombined first).

        Prints a warning if the settings are insufficient to allow all epochs to
        be sampled (i.e. 'n_bootstraps' * 'n_epochs_per_bootstrap' < n_epochs).

        PARAMETERS
        ----------
        n_bootstraps : int
        -   The number of bootstrapping iterations to perform (i.e. the number
            of windows that will be generated).

        n_epochs_per_bootstrap : int
        -   The number of epochs to sample each bootstrapping iteration (i.e.
            the number of epochs per window).

        random_seed : int; default 44
        -   Seed to set numpy random before sampling the epochs.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the data has not been epoched or has been windowed.

        ValueError
        -   Raised if 'n_bootstraps' or 'n_epochs_per_bootstrap' < 1.
        """
        if not self._epoched:
            raise ProcessingOrderError(
                "Data must be epoched before bootstrapping."
            )
        if self._windowed:
            raise ProcessingOrderError(
                "Data must not be windowed before bootstrapping."
            )
        if n_bootstraps < 1:
            raise ValueError(
                f"'n_bootstraps' must be at least 1, but is {n_bootstraps}."
            )
        if n_epochs_per_bootstrap < 1:
            raise ValueError(
                "'n_epochs_per_bootstrap' must be at least 1, but is "
                f"{n_epochs_per_bootstrap}."
            )
        n_epochs = self.data[0].get_data().shape[0]
        if n_epochs_per_bootstrap * n_bootstraps < n_epochs:
            Warning(
                "'n_bootstraps' and 'n_epochs_per_bootstrap' are insufficient "
                "to allow all available epochs in the data to be sampled."
            )

        epoch_picks = self._bootstrap_pick_epochs(
            n_bootstraps=n_bootstraps,
            n_epochs_per_bootstrap=n_epochs_per_bootstrap,
            random_seed=random_seed,
        )
        percentage_epoch_coverage = self._bootstrap_get_epoch_coverage(
            epoch_picks=epoch_picks
        )
        bootstrap_windows = self._bootstrap_create_windows(
            epoch_picks=epoch_picks
        )
        self._bootstrap_store_windows(bootstrap_windows=bootstrap_windows)

        self._bootstrapped = True
        self._windowed = True
        self._update_processing_steps(
            "bootstrap_data",
            {
                "n_bootstraps": n_bootstraps,
                "n_epochs_per_bootstrap": n_epochs_per_bootstrap,
                "random_seed": random_seed,
                "percentage_epoch_coverage": percentage_epoch_coverage,
            },
        )
        self.extra_info["bootstraps"] = epoch_picks.tolist()

        if self._verbose:
            n_epochs_dropped = int(n_epochs * percentage_epoch_coverage / 100)
            print(
                f"Bootstrapping the data {n_bootstraps} times, selecting "
                f"{n_epochs_per_bootstrap} epochs per bootstrap, covering "
                f"{int(percentage_epoch_coverage)}% of available epochs "
                f"({n_epochs_dropped} of {n_epochs})."
            )

    def _bootstrap_pick_epochs(
        self,
        n_bootstraps: int,
        n_epochs_per_bootstrap: int,
        random_seed: int,
    ) -> np.ndarray:
        """Generate an array of indices listing the epochs which should be
        picked for each bootstrapping iteration, sampling with replacement using
        a uniform distribution.

        PARAMETERS
        ----------
        n_bootstraps : int
        -   The number of bootstrapping iterations to perform (i.e. the number
            of windows that will be generated).

        n_epochs_per_bootstrap : int
        -   The number of epochs to sample each bootstrapping iteration (i.e.
            the number of epochs per window).

        random_seed : int; default 44
        -   Seed to set numpy random before sampling the epochs.

        RETURNS
        -------
        epoch_picks : numpy ndarray
        -   The sampled epochs, with shape ['n_bootstraps' x
            'n_epochs_per_bootstrap'].
        """
        random = np.random.RandomState(random_seed)

        epoch_picks = random.randint(
            self.data[0].get_data().shape[0],
            size=(n_bootstraps, n_epochs_per_bootstrap),
            dtype=np.int32,
        )

        return epoch_picks

    def _bootstrap_get_epoch_coverage(self, epoch_picks: np.ndarray) -> float:
        """Finds the percentage of available epochs present in the bootstrap
        picks.

        PARAMETERS
        ----------
        epoch_picks : numpy ndarray of int
        -   The sampled epochs, with shape ['n_bootstraps' x
            'n_epochs_per_bootstrap'].

        RETURNS
        -------
        percentage_epoch_coverage : float
        -   The percentage of available epochs present in the bootstrap picks.
        """
        return (
            np.unique(epoch_picks).shape[0] / self.data[0].get_data().shape[0]
        ) * 100

    def _bootstrap_create_windows(self, epoch_picks: np.ndarray) -> np.ndarray:
        """Allocates the chosen epochs to a window for each bootstrap iteration.

        PARAMETERS
        ----------
        epoch_picks : numpy ndarray of int
        -   The sampled epochs, with shape ['n_bootstraps' x
            'n_epochs_per_bootstrap'].

        RETURNS
        -------
        bootstrap_windows : numpy ndarray
        -   Windows of data with the chosen epochs for each bootstrap iteration.
        """
        data_array = self.data[0].get_data(picks=self.data[0].ch_names)
        bootstrap_windows = np.empty(
            (epoch_picks.shape[0], epoch_picks.shape[1], *data_array.shape[1:])
        )
        for bootstrap_idx, iteration_picks in enumerate(epoch_picks):
            bootstrap_windows[bootstrap_idx] = data_array[iteration_picks]

        return bootstrap_windows

    def _bootstrap_store_windows(self, bootstrap_windows: np.ndarray) -> None:
        """Converts the bootstrapped data to MNE Epochs objects and stores it,
        replacing the previous data.

        PARAMETERS
        ----------
        bootstrap_windows : numpy ndarray
        -   Windows of data with the chosen epochs for each bootstrap iteration.
        """
        epochs = []
        for window in bootstrap_windows:
            epochs_object, _ = create_mne_data_object(
                data=window,
                data_dimensions=["epochs", "channels", "timepoints"],
                ch_names=self.data[0].ch_names,
                ch_types=self.data[0].get_channel_types(),
                sfreq=self.data[0].info["sfreq"],
                ch_coords=self.get_coordinates(),
                tmin=self.data[0].tmin,
                subject_info=self.data[0].info["subject_info"],
                verbose=False,
            )
            epochs.append(epochs_object)
        self.data = epochs

    def shuffle(
        self,
        channels: list[str],
        n_shuffles: int,
        rng_seed: Union[int, float, None] = None,
    ) -> None:
        """Creates new channels by randomly reordering the epochs of channels,
        creating time-series data with a disrupted temporal order.

        PARAMETERS
        ----------
        channels : list[str]
        -   Names of the channels to create shuffled copies of.

        n_shuffles : int
        -   The number of shuffled copies to create for each channel.

        rng_seed : int | float
        -   The seed to use for the random number generator. The seed is set
            once before any shuffling takes place, and is not set again once the
            shuffling has begun.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the data has not been epoched.
        """
        if not self._epoched:
            raise ProcessingOrderError(
                "Error when shuffling the data:\nThe data has not been "
                "epoched, but shuffling requires the data be epoched."
            )

        if rng_seed is not None:
            np.random.seed(rng_seed)

        shuffled_data = []
        for window_i, data in enumerate(self.data):
            to_shuffle = deepcopy(data)
            to_shuffle.pick(channels)
            shuffled_data.append([])
            for shuffle_i in range(n_shuffles):
                epoch_order = np.arange(len(to_shuffle.events))
                np.random.shuffle(epoch_order)
                shuffled_data[window_i].append(
                    concatenate_epochs(
                        [to_shuffle[epoch_order]], verbose="ERROR"
                    )
                )
                shuffled_data[window_i][shuffle_i].rename_channels(
                    {
                        name: f"SHUFFLED[{shuffle_i}]_{name}"
                        for name in channels
                    }
                )
            if n_shuffles > 1:
                shuffled_data[window_i] = shuffled_data[window_i][
                    0
                ].add_channels(shuffled_data[window_i][1:])
            else:
                shuffled_data[window_i] = shuffled_data[window_i][0]

        ch_reref_types = (
            ordered_list_from_dict(channels, self.extra_info["ch_reref_types"])
            * n_shuffles
        )
        ch_regions = (
            ordered_list_from_dict(channels, self.extra_info["ch_regions"])
            * n_shuffles
        )
        ch_subregions = (
            ordered_list_from_dict(channels, self.extra_info["ch_subregions"])
            * n_shuffles
        )
        ch_hemispheres = (
            ordered_list_from_dict(channels, self.extra_info["ch_hemispheres"])
            * n_shuffles
        )
        self.add_channels(
            data=shuffled_data,
            data_dimensions=self._data_dimensions,
            ch_names=shuffled_data[0].ch_names,
            ch_types=shuffled_data[0].get_channel_types(),
            ch_coords=shuffled_data[0]._get_channel_positions(
                picks=shuffled_data[0].ch_names
            ),
            ch_reref_types=ch_reref_types,
            ch_regions=ch_regions,
            ch_subregions=ch_subregions,
            ch_hemispheres=ch_hemispheres,
            ch_epoch_orders=["shuffled"] * len(channels) * n_shuffles,
        )

        self._shuffled = True
        self._update_processing_steps(
            "shuffle_data",
            {
                "channels": channels,
                "n_shuffles": n_shuffles,
                "rng_seed": rng_seed,
            },
        )
        for n_shuffle in range(n_shuffles):
            for name in shuffled_data[n_shuffle].ch_names:
                self.extra_info["ch_epoch_orders"][name] = "shuffled"
        if self._verbose:
            print(
                "Creating epoch-shuffled data for the following channels over "
                f"{n_shuffles} iterations:\n{channels}\n"
            )

    def _extract_data(self, rearrange: Union[list[str], None]) -> NDArray:
        """Extracts the signals from the mne.io.Raw object.

        PARAMETERS
        ----------
        rearrange : list[str] | None; default None
        -   How to rearrange the axes of the data once extracted.
        -   E.g. ["channels", "epochs", "timepoints"] would give data in the
            format channels x epochs x timepoints
        -   If None, the data is taken as is.

        RETURNS
        -------
        extracted_data : numpy array
        -   The time-series signals extracted from the mne.io.Raw oject.
        """
        extracted_data = []
        for data in self.data:
            extracted_data.append(deepcopy(data.get_data()))
        if not self._windowed:
            extracted_data = extracted_data[0]
        else:
            extracted_data = np.asarray(extracted_data)

        if rearrange is not None:
            extracted_data = rearrange_axes(
                obj=extracted_data,
                old_order=self.data_dimensions,
                new_order=rearrange,
            )

        return extracted_data

    def save_object(
        self, fpath: str, ask_before_overwrite: Optional[bool] = None
    ) -> None:
        """Saves the Signal object as a .pkl file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved. The filetype extension
            (.pkl) can be included, otherwise it will be automatically added.

        ask_before_overwrite : bool | None; default the object's verbosity
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.
        -   By default, this is set to None, in which case the value of the
            verbosity when the Signal object was instantiated is used.
        """
        if not check_ftype_present(fpath):
            fpath += ".pkl"

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose
        if ask_before_overwrite:
            write = check_before_overwrite(fpath)
        else:
            write = True

        if write:
            save_as_pkl(to_save=self, fpath=fpath)

    def data_as_dict(self) -> dict:
        """Returns the data as a dictionary.

        RETURNS
        -------
        data_dict : dict
        -   The data as a dictionary.
        """
        rearrange = ["channels"]
        if self._windowed:
            rearrange.append("windows")
        if self._epoched:
            rearrange.append("epochs")
        rearrange.append("timepoints")
        extracted_data = self._extract_data(rearrange=rearrange).tolist()

        data_dict = {
            "data": extracted_data,
            "data_dimensions": rearrange,
            "ch_names": self.data[0].ch_names,
            "ch_types": self.data[0].get_channel_types(),
            "ch_coords": self.get_coordinates(),
            "samp_freq": self.data[0].info["sfreq"],
            "metadata": self.extra_info["metadata"],
            "processing_steps": self.processing_steps,
            "subject_info": self.data[0].info["subject_info"],
        }
        optional_extra_info_keys = [
            "ch_regions",
            "ch_subregions",
            "ch_hemispheres",
            "ch_reref_types",
            "ch_epoch_orders",
        ]
        for key in optional_extra_info_keys:
            if self.extra_info[key] is not None:
                data_dict[key] = self.extra_info[key]

        return data_dict

    def save_as_dict(
        self,
        fpath: str,
        ftype: Optional[str] = None,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the time-series data and additional information as a file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved.

        ftype : str | None; default None
        -   The filetype of the data that will be saved, without the leading
            period. E.g. for saving the file in the json format, this would be
            "json", not ".json".
        -   The information being saved must be an appropriate type for saving
            in this format.
        -   If None, the filetype is determined based on 'fpath', and so the
            extension must be included in the path.

        ask_before_overwrite : bool | None; default the object's verbosity
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.
        -   By default, this is set to None, in which case the value of the
            verbosity when the Signal object was instantiated is used.

        RAISES
        ------
        UnavailableProcessingError
        -   Raised if the given format for saving the file is in an unsupported
            format.
        """
        to_save = self.data_as_dict()

        if ftype is None:
            ftype = identify_ftype(fpath)
        if not check_ftype_present(fpath):
            fpath += ftype

        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose
        if ask_before_overwrite:
            write = check_before_overwrite(fpath)
        else:
            write = True

        if write:
            if ftype == "json":
                save_as_json(to_save=to_save, fpath=fpath)
            elif ftype == "pkl":
                save_as_pkl(to_save=to_save, fpath=fpath)
            else:
                raise UnavailableProcessingError(
                    f"Error when trying to save the raw signals:\nThe {ftype} "
                    "format for saving is not supported."
                )
            if self._verbose:
                print(f"Saving the raw signals to:\n'{fpath}'.\n")


def data_dict_to_signal(data: dict) -> Signal:
    """Converts a data dictionary into a Signal object.
    -   The signals themselves will be converted to either an MNE Raw object or
        MNE Epochs object, depending on whether the signals have been epoched.

    PARAMETERS
    ----------
    data : dict
    -   Data dictionary, following the same structure as a data dictionary
        derived from a Signal object.

    RETURNS
    -------
    signal : Signal
    -   The data dictionary as a Signal object.
    """
    signal = Signal()

    new_dims = ["channels", "timepoints"]
    windowed = False
    if "epochs" in data["data_dimensions"]:
        new_dims = ["epochs", *new_dims]
    if "windows" in data["data_dimensions"]:
        new_dims = ["windows", *new_dims]
        windowed = True
    data_array = rearrange_axes(
        obj=data["data"],
        old_order=data["data_dimensions"],
        new_order=new_dims,
    )
    if not windowed:
        data_array = [data_array]

    mne_objects = []
    for data_window in data_array:
        if windowed:
            mne_dims = new_dims[1:]
        else:
            mne_dims = new_dims
        mne_object, _ = create_mne_data_object(
            data=data_window,
            data_dimensions=mne_dims,
            ch_names=data["ch_names"],
            ch_types=data["ch_types"],
            ch_coords=data["ch_coords"],
            sfreq=data["samp_freq"],
            subject_info=data["subject_info"],
        )
        mne_objects.append(mne_object)

    extra_info = create_extra_info(data=data)

    if not windowed:
        mne_objects = mne_objects[0]
    signal.data_from_objects(
        data=mne_objects,
        processing_steps=data["processing_steps"],
        extra_info=extra_info,
    )

    return signal
