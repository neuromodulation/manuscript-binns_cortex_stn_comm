"""Methods for creating and handling objects."""

from datetime import datetime
from typing import Any, Union
import numpy as np
import mne
from numpy.typing import NDArray
from coh_handle_entries import create_lambda, rearrange_axes

extra_info_keys = [
    "ch_regions",
    "ch_subregions",
    "ch_hemispheres",
    "ch_reref_types",
    "ch_epoch_orders",
    "metadata",
]


class FillableObject:
    """Creates an empty object that can be filled with attributes.

    PARAMETERS
    ----------
    attrs : dict
    -   The attributes to fill the object with. Keys and values of the
        dictionary are added as the attributes of the object and their values,
        respectively.
    """

    def __init__(self, attrs: dict):
        for name, value in attrs.items():
            setattr(self, name, value)


def create_extra_info(data: dict) -> dict[dict]:
    """Create a dictionary for holding additional information used in Signal
    objects.

    PARAMETERS
    ----------
    data : dict
    -   Data dictionary in the same format as that derived from a Signal object
        to extract the extra information from.

    RETURNS
    -------
    extra_info : dict[dict]
    -   Additional information extracted from the data dictionary.
    """
    ch_dict_keys = [
        "ch_regions",
        "ch_subregions",
        "ch_hemispheres",
        "ch_reref_types",
        "ch_epoch_orders",
    ]
    extra_info = {}
    for key in extra_info_keys:
        if key in data.keys():
            if key in ch_dict_keys:
                extra_info[key] = {
                    name: data[key][name] for name in data["ch_names"]
                }
            else:
                extra_info[key] = data[key]

    return extra_info


def create_mne_data_object(
    data: Union[list, NDArray],
    data_dimensions: list[str],
    ch_names: list[str],
    ch_types: list[str],
    sfreq: Union[int, float],
    ch_coords: Union[list[list[Union[int, float]]], None] = None,
    annotations: mne.Annotations | None = None,
    meas_date: datetime | None = None,
    events: Union[np.ndarray, None] = None,
    tmin: float = 0.0,
    event_id: Union[dict, None] = None,
    subject_info: Union[dict, None] = None,
    verbose: bool = True,
) -> tuple[Union[mne.io.Raw, mne.Epochs], list[str]]:
    """Creates an MNE Raw or Epochs object, depending on the structure of the
    data.

    PARAMETERS
    ----------
    data : list | numpy array
    -   The data of the object.

    data_dimensions : list[str]
    -   Names of axes in 'data'.
    -   If "epochs" is in the dimensions, an MNE Epochs object is created,
        otherwise an MNE Raw object is created.
    -   MNE Epochs objects expect data to be in the dimensions ["epochs",
        "channels", "timepoints"], which the data axes are rearranged to if they
        do not match.
    -   MNE Raw objects expect data to be in the dimensions ["channels",
        "timepoints"], which the data axes are rearranged to if they do not
        match.

    ch_names : list[str]
    -   Names of the channels.

    ch_types : list[str]
    -   Types of the channels.

    sfreq : Union[int, float]
    -   Sampling frequency of the data.

    ch_coords : list[list[int | float]] | None; default None
    -   Coordinates of the channels.

    annotations : MNE Annotations | None (default None)
        Annotations to add to the Raw object.

    meas_date : datetime | None (default None)
        Measurement date of the date to add to the Raw object.

    events : numpy ndarray | None; default None
    -   Events to add to the Epochs object.

    tmin : float; default 0.0
    -   Time of the first sample before an event in the epochs to add to the
        Epochs object.

    event_id : dict | None; default None
    -   Event IDs to add to the Epochs object.

    subject_info : dict | None; default None
    -   Information about the subject from which the data was collected.

    verbose : bool; default True
    -   Verbosity setting of the generated MNE object.

    RETURNS
    -------
    data_object : MNE Raw | MNE Epochs
    -   The generated MNE object containing the data.

    new_dimensions : list[str]
    -   Names of the new axes in 'data'.
    -   ["epochs", "channels", "timepoints"] if 'data_object' is an MNE Epochs
        object.
    -   ["channels", "timepoints"] if 'data_object' is an MNE Raw object.
    """

    data_info = create_mne_data_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq,
        subject_info=subject_info,
        verbose=verbose,
    )

    if "epochs" in data_dimensions:
        new_dimensions = ["epochs", "channels", "timepoints"]
        data = rearrange_axes(
            obj=data,
            old_order=data_dimensions,
            new_order=new_dimensions,
        )
        data_object = mne.EpochsArray(
            data=data,
            info=data_info,
            events=events,
            tmin=tmin,
            event_id=event_id,
            verbose=verbose,
        )
    else:
        new_dimensions = ["channels", "timepoints"]
        data = rearrange_axes(
            obj=data,
            old_order=data_dimensions,
            new_order=new_dimensions,
        )
        data_object = mne.io.RawArray(
            data=data, info=data_info, verbose=verbose
        )
        data_object.set_meas_date(meas_date)
        if annotations is not None:
            data_object.set_annotations(annotations)

    if ch_coords is not None:
        data_object._set_channel_positions(pos=ch_coords, names=ch_names)

    return data_object, new_dimensions


def create_mne_data_info(
    ch_names: list[str],
    ch_types: list[str],
    sfreq: Union[int, float],
    subject_info: Union[dict, None] = None,
    verbose: bool = True,
) -> mne.Info:
    """Create an MNE Info object.

    PARAMETERS
    ----------
    ch_names : list[str]
    -   Names of the channels.

    ch_types : list[str]
    -   Types of the channels.

    sfreq : Union[int, float]
    -   Sampling frequency of the data.

    subject_info : dict | None; default None
    -   Information about the subject from which the data was collected.

    verbose : bool; default True
    -   Verbosity setting of the generated MNE object.

    RETURNS
    -------
    data_info : MNE Info
    -   Information about the data.
    """

    data_info = mne.create_info(
        ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, verbose=verbose
    )
    if subject_info is not None:
        data_info["subject_info"] = subject_info
    return data_info


def nested_changes_list(contents: list, changes: dict) -> None:
    """Makes changes to the specified values occuring within nested dictionaries
    of lists of a parent list.

    PARAMETERS
    ----------
    contents : list
    -   The list containing nested dictionaries and lists whose values should be
        changed.

    changes : dict
    -   Dictionary specifying the changes to make, with the keys being the
        values that should be changed, and the values being what the values
        should be changed to.
    """

    for value in contents:
        if isinstance(value, list):
            nested_changes_list(contents=value, changes=changes)
        elif isinstance(value, dict):
            nested_changes_dict(contents=value, changes=changes)
        else:
            if value in changes.keys():
                value = changes[value]


def nested_changes_dict(contents: dict, changes: dict) -> None:
    """Makes changes to the specified values occuring within nested
    dictionaries or lists of a parent dictionary.

    PARAMETERS
    ----------
    contents : dict
    -   The dictionary containing nested dictionaries and lists whose values
        should be changed.

    changes : dict
    -   Dictionary specifying the changes to make, with the keys being the
        values that should be changed, and the values being what the values
        should be changed to.
    """

    for key, value in contents.items():
        if isinstance(value, list):
            nested_changes_list(contents=value, changes=changes)
        elif isinstance(value, dict):
            nested_changes_dict(contents=value, changes=changes)
        else:
            if value in changes.keys():
                contents[key] = changes[value]


def nested_changes(contents: Union[dict, list], changes: dict) -> None:
    """Makes changes to the specified values occuring within nested
    dictionaries or lists of a parent dictionary or list.

    PARAMETERS
    ----------
    contents : dict | list
    -   The dictionary or list containing nested dictionaries and lists whose
        values should be changed.

    changes : dict
    -   Dictionary specifying the changes to make, with the keys being the
        values that should be changed, and the values being what the values
        should be changed to.
    """

    if isinstance(contents, dict):
        nested_changes_dict(contents=contents, changes=changes)
    elif isinstance(contents, list):
        nested_changes_list(contents=contents, changes=changes)
    else:
        raise TypeError(
            "Error when changing nested elements of an object:\nProcessing "
            f"objects of type '{type(contents)}' is not supported. Only 'list' "
            "and 'dict' objects can be processed."
        )


def numpy_to_python(obj: Union[dict, list, np.generic]) -> Any:
    """Iterates through all entries of an object and converts any numpy elements
    into their base Python object types, e.g. float32 into float, ndarray into
    list, etc...

    PARAMETERS
    ----------
    obj : dict | list | numpy generic
    -   The object whose entries should be iterated through and, if numpy
        objects, converted to their equivalent base Python types.

    RETURNS
    -------
    Any
    -   The object whose entries, if numpy objects, have been converted to their
        equivalent base Python types.
    """
    if isinstance(obj, dict):
        return numpy_to_python_dict(obj)
    elif isinstance(obj, list):
        return numpy_to_python_list(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        raise TypeError(
            "Error when changing nested elements of an object:\nProcessing "
            f"objects of type '{type(obj)}' is not supported. Only 'list' "
            ", 'dict', and 'numpy generic' objects can be processed."
        )


def numpy_to_python_dict(obj: dict) -> dict:
    """Iterates through all entries of a dictionary and converts any numpy
    elements into their base Python object types, e.g. float32 into float,
    ndarray into list, etc...

    PARAMETERS
    ----------
    obj : dict
    -   The dictionary whose entries should be iterated through and, if numpy
        objects, converted to their equivalent base Python types.

    RETURNS
    -------
    new_obj : dict
    -   The dictionary whose entries, if numpy objects, have been converted to
        their equivalent base Python types.
    """
    new_obj = {}
    for key, value in obj.items():
        if isinstance(value, list):
            new_obj[key] = numpy_to_python_list(value)
        elif isinstance(value, dict):
            new_obj[key] = numpy_to_python_dict(value)
        elif type(value).__module__ == np.__name__:
            new_obj[key] = getattr(value, "tolist", create_lambda(value))()
        else:
            new_obj[key] = value

    return new_obj


def numpy_to_python_list(obj: list) -> list:
    """Iterates through all entries of a list and converts any numpy elements
    into their base Python object types, e.g. float32 into float, ndarray into
    list, etc...

    PARAMETERS
    ----------
    obj : list
    -   The list whose entries should be iterated through and, if numpy objects,
        converted to their equivalent base Python types.

    RETURNS
    -------
    new_obj : list
    -   The list whose entries, if numpy objects, have been converted to their
        equivalent base Python types.
    """
    new_obj = []
    for value in obj:
        if isinstance(value, list):
            new_obj.append(numpy_to_python_list(value))
        elif isinstance(value, dict):
            new_obj.append(numpy_to_python_dict(value))
        elif type(value).__module__ == np.__name__:
            new_obj.append(getattr(value, "tolist", create_lambda(value))())
        else:
            new_obj.append(value)

    return new_obj
