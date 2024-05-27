"""Functions for dealing with files.

METHODS
-------
generate_raw_fpath
-   Generates an mne_bids.BIDSPath object for loading an mne.io.Raw object.

generate_sessionwise_fpath
-   Generates a filepath for an object that corresponds to an individual
    recording session based on the MNE data-storage filepath structure.

generate_analysiswise_fpath
-   Generates a filepath for an object that corresponds to a particular
    analysis spanning multiple recordings sessions.

check_ftype_present
-   Checks whether a filetype is present in a filepath string based on the
    presence of a period ('.').

identify_ftype
-   Finds what file type of a file is based on the filename extension.
"""

import os
import json
import pickle
from typing import Any, Optional
import numpy as np
import pandas as pd
import mne
import mne_bids
from coh_exceptions import (
    MissingFileExtensionError,
    UnavailableProcessingError,
    UnidenticalEntryError,
)
from coh_handle_entries import check_non_repeated_vals_lists
from coh_handle_objects import nested_changes


def generate_raw_fpath(
    folderpath: str,
    dataset: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
) -> mne_bids.BIDSPath:
    """Generates an mne_bids.BIDSPath object for loading an mne.io.Raw object.

    PARAMETERS
    ----------
    folderpath : str
    -   The path of the folder where the datasets are located.

    dataset : str
    -   The name of the dataset within the folder given in 'folderpath'.

    subject : str
    -   The name of the subject for which the mne_bids.BIDSPath object should be
        generated.

    session : str
    -   The name of the session for which the mne_bids.BIDSPath object should be
        generated.

    task : str
    -   The name of the task for which the mne_bids.BIDSPath object should be
        generated.

    acquisition : str
    -   The name of the acquisition mode for which the mne_bids.BIDSPath object
        should be generated.

    run : str
    -   The name of the run for which the mne_bids.BIDSPath object should be
        generated.

    RETURNS
    -------
    mne_bids.BIDSPath
    -   An mne_bids.BIDSPath object for loading an mne.io.Raw object.
    """

    return mne_bids.BIDSPath(
        subject=subject,
        session=session,
        task=task,
        acquisition=acquisition,
        run=run,
        root=os.path.join(folderpath, dataset, "rawdata"),
    )


def generate_sessionwise_fpath(
    folderpath: str,
    dataset: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    group_type: str,
    filetype: str,
    fpath_format: str = "personal",
) -> str:
    """Generates a filepath for an object that corresponds to an individual
    recording session.

    PARAMETERS
    ----------
    folderpath : str
    -   The path of the folder where the datasets are located.

    dataset : str
    -   The name of the dataset folder within the folder given in 'folderpath'.

    subject : str
    -   The name of the subject for which the filepath should be generated.

    session : str
    -   The name of the session for which the filepath should be generated.

    task : str
    -   The name of the task for which the filepath should be generated.

    acquisition : str
    -   The name of the acquisition mode for which the filepath should be
        generated.

    run : str
    -   The name of the run for which the filepath should be generated.

    group_type : str
    -   The name of the group of files for which the filepath should be
        generate, e.g. 'annotations', 'settings'.

    filetype : str
    -   The file extension, prefixed with a period, e.g. '.json', '.csv'.

    fpath_format : str; default "personal"
    -   The format the produced filepath should take.
    -   If "personal", a custom filepath format is used: "{dataset}/{subject}/
        {session}/task-{task}_acq-{acquisition}_run-{run}_{group_type}
        {filetype}".
    -   If "BIDS", the BIDS format is used: "{dataset}/{subject}/{session}/
        sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}_
        {group_type}{filetype}".

    RETURNS
    -------
    str
    -   The filepath of the object.

    RAISES
    ------
    NotImplementedError
    -   Raised if 'fpath_format' is not supported.
    """
    supported_fpath_formats = ["personal", "BIDS"]
    if fpath_format not in supported_fpath_formats:
        raise NotImplementedError(
            f"The filepath format '{fpath_format}' is not supported. The "
            f"supported formats are {supported_fpath_formats}."
        )

    subfolders = os.path.join(dataset, f"sub-{subject}", f"ses-{session}")

    if fpath_format == "personal":
        filename = (
            f"task-{task}_acq-{acquisition}_run-{run}_{group_type}{filetype}"
        )
    else:
        filename = (
            f"sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_"
            f"run-{run}_{group_type}{filetype}"
        )

    return os.path.join(folderpath, subfolders, filename)


def generate_analysiswise_fpath(
    folderpath: str, analysis_name: str, filetype: str
) -> str:
    """Generates a filepath for an object that corresponds to a particular
    analysis spanning multiple recordings sessions.

    PARAMETERS
    ----------
    folderpath : str
    -   The path of the folder where the datasets are located.

    analysis_name : str
    -   The name of the analysis folder within the folder given in
        "'folderpath'/settings".

    filetype : str
    -   The file extension, prefixed with a period, e.g. '.json', '.csv'.

    RETURNS
    -------
    str
    -   The filepath of the object.
    """

    return os.path.join(folderpath, analysis_name + filetype)


def generate_fpath_from_analysed(
    analysed: list[dict[str]],
    parent_folderpath: str,
    analysis: str,
    ftype: str,
    fpath_format: str = "personal",
) -> str:
    """Generates a filepath based on information that has been analysed.

    PARAMETERS
    ----------
    analysed : list[dict[str]]
    -   List in which each element is a dictionary containing information about
        a piece of information that has been analysed.
    -   Each dictionary contains the following keys regarding what the
        information was derived from: 'cohort' for the cohort of the subject;
        'sub' for the subject's name; 'ses' for the session name; 'task' for the
        task name; 'acq' for the acquisition name; 'run' for the run name.
    -   If multiple types of a single attribute (e.g. 'cohort', 'sub', etc...)
        are present, then these values will be replaced with "multi" to indicate
        the information has been derived from multiple sources.

    parent_folderpath : str
    -   Parent folderpath which the filepath will be appended to.

    analysis : str
    -   Name of the analysis that will be included in the filename.

    ftype : str
    -   Filetype extension of the file, with the period, e.g. a JSON file would
        have an ftype of '.json'.

    fpath_format : str; default "personal"
    -   The format the produced filepath should take.
    -   If "personal", a custom filepath format is used: "{dataset}/{subject}/
        {session}/task-{task}_acq-{acquisition}_run-{run}_{group_type}
        {filetype}".
    -   If "BIDS", the BIDS format is used: "{dataset}/{subject}/{session}/
        sub-{subject}_ses-{session}_task-{task}_acq-{acquisition}_run-{run}_
        {group_type}{filetype}".

    RETURNS
    -------
    str
    -   Filepath based on the information that has been analysed.

    RAISES
    ------
    NotImplementedError
    -   Raised if 'fpath_format' is not supported.
    """
    supported_fpath_formats = ["personal", "BIDS"]
    if fpath_format not in supported_fpath_formats:
        raise NotImplementedError(
            f"The filepath format '{fpath_format}' is not supported. The "
            f"supported formats are {supported_fpath_formats}."
        )

    required_info = ["cohort", "sub", "ses", "task", "acq", "run"]
    info_keys = [list(data_info.keys()) for data_info in analysed]
    check_non_repeated_vals_lists(
        lists=[required_info, *info_keys], allow_non_repeated=False
    )

    first = True
    for data_info in analysed:
        if first:
            analysed_info = {key: [value] for key, value in data_info.items()}
            first = False
        else:
            for key, value in data_info.items():
                analysed_info[key].append(value)

    for key, values in analysed_info.items():
        unique_values = np.unique(values).tolist()
        if len(unique_values) > 1:
            value = "multi"
        else:
            value = unique_values[0]
        analysed_info[key] = value

    cohort = analysed_info["cohort"]
    sub = analysed_info["sub"]
    ses = analysed_info["ses"]
    task = analysed_info["task"]
    acq = analysed_info["acq"]
    run = analysed_info["run"]
    folderpath = os.path.join(
        parent_folderpath, cohort, f"sub-{sub}", f"ses-{ses}"
    )

    if fpath_format == "personal":
        filename = f"task-{task}_acq-{acq}_run-{run}_{analysis}{ftype}"
    else:
        filename = (
            f"sub-{sub}_ses-{ses}_task-{task}_acq-{acq}_run-{run}_{analysis}"
            f"{ftype}"
        )

    return os.path.join(folderpath, filename)


def check_ftype_present(fpath: str) -> bool:
    """Checks whether a filetype is present in a filepath string based on the
    presence of a period ('.').

    PARAMETERS
    ----------
    fpath : str
    -   The filepath, including the filename.

    RETURNS
    -------
    ftype_present : bool
    -   Whether or not a filetype is present.
    """

    if "." in fpath:
        ftype_present = True
    else:
        ftype_present = False

    return ftype_present


def identify_ftype(fpath: str) -> str:
    """Finds what file type of a file is based on the filename extension.

    PARAMETERS
    ----------
    fpath : str
    -   The filepath, including the filename and extension.

    RETURNS
    -------
    str
    -   The file type.

    RAISES
    ------
    MissingFileExtensionError
    -   Raised if 'fpath' is missing the filetype extension.
    """

    if not check_ftype_present(fpath):
        raise MissingFileExtensionError(
            "Error when determining the filetype:\nNo filetype can be found in "
            f"the filepath '{fpath}'.\nFilepaths should be in the format "
            "'filename.filetype'."
        )

    return fpath[fpath.rfind(".") + 1 :]


def extra_deserialise_json(contents: dict) -> dict:
    """Performs custom deserialisation on a dictionary loaded from a json file
    with changes not present in the default deserialisation used in the 'load'
    method of the 'json' package.
    -   Current extra changes include: converting "INFINITY" strings into
        infinity floats.

    PARAMETERS
    ----------
    contents : dict
    -   The contents of the dictionary loaded from a json file.

    RETURNS
    -------
    dict
    -   The contents of the dictionary with additional changes made.
    """

    deserialise = {"INFINITY": float("inf")}

    nested_changes(contents=contents, changes=deserialise)

    return contents


def load_from_json(fpath: str) -> dict:
    """Loads the contents of a json file as a dictionary.

    PARAMETERS
    ----------
    fpath : str
    -   Location of the file to load.

    RETURNS
    -------
    contents : dict
    -   The contents of the file in a dictionary.
    """

    with open(fpath, encoding="utf8") as file:
        contents = extra_deserialise_json(contents=json.load(file))

    return contents


def load_from_pkl(fpath: str) -> Any:
    """Loads the contents of a pkl file.

    PARAMETERS
    ----------
    fpath : str
    -   Location of the file to load.

    RETURNS
    -------
    Any
    -   The contents of the file.
    """

    with open(fpath, "rb") as file:
        return pickle.load(file)


def load_file(
    fpath: str, ftype: Optional[str] = None, verbose: bool = True
) -> Any:
    """Loads the contents of a json or pkl file.

    PARAMETERS
    ----------
    fpath : str
        -   Location where the data should be loaded from.

    ftype : str
    -   The filetype of the data that will be loaded, without the leading
        period. E.g. for loading the file from the json format, this would be
        "json", not ".json".

    verbose : bool; default True
    -   Whether or not to print a note of the loading process.
    """

    if check_ftype_present(fpath) and ftype is not None:
        fpath_ftype = identify_ftype(fpath)
        if fpath_ftype != ftype:
            raise UnidenticalEntryError(
                "Error when trying to save the results of the analysis:\n "
                f"The filetypes in the filepath ({fpath_ftype}) and in the "
                f"requested filetype ({ftype}) do not match."
            )
    elif check_ftype_present(fpath) and ftype is None:
        ftype = identify_ftype(fpath)
    elif not check_ftype_present(fpath) and ftype is not None:
        fpath += ftype
    else:
        raise MissingFileExtensionError(
            "Error when trying to save the results of the analysis:\nNo "
            "filetype is given in the filepath and no filetype has been "
            "specified."
        )

    if ftype == "json":
        contents = load_from_json(fpath=fpath)
    elif ftype == "pkl":
        contents = load_from_pkl(fpath=fpath)
    else:
        raise UnavailableProcessingError(
            f"Error when trying to load the file:\nThe {ftype} format for "
            "loading is not supported."
        )

    if verbose:
        print(f"Loading the contents of the filepath:\n'{fpath}'.")

    return contents


def check_annots_empty(fpath: str) -> bool:
    """Load annotations in an MNE format from a csv file and checks whether any
    annotations are present

    PARAMETERS
    ----------
    fpath : str
    -   The filepath to a csv file to load the annotations from.

    RETURNS
    -------
    empty : bool
    -   Whether or not the annotations in the file are empty.
    """

    annots = pd.read_csv(fpath, keep_default_na=False)
    if annots.empty:
        empty = True
    else:
        empty = False

    return empty


def check_annots_orig_time(annots: mne.Annotations) -> mne.Annotations:
    """Checks whether a meaningful origin time (i.e. not 1970-01-01) is present
    in the MNE Annotations object, setting it to 'None' if this is not the case.

    PARAMETERS
    ----------
    annots : MNE Annotations
    -   The annotations to check.

    RETURNS
    -------
    annots : MNE Annotations
    -   The annotations, with non-meaningful origin time corrected, if
    applicable.
    """

    orig_time = annots.orig_time
    if orig_time.day == 1 and orig_time.month == 1 and orig_time.year == 1970:
        annots = mne.Annotations(
            onset=annots.onset,
            duration=annots.duration,
            description=annots.description,
            orig_time=None,
        )

    return annots
