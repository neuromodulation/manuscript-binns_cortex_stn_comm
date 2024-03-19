"""Methods for loading files and handling the resulting objects."""

import os
from coh_handle_files import generate_sessionwise_fpath, load_file
from coh_signal import data_dict_to_signal, Signal


def load_preprocessed_dict(
    folderpath_preprocessing: str,
    dataset: str,
    preprocessing: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    ftype: str,
) -> Signal:
    """Loads preprocessed data saved as a dictionary and converts it to a Signal
    object.

    PARAMETERS
    ----------
    folderpath_preprocessing : str
    -   Path to the preprocessing folder where the data and settings are found.

    dataset : str
    -   Name of the dataset to analyse.

    preprocessing : str
    -   Name of the preprocessing type to analyse.

    subject : str
    -   Name of the subject to analyse.

    session : str
    -   Name of the session to analyse.

    task : str
    -   Name of the task to analyse.

    acquisition : str
    -   Name of the acquisition to analyse.

    run : str
    -   Name of the run to analyse.

    ftype : str
    -   Filetype of the file.

    RETURNS
    -------
    signal : Signal
    -   The preprocessed data converted from a dictionary to a Signal object.
    """

    fpath = generate_sessionwise_fpath(
        folderpath=os.path.join(folderpath_preprocessing, "Data"),
        dataset=dataset,
        subject=subject,
        session=session,
        task=task,
        acquisition=acquisition,
        run=run,
        group_type=preprocessing,
        filetype=ftype,
    )

    data_dict = load_file(fpath=fpath)
    signal = data_dict_to_signal(data=data_dict)

    return signal
