"""An abstract class for implementing data processing methods.

CLASSES
-------
ProcMethod
-   Abstract class for implementing data processing methods.
"""

from copy import deepcopy
from typing import Union
import pandas as pd
from abc import ABC, abstractmethod
from coh_handle_entries import ordered_list_from_dict
from coh_saving import save_object
import coh_signal


class ProcMethod(ABC):
    """Abstract class for implementing data processing methods.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   A preprocessed Signal object whose data will be processed.

    verbose : bool; Optional, default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process (abstract)
    -   Processes the data.

    save_object (abstract)
    -   Saves the object as a .pkl file.

    save_results (abstract)
    -   Converts the results and additional information to a dictionary and
        saves them as a file.

    results_as_dict (abstract)
    -   Organises the results and additional information into a dictionary.
    """

    @abstractmethod
    def __init__(self, signal: coh_signal.Signal, verbose: bool) -> None:
        # Initialises aspects of the ProcMethod object that will be filled with
        # information as the data is processed.
        self.results = None
        self._results_dims = None
        self.processing_steps = None
        self.extra_info = None

        # Initialises inputs of the ProcMethod object.
        self.signal = deepcopy(signal)
        self.verbose = verbose

        self._n_windows = len(self.signal.data)

        # Initialises aspects of the ProcMethod object that indicate which
        # methods have been called (starting as 'False'), which can later be
        # updated.
        self._processed = False
        self._windows_averaged = False

    @abstractmethod
    def _sort_inputs(self) -> None:
        """Checks the inputs to the processing method object to ensure that they
        match the requirements for processing and assigns inputs."""
        self.processing_steps = deepcopy(self.signal.processing_steps)
        self.extra_info = deepcopy(self.signal.extra_info)

    def _features_to_df(self) -> pd.DataFrame:
        """Collates features of channels (e.g. names, types, regions, etc...)
        into a pandas DataFrame so that which channels belong to which groups
        can be easily checked.

        RETURNS
        -------
        pandas DataFrame
        -   DataFrame containing the features of each channel.
        """
        ch_names = self.signal.data[0].ch_names
        feature_dict = {
            "ch_names": ch_names,
            "ch_types": self.signal.data[0].get_channel_types(picks=ch_names),
        }
        for extra_info_key, extra_info_value in self.extra_info.items():
            if extra_info_key != "metadata":
                if extra_info_value is not None:
                    feature_dict[extra_info_key] = ordered_list_from_dict(
                        ch_names, extra_info_value
                    )

        return pd.DataFrame(feature_dict)

    @property
    def results_dims(self) -> list[str]:
        """Returns the dimensions of the results, corresponding to the results
        that will be returned with the 'get_results' method.

        RETURNS
        -------
        dims : list[str]
        -   Dimensions of the results.
        """
        if (
            self._windows_averaged
            or "windows" not in self.signal.data_dimensions
        ):
            dims = self._results_dims[1:]
        else:
            dims = self._results_dims

        return deepcopy(dims)

    def save_object(
        self,
        fpath: str,
        ask_before_overwrite: Union[bool, None] = None,
    ) -> None:
        """Saves the object as a .pkl file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved. The filetype extension
            (.pkl) can be included, otherwise it will be automatically added.

        ask_before_overwrite : bool
        -   Whether or not the user is asked to confirm to overwrite a
            pre-existing file if one exists.
        """
        if ask_before_overwrite is None:
            ask_before_overwrite = self.verbose

        save_object(
            to_save=self,
            fpath=fpath,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self.verbose,
        )

    @abstractmethod
    def save_results(self) -> None:
        """Converts the results and additional information to a dictionary and
        saves them as a file."""

    @abstractmethod
    def results_as_dict(self) -> None:
        """Organises the results and additional information into a
        dictionary."""

    def _get_optimal_dims(self) -> list[str]:
        """Finds the optimal order of dimensions for the results, following the
        order ["windows", "channels", "epochs", "frequencies", "timepoints"]
        based on which dimensions are present in the reuslts.

        RETURNS
        -------
        list[str]
        -   Optimal dimensions of the results.
        """
        possible_order = [
            "connections",
            "channels",
            "windows",
            "epochs",
            "frequencies",
            "timepoints",
            "segments",
        ]

        return [dim for dim in possible_order if dim in self.results_dims]
