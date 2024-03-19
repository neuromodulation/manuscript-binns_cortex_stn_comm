"""Class for viewing raw signals, and adding annotations to these signals.

CLASSES
-------
SignalViewer
-   Allows the user to view non-epoched raw signals with annotations (with the
    option for adding and editing annotations), and view the power spectra of
    these signals.
"""

import datetime

from typing import Union
from matplotlib import pyplot as plt
import numpy as np
import coh_signal
import mne
from coh_exceptions import UnsupportedFileExtensionError
from coh_handle_files import (
    check_annots_empty,
    check_annots_orig_time,
    check_ftype_present,
    identify_ftype,
)
from coh_saving import check_before_overwrite


class SignalViewer:
    """Allows the user to view non-epoched raw signals with annotations (with
    the option for adding and editing annotations), and view the power spectra
    of these signals.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The raw or preprocessed (but not epoched) signals to add annotations to.

    verbose : bool; default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    plot_raw
    -   Plots the raw signals along with the loaded annotations, if applicable.

    plot_power
    -   Plots the power spectra of the signals.

    load_annotations
    -   Loads annotations from a csv file.

    save_annotations
    -   Saves the annotations as a csv file.
    """

    def __init__(
        self, signal: coh_signal.Signal, verbose: bool = True
    ) -> None:
        # Initialises inputs of the object.
        self.signal = signal
        self._verbose = verbose
        self._sort_inputs()

    def _sort_inputs(self) -> None:
        """Checks that the inputs to the object match the requirements for
        processing

        RAISES
        ------
        InputTypeError
        -   Raised if the data contained in the Signal object has been windowed
            or epoched.
        """
        if self.signal._windowed:
            raise TypeError(
                "Error when trying to instantiate the Annotations object:\n"
                "The data in the Signal object being used has been windowed. "
                "Only non-windowed data is supported."
            )

    def _sort_fpath(self, fpath: str) -> str:
        """Checks whether the provided filepath for loading or saving
        annotations.
        -   If a filetype is present, checks if it is a supported type (i.e.
            '.csv').
        -   If a filetype is not present, add a '.csv' filetype ending.

        PARAMETERS
        ----------
        fpath : str
        -   The filepath to check

        RETURNS
        -------
        fpath : str
        -   The checked filepath, with filetype added if necessary.

        RAISES
        ------
        UnsupportedFileExtensionError
        -   Raised if the 'fpath' contains a file extension that is not '.csv'.
        """
        if check_ftype_present(fpath):
            fpath_ftype = identify_ftype(fpath)
            supported_ftypes = ["csv"]
            if fpath_ftype != "csv":
                raise UnsupportedFileExtensionError(
                    "Error when trying to save the annotations:\nThe filetype "
                    f"{fpath_ftype} is not supported. Annotations can only be "
                    f"saved as filetypes: {supported_ftypes}"
                )
        else:
            fpath += ".csv"

        return fpath

    def load_annotations(self, fpath: str) -> None:
        """Loads pre-existing annotations for the signals from a csv file.

        PARAMETERS
        ----------
        fpath : str
        -   The filepath to load the annotations from.
        """
        fpath = self._sort_fpath(fpath=fpath)

        if check_annots_empty(fpath):
            print("There are no events to read from the annotations file.")
        else:
            if self.signal.data[0].info["meas_date"] is None:
                print(
                    "The measurement date of the recording is not specified; "
                    "setting this to the default value (1970-01-01-0-0)."
                )
                self.signal.data[0].set_meas_date(
                    datetime.datetime(
                        1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc
                    )
                )
            self.signal.data[0].set_annotations(mne.read_annotations(fpath))

        if self._verbose:
            print(
                f"Loading {len(self.signal.data[0].annotations)} annotations "
                f"from the filepath:\n'{fpath}'"
            )

    def _sort_custom_annotations(self) -> None:
        """Checks the annotations and converts any named 'END' into a 'BAD'
        annotation spanning from the start of the 'END' annotation to the end of
        the recording and any named 'START' into a 'BAD' annotation spanning
        from the start of the recording to the start of the 'START'
        annotation."""
        start_time = self.signal.data[0].times[0]
        end_time = self.signal.data[0].times[-1]
        time_interval = (
            self.signal.data[0].times[1] - self.signal.data[0].times[0]
        )

        custom_labels = ["START", "END"]
        custom_annotations = {}
        for i, label in enumerate(self.signal.data[0].annotations.description):
            if label == "START":
                onset = start_time
                duration = (
                    self.signal.data[0].annotations.onset[i]
                    - start_time
                    + time_interval
                )
                description = "BAD_recording_start"
                if self._verbose:
                    print(
                        f"'START' annotation converted to a {description} "
                        "annotation covering the first "
                        f"{np.round(duration, 2)} seconds of the recording.\n"
                    )
            elif label == "END":
                onset = self.signal.data[0].annotations.onset[i]
                duration = end_time - onset + time_interval
                description = "BAD_recording_end"
                if self._verbose:
                    print(
                        f"'END' annotation converted to a {description} "
                        f"annotation in the {np.round(duration, 2)} seconds "
                        "prior to the end of the recording.\n"
                    )
            if label in custom_labels:
                custom_annotations[i] = {
                    "onset": onset,
                    "duration": duration,
                    "description": description,
                }

        if custom_annotations != {}:
            self._add_custom_annotations(custom_annotations)

    def _add_custom_annotations(self, custom_annotations: dict) -> None:
        """Combines the standard MNE annotations with the custom annotations
        into the same Annotations object which then replaces the Signal object's
        annotations.

        PARAMETERS
        ----------
        custom_annotations : dict
        -   Dictionary with integer keys representing indices in the original
            Annotations object which corresponds to the custom annotation
            entries, whose values are dictionaries with 'onset', 'duration',
            and 'description' keys containing the appropriate values for the
            new, custom annotations.
        """
        onsets = []
        durations = []
        descriptions = []
        for i, annotation in enumerate(self.signal.data[0].annotations):
            if i in custom_annotations.keys():
                onsets.append(custom_annotations[i]["onset"])
                durations.append(custom_annotations[i]["duration"])
                descriptions.append(custom_annotations[i]["description"])
            else:
                onsets.append(annotation["onset"])
                durations.append(annotation["duration"])
                descriptions.append(annotation["description"])
        self.signal.data[0].set_annotations(
            mne.Annotations(onsets, durations, descriptions)
        )

    def plot_raw(self) -> None:
        """Plots the raw signals along with the loaded annotations, if
        applicable.

        Supports the addition of two special annotations: "START"; and "END".
        These annotations are converted to "BAD_recording_start" and
        "BAD_recording_end" annotations, respectively, spanning from the
        start of the recording to the beginning of the "START" annotation,
        and from the beginning of the "END" annotations to the end of the
        recording, respectively.
        """
        # If mne-qt-browser is installed
        self.signal.data[0].plot(scalings="auto", block=True)

        # If mne-qt-browser is not installed
        # self.signal.data[0].plot(scalings="auto", show=False)
        # plt.tight_layout()
        # plt.show(block=True)

    def plot_power(
        self,
        mode: str = "multitaper",
        fmin: float = 0.0,
        fmax: float = np.inf,
        mode_kwargs: dict | None = None,
        pick_types: Union[list[str], None] = None,
        include_bads: bool = True,
        n_jobs: int = 1,
    ) -> None:
        """Plots the power spectra of the signals.

        PARAMETERS
        ----------
        mode : str; default "multitaper"
        -   Mode to use to compute power.

        fmin : float; default 0.0
        -   Lower frequency of interest.

        fmax : float; default numpy inf
        -   Upper frequency of interest.

        mode_kwargs : dict; default None
        -   Kwargs to pass to the MNE power computation for the requested mode.

        pick_types : list of str | None; default None
        -   Types of channels to plot.

        include_bads : bool; default True
        -   Whether or not to plot channels marked as bad.

        n_jobs : int; default 1
        -   The number of jobs to run in parallel. If '-1', this is set to the
            number of CPU cores.
        """
        picks = self.signal.data[0].ch_names
        if not include_bads:
            picks = [
                name
                for name in picks
                if name not in self.signal.data[0].info["bads"]
            ]

        if pick_types:
            ch_types = self.signal.data[0].get_channel_types()
            picks = [
                name
                for ch_i, name in enumerate(picks)
                if ch_types[ch_i] in pick_types
            ]

        self.signal.data[0].compute_psd(
            method=mode,
            fmin=fmin,
            fmax=fmax,
            picks=picks,
            n_jobs=n_jobs,
            verbose=self._verbose,
            **mode_kwargs,
        ).plot(show=False, spatial_colors=False)
        plt.show(block=True)

    def save_annotations(
        self, fpath: str, ask_before_overwrite: bool = True
    ) -> None:
        """Saves the annotations to a csv file.

        PARAMETERS
        ----------
        fpath : str
        -   The filepath to save the annotations to.

        ask_before_overwrite : bool; default True
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.
        """
        self._sort_custom_annotations()

        fpath = self._sort_fpath(fpath=fpath)

        if ask_before_overwrite:
            write = check_before_overwrite(fpath)
        else:
            write = True

        if write:
            self.signal.data[0].annotations.save(fname=fpath, overwrite=True)

            if self._verbose:
                print(
                    f"Saving {len(self.signal.data[0].annotations)} annotation(s) "
                    f"to:\n'{fpath}'"
                )
