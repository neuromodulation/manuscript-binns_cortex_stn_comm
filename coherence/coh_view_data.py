"""Plots data for annotating.

METHODS
-------
inspect_data
-   Plots power spectra of signals.

annotate_data
-   Plots non-epoched signals for annotating.
"""

from os.path import exists
from coh_handle_files import (
    generate_analysiswise_fpath,
    generate_sessionwise_fpath,
    load_file,
)
from coh_signal_viewer import SignalViewer
import coh_signal


def inspect_data(
    signal: coh_signal.Signal,
    folderpath_preprocessing: str,
    analysis: str,
) -> None:
    """Plots power spectra of signals.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The pre-processed data to plot.

    folderpath_preprocessing : str
    -   The folderpath to the location of the preprocessing folder.

    analysis : str
    -   The name of the analysis folder within "'folderpath_extras'/settings".
    """

    ### Analysis setup
    ## Gets the relevant filepaths
    analysis_settings = load_file(
        fpath=generate_analysiswise_fpath(
            f"{folderpath_preprocessing}\\Settings\\Generic", analysis, ".json"
        )
    )["power"]

    ### Data plotting
    ## Plots the power spectra
    signal_viewer = SignalViewer(signal=signal)
    signal_viewer.plot_power(
        mode=analysis_settings["mode"],
        fmin=analysis_settings["fmin"],
        fmax=analysis_settings["fmax"],
        mode_kwargs=analysis_settings["mode_kwargs"],
        pick_types=analysis_settings["pick_types"],
        include_bads=analysis_settings["include_bads"],
        n_jobs=analysis_settings["n_jobs"],
    )


def annotate_data(
    signal: coh_signal.Signal,
    folderpath_preprocessing: str,
    dataset: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    load_annotations: bool = True,
) -> None:
    """Plots non-epoched signals for annotating.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The pre-processed data to plot.

    folderpath_preprocessing : str
    -   The folderpath to the location of the preprocessing folder.

    dataset : str
    -   The name of the dataset folder found in 'folderpath_data'.

    subject : str
    -   The name of the subject whose data will be plotted.

    session : str
    -   The name of the session for which the data will be plotted.

    task : str
    -   The name of the task for which the data will be plotted.

    acquisition : str
    -   The name of the acquisition mode for which the data will be plotted.

    run : str
    -   The name of the run for which the data will be plotted.

    load_annotations : bool; default True
    -   Whether or not to load pre-existing annotations, if present, when
        viewing the signals.
    """

    ### Analysis setup
    ## Gets the relevant filepaths
    annotations_fpath = generate_sessionwise_fpath(
        f"{folderpath_preprocessing}\\Settings\\Specific",
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        "annotations",
        ".csv",
    )

    ### Data plotting
    ## Plots the data for annotating
    signal_viewer = SignalViewer(signal=signal)
    if load_annotations:
        if exists(annotations_fpath):
            signal_viewer.load_annotations(fpath=annotations_fpath)
        else:
            print(
                "No pre-existing annotations to load from the filepath:\n"
                f"{annotations_fpath}"
            )
    signal_viewer.plot_raw()
    signal_viewer.save_annotations(fpath=annotations_fpath)
