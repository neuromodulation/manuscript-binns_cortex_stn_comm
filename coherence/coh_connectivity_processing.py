"""Generates connectivity results from preprocessed data."""

import os
import numpy as np
from coh_handle_files import (
    generate_analysiswise_fpath,
    generate_sessionwise_fpath,
    load_file,
)
from coh_connectivity import (
    ConnectivityCoherence,
    ConnectivityGranger,
    ConnectivityMultivariateCoh,
    ConnectivityTDE,
)
import coh_signal


def coherence_processing(
    signal: coh_signal.Signal,
    folderpath_processing: str,
    dataset: str,
    preprocessing: str,
    analysis: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    verbose: bool,
    save: bool,
) -> None:
    """Performs processing to generate coherence results.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The pre-processed data to analyse.

    folderpath_processing : str
    -   The folderpath to the location of the datasets' 'extras', e.g. the
        annotations, processing settings, etc...

    dataset : str
    -   The name of the dataset folder found in 'folderpath_data'.

    preprocessing : str
    -   The name of the preprocessing used.

    analysis : str
    -   The name of the analysis folder within "'folderpath_extras'/settings".

    subject : str
    -   The name of the subject whose data will be analysed.

    session : str
    -   The name of the session for which the data will be analysed.

    task : str
    -   The name of the task for which the data will be analysed.

    acquisition : str
    -   The name of the acquisition mode for which the data will be analysed.

    run : str
    -   The name of the run for which the data will be analysed.

    verbose : bool
        Whether or not to show information about the processing.

    save : bool
    -   Whether or not to save the results of the analysis.
    """
    ### Analysis setup
    ## Gets the relevant filepaths
    generic_settings_fpath = generate_analysiswise_fpath(
        os.path.join(folderpath_processing, "Settings", "Generic"),
        analysis,
        ".json",
    )
    ## Loads the analysis settings
    analysis_settings = load_file(fpath=generic_settings_fpath)

    ### Data processing
    ## Coherence analysis
    if analysis_settings["cwt_freq_range"] is not None:
        cwt_freqs = np.arange(
            analysis_settings["cwt_freq_range"][0],
            analysis_settings["cwt_freq_range"][1]
            + analysis_settings["cwt_freq_resolution"],
            analysis_settings["cwt_freq_resolution"],
        )
    coherence = ConnectivityCoherence(signal, verbose=verbose)
    coherence.process(
        con_methods=analysis_settings["con_methods"],
        power_method=analysis_settings["power_method"],
        seeds=analysis_settings["seeds"],
        targets=analysis_settings["targets"],
        fmin=analysis_settings["fmin"],
        fmax=analysis_settings["fmax"],
        fskip=analysis_settings["fskip"],
        faverage=analysis_settings["faverage"],
        tmin=analysis_settings["tmin"],
        tmax=analysis_settings["tmax"],
        mt_bandwidth=analysis_settings["mt_bandwidth"],
        mt_adaptive=analysis_settings["mt_adaptive"],
        mt_low_bias=analysis_settings["mt_low_bias"],
        cwt_freqs=cwt_freqs,
        cwt_n_cycles=analysis_settings["cwt_n_cycles"],
        average_windows=analysis_settings["average_windows"],
        average_timepoints=analysis_settings["average_timepoints"],
        absolute_connectivity=analysis_settings["absolute_connectivity"],
        block_size=analysis_settings["block_size"],
        n_jobs=analysis_settings["n_jobs"],
    )
    if save:
        coherence_fpath = generate_sessionwise_fpath(
            os.path.join(folderpath_processing, "Data"),
            dataset,
            subject,
            session,
            task,
            acquisition,
            run,
            f"connectivity-{preprocessing}-{analysis}",
            ".pkl",
        )
        coherence.save_results(coherence_fpath)


def tde_processing(
    signal: coh_signal.Signal,
    folderpath_processing: str,
    dataset: str,
    preprocessing: str,
    analysis: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    verbose: bool,
    save: bool,
) -> None:
    """Generate time delay estimation (TDE) values.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The pre-processed data to analyse.

    folderpath_processing : str
    -   The folderpath to the location of the datasets' 'extras', e.g. the
        annotations, processing settings, etc...

    dataset : str
    -   The name of the dataset folder found in 'folderpath_data'.

    preprocessing : str
    -   The name of the preprocessing used.

    analysis : str
    -   The name of the analysis folder within "'folderpath_extras'/settings".

    subject : str
    -   The name of the subject whose data will be analysed.

    session : str
    -   The name of the session for which the data will be analysed.

    task : str
    -   The name of the task for which the data will be analysed.

    acquisition : str
    -   The name of the acquisition mode for which the data will be analysed.

    run : str
    -   The name of the run for which the data will be analysed.

    verbose : bool
        Whether or not to show information about the processing.

    save : bool
    -   Whether or not to save the results of the analysis.
    """

    ### Analysis setup
    ## Gets the relevant filepaths
    generic_settings_fpath = generate_analysiswise_fpath(
        os.path.join(folderpath_processing, "Settings", "Generic"),
        analysis,
        ".json",
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=generic_settings_fpath)

    ### Data processing
    tde = ConnectivityTDE(signal, verbose=verbose)
    tde.process(
        seeds=analysis_settings["seeds"],
        targets=analysis_settings["targets"],
        freq_bands=analysis_settings["freq_bands"],
        method=tuple(analysis_settings["method"]),
        antisym=tuple(analysis_settings["antisym"]),
        window_func=analysis_settings["window_func"],
        conf_interval=analysis_settings["conf_interval"],
        average_windows=analysis_settings["average_windows"],
        n_jobs=analysis_settings["n_jobs"],
    )
    if save:
        tde_fpath = generate_sessionwise_fpath(
            os.path.join(folderpath_processing, "Data"),
            dataset,
            subject,
            session,
            task,
            acquisition,
            run,
            f"connectivity-{preprocessing}-{analysis}",
            "",
        )
        tde.save_results(tde_fpath, "pkl", ask_before_overwrite=False)


def mim_mic_processing(
    signal: coh_signal.Signal,
    folderpath_processing: str,
    dataset: str,
    preprocessing: str,
    analysis: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    verbose: bool,
    save: bool,
) -> None:
    """Performs processing to generate multivariate connectivity results for
    the multivariate interaction measure (MIM) and/or the maximised imaginary
    coherence (MIC).

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The pre-processed data to analyse.

    folderpath_processing : str
    -   The folderpath to the location of the datasets' 'extras', e.g. the
        annotations, processing settings, etc...

    dataset : str
    -   The name of the dataset folder found in 'folderpath_data'.

    preprocessing : str
    -   The name of the preprocessing used.

    analysis : str
    -   The name of the analysis folder within "'folderpath_extras'/settings".

    subject : str
    -   The name of the subject whose data will be analysed.

    session : str
    -   The name of the session for which the data will be analysed.

    task : str
    -   The name of the task for which the data will be analysed.

    acquisition : str
    -   The name of the acquisition mode for which the data will be analysed.

    run : str
    -   The name of the run for which the data will be analysed.

    verbose : bool
        Whether or not to show information about the processing.

    save : bool
    -   Whether or not to save the results of the analysis.
    """

    ### Analysis setup
    ## Gets the relevant filepaths
    generic_settings_fpath = generate_analysiswise_fpath(
        os.path.join(folderpath_processing, "Settings", "Generic"),
        analysis,
        ".json",
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=generic_settings_fpath)

    ### Data processing
    ## Multivariate connectivity analysis
    if analysis_settings["cwt_freq_range"] is not None:
        cwt_freqs = np.arange(
            analysis_settings["cwt_freq_range"][0],
            analysis_settings["cwt_freq_range"][1]
            + analysis_settings["cwt_freq_resolution"],
            analysis_settings["cwt_freq_resolution"],
        )

    multivariate = ConnectivityMultivariateCoh(signal, verbose=verbose)
    multivariate.process(
        power_method=analysis_settings["power_method"],
        seeds=analysis_settings["seeds"],
        targets=analysis_settings["targets"],
        fmin=analysis_settings["fmin"],
        fmax=analysis_settings["fmax"],
        fskip=analysis_settings["fskip"],
        faverage=analysis_settings["faverage"],
        tmin=analysis_settings["tmin"],
        tmax=analysis_settings["tmax"],
        mt_bandwidth=analysis_settings["mt_bandwidth"],
        mt_adaptive=analysis_settings["mt_adaptive"],
        mt_low_bias=analysis_settings["mt_low_bias"],
        cwt_freqs=cwt_freqs,
        cwt_n_cycles=analysis_settings["cwt_n_cycles"],
        n_components=analysis_settings["n_components"],
        average_windows=analysis_settings["average_windows"],
        average_timepoints=analysis_settings["average_timepoints"],
        block_size=analysis_settings["block_size"],
        n_jobs=analysis_settings["n_jobs"],
    )
    if save:
        multivariate_fpath = generate_sessionwise_fpath(
            os.path.join(folderpath_processing, "Data"),
            dataset,
            subject,
            session,
            task,
            acquisition,
            run,
            f"connectivity-{preprocessing}-{analysis}",
            "",
        )
        multivariate.save_results(
            multivariate_fpath, "pkl", ask_before_overwrite=False
        )


def granger_processing(
    signal: coh_signal.Signal,
    folderpath_processing: str,
    dataset: str,
    preprocessing: str,
    analysis: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    verbose: bool,
    save: bool,
) -> None:
    """Performs processing to generate multivariate spectral Granger causality
    results.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The pre-processed data to analyse.

    folderpath_processing : str
    -   The folderpath to the location of the datasets' 'extras', e.g. the
        annotations, processing settings, etc...

    dataset : str
    -   The name of the dataset folder found in 'folderpath_data'.

    preprocessing : str
    -   The name of the preprocessing used.

    analysis : str
    -   The name of the analysis folder within "'folderpath_extras'/settings".

    subject : str
    -   The name of the subject whose data will be analysed.

    session : str
    -   The name of the session for which the data will be analysed.

    task : str
    -   The name of the task for which the data will be analysed.

    acquisition : str
    -   The name of the acquisition mode for which the data will be analysed.

    run : str
    -   The name of the run for which the data will be analysed.

    verbose : bool
        Whether or not to show information about the processing.

    save : bool
    -   Whether or not to save the results of the analysis.
    """

    ### Analysis setup
    ## Gets the relevant filepaths
    generic_settings_fpath = generate_analysiswise_fpath(
        os.path.join(folderpath_processing, "Settings", "Generic"),
        analysis,
        ".json",
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=generic_settings_fpath)

    ### Data processing
    ## Coherence analysis
    if analysis_settings["cwt_freq_range"] is not None:
        cwt_freqs = np.arange(
            analysis_settings["cwt_freq_range"][0],
            analysis_settings["cwt_freq_range"][1]
            + analysis_settings["cwt_freq_resolution"],
            analysis_settings["cwt_freq_resolution"],
        )
    granger = ConnectivityGranger(signal, verbose=verbose)
    granger.process(
        power_method=analysis_settings["power_method"],
        seeds=analysis_settings["seeds"],
        targets=analysis_settings["targets"],
        fmin=analysis_settings["fmin"],
        fmax=analysis_settings["fmax"],
        fskip=analysis_settings["fskip"],
        faverage=analysis_settings["faverage"],
        tmin=analysis_settings["tmin"],
        tmax=analysis_settings["tmax"],
        mt_bandwidth=analysis_settings["mt_bandwidth"],
        mt_adaptive=analysis_settings["mt_adaptive"],
        mt_low_bias=analysis_settings["mt_low_bias"],
        cwt_freqs=cwt_freqs,
        cwt_n_cycles=analysis_settings["cwt_n_cycles"],
        n_components=analysis_settings["n_components"],
        n_lags=analysis_settings["n_lags"],
        average_windows=analysis_settings["average_windows"],
        average_timepoints=analysis_settings["average_timepoints"],
        block_size=analysis_settings["block_size"],
        n_jobs=analysis_settings["n_jobs"],
    )
    if save:
        granger_fpath = generate_sessionwise_fpath(
            os.path.join(folderpath_processing, "Data"),
            dataset,
            subject,
            session,
            task,
            acquisition,
            run,
            f"connectivity-{preprocessing}-{analysis}",
            ".pkl",
        )
        granger.save_results(granger_fpath, ask_before_overwrite=False)
