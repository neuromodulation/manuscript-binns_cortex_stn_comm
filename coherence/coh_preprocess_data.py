"""Loads and preprocesses ECoG and LFP data stored in the MNE-BIDS format.

METHODS
-------
preprocessing
-   Loads an mne.io.Raw object and preprocesses it in preparation for analysis.

preprocessing_for_viewing
-   Loads an mne.io.Raw object and preprocesses it for inspecting data quality.
"""

import os
from warnings import warn
from coh_handle_files import (
    generate_analysiswise_fpath,
    generate_raw_fpath,
    generate_sessionwise_fpath,
    load_file,
)
from coh_settings import extract_metadata
from coh_signal import Signal


def preprocessing(
    folderpath_data: str,
    folderpath_preprocessing: str,
    dataset: str,
    analysis: str,
    settings: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
    save: bool = False,
) -> Signal:
    """Loads an mne.io.Raw object and preprocesses it in preparation for
    analysis.

    PARAMETERS
    ----------
    folderpath_data : str
    -   The folderpath to the location of the datasets.

    folderpath_preprocessing : str
    -   The folderpath to the location of the preprocessing settings and
        derivatives.

    dataset : str
    -   The name of the cohort's raw data and processing data folders.

    analysis : str
    -   The name of the analysis folder within "'folderpath_extras'/settings".

    settings : str
    -   The name of the type of settings that will be used.

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

    save : bool; default False
    -   Whether or not to save the preprocessed data

    RETURNS
    -------
    signal : Signal
    -   The preprocessed and epoched data.
    """
    ### Analysis setup
    ## Gets the relevant filepaths
    generic_analysis_folder = os.path.join(
        folderpath_preprocessing, "Settings", "Generic"
    )
    specific_analysis_folder = os.path.join(
        folderpath_preprocessing, "Settings", "Specific"
    )
    analysis_settings_fpath = generate_analysiswise_fpath(
        generic_analysis_folder, analysis, ".json"
    )
    data_settings_fpath = generate_sessionwise_fpath(
        specific_analysis_folder,
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        f"settings-{settings}",
        ".json",
    )
    raw_fpath = generate_raw_fpath(
        folderpath_data, dataset, subject, session, task, acquisition, run
    )
    annotations_fpath = generate_sessionwise_fpath(
        specific_analysis_folder,
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        "annotations",
        ".csv",
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=analysis_settings_fpath)
    data_settings = load_file(fpath=data_settings_fpath)

    ### Data Pre-processing
    signal = Signal()
    signal.raw_from_fpath(raw_fpath)
    signal.pick_channels(data_settings["ch_names"])
    if data_settings["ch_coords"] is not None:
        signal.set_coordinates(
            data_settings["ch_names"], data_settings["ch_coords"]
        )
    if data_settings["ch_regions"] is not None:
        signal.set_regions(
            data_settings["ch_names"], data_settings["ch_regions"]
        )
    if data_settings["ch_subregions"] is not None:
        signal.set_subregions(
            data_settings["ch_names"],
            data_settings["ch_subregions"],
        )
    if data_settings["ch_hemispheres"] is not None:
        signal.set_hemispheres(
            data_settings["ch_names"], data_settings["ch_hemispheres"]
        )

    for key, value in analysis_settings.items():
        if key == "load_annotations":
            if value:
                signal.load_annotations(annotations_fpath)
        elif key == "remove_bad_segments":
            if value:
                signal.remove_bad_segments()
        elif key == "combine_channels":
            if value:
                combine_settings = data_settings["combine_channels"]
                signal.combine_channels(
                    ch_names_old=combine_settings["ch_names_old"],
                    ch_names_new=combine_settings["ch_names_new"],
                    ch_types_new=combine_settings["ch_types_new"],
                    ch_coords_new=combine_settings["ch_coords_new"],
                    ch_regions_new=combine_settings["ch_regions_new"],
                    ch_subregions_new=combine_settings["ch_subregions_new"],
                )
        elif key == "rereference":
            if value:
                for reref_key in data_settings["rereferencing"].keys():
                    reref_settings = data_settings["rereferencing"][reref_key]
                    if reref_key == "pseudo" and value["pseudo"]:
                        reref_method = signal.rereference_pseudo
                    elif reref_key == "bipolar" and value["bipolar"]:
                        reref_method = signal.rereference_bipolar
                    elif (
                        reref_key == "common_average"
                        and value["common_average"]
                    ):
                        reref_method = signal.rereference_common_average
                    else:
                        raise NotImplementedError(
                            "Error when rereferencing data:\nThe following "
                            f"rereferencing method '{reref_key}' is not "
                            "implemented."
                        )
                    reref_method(
                        ch_names_old=reref_settings["ch_names_old"],
                        ch_names_new=reref_settings["ch_names_new"],
                        ch_types_new=reref_settings["ch_types_new"],
                        ch_reref_types=reref_settings["ch_reref_types"],
                        ch_coords_new=reref_settings["ch_coords_new"],
                        ch_regions_new=reref_settings["ch_regions_new"],
                        ch_subregions_new=reref_settings["ch_subregions_new"],
                        ch_hemispheres_new=reref_settings[
                            "ch_hemispheres_new"
                        ],
                        eligible_entries=value[reref_key]
                        if value[reref_key] is not True
                        else None,
                    )
        elif key == "drop_channels":
            if value:
                for criteria in value:
                    signal.drop_channels(
                        eligible_entries=criteria["eligible"],
                        conditions=criteria["conditions"],
                    )
        elif key == "reorder_channels":
            if value:
                signal.order_channels(data_settings["post_reref_organisation"])
        elif key == "line_noise_Hz":
            if value:
                signal.notch_filter(value)
        elif key == "bandpass":
            if value:
                for bandpass in value:
                    signal.bandpass_filter(
                        bandpass["freqs"][0],
                        bandpass["freqs"][1],
                        bandpass["picks"],
                    )
        elif key == "resample_Hz":
            if value:
                signal.resample(value)
        elif key == "explore_parrm":
            signal.parrm(
                stim_freq=value["stim_freq"],
                grouping=value["grouping"],
                eligible_entries=value["eligible_entries"],
                explore_params=True,
                n_jobs=value["n_jobs"],
            )
        elif key == "parrm":
            if data_settings["stim"] == "On" and value:
                parrm_settings = data_settings["parrm_settings"]
                signal.parrm(
                    stim_freq=parrm_settings["stim_freq"],
                    filter_half_width=parrm_settings["filter_half_width"],
                    omit_n_samples=parrm_settings["omit_n_samples"],
                    filter_direction=parrm_settings["filter_direction"],
                    period_half_width=parrm_settings["period_half_width"],
                    grouping=value["grouping"],
                    eligible_entries=value["eligible_entries"],
                    group_names=parrm_settings["group_names"],
                    explore_params=False,
                    n_jobs=value["n_jobs"],
                )
        elif key == "epoch":
            if value:
                signal.epoch(
                    length=value["length_s"], sd_outlier=value["sd_outlier"]
                )
        elif key == "bootstrap":
            if value:
                bootstrap_settings = analysis_settings["bootstrap"]
                signal.bootstrap(
                    n_bootstraps=bootstrap_settings["n_bootstraps"],
                    n_epochs_per_bootstrap=bootstrap_settings[
                        "n_epochs_per_bootstrap"
                    ],
                    random_seed=bootstrap_settings["random_seed"],
                )
        else:
            warn(f"The key {key} is not a recognised preprocessing step.")

    ## Adds metadata about the preprocessed data
    metadata = extract_metadata(settings=data_settings)
    signal.add_metadata(metadata)

    if save:
        preprocessed_data_folder = os.path.join(
            folderpath_preprocessing, "Data"
        )
        preprocessed_data_fpath = generate_sessionwise_fpath(
            preprocessed_data_folder,
            dataset,
            subject,
            session,
            task,
            acquisition,
            run,
            f"preprocessed-{settings}-{analysis}",
            ".pkl",
        )
        signal.save_as_dict(
            fpath=preprocessed_data_fpath, ask_before_overwrite=False
        )

    return signal


def preprocessing_for_viewing(
    folderpath_data: str,
    folderpath_preprocessing: str,
    dataset: str,
    analysis: str,
    settings: str,
    subject: str,
    session: str,
    task: str,
    acquisition: str,
    run: str,
) -> Signal:
    """Loads an mne.io.Raw object and preprocesses it for inspecting data
    quality.

    PARAMETERS
    ----------
    folderpath_data : str
    -   The folderpath to the location of the datasets.

    folderpath_preprocessing : str
    -   The folderpath to the location of the preprocessing settings and
        derivatives.

    dataset : str
    -   The name of the cohort's raw data and processing data folders.

    analysis : str
    -   The name of the analysis folder within "'folderpath_extras'/settings".

    settings : str
    -   The name of the type of settings that will be used.

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

    RETURNS
    -------
    signal : Signal
    -   The preprocessed and epoched data.
    """

    ### Analysis setup
    ## Gets the relevant filepaths
    generic_analysis_folder = os.path.join(
        folderpath_preprocessing, "Settings", "Generic"
    )
    specific_analysis_folder = os.path.join(
        folderpath_preprocessing, "Settings", "Specific"
    )
    analysis_settings_fpath = generate_analysiswise_fpath(
        generic_analysis_folder, analysis, ".json"
    )
    data_settings_fpath = generate_sessionwise_fpath(
        specific_analysis_folder,
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        f"settings-{settings}",
        ".json",
    )
    raw_fpath = generate_raw_fpath(
        folderpath_data, dataset, subject, session, task, acquisition, run
    )
    annotations_fpath = generate_sessionwise_fpath(
        specific_analysis_folder,
        dataset,
        subject,
        session,
        task,
        acquisition,
        run,
        "annotations",
        ".csv",
    )

    ## Loads the analysis settings
    analysis_settings = load_file(fpath=analysis_settings_fpath)
    data_settings = load_file(fpath=data_settings_fpath)

    ### Data Pre-processing
    signal = Signal()
    signal.raw_from_fpath(raw_fpath)
    if analysis_settings["load_annotations"]:
        signal.load_annotations(annotations_fpath)
    signal.pick_channels(data_settings["ch_names"])
    if data_settings["ch_coords"] is not None:
        signal.set_coordinates(
            data_settings["ch_names"], data_settings["ch_coords"]
        )
    signal.set_regions(data_settings["ch_names"], data_settings["ch_regions"])
    signal.set_subregions(
        data_settings["ch_names"], data_settings["ch_subregions"]
    )
    signal.set_hemispheres(
        data_settings["ch_names"], data_settings["ch_hemispheres"]
    )
    if "combine_channels" in data_settings.keys():
        combine_settings = data_settings["combine_channels"]
        signal.combine_channels(
            ch_names_old=combine_settings["ch_names_old"],
            ch_names_new=combine_settings["ch_names_new"],
            ch_types_new=combine_settings["ch_types_new"],
            ch_coords_new=combine_settings["ch_coords_new"],
            ch_regions_new=combine_settings["ch_regions_new"],
            ch_subregions_new=combine_settings["ch_subregions_new"],
        )
    if analysis_settings["rereference"]:
        for key in data_settings["rereferencing"].keys():
            reref_settings = data_settings["rereferencing"][key]
            if key == "pseudo":
                reref_method = signal.rereference_pseudo
            elif key == "bipolar":
                reref_method = signal.rereference_bipolar
            elif key == "common_average":
                reref_method = signal.rereference_common_average
            else:
                raise Exception(
                    "Error when rereferencing data:\nThe following "
                    f"rereferencing method '{key}' is not implemented."
                )
            reref_method(
                ch_names_old=reref_settings["ch_names_old"],
                ch_names_new=reref_settings["ch_names_new"],
                ch_types_new=reref_settings["ch_types_new"],
                ch_reref_types=reref_settings["ch_reref_types"],
                ch_coords_new=reref_settings["ch_coords_new"],
                ch_regions_new=reref_settings["ch_regions_new"],
                ch_subregions_new=reref_settings["ch_subregions_new"],
                ch_hemispheres_new=reref_settings["ch_hemispheres_new"],
            )
        signal.order_channels(data_settings["post_reref_organisation"])
    if analysis_settings["line_noise"] is not None:
        signal.notch_filter(analysis_settings["line_noise"])
    if analysis_settings["bandpass"] is not None:
        signal.bandpass_filter(
            analysis_settings["bandpass"][0], analysis_settings["bandpass"][1]
        )
    if analysis_settings["resample"] is not None:
        signal.resample(analysis_settings["resample"])
    if analysis_settings["pseudo_window_length"] is not None:
        signal.pseudo_window(analysis_settings["pseudo_window_length"])
    if analysis_settings["epoch_length"] is not None:
        signal.epoch(analysis_settings["epoch_length"])

    ## Adds metadata about the preprocessed data
    metadata = extract_metadata(settings=data_settings)
    signal.add_metadata(metadata)

    return signal
