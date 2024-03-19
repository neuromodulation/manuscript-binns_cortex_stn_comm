"""Analyses results."""

import os

import numpy as np

from coh_handle_files import generate_fpath_from_analysed, load_file
from coh_process_results import load_results_of_types


def analyse(
    folderpath_processing: str,
    folderpath_analysis: str,
    analysis: str,
    to_analyse: str,
    to_analyse_ftype: str,
    save: bool,
) -> None:
    """Analyses results.

    PARAMETERS
    ----------
    folderpath_processing : str
    -   Folderpath to the location where the results of the data processing are
        located.

    folderpath_analysis : str
    -   Folderpath to the location where the analysis settings are located and
        where the output of the analysis should be saved, if applicable.

    analysis : str
    -   Name of the analysis being performed, used for loading the analysis
        settings and, optionally, saving the output of the analysis.

    to_analyse : str
    -   Name of the file containing information on the set of results to
        analyse.

    to_analyse_ftype : str
    -   The filetype of the results, with the leading period, e.g. JSON would be
        specified as ".json".

    save : bool
    -   Whether or not to save the output of the analysis.
    """
    ## Loads the analysis settings
    analysis_settings = load_file(
        os.path.join(folderpath_analysis, "Settings", f"{analysis}.json")
    )
    to_analyse = load_file(
        os.path.join(
            folderpath_analysis,
            "Settings",
            "Data File Presets",
            f"{to_analyse}.json",
        )
    )

    results = load_results_of_types(
        folderpath_processing=os.path.join(folderpath_processing, "Data"),
        to_analyse=to_analyse,
        result_types=analysis_settings["result_types"],
        extract_from_dicts=analysis_settings["extract_from_dicts"],
        identical_keys=analysis_settings["identical_keys"],
        discard_keys=analysis_settings["discard_keys"],
        result_ftype=to_analyse_ftype,
    )

    accepted_methods = [
        "average_within_nodes",
        "average_over_nodes",
        "subtract",
        "log10",
        "absolute",
        "find_value",
        "find_index_of_value",
        "isolate_bands",
        "percentile",
        "interpolate",
        "gaussianise",
        "zscore_within_nodes",
        "zscore_over_nodes",
        "project_to_mesh",
        "track_fibres_within_radius",
        "track_closest_fibres",
    ]
    for step in analysis_settings["steps"]:
        if step["method"] not in accepted_methods:
            raise NotImplementedError(
                f"The method {step['method']} is not recognised. Accepted "
                f"methods are {accepted_methods}."
            )
        if step["method"] == "average_over_nodes":
            results.average_over_nodes(
                over_key=step["over_key"],
                data_keys=step["data_keys"],
                group_keys=step["group_keys"],
                eligible_entries=step["eligible_entries"],
                identical_keys=step["identical_keys"],
                ignore_nan=step["ignore_nan"],
                var_measures=step["var_measures"],
            )
        elif step["method"] == "average_within_nodes":
            results.average_within_nodes(
                data_keys=step["data_keys"],
                average_dimension=step["average_dimension"],
                eligible_entries=step["eligible_entries"],
                ignore_nan=step["ignore_nan"],
                var_measures=step["var_measures"],
            )
        elif step["method"] == "subtract":
            results.subtract(
                over_key=step["over_key"],
                data_keys=step["data_keys"],
                group_keys=step["group_keys"],
                eligible_entries=step["eligible_entries"],
                identical_keys=step["identical_keys"],
            )
        elif step["method"] == "log10":
            results.log10(
                data_keys=step["data_keys"],
                eligible_entries=step["eligible_entries"],
            )
        elif step["method"] == "absolute":
            results.absolute(
                data_keys=step["data_keys"],
                eligible_entries=step["eligible_entries"],
            )
        elif step["method"] == "find_value":
            results.find_value(
                value_method=step["value_method"],
                data_keys=step["data_keys"],
                find_in_dimension=step["find_in_dimension"],
                eligible_entries=step["eligible_entries"],
            )
        elif step["method"] == "find_index_of_value":
            results.find_index_of_value(
                value_method=step["value_method"],
                data_keys=step["data_keys"],
                find_in_dimension=step["find_in_dimension"],
                eligible_entries=step["eligible_entries"],
            )
        elif step["method"] == "isolate_bands":
            results.isolate_bands(
                data_keys=step["data_keys"],
                isolate_dimension=step["isolate_dimension"],
                bands=step["bands"],
                eligible_entries=step["eligible_entries"],
            )
        elif step["method"] == "percentile":
            results.percentile(
                over_key=step["over_key"],
                data_keys=step["data_keys"],
                group_keys=step["group_keys"],
                percentile_interval=step["percentile_interval"],
                eligible_entries=step["eligible_entries"],
                identical_keys=step["identical_keys"],
                ignore_nan=step["ignore_nan"],
            )
        elif step["method"] == "zscore_within_nodes":
            results.zscore_within_nodes(
                data_keys=step["data_keys"],
                zscore_dimension=step["zscore_dimension"],
                eligible_entries=step["eligible_entries"],
                ignore_nan=step["ignore_nan"],
            )
        elif step["method"] == "zscore_over_nodes":
            results.zscore_over_nodes(
                data_keys=step["data_keys"],
                group_keys=step["group_keys"],
                eligible_entries=step["eligible_entries"],
                identical_keys=step["identical_keys"],
                ignore_nan=step["ignore_nan"],
            )
        elif step["method"] == "interpolate":
            if isinstance(step["interpolate_to"], np.ndarray):
                interpolation_coords = step["interpolate_to"].copy()
            else:
                supported_interpolations = [
                    "central_cortex_lh",
                    "cortex_bl",
                    "cortex_rh",
                    "cortex_lh",
                    "STN_bl",
                    "STN_rh",
                    "STN_lh",
                ]
                if step["interpolate_to"] not in supported_interpolations:
                    raise ValueError(
                        "The requested interpolation is not recognised. "
                        "Supported interpolations are "
                        f"{supported_interpolations}."
                    )
                interpolation_coords = np.load(
                    os.path.join(
                        "coherence",
                        "interpolation_coords",
                        f"{step['interpolate_to']}.npy",
                    )
                )
            results.interpolate(
                over_key=step["over_key"],
                data_keys=step["data_keys"],
                group_keys=step["group_keys"],
                coords_key=step["coords_key"],
                interpolation_coords=interpolation_coords,
                pin_to_hemisphere=step["pin_to_hemisphere"],
                interpolation_settings=step["interpolation_settings"],
                eligible_entries=step["eligible_entries"],
                identical_keys=step["identical_keys"],
            )
        elif step["method"] == "gaussianise":
            results.gaussianise(
                over_key=step["over_key"],
                data_keys=step["data_keys"],
                gaussianise_dimension=step["gaussianise_dimension"],
                group_keys=step["group_keys"],
                eligible_entries=step["eligible_entries"],
                identical_keys=step["identical_keys"],
            )
        elif step["method"] == "project_to_mesh":
            results.project_to_mesh(
                mesh=step["mesh"],
                coords_key=step["coords_key"],
                pin_to_hemisphere=step["pin_to_hemisphere"],
                eligible_entries=step["eligible_entries"],
            )
        elif step["method"] == "track_fibres_within_radius":
            results.track_fibres_within_radius(
                atlas=step["atlas"],
                seeds_key=step["seeds_key"],
                targets_key=step["targets_key"],
                seeds_coords_key=step["seeds_coords_key"],
                targets_coords_key=step["targets_coords_key"],
                seeds_types_key=step["seeds_types_key"],
                targets_types_key=step["targets_types_key"],
                sphere_radii=step["sphere_radii"],
                allow_bypassing_fibres=step["allow_bypassing_fibres"],
                pin_to_hemisphere=step["pin_to_hemisphere"],
                eligible_entries=step["eligible_entries"],
            )
        elif step["method"] == "track_closest_fibres":
            results.track_closest_fibres(
                atlas=step["atlas"],
                seeds_key=step["seeds_key"],
                seeds_coords_key=step["seeds_coords_key"],
                normalise_distance=step["normalise_distance"],
                pin_to_hemisphere=step["pin_to_hemisphere"],
                eligible_entries=step["eligible_entries"],
            )

    if save:
        results_fpath = generate_fpath_from_analysed(
            analysed=to_analyse,
            parent_folderpath=os.path.join(folderpath_analysis, "Results"),
            analysis=analysis,
            ftype=".pkl",
            fpath_format="personal",
        )
        results.save_results(fpath=results_fpath)
