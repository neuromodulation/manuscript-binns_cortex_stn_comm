"""Plots results on graphs.

METHODS
-------
result_plotting
-   Plots results on graphs as line plots and/or box plots.
"""

from coh_handle_files import (
    generate_analysiswise_fpath,
    load_file,
)
from coh_plotting import BoxPlot, LinePlot
from coh_process_results import load_results_of_types


def result_plotting(
    folderpath_analysis: str,
    folderpath_plotting: str,
    plotting: str,
    to_plot: str,
    to_plot_ftype: str,
) -> None:
    """Plots results on graphs as line plots and/or box plots.

    PARAMETERS
    ----------
    folderpath_analysis : str
    -   Folderpath to the location where the analysed results are located.

    folderpath_plotting : str
    -   Folderpath to the location where the plotting settings are located and
        where the plots should be saved, if applicable.

    plotting : str
    -   Name of the plotting being performed, used for loading the plotting
        settings and, optionally, saving the plots.

    to_plot : str
    -   Name of the file containing information on the set of results to plot.

    to_plot_ftype : str
    -   The filetype of the results, with the leading period, e.g. JSON would be
        specified as ".json".

    to_plot : str
    -   The set of results to plot.
    """
    ### Plotting setup
    ## Gets the relevant filepaths and loads the plotting settings
    plotting_settings_fpath = generate_analysiswise_fpath(
        f"{folderpath_plotting}\\Settings", plotting, ".json"
    )
    plotting_settings = load_file(fpath=plotting_settings_fpath)
    to_plot_fpath = generate_analysiswise_fpath(
        f"{folderpath_plotting}\\Settings\\Data File Presets", to_plot, ".json"
    )
    to_analyse = load_file(fpath=to_plot_fpath)

    ## Loads the results to plot
    results = load_results_of_types(
        folderpath_processing=f"{folderpath_analysis}\\Results",
        to_analyse=to_analyse,
        result_types=plotting_settings["result_types"],
        extract_from_dicts=None,
        identical_keys=None,
        discard_keys=None,
        result_ftype=to_plot_ftype,
    ).results_as_dict()

    ### Plotting
    supported_plot_types = ["line_plot", "box_plot"]
    for plot_settings in plotting_settings["plotting"]:
        plot_type = plot_settings["type"]
        if plot_type not in supported_plot_types:
            raise ValueError(
                "Error when trying to plot the analysis results:\nThe plot "
                f"type '{plot['plot_type']}' is not supported. Only plots of "
                f"type(s) {supported_plot_types} are supported.\n"
            )
        if plot_type == "line_plot":
            plot = LinePlot(results=results)
            plot.plot(
                x_axis_var=plot_settings["x_axis_var"],
                y_axis_vars=plot_settings["y_axis_vars"],
                x_axis_limits=plot_settings["x_axis_limits"],
                x_axis_label=plot_settings["x_axis_label"],
                x_axis_tick_interval=plot_settings["x_axis_tick_interval"],
                y_axis_limits=plot_settings["y_axis_limits"],
                y_axis_labels=plot_settings["y_axis_labels"],
                y_axis_tick_intervals=plot_settings["y_axis_tick_intervals"],
                y_axis_cap_max=plot_settings["y_axis_cap_max"],
                y_axis_cap_min=plot_settings["y_axis_cap_min"],
                x_axis_scale=plot_settings["x_axis_scale"],
                y_axis_scales=plot_settings["y_axis_scales"],
                var_measure=plot_settings["var_measure"],
                y_axis_limits_grouping=plot_settings["y_axis_limits_grouping"],
                figure_grouping=plot_settings["figure_grouping"],
                subplot_grouping=plot_settings["subplot_grouping"],
                analysis_keys=plot_settings["analysis_keys"],
                legend_properties=plot_settings["legend_properties"],
                identical_keys=plot_settings["identical_keys"],
                eligible_values=plot_settings["eligible_values"],
                order_values=plot_settings["order_values"],
                figure_layout=plot_settings["figure_layout"],
                average_as_equal=plot_settings["average_as_equal"],
                save=plot_settings["save"],
                save_folderpath=f"{folderpath_plotting}\\Figures\\{plotting}",
                save_ftype=plot_settings["save_ftype"],
            )
        elif plot_type == "box_plot":
            plot = BoxPlot(results=results)
            plot.plot(
                x_axis_var=plot_settings["x_axis_var"],
                y_axis_vars=plot_settings["y_axis_vars"],
                x_axis_label=plot_settings["x_axis_label"],
                y_axis_limits=plot_settings["y_axis_limits"],
                y_axis_labels=plot_settings["y_axis_labels"],
                y_axis_scales=plot_settings["y_axis_scales"],
                y_axis_tick_intervals=plot_settings["y_axis_tick_intervals"],
                y_axis_cap_max=plot_settings["y_axis_cap_max"],
                y_axis_cap_min=plot_settings["y_axis_cap_min"],
                y_axis_limits_grouping=plot_settings["y_axis_limits_grouping"],
                figure_grouping=plot_settings["figure_grouping"],
                subplot_grouping=plot_settings["subplot_grouping"],
                analysis_keys=plot_settings["analysis_keys"],
                identical_keys=plot_settings["identical_keys"],
                eligible_values=plot_settings["eligible_values"],
                order_values=plot_settings["order_values"],
                figure_layout=plot_settings["figure_layout"],
                average_as_equal=plot_settings["average_as_equal"],
                save=plot_settings["save"],
                save_folderpath=f"{folderpath_plotting}\\Figures\\{plotting}",
                save_ftype=plot_settings["save_ftype"],
            )
