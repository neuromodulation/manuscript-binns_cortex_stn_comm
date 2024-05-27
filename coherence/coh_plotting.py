"""Classes for plotting results.

CLASSES
-------
Plotting : Abstract Base Class
-   Abstract class for plotting results.

LinePlot : subclass of Plotting
-   Class for plotting results on line plots.

BoxPlot : subclass of Plotting
-   Class for plotting results on box plots.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union
from matplotlib import pyplot as plt
import numpy as np
from coh_exceptions import EntryLengthError
from coh_handle_entries import (
    check_master_entries_in_sublists,
    check_non_repeated_vals_lists,
    check_vals_identical_list,
    combine_col_vals_df,
    get_eligible_idcs_lists,
    get_group_names_idcs,
    loop_index,
    reorder_rows_dataframe,
    sort_inputs_results,
    dict_to_df,
)
from coh_handle_plots import get_plot_colours, maximise_figure_windows
from coh_saving import check_before_overwrite


class Plotting(ABC):
    """Abstract class for plotting results.

    PARAMETERS
    ----------
    results : dict
    -   A dictionary containing results to process.
    -   The entries in the dictionary should be either lists, numpy arrays, or
        dictionaries.
    -   Entries which are dictionaries will have their values treated as being
        identical for all values in the 'results' dictionary, given they are
        extracted from these dictionaries into the results.
    -   Keys ending with "_dimensions" are treated as containing information
        about the dimensions of other attributes in the results, e.g.
        'X_dimensions' would specify the dimensions for attribute 'X'. The
        dimensions should be a list of strings containing the values "channels"
        and "frequencies" in the positions corresponding to the axis of these
        dimensions in 'X'. A single list should be given, i.e. 'X_dimensions'
        should hold for all entries of 'X'.If no dimensions, are given, the 0th
        axis is assumed to correspond to channels and the 1st axis to
        frequencies.
    -   E.g. if 'X' has shape [25, 10, 50, 300] with an 'X_dimensions' of
        ['epochs', 'channels', 'frequencies', 'timepoints'], the shape of 'X'
        would be rearranged to [10, 50, 25, 300], corresponding to the
        dimensions ["channels", "frequencies", "epochs", "timepoints"].

    extract_from_dicts : dict[list[str]] | None; default None
    -   The keys of dictionaries within 'results' to include in the processing.
    -   Keys which are extracted are treated as being identical for all values
        in the 'results' dictionary.

    identical_keys : list[str] | None; default None
    -   The keys in 'results' which are identical across channels and for which
        only one copy is present.
    -   If any dimension attributes are present, these should be included as an
        identical entry, as they will be added automatically.

    discard_keys : list[str] | None; default None
    -   The keys which should be discarded immediately without processing.

    verbose : bool; default True
    -   Whether or not to print updates about the plotting process.

    METHODS
    -------
    plot
    -   Abstract method for plotting the results.
    """

    def __init__(
        self,
        results: dict,
        extract_from_dicts: Union[dict[list[str]], None] = None,
        identical_keys: Union[list[str], None] = None,
        discard_keys: Union[list[str], None] = None,
        verbose: bool = True,
    ) -> None:
        # Initialises inputs of the object.
        results = sort_inputs_results(
            results=results,
            extract_from_dicts=extract_from_dicts,
            identical_keys=identical_keys,
            discard_keys=discard_keys,
            verbose=verbose,
        )
        self._results = dict_to_df(obj=results)
        self._verbose = verbose

        # Initialises input settings for plotting.
        self.x_axis_var = None
        self.y_axis_vars = None
        self.x_axis_label = None
        self.y_axis_limits = None
        self.y_axis_labels = None
        self.y_axis_tick_intervals = None
        self.y_axis_cap_max = None
        self.y_axis_cap_min = None
        self.x_axis_scale = None
        self.y_axis_scales = None
        self.y_axis_limits_grouping = None
        self.var_measure = None
        self.figure_grouping = None
        self.subplot_grouping = None
        self.analysis_keys = None
        self.legend_properties = None
        self._legend_location = None
        self._legend_font_properties = None
        self.identical_keys = None
        self.eligible_values = None
        self.order_values = None
        self.figure_layout = None
        self.average_as_equal = None
        self.save = None
        self.save_folderpath = None
        self.save_ftype = None
        self._special_values = None

        # Initialises aspects of the object that will be filled with information
        # as the results are plotted.
        self._plot_type = None
        self._eligible_idcs = None
        self._y_axis_limits_idcs = None
        self._plot_grouping = None

        # Initialises aspects of the object that indicate which methods have
        # been called (starting as 'False'), which can later be updated.
        self._plotted = False

    @abstractmethod
    def plot(self) -> None:
        """Abstract method for plotting the results."""

    def _reinitialise_aspects(self) -> None:
        """Reinitialises aspects of the object that will be filled with
        information as the results are plotted by setting the aspects to
        'None'."""
        self._eligible_idcs = None
        self._y_axis_limits_idcs = None
        self._plot_grouping = None

    def _sort_plot_inputs(self) -> None:
        """Sorts the plotting settings."""
        if self.legend_properties is not None:
            self._sort_legend_properties()
        self._sort_special_indexing_values()
        self._sort_saving_inputs()
        self._sort_values_order()
        self._discard_keys(keys=self._get_missing_keys())
        self._check_identical_keys()
        self._get_eligible_indices()
        self._sort_y_axis_limits_grouping()
        self._sort_y_axis_labels()
        self._sort_y_axis_tick_intervals()
        self._sort_y_axis_scales_inputs()

    def _sort_legend_properties(self) -> None:
        """Sorts the inputs for legend properties into a form accepted by
        matplotlib."""
        font_properties_params = [
            "family",
            "style",
            "variant",
            "weight",
            "stretch",
            "size",
            "fname",
            "math_fontfamily",
        ]
        for param in self.legend_properties.keys():
            if param not in font_properties_params and param != "loc":
                raise ValueError(
                    f"The parameter '{param}' is not recognised as a "
                    "matplotlib FontProperties class parameter. The accepted "
                    f"parameters are: {font_properties_params}"
                )

        self._legend_font_properties = {
            key: None for key in font_properties_params
        }
        for param, value in self.legend_properties.items():
            if param == "loc":
                self._legend_location = value
            else:
                self._legend_font_properties[param] = value

    def _sort_special_indexing_values(self) -> None:
        """Sorts special values to use when finding rows of results that belong
        to the same group.

        If 'average_as_equal' is 'True', converts any string beginning with
        "avg[" (indicating that data has been averaged across multipleresult
        types) in the columns into the string "avg_", followed by the name of
        the column, thereby allowing averaged results to be treated as belonging
        to the same type, regardless of the specific types of results they were
        averaged from.
        """
        if self.average_as_equal:
            self._special_values = {"avg[": "avg_"}
        else:
            self._special_values = None

    def _sort_saving_inputs(self) -> None:
        """Sorts the inputs associated with saving figures, making sure they are
        in the correct format.

        RAISES
        ------
        ValueError
        -   Raised if the figures will be saved, but no folderpath is given.
        -   Raised if the figures will be saved, but no filetype extension is
            given.
        """
        if self.save:
            if self.save_folderpath is None:
                raise ValueError(
                    "Error when trying to plot results:\nIt has been requested "
                    "for the figures to be saved, but no folderpath for saving "
                    "has been specified."
                )
            if self.save_ftype is None:
                raise ValueError(
                    "Error when trying to plot results:\nIt has been requested "
                    "for the figures to be saved, but no filetype extension "
                    "for saving has been specified."
                )

    def _sort_values_order(self) -> None:
        """Reorders the rows in the results based on the order of the values in
        the key specified in 'self.order_values', such that results belonging to
        certain values are plotted in a certain order."""
        if self.order_values is not None:
            self._results = reorder_rows_dataframe(
                dataframe=self._results,
                key=list(self.order_values.keys())[0],
                values_order=list(self.order_values.values())[0],
            )

    def _get_missing_keys(self) -> list[str]:
        """Finds which keys in the results are not accounted for in the plotting
        settings.

        RETURNS
        -------
        list[str]
        -   Names of keys in the results not accounted for by the plotting
            settings.
        """
        return [
            key
            for key in self._results.keys()
            if key not in self._get_present_keys()
        ]

    def _get_present_keys(self) -> list[str]:
        """Finds which keys in the results have been accounted for in the
        plotting settings.

        RETURNS
        -------
        present_keys : list[str]
        -   Names of the keys in the results accounted for by the plotting
            settings.
        """
        settings_inputs = [
            "x_axis_var",
            "y_axis_vars",
            "y_axis_limits_grouping",
            "figure_grouping",
            "subplot_grouping",
            "analysis_keys",
            "identical_keys",
        ]

        present_keys = ["n_from"]
        if self.eligible_values is not None:
            present_keys.extend(list(self.eligible_values.keys()))
        for setting in settings_inputs:
            key = getattr(self, setting)
            if key is not None:
                if isinstance(key, list):
                    present_keys.extend(key)
                else:
                    present_keys.append(key)

        add_keys = []
        for key in present_keys:
            if f"{key}_dimensions" in self._results.keys():
                add_keys.append(f"{key}_dimensions")
            if self.var_measure is not None:
                if f"{key}_{self.var_measure}" in self._results.keys():
                    add_keys.append(f"{key}_{self.var_measure}")
        present_keys.extend(add_keys)

        return present_keys

    def _discard_keys(self, keys: list[str]) -> None:
        """Drops keys from the results DataFrame and resets the DataFrame
        index."""
        self._results = self._results.drop(columns=keys)
        self._results = self._results.reset_index()
        if self._verbose:
            print(f"Discarding the following keys from the results: {keys}\n")

    def _check_identical_keys(self) -> None:
        """Checks that keys in the results marked as identical are identical."""
        for key in self.identical_keys:
            values = deepcopy(self._results[key])
            if self.average_as_equal:
                for i, val in enumerate(values):
                    if isinstance(val, str):
                        if val[:4] == "avg[":
                            values[i] = f"avg_{key}"
            is_identical, vals = check_vals_identical_list(to_check=values)
            if not is_identical:
                raise ValueError(
                    "Error when trying to plot the results:\nThe results key "
                    f"'{key}' is marked as an identical key, however its "
                    "values are not identical for all results:\n- Unique "
                    f"values: {vals}\n"
                )

    def _get_eligible_indices(self) -> None:
        """Finds which indices in the results contain values designated as
        eligible for plotting."""
        if self.eligible_values is not None:
            to_check = {
                key: self._results[key] for key in self.eligible_values.keys()
            }
            self._eligible_idcs = get_eligible_idcs_lists(
                to_check=to_check, eligible_vals=self.eligible_values
            )
        else:
            self._eligible_idcs = np.arange(len(self._results.index)).tolist()

    def _sort_y_axis_limits_grouping(self) -> None:
        """Finds the names and indices of groups that will share y-axis limits.

        RAISES
        ------
        ValueError
        -   Raised if an entry of the grouping factor for the y-axis limits is
            missing from the grouping factors for the figures or subplots.
        """
        self._y_axis_limits_idcs = {}
        if self.y_axis_limits_grouping is not None:
            all_present, absent_entries = check_master_entries_in_sublists(
                master_list=self.y_axis_limits_grouping,
                sublists=[self.figure_grouping, self.subplot_grouping],
                allow_duplicates=False,
            )
            if not all_present:
                raise ValueError(
                    "Error when trying to plot results:\nThe entry(ies) in the "
                    f"results {self.y_axis_limits_grouping} for creating "
                    "groups that will share the y-axis limits must also be "
                    "accounted for in the entry(ies) for creating groups that "
                    "will be plotted on the same figures/subplots, as plotting "
                    "results with multiple y-axes on the same plot is not yet "
                    "supported.\nThe following entries are unaccounted for: "
                    f"{absent_entries}\n"
                )
            grouping_entries = [
                entry
                for entry in self.y_axis_limits_grouping
                if entry != "Y_AXIS_VARS"
            ]
            if grouping_entries != []:
                self._y_axis_limits_idcs = get_group_names_idcs(
                    dataframe=self._results,
                    keys=grouping_entries,
                    eligible_idcs=self._eligible_idcs,
                    replacement_idcs=self._eligible_idcs,
                    special_vals=self._special_values,
                )

        if not self._y_axis_limits_idcs:
            self._y_axis_limits_idcs["ALL"] = deepcopy(self._eligible_idcs)

    def _sort_y_axis_limits(self) -> None:
        """Checks that the limits for the y-axis variables are in the correct
        format.

        RAISES
        ------
        KeyError
        -   Raised if the keys of the dictionary in the provided 'y_axis_limits'
            do not match the names of the groups in the
            automatically-generated y-axis limit group indices.
        -   Raised if the keys of the dictionary within each group dictionary do
            not contain the names (and hence limits) for each y-axis variable
            being plotted.
        """
        all_repeated = check_non_repeated_vals_lists(
            lists=[
                self.y_axis_limits.keys(),
                self._y_axis_limits_idcs.keys(),
            ],
            allow_non_repeated=True,
        )
        if not all_repeated:
            raise KeyError(
                "Error when trying to plot results:\nNames of the groups "
                "in the specified y-axis limits do not match those "
                "generated from the results:\n- Provided names: "
                f"{self.y_axis_limits.keys()}\n- Names should be: "
                f"{self._y_axis_limits_idcs.keys()}\n"
            )
        for group_name, var_lims in self.y_axis_limits.items():
            for var in self.y_axis_vars:
                if var not in var_lims.keys():
                    raise KeyError(
                        "Error when trying to plot results:\nMissing "
                        f"limits for the y-axis variable '{var}' in the "
                        f"group '{group_name}'.\n"
                    )

    def _sort_y_axis_tick_intervals(self) -> None:
        """Checks that the tick intervals for the y-axis variables are in the
        correct format.

        RAISES
        ------
        KeyError
        -   Raised if the keys of the dictionary in the provided
            'y_axis_tick_intervals' do not match the names of the groups in the
            automatically-generated y-axis group indices.
        -   Raised if the keys of the dictionary within each group dictionary do
            not contain the names (and hence tick intervals) for each y-axis
            variable being plotted.
        """
        if self.y_axis_tick_intervals:
            all_repeated = check_non_repeated_vals_lists(
                lists=[
                    self.y_axis_tick_intervals.keys(),
                    self._y_axis_limits_idcs.keys(),
                ],
                allow_non_repeated=True,
            )
            if not all_repeated:
                raise KeyError(
                    "Error when trying to plot results:\nNames of the groups "
                    "in the specified y-axis limits do not match those "
                    "generated from the results:\n- Provided names: "
                    f"{list(self.y_axis_tick_intervals.keys())}\n- Names "
                    f"should be: {list(self._y_axis_limits_idcs.keys())}\n"
                )
            for group_name, var_lims in self.y_axis_tick_intervals.items():
                for var in self.y_axis_vars:
                    if var not in var_lims.keys():
                        raise KeyError(
                            "Error when trying to plot results:\nMissing "
                            f"tick interval for the y-axis variable '{var}' in "
                            f"the group '{group_name}'.\n"
                        )
        else:
            self.y_axis_tick_intervals = {}
            for group in self._y_axis_limits_idcs.keys():
                self.y_axis_tick_intervals[group] = {}
                for var in self.y_axis_vars:
                    self.y_axis_tick_intervals[group][var] = None

    def _sort_y_axis_limits_inputs(self) -> tuple[bool, list[list[str]]]:
        """Sorts inputs for setting the y-axis limits.

        RETURNS
        -------
        share_across_vars : bool
        -   Whether or not the y-axis limits should be shared across all
            variables being plotted on the y-axes.

        extra_vars : list[list[str]]
        -   Additional values to combine with those of the variables being
            plotted on the y-axes (such as standard error or standard deviation
            values) when determining the y-axis limits.
        """
        if self.y_axis_cap_min is None:
            self.y_axis_cap_min = float("-inf")
        if self.y_axis_cap_max is None:
            self.y_axis_cap_max = float("inf")

        share_across_vars = True
        if self.y_axis_limits_grouping is not None:
            if "Y_AXIS_VARS" in self.y_axis_limits_grouping:
                share_across_vars = False

        if self.var_measure is not None:
            extra_vars = [
                f"{var}_{self.var_measure}" for var in self.y_axis_vars
            ]
        else:
            extra_vars = None

        return share_across_vars, extra_vars

    def _sort_y_axis_labels(self) -> None:
        """Checks that y-axis labels are in the appropriate format,
        automatically generating them based on the variables being plotted if
        the labels are not specified.

        RAISES
        ------
        EntryLengthError
        -   Raised if the lengths of the y-axis labels does not match the number
            of different variables being plotted on the y-axes. Only raised if
            the y-axis labels are not 'None'.
        -   Raised if multiple variables are being plotted on the same y-axis,
            but more than one y-axis label is given. Only raised if the y-axis
            labels are not 'None'.
        """
        if self.y_axis_labels is None:
            if (
                "Y_AXIS_VARS" in self.figure_grouping
                or "Y_AXIS_VARS" in self.subplot_grouping
            ):
                self.y_axis_labels = self.y_axis_vars
            else:
                self.y_axis_labels = [""]
                self.y_axis_labels.extend(
                    f"{var} / " for var in self.y_axis_vars
                )
                self.y_axis_labels = self.y_axis_labels[:-3]
        else:
            if (
                "Y_AXIS_VARS" in self.figure_grouping
                or "Y_AXIS_VARS" in self.subplot_grouping
            ):
                if len(self.y_axis_labels) != len(self.y_axis_vars):
                    raise EntryLengthError(
                        "Error when trying to plot results:\nThe number of "
                        "different variables being plotted separately on the "
                        f"y-axes ({len(self.y_axis_vars)}) and the number of "
                        f"y-axis labels ({len(self.y_axis_labels)}) do not "
                        "match."
                    )
            else:
                if len(self.y_axis_labels) != 1:
                    raise EntryLengthError(
                        "Error when trying to plot results:\nThe different "
                        "variables are being plotted together on the y-axes, "
                        "and so there can only be a single y-axis label, but "
                        f"there are {len(self.y_axis_labels)} labels."
                    )

    def _sort_y_axis_scales_inputs(self) -> None:
        """Checks that y-axis scales are in the appropriate format,
        automatically generating them based on the variables being plotted if
        the scales are not specified.

        RAISES
        ------
        EntryLengthError
        -   Raised if the lengths of the y-axis scales do not match the number
            of different variables being plotted on the y-axes. Only raised if
            the y-axis scales are not 'None'.
        -   Raised if multiple variables are being plotted on the same y-axis,
            but more than one y-axis scale is given. Only raised if the y-axis
            scales are not 'None'.
        """
        if self.y_axis_scales is None:
            self.y_axis_scales = ["linear" for var in self.y_axis_vars]
        else:
            if (
                "Y_AXIS_VARS" in self.figure_grouping
                or "Y_AXIS_VARS" in self.subplot_grouping
            ):
                if len(self.y_axis_scales) != len(self.y_axis_vars):
                    raise EntryLengthError(
                        "Error when trying to plot results:\nThe number of "
                        "different variables being plotted separately on the "
                        f"y-axes ({len(self.y_axis_vars)}) and the number of "
                        f"y-axis scales ({len(self.y_axis_labels)}) do not "
                        "match."
                    )
            else:
                if len(self.y_axis_scales) != 1:
                    raise EntryLengthError(
                        "Error when trying to plot results:\nThe different "
                        "variables are being plotted together on the y-axes, "
                        "and so there can only be a single y-axis scale, but "
                        f"there are {len(self.y_axis_labels)} scales."
                    )

    def _sort_plot_grouping(self) -> None:
        """Sorts the figure and subplot groups, finding the indices of the
        corresponding rows in the results."""
        self._plot_grouping = {}
        self._sort_figure_grouping()
        self._sort_subplot_grouping()

    def _sort_figure_grouping(self) -> None:
        """Sorts the groups for which indices of rows in the results should be
        plotted on the same set of figures."""
        if self.figure_grouping is not None:
            figure_grouping_entries = [
                entry
                for entry in self.figure_grouping
                if entry != "Y_AXIS_VARS"
            ]
        else:
            figure_grouping_entries = []

        if figure_grouping_entries != []:
            self._plot_grouping = get_group_names_idcs(
                dataframe=self._results,
                keys=figure_grouping_entries,
                eligible_idcs=self._eligible_idcs,
                replacement_idcs=self._eligible_idcs,
                special_vals=self._special_values,
            )
        else:
            self._plot_grouping["ALL"] = self._eligible_idcs

    def _sort_subplot_grouping(self) -> None:
        """Sorts the groups for which indices of rows in the results should be
        plotted on the same subplots on each set of figures."""
        if self.subplot_grouping is not None:
            subplot_grouping_entries = [
                entry
                for entry in self.subplot_grouping
                if entry != "Y_AXIS_VARS"
            ]
        else:
            subplot_grouping_entries = []

        for fig_group, idcs in self._plot_grouping.items():
            if subplot_grouping_entries != []:
                self._plot_grouping[fig_group] = get_group_names_idcs(
                    dataframe=self._results,
                    keys=subplot_grouping_entries,
                    eligible_idcs=self._plot_grouping[fig_group],
                    replacement_idcs=self._plot_grouping[fig_group],
                    special_vals=self._special_values,
                )
            else:
                self._plot_grouping[fig_group] = {"ALL": deepcopy(idcs)}

    def _plot_results(self) -> None:
        """Plots the results of the figure and subplot groups."""
        if self.figure_layout is None:
            self.figure_layout = [1, 1]

        for figure_group, _ in self._plot_grouping.items():
            if "Y_AXIS_VARS" in self.figure_grouping:
                for y_axis_var in self.y_axis_vars:
                    self._plot_figure(
                        figure_group_name=figure_group,
                        y_axis_vars=[y_axis_var],
                    )
            else:
                self._plot_figure(
                    figure_group_name=figure_group,
                    y_axis_vars=self.y_axis_vars,
                )

    def _plot_figure(
        self, figure_group_name: str, y_axis_vars: list[str]
    ) -> None:
        """Plots the results of the subplot groups belonging to a specified
        figure group.

        PARAMETERS
        ----------
        figure_group_name : str
        -   Name of the figure group whose results should be plotted.

        y_axis_vars : list[str]
        -   Names of the y-axis variables to plot on the same figure group.
        """
        (
            subplot_group_names,
            subplot_group_indices,
        ) = self._get_subplot_group_info(
            figure_group_name=figure_group_name, n_y_axis_vars=len(y_axis_vars)
        )

        n_subplot_groups = len(subplot_group_names)
        n_subplots_per_fig = self.figure_layout[0] * self.figure_layout[1]
        n_figs = int(np.ceil(n_subplot_groups / n_subplots_per_fig))
        n_rows = self.figure_layout[0]
        n_cols = self.figure_layout[1]

        if self._verbose:
            print(
                f"Plotting {n_subplot_groups} subplot group(s) in a {n_rows} "
                f"by {n_cols} pattern across {n_figs} figure(s).\n- Figure "
                f"group: {figure_group_name}\n- Subplot groups: "
                f"{subplot_group_names}\n"
            )

        still_to_plot = True
        for fig_i in range(n_figs):
            fig, axes = self._establish_figure(
                n_rows=n_rows, n_cols=n_cols, title=figure_group_name
            )
            subplot_group_i = 0
            for row_i in range(n_rows):
                for col_i in range(n_cols):
                    if still_to_plot:
                        subplot_group_name = subplot_group_names[
                            subplot_group_i
                        ]
                        subplot_group_idcs = subplot_group_indices[
                            subplot_group_i
                        ]
                        self._plot_subplot(
                            axes[row_i, col_i],
                            subplot_group_name,
                            subplot_group_idcs,
                            y_axis_vars,
                        )
                        subplot_group_i += 1
                        if subplot_group_i == n_subplot_groups:
                            still_to_plot = False
                            extra_subplots = (
                                n_subplots_per_fig * n_figs - n_subplot_groups
                            )
                    else:
                        if extra_subplots > 0:
                            fig.delaxes(axes[row_i, col_i])
            try:
                maximise_figure_windows()
            except NotImplementedError:
                print(
                    "The figure could not be made fullscreen automatically "
                    "with the current combination of operating system and "
                    "plotting backend."
                )
            plt.tight_layout()
            plt.show()
            if self.save:
                self._save_figure(
                    figure=fig,
                    figure_group=figure_group_name,
                    figure_n=fig_i + 1,
                    n_figures=n_figs,
                    y_axis_vars=y_axis_vars,
                )

    def _get_subplot_group_info(
        self, figure_group_name: str, n_y_axis_vars: int
    ) -> tuple[list[str], list[list[int]]]:
        """Gets information about subplot groups for a given figure group.

        PARAMETERS
        ----------
        figure_group_name : str
        -   Name of the figure group.

        n_y_axis_vars : int
        -   Number of y-axis variables being plotted.
        -   If multiple y-axis variables are not being plotted on the same
            subplots, subplot group names and indices are multiplied by how many
            y-axis variables are being plotted so that the correct number of
            subplots allowing each y-axis variable to be plotted on a separate
            subplot are created.

        RETURNS
        -------
        subplot_group_names : list[str]
        -   Names of the subplot groups to plot.

        subplot_group_indices : list[str]
        -   Indices of the rows of results in the subplot groups to plot.
        """
        if "Y_AXIS_VARS" in self.subplot_grouping:
            subplot_group_names = []
            subplot_group_indices = []
            subplot_group_names.extend(
                [
                    [name] * n_y_axis_vars
                    for name in self._plot_grouping[figure_group_name].keys()
                ]
            )
            subplot_group_indices.extend(
                [
                    [idcs] * n_y_axis_vars
                    for idcs in self._plot_grouping[figure_group_name].values()
                ]
            )
        else:
            subplot_group_names = list(
                self._plot_grouping[figure_group_name].keys()
            )
            subplot_group_indices = list(
                self._plot_grouping[figure_group_name].values()
            )

        return subplot_group_names, subplot_group_indices

    def _establish_figure(
        self, n_rows: int, n_cols: int, title: str
    ) -> tuple[plt.figure, plt.Axes]:
        """Creates a figure with the desired number of subplot rows and columns,
        with a specified title.

        PARAMETERS
        ----------
        n_rows : int
        -   Number of subplot rows in the figure.

        n_cols : int
        -   Number of subplot columns in the figure.

        title : str
        -   Title of the figure.

        RETURNS
        -------
        fig : matplotlib pyplot figure
        -   The created figure.

        axes : matplotlib pyplot Axes
        -   The Axes object of subplots on the figure.
        """
        fig, axes = plt.subplots(n_rows, n_cols)
        plt.tight_layout()
        fig.suptitle(self._get_figure_title(group_name=title), fontsize=10)
        axes = self._sort_figure_axes(axes=axes, n_rows=n_rows, n_cols=n_cols)

        return fig, axes

    def _get_figure_title(self, group_name: str) -> str:
        """Generates a title for a figure based on the identical entries in the
        results (if any), and the group of results being plotted on the figure.

        PARAMETERS
        ----------
        group_name : str
        -   Name of the group of results being plotted on the figure.

        RETURNS
        -------
        str
        -   Title of the figure in two lines. Line one: identical entry names
            and values. Line two: group name.
        """
        if self.identical_keys is not None:
            identical_entries_title = combine_col_vals_df(
                dataframe=self._results,
                keys=self.identical_keys,
                idcs=[0],
                special_vals=self._special_values,
            )[0]
        else:
            identical_entries_title = ""

        return f"{identical_entries_title}\n{group_name}"

    def _sort_figure_axes(
        self, axes: plt.Axes, n_rows: int, n_cols: int
    ) -> plt.Axes:
        """Sorts the pyplot Axes object by adding an additional row and/or
        column if there is only one row and/or column in the object, useful for
        later indexing.

        PARAMETERS
        ----------
        axes : matplotlib pyplot Axes
        -   The axes to sort.

        n_rows : int
        -   Number of rows in the axes.

        n_cols : int
        -   Number of columns in the axes.

        RETURNS
        -------
        axes : matplotlib pyplot Axes
        -   The sorted axes.
        """
        if n_rows == 1 and n_cols == 1:
            axes = np.asarray([[axes]])
        elif n_rows == 1 and n_cols > 1:
            axes = np.vstack((axes, [None] * n_cols))
        elif n_cols == 1 and n_rows > 1:
            axes = np.vstack((axes, [None] * n_rows)).T

        return axes

    @abstractmethod
    def _plot_subplot(
        self,
        subplot: plt.Axes,
        group_name: str,
        group_idcs: list[int],
        y_axis_vars: list[str],
    ) -> None:
        """Plots the results of a specified subplot group belonging to a single
        figure group.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot to add features to.

        group_name : str
        -   Name of the subplot group whose results are being plotted.

        group_idcs : list[int]
        -   Indices of the results in the subplot group to plot.

        y_axis_vars : list[str]
        -   Names of the y-axis variables to plot on the same figure group.
        """

    def _establish_subplot(
        self,
        subplot: plt.Axes,
        title: str,
        y_axis_vars: list[str],
    ) -> None:
        """Adds a title and axis labels to the subplot.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot to add features to.

        title : str
        -   Title of the subplot.

        y_axis_vars : list[str]
        -   Names of the y-axis variables being plotted, used to generate the
            y-axis label.
        """
        subplot.set_title(title)
        subplot.set_xlabel(self.x_axis_label)
        if self.x_axis_scale is not None:
            subplot.set_xscale(self.x_axis_scale)
        if len(y_axis_vars) == 1:
            y_label = self.y_axis_labels[self.y_axis_vars.index(y_axis_vars[0])]
            y_scale = self.y_axis_scales[self.y_axis_vars.index(y_axis_vars[0])]
        else:
            y_label = self.y_axis_labels
            y_scale = self.y_axis_scales
        subplot.set_ylabel(y_label)
        subplot.set_yscale(y_scale)

    def _get_y_axis_limits_group(self, idcs: list[int]) -> str:
        """Finds which y-axis limit group the results of the given indices
        belongs to.

        PARAMETERS
        ----------
        idcs : list[int]
        -   Indices in the results to find their y-axis limit group.

        RETURNS
        -------
        group_name : str
        -   Name of the y-axis limit group.

        RAISES
        ------
        ValueError
        -   Raised if the indices do not belong to any y-axis limit group.
        """
        group_found = False
        for group_name, group_val in self._y_axis_limits_idcs.items():
            if set(idcs) <= set(group_val):
                group_found = True
                break

        if not group_found:
            raise ValueError(
                "Error when trying to plot the results:\nThe row indices of "
                f"results {idcs} are not present in any of the y-axis limit "
                "groups."
            )

        return group_name

    def _set_y_axis_tick_interval(
        self, subplot: plt.Axes, interval: Union[int, float, None]
    ) -> None:
        """Sets the tick intervals for the y-axis of a subplot.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot to add features to.

        interval : int | float | None
        -   The tick interval to use. If None, the current interval is used.
        """
        if interval:
            yticks = subplot.get_yticks()
            subplot.set_yticks(
                np.arange(yticks[1], yticks[-2] + interval / 100, interval)
            )

    def _save_figure(
        self,
        figure: plt.figure,
        figure_group: str,
        figure_n: int,
        n_figures: int,
        y_axis_vars: list[str],
    ) -> None:
        """Saves a figure.

        PARAMETERS
        ----------
        figure : matplotlib pyplot figure
        -   The figure to save.

        figure_group : str
        -   Name of the figure group.

        figure_n : int
        -   The number of the current figure in the group.

        n_figures : int
        -   The total number of figures in the group.

        y_axis_vars : list[str]
        -   The y-axis variables being plotted.
        """
        save_fpath = self._get_save_fpath(
            figure_group=figure_group,
            figure_n=figure_n,
            n_figures=n_figures,
            y_axis_vars=y_axis_vars,
        )

        write = check_before_overwrite(fpath=save_fpath)
        if write:
            figure.savefig(save_fpath, bbox_inches="tight")
            if self._verbose:
                print(f"Saving the figure to: {save_fpath}\n")

    def _get_save_fpath(
        self,
        figure_group: str,
        figure_n: int,
        n_figures: int,
        y_axis_vars: list[str],
    ) -> str:
        """Generates a filepath for saving a figure based on the name of the
        figure group, the y-axis variables being plotted, and the number of the
        figure in the group.

        PARAMETERS
        ----------
        figure_group : str
        -   Name of the figure group.

        figure_n : int
        -   The number of the current figure in the group.

        n_figures : int
        -   The total number of figures in the group.

        y_axis_vars : list[str]
        -   The y-axis variables being plotted.

        RETURNS
        -------
        str
        -   Filepath for the location to save the figure.
        """
        filename = figure_group
        for var in y_axis_vars:
            filename += f" & {var}"

        return (
            f"{self.save_folderpath}\\{filename}_{str(figure_n)}of"
            f"{str(n_figures)}.{self.save_ftype}"
        )


class LinePlot(Plotting):
    """Class for plotting results on line plots.

    PARAMETERS
    ----------
    results : dict
    -   A dictionary containing results to process.
    -   The entries in the dictionary should be either lists, numpy arrays, or
        dictionaries.
    -   Entries which are dictionaries will have their values treated as being
        identical for all values in the 'results' dictionary, given they are
        extracted from these dictionaries into the results.
    -   Keys ending with "_dimensions" are treated as containing information
        about the dimensions of other attributes in the results, e.g.
        'X_dimensions' would specify the dimensions for attribute 'X'. The
        dimensions should be a list of strings containing the values "channels"
        and "frequencies" in the positions corresponding to the axis of these
        dimensions in 'X'. A single list should be given, i.e. 'X_dimensions'
        should hold for all entries of 'X'.If no dimensions, are given, the 0th
        axis is assumed to correspond to channels and the 1st axis to
        frequencies.
    -   E.g. if 'X' has shape [25, 10, 50, 300] with an 'X_dimensions' of
        ['epochs', 'channels', 'frequencies', 'timepoints'], the shape of 'X'
        would be rearranged to [10, 50, 25, 300], corresponding to the
        dimensions ["channels", "frequencies", "epochs", "timepoints"].

    extract_from_dicts : dict[list[str]] | None; default None
    -   The entries of dictionaries within 'results' to include in the
        processing.
    -   Entries which are extracted are treated as being identical for all
        values in the 'results' dictionary.

    identical_entries : list[str] | None; default None
    -   The entries in 'results' which are identical across channels and for
        which only one copy is present.
    -   If any dimension attributes are present, these should be included as an
        identical entry, as they will be added automatically.

    discard_entries : list[str] | None; default None
    -   The entries which should be discarded immediately without processing.

    verbose : bool; default True
    -   Whether or not to print updates about the plotting process.
    """

    def __init__(
        self,
        results: dict,
        extract_from_dicts: Union[dict[list[str]], None] = None,
        identical_keys: Union[list[str], None] = None,
        discard_keys: Union[list[str], None] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            results=results,
            extract_from_dicts=extract_from_dicts,
            identical_keys=identical_keys,
            discard_keys=discard_keys,
            verbose=verbose,
        )
        # Initialises input settings for plotting.
        self.x_axis_limits = None
        self.x_axis_tick_interval = None

        # Initialises aspects of the object that will be filled with information
        # as the results are plotted.
        self._plot_type = "line"
        self._x_axis_limit_idcs = None

    def plot(
        self,
        x_axis_var: str,
        y_axis_vars: list[str],
        x_axis_limits: Union[list[Union[int, float]], None] = None,
        x_axis_label: Union[str, None] = None,
        x_axis_tick_interval: Union[int, float, None] = None,
        y_axis_limits: Union[dict[dict[list[Union[int, float]]]], None] = None,
        y_axis_labels: Union[list[str], None] = None,
        y_axis_tick_intervals: Union[
            dict[dict[Union[int, float]]], None
        ] = None,
        x_axis_scale: Union[str, None] = None,
        y_axis_scales: Union[list[str], None] = None,
        y_axis_cap_max: Union[int, float, None] = None,
        y_axis_cap_min: Union[int, float, None] = None,
        var_measure: Union[str, None] = None,
        y_axis_limits_grouping: Union[list[str], None] = None,
        figure_grouping: Union[list[str], None] = None,
        subplot_grouping: Union[list[str], None] = None,
        analysis_keys: Union[list[str], None] = None,
        legend_properties: Union[dict, None] = None,
        identical_keys: Union[list[str], None] = None,
        eligible_values: Union[dict[list[str]], None] = None,
        order_values: Union[dict[list[str]], None] = None,
        average_as_equal: bool = True,
        figure_layout: Union[list[int], None] = None,
        save: bool = False,
        save_folderpath: Union[str, None] = None,
        save_ftype: Union[str, None] = None,
    ) -> None:
        """Plots the results as line graphs.

        Keys not present in 'analysis_keys', 'identical_keys',
        'y_axis_limits_grouping', 'figure_grouping', 'subplot_grouping', or
        'eligible_values' will be excluded from the results.

        PARAMETERS
        ----------
        x_axis_var : str
        -   Key in the results to plot on the x-axis.

        y_axis_vars : list[str]
        -   Key(s) in the results to plot on the y-axis.

        x_axis_limits : list[int | float] | None; default None
        -   Lower- and upper-boundary limits, respectively, to plot for the
            x-axis variable in a list with two entries. If 'None', no limits are
            imposed.

        x_axis_label : str | None; default None
        -   X-axis label. If 'None', the name of the x-axis variable is used.

        x_axis_tick_interval : int | float | None; default None
        -   X-axis tick interval in units of the x-axis variable.

        y_axis_limits : dict[dict[list[int | float]] | None] | None; default
        None
        -   Y-axis limits. Each key in the dictionary corresponds to a name of a
            y-axis limit group, whose value is a dictionary where each key
            corresponds to the name of a y-axis variable being plotted, whose
            value is a list with two entries corresponding to the lower- and
            upper-boundary limits, respectively.
        -   Dictionaries for the y-axis limit groups can also have values of
            'None', in which case no limits are imposed.

        y_axis_labels : list[str] | None; default None
        -   Y-axis labels. If multiple y-axis variables are being plotted
            together on the same plots, only one label should be present,
            otherwise a label should be present for each y-axis variable.
        -   If 'None', the names of the y-axis variables are used.

        y_axis_tick_intervals : dict[dict[int | float] | None] | None; default
        None
        -   Y-axis tick intervals in the units of the y-axis. Each key in the
            dictionary corresponds to a name of a y-axis tick interval group,
            whose value is a dictionary where each key corresponds to the name
            of a y-axis variable being plotted, whose value is a number
            corresponding to the tick interval.
        -   Dictionaries for the y-axis tick interval groups can also have
            values of 'None', in which case the default tick intervals are used.

        x_axis_scale : str | None; default None
        -   X-axis scale. Can be 'linear' or 'log'.
        -   If 'None', 'linear' scale is used.

        y_axis_scales : list[str] | None; default None
        -   Y-axis scales. Can be 'linear' or 'log'. If multiple y-axis
            variables are being plotted together on the same plots, only one
            scale should be present, otherwise a scale should be present for
            each y-axis variable.
        -   If 'None', 'linear' scale is used for all y-axis variables.

        y_axis_cap_max : int | float | None; default None
        -   Value to cap the maximum of the y-axis at. I.e. If the cap would be
            exceeded by the plot limits, the cap is imposed, but the limits
            would be kept as-is if the cap were not exceeded.
        -   If 'None', no cap is imposed.

        y_axis_cap_min : int | float | None; default None
        -   Value to cap theminimum of the y-axis at. I.e. If the cap would be
            exceeded by the plot limits, the cap is imposed, but the limits
            would be kept as-is if the cap were not exceeded.
        -   If 'None', no cap is imposed.

        var_measure : str | None; default None
        -   Name of the variability measure (e.g. standard deviation, standard
            error of the mean) to plot alongside the results.

        y_axis_limits_grouping : list[str] | None; default None
        -   Keys in the results to use to group the y-axis limits.
        -   A special string "Y_AXIS_VARS" is permitted, in which case results
            belonging to different y-axis variables will have their own y-axis
            limits. If present here, this string must also be present in either
            'figure_grouping' or 'subplot'grouping'.
        -   If 'None', all plots share the same y-axis limits.

        figure_grouping : list[str] | None; default None
        -   Keys in the results to use to group the results which are plotted on
            the same figures, assuming this is permitted by 'figure_layout'.
        -   A special string "Y_AXIS_VARS" is permitted, in which case results
            belonging to different y-axis variables will be plotted on different
            figures. If present here, this string cannot be present in
            'subplot_grouping'.
        -   If 'None', results belonging to all y-axis variables will try to be
            plotted on the same figure.

        subplot_grouping : list[str] | None; default None
        -   Keys in the results to use to group the results which are plotted on
            the same subplots, on the same figure.
        -   A special string "Y_AXIS_VARS" is permitted, in which case results
            belonging to different y-axis variables will be plotted on different
            subplots. If present here, this string cannot be present in
            'figure_grouping'.
        -   If 'None', results belonging to all y-axis variables will be plotted
            on the same subplots.

        analysis_keys : list[str] | None; default None
        -   Keys for which variables of multiple types can be plotted on the
            same plots, and which will be included in the plot legend.

        legend_properties : dict | None; default None
        -   Properties of the subplot legends. Can contain as keys all the
            attributed of matplotlib's FontProperties class ("family", "style",
            "variant", "weight", "stretch", "size", "fname", and
            "math_fontfamily"), as well as a "loc" key specifying the location
            of the legend according to the arguments accepted by matplotlib.

        identical_keys : list[str] | None; default None
        -   Keys for which only a single variable type should be present in the
            data, and which will be included in the figure title.

        eligible_values : dict[list[str]] | None; default None
        -   A dictionary where the keys are keys in the results, and the values
            are values belonging to those keys in the results that should be
            plotted.
        -   If 'None', all results are considered eligible for plotting.

        order_values : dict[list[str]] | None; default None
        -   Dictionary containing a single key which is a key in the results,
            and whose value is a list of strings specifying how the rows in the
            results should be reordered, such that rows where the value is the
            first entry of the list is reordered into the first positions, the
            rows where the value is the second entry of the list is reordered
            into subsequent positions, etc...
        -   Used to control in what order results are plotted on the figures.

        average_as_equal : bool; default True
        -   Whether or not to treat results which have been averaged across
            multiple types beginning with the string "avg[", followed by the
            names of the types the results were averaged from, as belonging to
            the same type.

        figure_layout : list[int] | None; default None
        -   Structure of the subplots on each figure. A list with two entries
            specifying the number of rows and columns of subplots, respectively.
        -   If 'None', a 1x1 structure is used.

        save : bool; default False
        -   Whether or not to save the figures.

        save_folderpath : str | None; default None
        -   Folderpath to save the figures at.

        save_ftype : str | None; default None
        -   Filetype extension to save the figures as. Accepts whichever
            filetypes are supported by the matplotlib 'save_figure' methods.
        -   Filetypes should be given without the leading period, e.g. saving as
            a .png file should be declared as "png", not ".png".
        """
        self.x_axis_var = x_axis_var
        self.y_axis_vars = y_axis_vars
        self.x_axis_limits = x_axis_limits
        self.x_axis_label = x_axis_label
        self.x_axis_tick_interval = x_axis_tick_interval
        self.y_axis_limits = y_axis_limits
        self.y_axis_labels = y_axis_labels
        self.y_axis_tick_intervals = y_axis_tick_intervals
        self.x_axis_scale = x_axis_scale
        self.y_axis_scales = y_axis_scales
        self.y_axis_cap_max = y_axis_cap_max
        self.y_axis_cap_min = y_axis_cap_min
        self.var_measure = var_measure
        self.y_axis_limits_grouping = y_axis_limits_grouping
        self.figure_grouping = figure_grouping
        self.subplot_grouping = subplot_grouping
        self.analysis_keys = analysis_keys
        self.legend_properties = legend_properties
        self.identical_keys = identical_keys
        self.eligible_values = eligible_values
        self.order_values = order_values
        self.figure_layout = figure_layout
        self.average_as_equal = average_as_equal
        self.save = save
        self.save_folderpath = save_folderpath
        self.save_ftype = save_ftype

        if self._plotted:
            self._reinitialise_aspects()

        self._sort_plot_inputs()

        self._plot_results()

    def _reinitialise_aspects(self) -> None:
        """Reinitialises aspects of the object that will be filled with
        information as the results are plotted by setting the aspects to
        'None'."""
        self._x_axis_limit_idcs = None
        super()._reinitialise_aspects()

    def _sort_plot_inputs(self) -> None:
        """Sorts the plotting settings."""
        super()._sort_plot_inputs()
        self._sort_x_axis_limit_idcs()
        self._sort_y_axis_limits()
        self._sort_x_axis_scale_inputs()
        self._sort_plot_grouping()

    def _sort_x_axis_limit_idcs(self) -> None:
        """Finds the indices of the x-axis limits"""
        self._x_axis_limit_idcs = {}
        for eligible_idx in self._eligible_idcs:
            x_axis_vals = self._results[self.x_axis_var][eligible_idx]
            if self.x_axis_limits is not None:
                limit_idcs = [
                    x_axis_vals.index(limit) for limit in self.x_axis_limits
                ]
            else:
                limit_idcs = [0, len(x_axis_vals) - 1]
            self._x_axis_limit_idcs[eligible_idx] = limit_idcs

    def _sort_y_axis_limits(self) -> None:
        """Checks that the limits for the y-axis variables are in the correct
        format, if provided, or generates the limits if the 'y_axis_limits'
        input is 'None'.

        RAISES
        ------
        KeyError
        -   Raised if the keys of the dictionary in the provided 'y_axis_limits'
            do not match the names of the groups in the
            automatically-generated y-axis limit group indices.
        -   Raised if the keys of the dictionary within each group dictionary do
            not contain the names (and hence limits) for each y-axis variable
            being plotted.
        """
        if self.y_axis_limits is not None:
            super()._sort_y_axis_limits()
        else:
            self.y_axis_limits = {}
            share_across_vars, extra_vars = self._sort_y_axis_limits_inputs()
            for group_name, idcs in self._y_axis_limits_idcs.items():
                self.y_axis_limits[group_name] = self._get_extremes_vars(
                    var_names=self.y_axis_vars,
                    extra_vars=extra_vars,
                    idcs=idcs,
                    share_across_vars=share_across_vars,
                    min_cap=self.y_axis_cap_min,
                    max_cap=self.y_axis_cap_max,
                )

    def _get_extremes_vars(
        self,
        var_names: list[str],
        extra_vars: Union[list[Union[list[str], None]], None] = None,
        idcs: Union[list[int], None] = None,
        share_across_vars: bool = False,
        min_cap: Union[int, float, None] = None,
        max_cap: Union[int, float, None] = None,
    ) -> dict[list[Union[int, float]]]:
        """Finds the minimum and maximum values within the results for a
        specified set of columns and set of rows.

        PARAMETERS
        ----------
        var_names : list[str]
        -   Names of the column in the results to check.

        extra_vars : list[list[str] | None] | None; default None
        -   List of lists containing the names of the columns in the results
            which should be added to and subtracted from the results,
            respectively, such as standard error or standard deviation measures.
        -   Each list corresponds to the variable in the same position in
            'var_names'.

        idcs : list[int] | None; default None
        -   Indices of the rows in the results to check.
        -   If 'None', all rows are checked.

        share_across_vars : bool; default False
        -   Whether or not to have the minimum and maximum values shared across
            the variables.

        min_cap : int | float | None; default None
        -   Minimum value that can be set.

        max_cap : int | float | None; default None
        -   Maximum value that can be set.

        RETURNS
        -------
        extremes : dict[int | float]
        -   Dictionary where the keys are the variables checked, and the values
            a list with two entries corresponding to the minimum and maximum
            values of the checked results, for the corresponding variable.
        """
        if idcs is None:
            idcs = np.arange(len(self._results.index)).tolist()

        extremes = {}
        for idx, var_name in enumerate(var_names):
            try:
                extra_var = extra_vars[idx]
            except TypeError:
                extra_var = None
            extremes[var_name] = self._get_extremes_var(
                var_name=var_name,
                extra_var=extra_var,
                idcs=idcs,
                min_cap=min_cap,
                max_cap=max_cap,
            )

        if share_across_vars:
            min_val = min(np.ravel(list(extremes.values())).tolist())
            max_val = max(np.ravel(list(extremes.values())).tolist())
            for var in var_names:
                extremes[var] = [min_val, max_val]

        return extremes

    def _get_extremes_var(
        self,
        var_name: str,
        extra_var: Union[str, None] = None,
        idcs: Union[list[int], None] = None,
        min_cap: Union[int, float, None] = None,
        max_cap: Union[int, float, None] = None,
    ) -> list[Union[int, float]]:
        """Finds the minimum and maximum values within the results for a
        specified column and set of rows.

        PARAMETERS
        ----------
        var_name : str
        -   Name of the column in the results to check.

        extra_var : str | None; default None
        -   Names of the column in the results which should be added to and
            subtracted from the results, respectively, such as standard error
            or standard deviation measures.

        idcs : list[int] | None; default None
        -   Indices of the rows in the results to check.
        -   If 'None', all rows are checked.

        min_cap : int | float | None; default None
        -   Minimum value that can be set.

        max_cap : int | float | None; default None
        -   Maximum value that can be set.

        RETURNS
        -------
        list
        -   List with two entries corresponding to the minimum and maximum
            values of the checked results, respectively.
        """
        if idcs is None:
            idcs = np.arange(len(self._results.index)).tolist()

        min_val = float("inf")
        max_val = float("-inf")

        for row_i in idcs:
            x_lim_idcs = self._x_axis_limit_idcs[row_i]
            main_vals = self._results[var_name][row_i][
                x_lim_idcs[0] : x_lim_idcs[1] + 1
            ]
            minima = [min(main_vals)]
            maxima = [max(main_vals)]
            if extra_var is not None:
                extra_vals = self._results[extra_var][row_i]
                if extra_vals:
                    extra_vals = extra_vals[x_lim_idcs[0] : x_lim_idcs[1] + 1]
                    subbed_vals = np.subtract(main_vals, extra_vals).tolist()
                    added_vals = np.add(main_vals, extra_vals).tolist()
                else:
                    subbed_vals = main_vals
                    added_vals = main_vals
                minima.append(min(subbed_vals))
                maxima.append(max(added_vals))
            minimum = min(minima)
            maximum = max(maxima)
            if minimum < min_val:
                min_val = minimum
            if maximum > max_val:
                max_val = maximum

        boundary_val = (max_val - min_val) * 0.05
        min_val = min_val - boundary_val
        max_val = max_val + boundary_val

        if min_val < min_cap:
            min_val = min_cap
        if max_val > max_cap:
            max_val = max_cap

        return [min_val, max_val]

    def _sort_x_axis_scale_inputs(self) -> None:
        """Sorts the inputs associated with the x-axis scale, setting to
        'linear' if no scale is specified."""
        if self.x_axis_scale is None:
            self.x_axis_scale = "linear"

    def _plot_subplot(
        self,
        subplot: plt.Axes,
        group_name: str,
        group_idcs: list[int],
        y_axis_vars: list[str],
    ) -> None:
        """Plots the results of a specified subplot group belonging to a single
        figure group.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot to add features to.

        group_name : str
        -   Name of the subplot group whose results are being plotted.

        group_idcs : list[int]
        -   Indices of the results in the subplot group to plot.

        y_axis_vars : list[str]
        -   Names of the y-axis variables to plot on the same figure group.
        """
        self._establish_subplot(
            subplot=subplot, title=group_name, y_axis_vars=y_axis_vars
        )

        colour_set = get_plot_colours()
        colours = [*colour_set]
        for _ in range(int(np.ceil(len(group_idcs) / len(colour_set))) - 1):
            colours.extend(colour_set)
        colour_i = 0
        for idx in group_idcs:
            for y_var in y_axis_vars:
                if len(y_axis_vars) == 1:
                    values_label = self._get_value_labels(idx=idx)
                else:
                    values_label = self._get_value_labels(
                        idx=idx, var_name=y_var
                    )
                (x_vals, y_vals, var_measure_vals) = self._get_vals_to_plot(
                    y_var=y_var, idx=idx
                )
                subplot.plot(
                    x_vals,
                    y_vals,
                    color=colours[colour_i],
                    linewidth=2,
                    label=values_label,
                )
                if (
                    self.var_measure is not None
                    and var_measure_vals is not None
                ):
                    subplot.fill_between(
                        x_vals,
                        var_measure_vals[0],
                        var_measure_vals[1],
                        color=colours[colour_i],
                        alpha=0.3,
                    )
                colour_i += 1
        subplot.legend(
            labelspacing=0,
            loc=self._legend_location,
            prop=self._legend_font_properties,
        )
        self._set_x_axis_tick_interval(subplot, self.x_axis_tick_interval)
        y_lim_group = self._get_y_axis_limits_group(idcs=group_idcs)
        self._set_y_axis_tick_interval(
            subplot, self.y_axis_tick_intervals[y_lim_group][y_axis_vars[0]]
        )
        subplot.set_ylim(self.y_axis_limits[y_lim_group][y_axis_vars[0]])

    def _get_value_labels(
        self, idx: int, var_name: Union[str, None] = None
    ) -> str:
        """Generates a label for a value based on the conditions from which this
        value was derived.

        PARAMETERS
        ----------
        idx : int
        -   Index of the row of the results for the values being plotted.

        var_name : str | None; default None
        -   Name of the variable being plotted. Optional, as you may not want to
            include the variable name if you are only plotting values from a
            single variable type.

        RETURNS
        -------
        str
        -   Label of the values being plotted.
        """
        label = combine_col_vals_df(
            dataframe=self._results,
            keys=self.analysis_keys,
            idcs=[idx],
            special_vals=self._special_values,
        )[0]

        if var_name is not None:
            label = f"{var_name}: {label}"

        return label

    def _get_vals_to_plot(self, y_var: str, idx: int) -> tuple[
        list[Union[int, float]],
        list[Union[int, float]],
        list[list[Union[int, float]]],
    ]:
        """For a given row of results, gets the values to plot on the x- and
        y-axes, plus the boundaries for filling in for the variability measure,
        if applicable.

        PARAMETERS
        ----------
        y_var : str
        -   Variable whose values are being plotted.

        idx : int
        -   Index of the row of the values in the results being plotted.

        RETURNS
        -------
        x_vals : list[int | float]
        -   Values to plot on the x-axis.

        y_vals : list[int | float]
        -   Values to plot on the y-axis.

        var_values : list[list[int | float]]
        -   Y-axis values with the addition of the variability measures. The
            first entry is the values in 'y_vals' minus the corresponding values
            in the variability measure, whilst the second entry is the values in
            'y_vals' plus the corresponding values in the variability measure.
        """
        x_idcs = self._x_axis_limit_idcs[idx]
        x_vals = self._results[self.x_axis_var][idx][x_idcs[0] : x_idcs[1] + 1]
        y_vals = self._results[y_var][idx][x_idcs[0] : x_idcs[1] + 1]

        var_measure = f"{y_var}_{self.var_measure}"
        if (
            self.var_measure is not None
            and self._results[var_measure][idx] is not None
        ):
            var_values = [[], []]
            var_values[0] = np.subtract(
                y_vals,
                self._results[var_measure][idx][x_idcs[0] : x_idcs[1] + 1],
            )
            var_values[1] = np.add(
                y_vals,
                self._results[var_measure][idx][x_idcs[0] : x_idcs[1] + 1],
            )
        else:
            var_values = None

        return x_vals, y_vals, var_values

    def _set_x_axis_tick_interval(
        self, subplot: plt.Axes, interval: Union[int, float, None]
    ) -> None:
        """Sets the tick intervals for the x-axis of a subplot.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot to add features to.

        interval : int | float | None
        -   The tick interval to use. If None, the current interval is used.
        """
        if interval:
            subplot.set_xticks(
                np.arange(
                    self.x_axis_limits[0],
                    self.x_axis_limits[1] + interval / 100,
                    interval,
                )
            )


class BoxPlot(Plotting):
    """Class for plotting results on box plots.

    PARAMETERS
    ----------
    results : dict
    -   A dictionary containing results to process.
    -   The entries in the dictionary should be either lists, numpy arrays, or
        dictionaries.
    -   Entries which are dictionaries will have their values treated as being
        identical for all values in the 'results' dictionary, given they are
        extracted from these dictionaries into the results.
    -   Keys ending with "_dimensions" are treated as containing information
        about the dimensions of other attributes in the results, e.g.
        'X_dimensions' would specify the dimensions for attribute 'X'. The
        dimensions should be a list of strings containing the values "channels"
        and "frequencies" in the positions corresponding to the axis of these
        dimensions in 'X'. A single list should be given, i.e. 'X_dimensions'
        should hold for all entries of 'X'.If no dimensions, are given, the 0th
        axis is assumed to correspond to channels and the 1st axis to
        frequencies.
    -   E.g. if 'X' has shape [25, 10, 50, 300] with an 'X_dimensions' of
        ['epochs', 'channels', 'frequencies', 'timepoints'], the shape of 'X'
        would be rearranged to [10, 50, 25, 300], corresponding to the
        dimensions ["channels", "frequencies", "epochs", "timepoints"].

    extract_from_dicts : dict[list[str]] | None; default None
    -   The entries of dictionaries within 'results' to include in the
        processing.
    -   Entries which are extracted are treated as being identical for all
        values in the 'results' dictionary.

    identical_entries : list[str] | None; default None
    -   The entries in 'results' which are identical across channels and for
        which only one copy is present.
    -   If any dimension attributes are present, these should be included as an
        identical entry, as they will be added automatically.

    discard_entries : list[str] | None; default None
    -   The entries which should be discarded immediately without processing.

    verbose : bool; default True
    -   Whether or not to print updates about the plotting process.
    """

    def __init__(
        self,
        results: dict,
        extract_from_dicts: Union[dict[list[str]], None] = None,
        identical_keys: Union[list[str], None] = None,
        discard_keys: Union[list[str], None] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            results=results,
            extract_from_dicts=extract_from_dicts,
            identical_keys=identical_keys,
            discard_keys=discard_keys,
            verbose=verbose,
        )

        # Initialises aspects of the object that will be filled with information
        # as the results are plotted.
        self._plot_type = "box"

    def plot(
        self,
        x_axis_var: str,
        y_axis_vars: list[str],
        x_axis_label: Union[str, None] = None,
        y_axis_limits: Union[dict[dict[list[Union[int, float]]]], None] = None,
        y_axis_labels: Union[list[str], None] = None,
        y_axis_tick_intervals: Union[
            dict[dict[Union[int, float]]], None
        ] = None,
        y_axis_scales: Union[list[str], None] = None,
        y_axis_cap_max: Union[int, float, None] = None,
        y_axis_cap_min: Union[int, float, None] = None,
        y_axis_limits_grouping: Union[list[str], None] = None,
        figure_grouping: Union[list[str], None] = None,
        subplot_grouping: Union[list[str], None] = None,
        analysis_keys: Union[list[str], None] = None,
        identical_keys: Union[list[str], None] = None,
        eligible_values: Union[dict[list[str]], None] = None,
        order_values: Union[dict[list[str]], None] = None,
        average_as_equal: bool = True,
        figure_layout: Union[list[int], None] = None,
        save: bool = False,
        save_folderpath: Union[str, None] = None,
        save_ftype: Union[str, None] = None,
    ) -> None:
        """Plots the results as line graphs.

        Keys not present in 'analysis_keys', 'identical_keys',
        'y_axis_limits_grouping', 'figure_grouping', 'subplot_grouping', or
        'eligible_values' will be excluded from the results.

        PARAMETERS
        ----------
        x_axis_var : str
        -   Key in the results to plot on the x-axis, also used for creating
            groups of boxes on each plot.

        y_axis_vars : list[str]
        -   Key(s) in the results to plot on the y-axis.

        x_axis_label : str | None: default None
        -   X-axis label. If 'None', the name of the x-axis variable is used.

        y_axis_limits : dict[dict[list[int | float]] | None] | None; default
        None
        -   Y-axis limits. Each key in the dictionary corresponds to a name of a
            y-axis limit group, whose value is a dictionary where each key
            corresponds to the name of a y-axis variable being plotted, whose
            value is a list with two entries corresponding to the lower- and
            upper-boundary limits, respectively.
        -   Dictionaries for the y-axis limit groups can also have values of
            'None', in which case no limits are imposed.

        y_axis_labels : list[str] | None; default None
        -   Y-axis labels. If multiple y-axis variables are being plotted
            together on the same plots, only one label should be present,
            otherwise a label should be present for each y-axis variable.
        -   If 'None', the names of the y-axis variables are used.

        y_axis_tick_intervals : dict[dict[int | float] | None] | None; default
        None
        -   Y-axis tick intervals in the units of the y-axis. Each key in the
            dictionary corresponds to a name of a y-axis tick interval group,
            whose value is a dictionary where each key corresponds to the name
            of a y-axis variable being plotted, whose value is a number
            corresponding to the tick interval.
        -   Dictionaries for the y-axis tick interval groups can also have
            values of 'None', in which case the default tick intervals are used.

        y_axis_cap_max : int | float | None; default None
        -   Value to cap the maximum of the y-axis at. I.e. If the cap would be
            exceeded by the plot limits, the cap is imposed, but the limits
            would be kept as-is if the cap were not exceeded.
        -   If 'None', no cap is imposed.

        y_axis_cap_min : int | float | None; default None
        -   Value to cap theminimum of the y-axis at. I.e. If the cap would be
            exceeded by the plot limits, the cap is imposed, but the limits
            would be kept as-is if the cap were not exceeded.
        -   If 'None', no cap is imposed.

        y_axis_limits_grouping : list[str] | None; default None
        -   Keys in the results to use to group the y-axis limits.
        -   A special string "Y_AXIS_VARS" is permitted, in which case results
            belonging to different y-axis variables will have their own y-axis
            limits. If present here, this string must also be present in either
            'figure_grouping' or 'subplot'grouping'.
        -   If 'None', all plots share the same y-axis limits.

        figure_grouping : list[str] | None; default None
        -   Keys in the results to use to group the results which are plotted on
            the same figures, assuming this is permitted by 'figure_layout'.
        -   A special string "Y_AXIS_VARS" is permitted, in which case results
            belonging to different y-axis variables will be plotted on different
            figures. If present here, this string cannot be present in
            'subplot_grouping'.
        -   If 'None', results belonging to all y-axis variables will try to be
            plotted on the same figure.

        subplot_grouping : list[str] | None; default None
        -   Keys in the results to use to group the results which are plotted on
            the same subplots, on the same figure.
        -   A special string "Y_AXIS_VARS" is permitted, in which case results
            belonging to different y-axis variables will be plotted on different
            subplots. If present here, this string cannot be present in
            'figure_grouping'.
        -   If 'None', results belonging to all y-axis variables will be plotted
            on the same subplots.

        analysis_keys : list[str] | None; default None
        -   Keys for which variables of multiple types can be plotted on the
            same plots, and which will be included in the plot legend.

        identical_keys : list[str] | None; default None
        -   Keys for which only a single variable type should be present in the
            data, and which will be included in the figure title.

        eligible_values : dict[list[str]] | None; default None
        -   A dictionary where the keys are keys in the results, and the values
            are values belonging to those keys in the results that should be
            plotted.
        -   If 'None', all results are considered eligible for plotting.

        order_values : dict[list[str]] | None; default None
        -   Dictionary containing a single key which is a key in the results,
            and whose value is a list of strings specifying how the rows in the
            results should be reordered, such that rows where the value is the
            first entry of the list is reordered into the first positions, the
            rows where the value is the second entry of the list is reordered
            into subsequent positions, etc...
        -   Used to control in what order results are plotted on the figures.

        average_as_equal : bool; default True
        -   Whether or not to treat results which have been averaged across
            multiple types beginning with the string "avg[", followed by the
            names of the types the results were averaged from, as belonging to
            the same type.

        figure_layout : list[int] | None; default None
        -   Structure of the subplots on each figure. A list with two entries
            specifying the number of rows and columns of subplots, respectively.
        -   If 'None', a 1x1 structure is used.

        save : bool; default False
        -   Whether or not to save the figures.

        save_folderpath : str | None; default None
        -   Folderpath to save the figures at.

        save_ftype : str | None; default None
        -   Filetype extension to save the figures as. Accepts whichever
            filetypes are supported by the matplotlib 'save_figure' methods.
        -   Filetypes should be given without the leading period, e.g. saving as
            a .png file should be declared as "png", not ".png".
        """
        self.x_axis_var = x_axis_var
        self.y_axis_vars = y_axis_vars
        self.x_axis_label = x_axis_label
        self.y_axis_limits = y_axis_limits
        self.y_axis_labels = y_axis_labels
        self.y_axis_tick_intervals = y_axis_tick_intervals
        self.y_axis_scales = y_axis_scales
        self.y_axis_cap_max = y_axis_cap_max
        self.y_axis_cap_min = y_axis_cap_min
        self.y_axis_limits_grouping = y_axis_limits_grouping
        self.figure_grouping = figure_grouping
        self.subplot_grouping = subplot_grouping
        self.analysis_keys = analysis_keys
        self.identical_keys = identical_keys
        self.eligible_values = eligible_values
        self.order_values = order_values
        self.figure_layout = figure_layout
        self.average_as_equal = average_as_equal
        self.save = save
        self.save_folderpath = save_folderpath
        self.save_ftype = save_ftype

        if self._plotted:
            self._reinitialise_aspects()

        self._sort_plot_inputs()

        self._plot_results()

    def _sort_plot_inputs(self) -> None:
        """Sorts the plotting settings."""
        super()._sort_plot_inputs()
        self._sort_y_axis_limits()
        self._sort_plot_grouping()

    def _sort_y_axis_limits(self) -> None:
        """Checks that the limits for the y-axis variables are in the correct
        format, if provided, or generates the limits if the 'y_axis_limits'
        input is 'None'.

        RAISES
        ------
        KeyError
        -   Raised if the keys of the dictionary in the provided 'y_axis_limits'
            do not match the names of the groups in the
            automatically-generated y-axis limit group indices.
        -   Raised if the keys of the dictionary within each group dictionary do
            not contain the names (and hence limits) for each y-axis variable
            being plotted.
        """
        if self.y_axis_limits is not None:
            super()._sort_y_axis_limits()
        else:
            self.y_axis_limits = {}
            share_across_vars, _ = self._sort_y_axis_limits_inputs()
            for group_name, idcs in self._y_axis_limits_idcs.items():
                self.y_axis_limits[group_name] = self._get_extremes_vars(
                    var_names=self.y_axis_vars,
                    idcs=idcs,
                    share_across_vars=share_across_vars,
                    min_cap=self.y_axis_cap_min,
                    max_cap=self.y_axis_cap_max,
                )

    def _get_extremes_vars(
        self,
        var_names: list[str],
        idcs: Union[list[int], None] = None,
        share_across_vars: bool = False,
        min_cap: Union[int, float, None] = None,
        max_cap: Union[int, float, None] = None,
    ) -> dict[list[Union[int, float]]]:
        """Finds the minimum and maximum values within the results for a
        specified set of columns and set of rows.

        PARAMETERS
        ----------
        var_names : list[str]
        -   Names of the column in the results to check.

        idcs : list[int] | None; default None
        -   Indices of the rows in the results to check.
        -   If 'None', all rows are checked.

        share_across_vars : bool; default False
        -   Whether or not to have the minimum and maximum values shared across
            the variables.

        min_cap : int | float | None; default None
        -   Minimum value that can be set.

        max_cap : int | float | None; default None
        -   Maximum value that can be set.

        RETURNS
        -------
        extremes : dict[int | float]
        -   Dictionary where the keys are the variables checked, and the values
            a list with two entries corresponding to the minimum and maximum
            values of the checked results, for the corresponding variable.
        """
        if idcs is None:
            idcs = np.arange(len(self._results.index)).tolist()

        extremes = {}
        for var_name in var_names:
            extremes[var_name] = self._get_extremes_var(
                var_name=var_name,
                idcs=idcs,
                min_cap=min_cap,
                max_cap=max_cap,
            )

        if share_across_vars:
            min_val = np.nanmin(np.ravel(list(extremes.values())).tolist())
            max_val = np.nanmax(np.ravel(list(extremes.values())).tolist())
            for var in var_names:
                extremes[var] = [min_val, max_val]

        return extremes

    def _get_extremes_var(
        self,
        var_name: str,
        idcs: Union[list[int], None] = None,
        min_cap: Union[int, float, None] = None,
        max_cap: Union[int, float, None] = None,
    ) -> list[Union[int, float]]:
        """Finds the minimum and maximum values within the results for a
        specified column and set of rows.

        PARAMETERS
        ----------
        var_name : str
        -   Name of the column in the results to check.

        idcs : list[int] | None; default None
        -   Indices of the rows in the results to check.
        -   If 'None', all rows are checked.

        min_cap : int | float | None; default None
        -   Minimum value that can be set.

        max_cap : int | float | None; default None
        -   Maximum value that can be set.

        RETURNS
        -------
        list[int | float | NaN]
        -   List with two entries corresponding to the minimum and maximum
            values of the checked results, respectively. If no non-NaN values
            are present in the checked results, a list with two NaN entries is
            returned.
        """
        if idcs is None:
            idcs = np.arange(len(self._results.index)).tolist()

        min_val = float("inf")
        max_val = float("-inf")

        for row_i in idcs:
            main_vals = self._results[var_name][row_i]
            if not np.isnan(main_vals).all():
                minimum = np.nanmin([np.nanmin(main_vals)])
                maximum = np.nanmax([np.nanmax(main_vals)])
                if minimum < min_val:
                    min_val = minimum
                if maximum > max_val:
                    max_val = maximum

        boundary_val = (max_val - min_val) * 0.05
        min_val = min_val - boundary_val
        max_val = max_val + boundary_val

        min_val = max(min_val, min_cap)
        max_val = min(max_val, max_cap)

        if min_val == float("inf"):
            min_val = np.nan
        if max_val == float("-inf"):
            max_val = np.nan

        return [min_val, max_val]

    def _sort_plot_grouping(self) -> None:
        """Sorts the figure and subplot groups, finding the indices of the
        corresponding rows in the results."""
        super()._sort_plot_grouping()
        self._sort_subplot_subgrouping()

    def _sort_subplot_subgrouping(self) -> None:
        """Sorts the subgroups for the indices of rows in the results being
        plotted on the same subplots on each set of figures (based on the
        different x-axis variable groups)."""
        for _, subplot_groups in self._plot_grouping.items():
            for subplot_group, subplot_idcs in subplot_groups.items():
                if self.x_axis_var is not None:
                    subplot_groups[subplot_group] = get_group_names_idcs(
                        dataframe=self._results,
                        keys=[self.x_axis_var],
                        eligible_idcs=subplot_idcs,
                        replacement_idcs=subplot_idcs,
                        special_vals=self._special_values,
                        keys_in_names=False,
                    )
                else:
                    subplot_groups[subplot_group] = {"ALL": subplot_idcs}
                for (
                    subplot_subgroup,
                    subgroup_idcs,
                ) in subplot_groups[subplot_group].items():
                    subplot_groups[subplot_group][subplot_subgroup] = (
                        get_group_names_idcs(
                            dataframe=self._results,
                            keys=self.analysis_keys,
                            eligible_idcs=subgroup_idcs,
                            replacement_idcs=subgroup_idcs,
                            special_vals=self._special_values,
                            keys_in_names=True,
                        )
                    )

    def _plot_subplot(
        self,
        subplot: plt.Axes,
        group_name: str,
        group_idcs: list[int],
        y_axis_vars: list[str],
    ) -> None:
        """Plots the results of a specified subplot group belonging to a single
        figure group.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot to add features to.

        group_name : str
        -   Name of the subplot group whose results are being plotted.

        group_idcs : list[int]
        -   Indices of the results in the subplot group to plot.

        y_axis_vars : list[str]
        -   Names of the y-axis variables to plot on the same figure group.
        """
        self._establish_subplot(
            subplot=subplot, title=group_name, y_axis_vars=y_axis_vars
        )
        plotted_idcs = []
        assigned_colours = {}
        colour_set = get_plot_colours()
        box_colours = []
        subgroup_centre_positions = {}
        start_x_position = 1
        subgroup_x_separation = 1.25
        for subgroup_name, analysis_groups in group_idcs.items():
            plotted_idcs.extend(
                [idx for idcs in analysis_groups.values() for idx in idcs]
            )
            subgroup_x_positions = []
            for y_var in y_axis_vars:
                analysis_group_names = list(analysis_groups.keys())
                if len(y_axis_vars) == 1:
                    value_labels = analysis_group_names
                else:
                    value_labels = [
                        f"{y_var}: {group}" for group in analysis_group_names
                    ]
                assigned_colours = self._assign_colours(
                    assigned_colours=assigned_colours,
                    colour_set=colour_set,
                    value_labels=value_labels,
                )
                current_x_positions = [
                    start_x_position + value_n
                    for value_n in range(len(value_labels))
                ]
                y_vals = self._get_y_vals_to_plot(
                    analysis_groups=analysis_groups, y_var=y_var
                )
                box_colours.extend(
                    [assigned_colours[label] for label in value_labels]
                )
                subplot.boxplot(
                    list(y_vals.values()),
                    positions=current_x_positions,
                    widths=0.9,
                    patch_artist=True,
                    medianprops={"color": "k"},
                )
                subgroup_x_positions.extend(current_x_positions)
                start_x_position = (
                    subgroup_x_positions[-1] + subgroup_x_separation
                )
            subgroup_centre_positions[subgroup_name] = np.mean(
                subgroup_x_positions
            )
            start_x_position += subgroup_x_separation
        self._add_subgroup_labels(
            subplot=subplot,
            labels=list(subgroup_centre_positions.keys()),
            positions=list(subgroup_centre_positions.values()),
        )
        self._colour_boxes(subplot=subplot, colours=box_colours)
        self._pad_xlim(subplot=subplot)
        self._make_legend(
            subplot=subplot, assigned_colours=assigned_colours, offscreen_x=-999
        )
        y_lim_group = self._get_y_axis_limits_group(idcs=plotted_idcs)
        self._set_ylim(subplot, self.y_axis_limits[y_lim_group][y_axis_vars[0]])

    def _get_y_vals_to_plot(
        self, analysis_groups: dict[int], y_var: str
    ) -> dict:
        """Gets the values to plot against the y-axis for a set of analysis
        groups belonging to a single subgroup.

        PARAMETERS
        ----------
        analysis_groups : dict[int]
        -   Dictionary where the keys are the labels of the analysis groups and
            the values the indices of rows in the results corresponding to this
            analysis group for a single subgroup.

        y_var : str
        -   Name of the y-axis variable being plotted.

        RETURNS
        -------
        y_values : dict
        -   Dictionary where the keys are the labels of the analysis groups and
            the values the results that will be plotted, stored in a list.

        NOTES
        -----
        -   If the results being plotted contains NaN values, these will be
            removed (allowing data that is present to actually be plotted).
            However, if the results consists of only a single NaN value, no
            change is made (leaving a blank space in the plot).
        """
        y_values = {}
        for group, idcs in analysis_groups.items():
            values = [self._results.at[idx, y_var] for idx in idcs]
            if values != [np.nan]:
                values = [value for value in values if not np.isnan(value)]
            y_values[group] = values

        return y_values

    def _assign_colours(
        self, assigned_colours: dict, colour_set: list, value_labels: list
    ) -> dict:
        """Assigns colours to groups of results being plotted.

        If colours have already been assigned to groups, these appropriate
        colours for the requested groups will be returned, otherwise, new
        colours will be assigned.

        PARAMETERS
        ----------
        assigned_colours : dict
        -   Dictionary where the keys are labels of the groups of results and
            the values colour values used when plotting these results.
        -   If no colours have been assigned yet, this can be an empty
            dictionary.

        colour_set : list
        -   Colour values that can be assigned to groups.

        value_labels : list
        -   Labels of the groups of results whose colours should be assigned.

        RETURNS
        -------
        assigned_colours : dict
        -   Dictionary where the keys are labels of the groups of results and
            the values colour values used when plotting these results, updated
            to include colours for groups that were not previously assigned a
            colour.
        """
        colour_set = get_plot_colours()
        colour_i = 0
        for label in value_labels:
            colour_i = loop_index(reset_at=len(colour_set), idx=colour_i)
            if label not in assigned_colours.keys():
                assigned_colours[label] = colour_set[colour_i]
            colour_i += 1

        return assigned_colours

    def _add_subgroup_labels(
        self,
        subplot: plt.Axes,
        labels: list[str],
        positions: list[Union[int, float]],
    ) -> None:
        """Adds labels for subgroups on a subplot.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot on which the boxes are plotted.

        labels : list[str]
        -   Names of the subgroups.

        positions : list[int | float]
        -   Positions of the subgroup labels.
        """
        subplot.set_xticks(positions)
        subplot.set_xticklabels(labels)

    def _colour_boxes(self, subplot: plt.Axes, colours: list) -> None:
        """Colours boxes on a subplot.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot on which the boxes are plotted.

        colours : list
        -   The colours to use. Can be any colour label interpretable by
            matplotlib.

        RAISES
        ------
        ValueError
        -   Raised if the number of colours to use does not match the number of
            boxes to colour.
        """
        if len(subplot.patches) != len(colours):
            raise ValueError(
                "Warning: boxes are being coloured, however the number of "
                f"colours provided ({len(colours)}) does not match the number "
                f"of boxes to colour ({len(subplot['boxes'])})."
            )
        for patch, colour in zip(subplot.patches, colours):
            patch.set_facecolor(colour)

    def _pad_xlim(self, subplot: plt.Axes) -> None:
        """Adds a small boundary to the x-axis limits so that the plotted data
        does not touch the sides.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot on which the data is plotted.
        """
        xlim = subplot.get_xlim()
        xlim_range = xlim[1] - xlim[0]
        boundary = [-xlim_range * 0.05, xlim_range * 0.025]
        subplot.set_xlim(np.add(xlim, boundary))

    def _make_legend(
        self,
        subplot: plt.Axes,
        assigned_colours: dict,
        offscreen_x: Union[int, float],
    ) -> None:
        """Adds a legend to a subplot, with one entry for each analysis group.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot on which the data are plotted.

        assigned_colours : dict
        -   Dictionary where the keys are labels of the groups of results and
            the values colour values used when plotting the data.

        offscreen_x : int | float
        -   An x-axis coordinate off-screen of the plotted data where
            pseudo-data with a label can be plotted.
        """
        xlim = subplot.get_xlim()
        ylim = subplot.get_ylim()
        for group, colour in assigned_colours.items():
            subplot.plot(offscreen_x, 1, label=group, color=colour, linewidth=2)
        subplot.set_xlim(xlim)
        subplot.set_ylim(ylim)
        subplot.legend(
            labelspacing=0,
            loc=self._legend_location,
            prop=self._legend_font_properties,
        )

    def _set_ylim(
        self, subplot: plt.Axes, limits: list[Union[int, float]]
    ) -> None:
        """Sets the y-axis limits for a subplot.

        PARAMETERS
        ----------
        subplot : matplotlib pyplot Axes
        -   The subplot on which the data are plotted.

        limits : list[int | float | NaN]
        -   The lower and upper limits of the y-axis, respectively. If either
            entry is NaN, the value will be derived from the current subplot
            limits.
        """
        ylim = subplot.get_ylim()
        new_ylim = deepcopy(limits)
        if not np.isnan(limits).all():
            if np.isnan(limits[0]):
                new_ylim[0] = ylim[0]
            elif np.isnan(limits[1]):
                new_ylim[1] = ylim[1]
            subplot.set_ylim(new_ylim)
