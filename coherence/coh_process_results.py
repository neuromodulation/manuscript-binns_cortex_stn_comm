"""Classes and methods for applying post-processing to results.

CLASSES
-------
PostProcess
-   Class for the post-processing of results derived from raw signals.

METHODS
-------
load_results_of_types
-   Loads results of a multiple types of data and merges them into a single
    PostProcess object.

load_results_of_type
-   Loads results of a single type of data and appends them into a single
    PostProcess object
"""

import os
from copy import deepcopy
from typing import Any, Callable, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import RBFInterpolator
import mne
import trimesh
from coh_exceptions import (
    DuplicateEntryError,
    EntryLengthError,
    UnavailableProcessingError,
    UnidenticalEntryError,
)
from coh_handle_entries import (
    combine_col_vals_df,
    check_non_repeated_vals_lists,
    check_vals_identical_df,
    dict_to_df,
    get_eligible_idcs_list,
    get_group_idcs,
    sort_inputs_results,
    unique,
)
from coh_handle_files import generate_sessionwise_fpath, load_file
from coh_normalisation import gaussian_transform
from coh_saving import save_dict, save_object
from coh_track_fibres import TrackFibres


class PostProcess:
    """Class for the post-processing of results derived from raw signals.

    PARAMETERS
    ----------
    results : dict
    -   A dictionary containing results to process.
    -   The entries in the dictionary should be either lists, numpy arrays, or
        dictionaries.
    -   Entries which are dictionaries will have their values treated as being
        identical for all values in the 'results' dictionary, given they are
        extracted from these dictionaries into the results.
    -   A key with the name "dimensions" is treated as containing information
        about the dimensions of other attributes in the results, e.g. with value
        ["windows", "channels", "frequencies"].

    extract_from_dicts : dict[list[str]] | None; default None
    -   The entries of dictionaries within 'results' to include in the
        processing.
    -   Entries which are extracted are treated as being identical for all
        values in the 'results' dictionary.

    identical_keys : list[str] | None; default None
    -   The keys in 'results' which are identical across channels and for
        which only one copy is present.
    -   If any dimension attributes are present, these should be included as an
        identical entry, as they will be added automatically.

    discard_keys : list[str] | None; default None
    -   The keys which should be discarded immediately without processing.

    METHODS
    -------
    average
    -   Averages results.

    subtract
    -   Subtracts results.

    log
    -   Log transforms results with base 10.

    isolate_bands
    -   Isolates data from bands (i.e. portions) of the results (e.g frequency
        bands) into a new DataFrame.

    append
    -   Appends other dictionaries of results to the list of result dictionaries
        stored in the PostProcess object.

    merge
    -   Merge dictionaries of results containing different keys into the
        results.
    """

    def __init__(
        self,
        results: dict,
        extract_from_dicts: Optional[dict[list[str]]] = None,
        identical_keys: Optional[list[str]] = None,
        discard_keys: Optional[list[str]] = None,
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

        # Initialises aspects of the object that will be filled with information
        # as the data is processed.
        self._process_measures = []
        self._var_measures = []
        self._var_columns = []
        self._desc_measures = []
        self._desc_process_measures = []
        self._desc_var_measures = ["std", "sem"]
        self._band_results = None

    def append_from_dict(
        self,
        new_results: dict,
        extract_from_dicts: Optional[dict[list[str]]] = None,
        identical_keys: Optional[list[str]] = None,
        discard_keys: Optional[list[str]] = None,
    ) -> None:
        """Appends a dictionary of results to the results stored in the
        PostProcess object.
        -   Cannot be called after frequency band results have been computed.

        PARAMETERS
        ----------
        new_results : dict
        -   A dictionary containing results to add.
        -   The entries in the dictionary should be either lists, numpy arrays,
            or dictionaries.
        -   Entries which are dictionaries will have their values treated as
            being identical for all values in the 'results' dictionary, given
            they are extracted from these dictionaries into the results.

        extract_from_dicts : dict[list[str]] | None; default None
        -   The entries of dictionaries within 'results' to include in the
            processing.
        -   Entries which are extracted are treated as being identical for all
            values in the 'results' dictionary.

        identical_keys : list[str] | None; default None
        -   The keys in 'results' which are identical across channels and for
            which only one copy is present.

        discard_keys : list[str] | None; default None
        -   The keys which should be discarded immediately without
            processing.
        """
        new_results = sort_inputs_results(
            results=new_results,
            extract_from_dicts=extract_from_dicts,
            identical_keys=identical_keys,
            discard_keys=discard_keys,
            verbose=self._verbose,
        )

        check_non_repeated_vals_lists(
            lists=[list(self._results.keys()), list(new_results.keys())],
            allow_non_repeated=False,
        )

        new_results = dict_to_df(obj=new_results)

        self._results = pd.concat(
            objs=[self._results, new_results], ignore_index=True
        )

    def append_from_df(
        self,
        new_results: pd.DataFrame,
    ) -> None:
        """Appends a DataFrame of results to the results stored in the
        PostProcess object.
        -   Cannot be called after frequency band results have been computed.

        PARAMETERS
        ----------
        new_results : pandas DataFrame
        -   The new results to append.
        """
        check_non_repeated_vals_lists(
            lists=[self._results.keys().tolist(), new_results.keys().tolist()],
            allow_non_repeated=False,
        )

        self._results = pd.concat(
            objs=[self._results, new_results], ignore_index=True
        )

    def _make_results_mergeable(
        self, results_1: pd.DataFrame, results_2: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Converts results DataFrames into a format that can be handled by the
        pandas function 'merge' by converting any lists into tuples.

        PARAMETERS
        ----------
        results_1: pandas DataFrame
        -   The first DataFrame to make mergeable.

        results_2: pandas DataFrame
        -   The second DataFrame to make mergeable.

        RETURNS
        -------
        pandas DataFrame
        -   The first DataFrame made mergeable.

        pandas DataFrame
        -   The second DataFrame made mergeable.
        """
        dataframes = [results_1, results_2]

        for df_i, dataframe in enumerate(dataframes):
            for row_i in dataframe.index:
                for key in dataframe.keys():
                    if isinstance(dataframe[key][row_i], list):
                        dataframe.at[row_i, key] = tuple(dataframe[key][row_i])
            dataframes[df_i] = dataframe

        return dataframes[0], dataframes[1]

    def _restore_results_after_merge(
        self, results: pd.DataFrame
    ) -> pd.DataFrame:
        """Converts a results DataFrame into its original format after merging
        by converting any tuples back to lists.

        PARAMETERS
        ----------
        results : pandas DataFrame
        -   The DataFrame with lists to restore from tuples.

        RETURNS
        -------
        results : pandas DataFrame
        -   The restored DataFrame.
        """
        for row_i in results.index:
            for key in results.keys():
                if isinstance(results[key][row_i], tuple):
                    results.at[row_i, key] = list(results[key][row_i])

        return results

    def _check_missing_before_merge(
        self, results_1: pd.DataFrame, results_2: pd.DataFrame
    ) -> None:
        """Checks that merging pandas DataFrames with the 'merge' method and
        the 'how' parameter set to 'outer' will not introduce new rows into the
        merged results DataFrame, resulting in some rows having NaN values for
        columns not present in their original DataFrame, but present in the
        other DataFrames being merged.
        -   This can occur if the column names which are shared between the
            DataFrames do not have all the same entries between the DataFrames,
            leading to new rows being added to the merged DataFrame.

        PARAMETERS
        ----------
        results_1 : pandas DataFrame
        -   The first DataFrame to check.

        results_2 : pandas DataFrame
        -   The second DataFrame to check.

        RAISES
        ------
        MissingEntryError
        -   Raised if the DataFrames' shared columns do not have values that are
            identical in the other DataFrame, leading to rows being excluded
            from the merged DataFrame.
        """
        if len(results_1.index) == len(results_2.index):
            test_merge = pd.merge(results_1, results_2, how="inner")
            if len(test_merge.index) != len(results_1.index):
                raise EntryLengthError(
                    "Error when trying to merge two sets of results with "
                    "'allow_missing' set to 'False':\nThe shared columns of "
                    "the DataFrames being merged do not have identical values "
                    "in the other DataFrame, leading to "
                    f"{len(test_merge.index)-len(results_1.index)} new row(s) "
                    "being included in the merged DataFrame.\nIf you still "
                    "want to merge these results, set 'allow_missing' to "
                    "'True'."
                )
        else:
            raise EntryLengthError(
                "Error when trying to merge two sets of results with "
                "'allow_missing' set to 'False':\nThere is an unequal number "
                "of channels present in the two sets of results being merged "
                f"({len(results_1.index)} and {len(results_2.index)}). Merging "
                "these results will lead to some attributes of the results "
                "having NaN values.\nIf you still want to merge these results, "
                "set 'allow_missing' to 'True'."
            )

    def _check_keys_before_merge(self, new_results: pd.DataFrame) -> None:
        """Checks that the column names in the DataFrames being merged are not
        identical.

        PARAMETERS
        ----------
        new_results : pandas DataFrame
        -   The new results being added.

        RAISES
        ------
        DuplicateEntryError
        -   Raised if there are no columns that are unique to the DataFrames
            being merged.
        """
        all_repeated = check_non_repeated_vals_lists(
            lists=[self._results.keys().tolist(), new_results.keys().tolist()],
            allow_non_repeated=True,
        )

        if all_repeated:
            raise DuplicateEntryError(
                "Error when trying to merge results:\nThere are no new columns "
                "in the results being added. If you still want to add the "
                "results, use the append methods."
            )

    def merge_from_dict(
        self,
        new_results: dict,
        extract_from_dicts: Optional[dict[list[str]]] = None,
        identical_keys: Optional[list[str]] = None,
        discard_keys: Optional[list[str]] = None,
        allow_missing: bool = False,
    ) -> None:
        """Merges a dictionary of results to the results stored in the
        PostProcess object.
        -   Cannot be called after frequency band results have been computed.

        PARAMETERS
        ----------
        new_results : dict
        -   A dictionary containing results to add.
        -   The entries in the dictionary should be either lists, numpy arrays,
            or dictionaries.
        -   Entries which are dictionaries will have their values treated as
            being identical for all values in the 'results' dictionary, given
            they are extracted from these dictionaries into the results.

        extract_from_dicts : dict[list[str]] | None; default None
        -   The entries of dictionaries within 'results' to include in the
            processing.
        -   Entries which are extracted are treated as being identical for all
            values in the 'results' dictionary.

        identical_keys : list[str] | None; default None
        -   The keys in 'results' which are identical across channels and for
            which only one copy is present.

        discard_keys : list[str] | None; default None
        -   The keys which should be discarded immediately without
            processing.

        allow_missing : bool; default False
        -   Whether or not to allow new rows to be present in the merged results
            with NaN values for columns not shared between the results being
            merged if the shared columns do not have matching values.
        -   I.e. if you want to make sure you are merging results from the same
            channels, set this to False, otherwise results from different
            channels will be merged and any missing information will be set to
            NaN.
        """
        new_results = sort_inputs_results(
            results=new_results,
            extract_from_dicts=extract_from_dicts,
            identical_keys=identical_keys,
            discard_keys=discard_keys,
            verbose=self._verbose,
        )

        new_results = dict_to_df(obj=new_results)

        self._check_keys_before_merge(new_results=new_results)

        current_results, new_results = self._make_results_mergeable(
            results_1=self._results, results_2=new_results
        )

        if not allow_missing:
            self._check_missing_before_merge(
                results_1=current_results, results_2=new_results
            )

        merged_results = pd.merge(current_results, new_results, how="outer")

        self._results = self._restore_results_after_merge(
            results=merged_results
        )

    def merge_from_df(
        self,
        new_results: pd.DataFrame,
        allow_missing: bool = False,
    ) -> None:
        """Merges a dictionary of results to the results stored in the
        PostProcess object.
        -   Cannot be called after frequency band results have been computed.

        PARAMETERS
        ----------
        new_results : pandas DataFrame
        -   A DataFrame containing results to add.

        allow_missing : bool; default False
        -   Whether or not to allow new rows to be present in the merged results
            with NaN values for columns not shared between the results being
            merged if the shared columns do not have matching values.
        -   I.e. if you want to make sure you are merging results from the same
            channels, set this to False, otherwise results from different
            channels will be merged and any missing information will be set to
            NaN.
        """
        self._check_keys_before_merge(new_results=new_results)

        current_results, new_results = self._make_results_mergeable(
            results_1=self._results, results_2=new_results
        )

        if not allow_missing:
            self._check_missing_before_merge(
                results_1=current_results, results_2=new_results
            )

        merged_results = pd.merge(current_results, new_results, how="outer")

        self._results = self._restore_results_after_merge(
            results=merged_results
        )

    def _populate_columns(
        self,
        attributes: list[str],
        fill: Optional[Any] = None,
    ) -> None:
        """Creates placeholder columns to add to the results DataFrame.

        PARAMETERS
        ----------
        attributes : list[str]
        -   Names of the columns to add.

        fill : Any; default None
        -   Placeholder values in the columns.
        """
        for attribute in attributes:
            self._results[attribute] = [deepcopy(fill)] * len(
                self._results.index
            )

    def _refresh_desc_measures(self, keys: list[str]) -> None:
        """Refreshes a list of the descriptive measures (e.g. variability
        measures such as standard error of the mean) that need to be
        re-calculated after any processing steps (e.g. averaging) are applied.

        PARAMETERS
        ----------
        keys : list of str
        -   Attributes of the data which are being processed and which will
            have the number of events the values are derived from added to the
            results.
        """
        present_desc_measures = []
        all_measures = [
            *self._process_measures,
            *self._var_measures,
        ]
        all_desc_measures = [
            *self._desc_process_measures,
            *self._desc_var_measures,
        ]
        for measure in all_measures:
            if measure in all_desc_measures:
                present_desc_measures.append(measure)

        for key in keys:
            present_desc_measures.append(f"{key}_n")

        self._desc_measures = np.unique(present_desc_measures).tolist()

    def _prepare_var_measures(
        self,
        measures: list[str],
        idcs: list[list[int]],
        keys: list[str],
    ) -> None:
        """Prepares for the calculation of variabikity measures, checking that
        the required attributes are present in the data (adding them if not)
        and checking that the requested measure is supported.

        PARAMETERS
        ----------
        measures : list[str]
        -   Types of measures to compute.
        -   Supported types are: 'std' for standard deviation; and 'sem' for
            standard error of the mean.

        idcs : list[list[int]]
        -   List containing sublists of indices for the data that has been
            grouped and processed together.

        keys : list[str]
        -   Attributes of the results to calculate variability measures for.

        RAISES
        ------
        UnavailableProcessingError
        -   Raised if a requested variability measure is not supported.
        """
        supported_measures = ["std", "sem", "ci_"]
        for measure in measures:
            if measure[:3] == "ci_":
                ci_percent = int(measure[3:])
                if ci_percent <= 0 or ci_percent > 100:
                    raise ValueError(
                        "Confidence interval percentages must be > 0 and "
                        "<= 100."
                    )
            elif measure not in supported_measures:
                raise UnavailableProcessingError(
                    "Error when calculating variability measures of the "
                    f"averaged data:\nComputing the measure '{measure}' is "
                    "not supported. Supported measures are: "
                    f"{supported_measures}"
                )

        current_measures = np.unique([*self._var_measures, *measures]).tolist()
        for measure in current_measures:
            if measure[:3] == "ci_":
                suffixes = ["_low", "_high"]
            else:
                suffixes = [""]
            for key in keys:
                for suffix in suffixes:
                    attribute_name = f"{key}_{measure}{suffix}"
                    if attribute_name not in self._results.keys():
                        self._populate_columns(attributes=[attribute_name])
                    else:
                        for group_idcs in idcs:
                            self._results.at[group_idcs[0], attribute_name] = (
                                None
                            )

    def _reset_var_measures(
        self,
        idcs: list[list[int]],
        keys: list[str],
    ) -> None:
        """Resets the values of variability measures for data that has been
        processed to 'None'.

        PARAMETERS
        ----------
        idcs : list[list[int]]
        -   List containing sublists of indices for the data that has been
            grouped and processed together.

        keys : list[str]
        -   Attributes of the results to calculate variability measures for.
        """
        for measure in self._var_measures:
            if measure[:3] == "ci_":
                suffixes = ["_low", "_high"]
            else:
                suffixes = [""]
            for key in keys:
                for suffix in suffixes:
                    for group_idcs in idcs:
                        self._results.at[
                            group_idcs[0], f"{key}_{measure}{suffix}"
                        ] = None

    def _compute_var_measures_over_nodes(
        self,
        measures: list[str],
        idcs: list[list[int]],
        keys: list[str],
    ) -> None:
        """Computes the variability measures over nodes in the results.

        PARAMETERS
        ----------
        measures : list[str]
        -   Types of variabilty measures to compute.
        -   Supported types are: 'std' for standard deviation; and 'sem' for
            standard error of the mean.

        idcs: list[list[int]]
        -   Unique indices of nodes in the results that should be processed.

        keys : list[str]
        -   Attributes of the results to calculate variability measures for.
        """
        self._prepare_var_measures(
            measures=measures,
            idcs=idcs,
            keys=keys,
        )

        for measure in measures:
            for key in keys:
                results_name = f"{key}_{measure}"
                for group_idcs in idcs:
                    if len(group_idcs) > 1:
                        entries = self._results.loc[group_idcs, key].tonumpy()

                        if measure == "std":
                            value = np.std(entries, axis=0).tolist()

                        elif measure == "sem":
                            value = stats.sem(entries, axis=0).tolist()

                        self._results.at[idcs[0], results_name] = value

        self._var_measures = np.unique(
            [*self._var_measures, *measures]
        ).tolist()

        self._refresh_desc_measures(keys)
        if self._verbose:
            print(
                "Computing the following variability measures on attributes "
                "for the processed data over nodes:\n- Variability measure(s): "
                f"{measures}\n- On attribute(s): {keys}\n"
            )

    def _compute_var_measures_within_nodes(
        self,
        measures: list[str],
        idcs: list[int],
        keys: list[str],
        dimension: str,
        axis_of_dimension: np.ndarray,
    ) -> None:
        """Computes the variability measures over dimensions within in each
        node of the results.

        PARAMETERS
        ----------
        measures : list[str]
        -   Types of variabilty measures to compute.
        -   Supported types are: 'std' for standard deviation; and 'sem' for
            standard error of the mean.

        idcs : list[int]
        -   Unique indices of nodes in the results that should be processed.

        keys : list[str]
        -   Attributes of the results to calculate variability measures for.

        dimension : str
        -   The dimension which variability is being computed over.

        axis_of_dimension: numpy ndarray
        -   The axis of the data to compute variability over with shape
            [len('process_entry_idcs') x len('process_keys')], giving an axis
            to compute variability over for each entry being processed.
        """
        self._prepare_var_measures(
            measures=measures,
            idcs=[[idx] for idx in idcs],
            keys=keys,
        )

        append_measures = []
        for measure in measures:
            if measure[:3] == "ci_":
                append_measures.append(f"{measure}_low")
                append_measures.append(f"{measure}_high")
            else:
                append_measures.append(measure)
            for key_i, key in enumerate(keys):
                results_name = f"{key}_{measure}"
                for idx_i, idx in enumerate(idcs):
                    axis = axis_of_dimension[idx_i, key_i]
                    entries = np.array(self._results.loc[idx, key])
                    already_added = False

                    if entries.shape[axis] > 1:
                        if measure == "std":
                            value = np.std(entries, axis=axis).tolist()

                        elif measure == "sem":
                            value = stats.sem(entries, axis=axis).tolist()

                        elif measure[:3] == "ci_":
                            value = self._compute_cis(
                                entries,
                                alpha=int(measure[3:]) / 100,
                                axis=axis,
                            )
                            self._results.at[idx, f"{results_name}_low"] = (
                                value[0]
                            )
                            self._results.at[idx, f"{results_name}_high"] = (
                                value[1]
                            )
                            already_added = True

                        if not already_added:
                            self._results.at[idx, results_name] = value

        self._var_measures = np.unique(
            [*self._var_measures, *append_measures]
        ).tolist()

        self._refresh_desc_measures(keys)
        if self._verbose:
            print(
                "Computing the following variability measures on attributes "
                f"for the processed data within node dimension {dimension}:\n- "
                f"Variability measure(s): {measures}\n- On attribute(s): "
                f"{keys}\n"
            )

    def _compute_cis(
        self, data: np.ndarray, alpha: int, axis: int
    ) -> np.ndarray:
        """Compute confidence intervals

        Parameters
        ----------
        data : numpy ndarray
        -   The data to compute the confidence intervals for.

        alpha : int
        -   The confidence level to compute the intervals for (between 0 and
            1).

        axis : int
        -   The axis of the data to compute the confidence intervals over.

        Returns
        -------
        confidence_intervals : tuple
        -   Lower and upper confidence intervals, respectively.
        """
        return stats.t.interval(
            alpha=alpha,
            df=np.shape(data)[axis] - 1,
            loc=np.mean(data, axis=axis),
            scale=stats.sem(data, axis=axis),
        )

    def _get_eligible_idcs(self, eligible_entries: dict) -> list[int]:
        """Finds the entries with eligible values for processing.

        PARAMETERS
        ----------
        eligible_entries : dict | None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        RETURNS
        -------
        eligible_idcs : list of int
        -   The rows of the results with values eligible for processing.

        RAISES
        ------
        TypeError
        -   Raised if 'eligible_entries' is not a dict and is not None.
        """
        if eligible_entries is None:
            eligible_idcs = np.arange(len(self._results)).tolist()

        else:
            if not isinstance(eligible_entries, dict):
                raise TypeError(
                    "'eligible_entries' must be of type dict, not "
                    f"{type(eligible_entries)}."
                )

            eligible_idcs = []
            for key, values in eligible_entries.items():
                eligible_idcs.append(
                    get_eligible_idcs_list(
                        vals=self._results[key], eligible_vals=values
                    )
                )

            eligible_idcs = list(
                set(eligible_idcs[0]).intersection(*eligible_idcs[1:])
            )

        return eligible_idcs

    def _prepare_for_nongroup_method(
        self, eligible_entries: Union[dict, None]
    ) -> list[int]:
        """Finds what data should be processed, and, if applicable, how that
        data should be grouped in preparation for applying a processing method.

        PARAMETERS
        ----------
        eligible_entries : dict | None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        RETURNS
        -------
        process_idcs : list[int]
        -   Indices of the data that should be processed.
        """
        return self._get_eligible_idcs(eligible_entries)

    def _prepare_for_group_method(
        self,
        method: str,
        over_key: str,
        data_keys: list[str],
        group_keys: list[str],
        eligible_entries: Union[dict, None],
        identical_keys: list[str],
        var_measures: list[str],
    ) -> list[list[int]]:
        """Finds the indices of results that should be grouped and processed
        together.

        PARAMETERS
        ----------
        method : str
        -   Type of processing to apply, e.g. 'average', 'subtract'.

        over_key : str
        -   Name of the attribute in the results to process over.

        data_keys : list[str]
        -   Names of the attributes in the results containing data that should
            be processed, and any variability measures computed on.

        group_keys : [list[str]]
        -   Names of the attributes in the results to use to group results that
            will be processed.

        eligible_entries : dict | None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        identical_keys : list[str]
        -   The names of the attributes in the results that will be checked if
            they are identical across the results being processed. If they are
            not identical, an error will be raised.

        var_measures : list[str]
        -   Names of measures of variability to be computed alongside the
            processing of the results.
        -   Supported measures are: 'std' for standard deviation; and 'sem' for
            standard error of the mean.

        RETURNS
        -------
        group_idcs : list[list[int]]
        -   A list of sublists corresponding to each group containing the
            indices of results to process together.

        RAISES
        ------
        ValueError
        -   Raised if the 'over_key' attribute is present in the 'group_keys' or
            'identical_keys'.
        """
        if group_keys is not None and over_key in group_keys:
            raise ValueError(
                "The attribute being processed over cannot be a member of the "
                "attributes used for grouping results."
            )
        if identical_keys is not None and over_key in identical_keys:
            raise ValueError(
                "The attribute being processed over cannot be a member of the "
                "attributes marked as identical."
            )

        if self._verbose:
            if eligible_entries is None:
                eligible_entries_msg = "all entries"
            else:
                eligible_entries_msg = eligible_entries
            print(
                f"Applying the method {method} for groups of results:\n"
                f"{method}-over attribute: {over_key}\n{method}-over attribute "
                f"with value(s): {eligible_entries_msg}.\nData attribute(s): "
                f"{data_keys}\nGrouping attribute(s): {group_keys}\nCheck "
                f"identical across results attribute(s): {identical_keys}\n"
                f"Variability measure(s): {var_measures}\n"
            )

        self._refresh_desc_measures(data_keys)

        eligible_idcs = self._get_eligible_idcs(eligible_entries)

        combined_vals = combine_col_vals_df(
            dataframe=self._results,
            keys=group_keys,
            idcs=eligible_idcs,
            special_vals={"avg[": "avg_"},
        )
        group_idcs, _ = get_group_idcs(
            vals=combined_vals, replacement_idcs=eligible_idcs
        )
        if identical_keys is not None:
            check_vals_identical_df(
                dataframe=self._results,
                keys=identical_keys,
                idcs=group_idcs,
            )

        return group_idcs

    def average_over_nodes(
        self,
        over_key: str,
        data_keys: list[str],
        group_keys: list[str],
        eligible_entries: Union[dict, None] = None,
        identical_keys: Union[list[str], None] = None,
        ignore_nan: bool = True,
        var_measures: Union[list[str], None] = None,
    ) -> None:
        """Averages results over nodes in the data.

        PARAMETERS
        ----------
        over_key : str
        -   Name of the attribute in the results to average over.

        data_keys : list[str]
        -   Names of the attributes in the results containing data that should
            be averaged, and any variability measures computed on.

        group_keys : [list[str]]
        -   Names of the attributes in the results to use to group results that
            will be averaged over.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        identical_keys : list[str] | None; default None
        -   The names of the attributes in the results that will be checked if
            they are identical across the results being averaged. If they are
            not identical, an error will be raised.

        ignore_nan : bool; default True
        -   Whether or not to ignore NaN values when averaging. If True, numpy's
            nanmean method is used to compute the average, else numpy's mean
            method is used.

        var_measures : list[str] | None; default None
        -   Names of measures of variability to be computed alongside the
            averaging of the results.
        -   Supported measures are: 'std' for standard deviation; and 'sem' for
            standard error of the mean.
        """
        group_idcs = self._prepare_for_group_method(
            method="average_over_nodes",
            over_key=over_key,
            data_keys=data_keys,
            eligible_entries=eligible_entries,
            group_keys=group_keys,
            identical_keys=identical_keys,
            var_measures=var_measures,
        )

        if self._var_measures:
            self._reset_var_measures(idcs=group_idcs, keys=data_keys)

        if var_measures:
            self._compute_var_measures_over_nodes(
                measures=var_measures,
                idcs=group_idcs,
                keys=data_keys,
            )

        self._compute_average_over_nodes(
            idcs=group_idcs,
            over_key=over_key,
            average_keys=data_keys,
            group_keys=group_keys,
            ignore_nan=ignore_nan,
        )

    def average_within_nodes(
        self,
        data_keys: list[str],
        average_dimension: str,
        eligible_entries: Union[dict, None] = None,
        ignore_nan: bool = True,
        var_measures: Union[list[str], None] = None,
    ) -> None:
        """Averages results of nodes separately over a specific dimension.

        PARAMETERS
        ----------
        data_keys : list[str]
        -   Names of the attributes in the results containing data that should
            be averaged, and any variability measures computed on.

        average_dimension : str
        -   The dimension of the attributes in 'data_keys' to average over.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        ignore_nan : bool; default True
        -   Whether or not to ignore NaN values when averaging. If True, numpy's
            nanmean method is used to compute the average, else numpy's mean
            method is used.

        var_measures : list[str] | None; default None
        -   Names of measures of variability to be computed alongside the
            averaging of the results.
        -   Supported measures are: 'std' for standard deviation; and 'sem' for
            standard error of the mean.
        """
        if not isinstance(average_dimension, str):
            raise TypeError(
                "The dimension to average over should be of type str, not "
                f"{type(average_dimension)}."
            )

        if self._verbose:
            if eligible_entries is None:
                eligible_entries_msg = "all entries"
            print(
                f"Averaging results over the {average_dimension} dimension.\n- "
                f"Eligible entries: {eligible_entries_msg}\n"
            )

        process_idcs = self._prepare_for_nongroup_method(
            eligible_entries=eligible_entries
        )
        self._refresh_desc_measures(data_keys)

        process_axis = self._find_dimension_axis(
            idcs=process_idcs, keys=data_keys, dimension=average_dimension
        )

        if self._var_measures:
            self._reset_var_measures(keys=data_keys, idcs=process_idcs)

        if var_measures:
            self._compute_var_measures_within_nodes(
                measures=var_measures,
                idcs=process_idcs,
                keys=data_keys,
                dimension=average_dimension,
                axis_of_dimension=process_axis,
            )

        self._compute_average_within_nodes(
            idcs=process_idcs,
            keys=data_keys,
            ignore_nan=ignore_nan,
            dimension=average_dimension,
            axis_of_dimension=process_axis,
        )

        check_keys = [
            average_dimension,
            *[f"{key}_dimensions" for key in data_keys],
        ]
        for key in check_keys:
            if not np.count_nonzero(
                np.array(self._results[key].tolist()) is not None
            ):
                self._results = self._results.drop(columns=key)

    def _find_dimension_axis(
        self, idcs: list[int], keys: list[str], dimension: str
    ) -> np.ndarray:
        """Finds the axis position corresponding to a dimension of the results.

        PARAMETERS
        ----------
        idcs : list of int
        -   The indices in the results to find the axis of interest's position.

        keys : list of str
        -   The attributes in the data whose dimensions should be analysed.

        dimension : str
        -   The name of the dimension whose axis position should be found.

        RETURNS
        -------
        axis : numpy ndarray
        -   A 2D array containing the position of the dimension for each of the
            indices with shape [len('idcs') x len('data_keys')].

        RAISES
        ------
        ValueError
        -   Raised if 'dimension' is not present in the dimensions for at least
            one of 'data_keys'.
        """
        axis = np.zeros((len(idcs), len(keys)), dtype=int)
        for idx_i, idx in enumerate(idcs):
            assert isinstance(idx, int), (
                "Finding the axis of interest is only supported for "
                "non-grouped results."
            )
            for key_i, key in enumerate(keys):
                try:
                    axis[idx_i, key_i] = self._results.at[
                        idx, f"{key}_dimensions"
                    ].index(dimension)
                except ValueError:
                    ValueError(
                        f"The dimension '{dimension}' is not present in the "
                        f"'{key}' results in row {idx}."
                    )

        return axis

    def _compute_average_over_nodes(
        self,
        idcs: list[list[int]],
        over_key: str,
        average_keys: list[str],
        group_keys: list[str],
        ignore_nan: bool,
    ) -> None:
        """Computes the average results over nodes.

        PARAMETERS
        ----------
        idcs : list[list[int]]
        -   Unique indices of nodes in the results that should be processed.

        over_key : str
        -   The attribute of the results to average over.

        average_keys : list[str]
        -   Attributes of the results to average.

        group_keys : list[str]
        -   Attributes of the results whose entries should be changed to reflect
            which entries have been averaged over.

        ignore_nan : bool
        -   Whether or not to ignore NaN values when averaging. If True, numpy's
            nanmean method is used to compute the average, else numpy's mean
            method is used.
        """
        if ignore_nan:
            avg_method = np.nanmean
        else:
            avg_method = np.mean

        drop_idcs = []
        for group_idcs in idcs:
            if len(group_idcs) > 1:
                for key in average_keys:
                    entries = np.array(
                        self._results.loc[group_idcs, key].tolist()
                    )

                    self._results.at[group_idcs[0], key] = avg_method(
                        entries, axis=0
                    ).tolist()

                    self._results.at[group_idcs[0], f"{key}_n"] = len(
                        group_idcs
                    )

                drop_idcs.extend(group_idcs[1:])
                for key in group_keys:
                    self._set_averaged_key_value(
                        idcs=group_idcs, key=key, combine_single_values=False
                    )

            self._set_averaged_key_value(
                idcs=group_idcs, key=over_key, combine_single_values=True
            )

        self._results = self._results.drop(index=drop_idcs)
        self._results = self._results.reset_index(drop=True)

    def _compute_average_within_nodes(
        self,
        idcs: list[int],
        keys: list[str],
        ignore_nan: bool,
        dimension: str,
        axis_of_dimension: np.ndarray,
    ) -> None:
        """Computes the average results across a dimension within each node.

        PARAMETERS
        ----------
        indices : list[int]
        -   Unique indices of nodes in the results that should be processed.

        keys : list[str]
        -   Attributes of the results to average.

        ignore_nan : bool
        -   Whether or not to ignore NaN values when averaging. If True, numpy's
            nanmean method is used to compute the average, else numpy's mean
            method is used.

        dimension : str
        -   The name of the dimension being averaged out.

        axis_of_dimension : numpy ndarray
        -   The axis of the data to average with shape
            [len('average_entry_idcs') x len('data_keys')], giving an axis to
            average across for each entry being processed.
        """
        if ignore_nan:
            avg_method = np.nanmean
        else:
            avg_method = np.mean

        for idx_i, idx in enumerate(idcs):
            for key_i, key in enumerate(keys):
                axis = axis_of_dimension[idx_i, key_i]
                entry = np.array(self._results.loc[idx, key])

                if entry.shape[axis] > 1:
                    self._results.at[idx, key] = avg_method(
                        entry, axis=axis
                    ).tolist()
                else:
                    self._results.at[idx, key] = np.squeeze(
                        entry, axis
                    ).tolist()

                self._results.at[idx, f"{key}_dimensions"].pop(axis)
                self._results.at[idx, f"{key}_n"] = int(entry.shape[axis])

        self._results.loc[idcs, dimension] = [None for _ in idcs]

    def _set_averaged_key_value(
        self, idcs: list[int], key: str, combine_single_values: bool = True
    ) -> None:
        """Sets the value for attributes in the nodes of the results being
        averaged together based on the unique values of the attributes at these
        nodes.

        E.g. averaging over two nodes of an attribute with the values '1' and
        '2', respectively, would be transformed into: 'avg[1, 2]'. Equally,
        averaging over three nodes of an attribute with the values '1', '1', and
        '2' would be transformed into: 'avg[1, 2]', as only the unique values
        are accounted for.

        PARAMETERS
        ----------
        idcs : list[int]
        -   Indices of nodes in the results being averaged together.

        key : str
        -   Name of the attribute in the data.

        combine_single_values : bool; default True
        -   Whether or not to change the key value based on the attribute of the
            nodes in the form 'avg[values]' if the unique values of the nodes
            have the same entry.
        """
        if combine_single_values:
            min_length = 1
        else:
            min_length = 2

        entries = np.unique(
            [str(self._results[key][idx]) for idx in idcs]
        ).tolist()
        if len(entries) >= min_length:
            value = "avg["
            for entry in entries:
                value += f"{entry}, "
            value = value[:-2] + "]"
            self._results.at[idcs[0], key] = value

    def subtract(
        self,
        over_key: str,
        data_keys: list[str],
        group_keys: list[str],
        eligible_entries: Union[dict, None] = None,
        identical_keys: Union[list[str], None] = None,
    ) -> None:
        """Subtracts results.

        PARAMETERS
        ----------
        over_key : str
        -   Name of the attribute in the results to subtract.

        data_keys : list[str]
        -   Names of the attributes in the results containing data that should
            be subtracted.

        group_keys : [list[str]]
        -   Names of the attributes in the results to use to group results that
            will be subtracted.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        identical_keys : list[str] | None; default None
        -   The names of the attributes in the results that will be checked if
            they are identical across the results being subtracted. If they are
            not identical, an error will be raised.
        """
        group_idcs = self._prepare_for_group_method(
            method="subtract",
            over_key=over_key,
            data_keys=data_keys,
            eligible_entries=eligible_entries,
            group_keys=group_keys,
            identical_keys=identical_keys,
            var_measures=None,
        )

        altered_group_idcs = self._compute_subtraction(
            subtract_entry_idcs=group_idcs,
            over_key=over_key,
            subtract_keys=data_keys,
        )

        if altered_group_idcs:
            self._reset_var_measures(
                idcs=altered_group_idcs,
                keys=data_keys,
            )

    def _compute_subtraction(
        self,
        subtract_entry_idcs: list[list[int]],
        over_key: str,
        subtract_keys: list[str],
    ) -> list[list[int]]:
        """Subtracts results over the unique node indices.

        PARAMETERS
        ----------
        subtract_entry_indices : list[list[int]]
        -   Unique indices of nodes in the results that should be processed.

        over_key : str
        -   The attribute of the results to subtract.

        subtract_keys : list[str]
        -   Attributes of the results to subtract.

        RETURNS
        -------
        altered_group_idcs : list[list[int]]
        -   Indices of nodes in the results that have been processed (i.e. two
            sets of results were present to subtract from one another).
        """
        drop_idcs = []
        altered_group_idcs = []
        for idcs in subtract_entry_idcs:
            if len(idcs) == 2:
                for key in subtract_keys:
                    entries = self._results.loc[idcs, key].tonumpy()
                    self._results.at[idcs[0], key] = np.subtract(
                        entries[0], entries[1]
                    ).tolist()
                    self._results.at[idcs[0], f"{key}_n"] = None
                drop_idcs.append(idcs[1])
                altered_group_idcs.append(idcs)
                self._set_subtracted_key_value(key=over_key, idcs=idcs)
            elif len(idcs) == 1:
                if self._verbose:
                    print(
                        f"Only one '{over_key}' value is present in a group,  "
                        "so no subtraction will be performed."
                    )
            else:
                raise ValueError(
                    "Only two values can be processed in the subtraction "
                    "method (i.e. one value can be subtracted from another), "
                    f"however {len(idcs)} values are being processed."
                )

        self._results = self._results.drop(index=drop_idcs)
        self._results = self._results.reset_index(drop=True)

        return altered_group_idcs

    def _set_subtracted_key_value(self, key: str, idcs: list[int]) -> None:
        """Sets the value for attributes in the nodes of the results being
        subtracted based on the unique values of the attributes at these nodes.

        PARAMETERS
        ----------
        key : str
        -   Name of the attribute in the data.

        idcs : list[int]
        -   Indices of nodes in the results being subtracted.

        NOTES
        -----
        -   E.g. subtracting values belonging to med-Off and -On conditions
            would give med-(Off - On).
        """
        entries = np.unique([self._results[key][idx] for idx in idcs]).tolist()
        self._results.at[idcs[0], key] = f"({entries[0]} - {entries[1]})"

    def log10(
        self, data_keys: list[str], eligible_entries: Union[dict, None] = None
    ) -> None:
        """Log transforms results with base 10.

        PARAMETERS
        ----------
        data_keys : list[str]
        -   Names of the attributes in the results containing data that should
            be transformed.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.
        """
        if self._verbose:
            if eligible_entries is None:
                eligible_entries_msg = "all entries"
            print(
                "Log transforming the results with base 10.\n- Eligible "
                f"entries: {eligible_entries_msg}\n"
            )

        method = "log10"

        process_idcs = self._prepare_for_nongroup_method(
            eligible_entries=eligible_entries
        )

        self._apply_transformation(
            method=method, idcs=process_idcs, keys=data_keys
        )
        self._transform_var_measures(
            method=method,
            idcs=process_idcs,
            keys=data_keys,
        )

    def absolute(
        self, data_keys: list[str], eligible_entries: Union[dict, None] = None
    ) -> None:
        """Takes the absolute value of results.

        PARAMETERS
        ----------
        data_keys : list[str]
        -   Names of the attributes in the results containing data that should
            be transformed.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.
        """
        if self._verbose:
            if eligible_entries is None:
                eligible_entries_msg = "all entries"
            print(
                "Taking the absolute value of the results.\n- Eligible "
                f"entries: {eligible_entries_msg}\n"
            )

        method = "absolute"

        process_idcs = self._prepare_for_nongroup_method(
            eligible_entries=eligible_entries
        )

        self._apply_transformation(
            method=method, idcs=process_idcs, keys=data_keys
        )

        self._transform_var_measures(
            method=method,
            idcs=process_idcs,
            keys=data_keys,
        )

    def _apply_transformation(
        self, method: str, idcs: list[int], keys: list[str]
    ) -> None:
        """Transforms data of the specified indices and keys.

        PARAMETERS
        ----------
        method : str
        -   The transformation to perform. Accepted values are: "log" for log
            transformation; and "absolute" for taking the absolute values.

        idcs : list[int]
        -   Indices of the data to transform for the keys in 'keys'.

        keys : list[str]
        -   Keys of the data to transform for the indices in 'idcs'.
        """
        transformation = self._get_transformation_function(method)

        for key in keys:
            entries = np.array(self._results.loc[idcs, key].to_list())
            try:
                self._results.loc[idcs, key] = transformation(entries).tolist()
            except TypeError:
                raise TypeError(
                    f"Unable to apply the transformation '{method}' to the "
                    f"'{key}' results, possibly because they contain None "
                    "values."
                ) from None

    def _transform_var_measures(
        self,
        method: str,
        idcs: list[int],
        keys: list[str],
    ) -> None:
        """Applies a transformation to the variability measures of the data.

        PARAMETERS
        ----------
        method : str
        -   The transformation to perform. Accepted values are: "log10" for log
            transformation with base 10; and "absolute", in which case it does
            not make sense to take the absolute values, which are instead set to
            None.

        idcs : list[int]
        -   Indices of the data to transform for the keys in 'keys'.

        keys : list[str]
        -   Keys of the data to transform for the indices in 'idcs'.
        """
        transformation = self._get_transformation_function(method)

        var_attributes = [
            f"{key}_{measure}" for key in keys for measure in self._var_measures
        ]
        for var_attribute in var_attributes:
            if method == "absolute":
                self._results.loc[idcs, var_attribute] = [None for _ in idcs]

            else:
                entries = self._results.loc[idcs, var_attribute].tolist()
                process_idcs = [
                    idx
                    for idx, value in enumerate(entries)
                    if value is not None
                ]
                entries = np.array(entries[process_idcs])

                try:
                    self._results.loc[process_idcs, var_attribute] = (
                        transformation(entries).tolist()
                    )
                except TypeError:
                    raise TypeError(
                        f"Unable to apply the transformation '{method}' to the "
                        f"'{var_attribute}' results, possibly because they "
                        "contain None values."
                    ) from None

    def _get_transformation_function(self, method: str) -> Any:
        """Gets the function to use for transforming values.

        PARAMETERS
        ----------
        method : str
        -   How to transform the results. Accepted methods are: "log10" for
            taking the log values with base 10; and "abs" for taking the
            absolute values.

        RETURNS
        -------
        function : Any
        -   The function to use to transform the results.

        RAISES
        ------
        ValueError
        -   Raised if 'method' is not supported.
        """
        method_function_mapping = {"log10": np.log10, "absolute": np.abs}

        try:
            return method_function_mapping[method]
        except KeyError as unsupported_method_error_msg:
            unsupported_method_error_msg = (
                f"Finding the {method} of results is not supported. Accepted "
                "forms of indexing are "
                f"{list[method_function_mapping.keys()]}."
            )
            raise ValueError(unsupported_method_error_msg) from None

    def find_value(
        self,
        value_method: str,
        data_keys: list[str],
        find_in_dimension: str,
        eligible_entries: Union[dict, None] = None,
    ) -> None:
        """Finds a value in a dimension of the results.

        PARAMETERS
        ----------
        value_method : str
        -   The type of value of interest. Accepted methods are: "max" for the
            maximum value; and "min" for the minimum value.

        data_keys : list[str]
        -   Names of the attributes in the results containing data whose values
            of interest should be found.

        find_in_dimension : str
        -   Name of the dimension of the attributes in 'data_keys' in which the
            values of interest should be found. E.g. if "frequencies", values of
            interest would be found in the frequency dimension of the results.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        RAISES
        ------
        TypeError
        -   Raised if 'find_in_dimension' is not a str.
        """
        if self._verbose:
            if eligible_entries is None:
                eligible_entries_msg = "all entries"
            print(
                f"Finding the {value_method} values in the {find_in_dimension} "
                f"of the results in {data_keys}.\n- Eligible entries: "
                f"{eligible_entries_msg}\n"
            )

        if not isinstance(find_in_dimension, str):
            raise TypeError(
                "'find_in_dimension' must be a str, not "
                f"{type(find_in_dimension)}."
            )

        find_index_of_values = False

        process_idcs = self._prepare_for_nongroup_method(
            eligible_entries=eligible_entries
        )

        result_names = self._prepare_values_of_interest_results(
            value_method=value_method,
            data_keys=data_keys,
            find_in_dimension=find_in_dimension,
            find_index_of_values=find_index_of_values,
        )

        find_in_axis = self._find_dimension_axis(
            idcs=process_idcs, keys=data_keys, dimension=find_in_dimension
        )

        self._find_values_of_interest(
            value_method=value_method,
            process_idcs=process_idcs,
            data_keys=data_keys,
            find_in_axis=find_in_axis,
            result_names=result_names,
        )

    def find_index_of_value(
        self,
        value_method: str,
        data_keys: list[str],
        find_in_dimension: str,
        eligible_entries: Union[dict, None] = None,
    ) -> None:
        """Finds the index of a value in a dimension of the results.

        PARAMETERS
        ----------
        value_method : str
        -   The type of value whose index should be found. Accepted methods are:
            "max" for the maximum value; and "min" for the minimum value.

        data_keys : list[str]
        -   Names of the attributes in the results containing data whose values
            of interest should be found.

        find_in_dimension : str
        -   Name of the dimension of the attributes in 'data_keys' in which the
            values of interest, and then the index of these values, should be
            found. E.g. if "frequencies", values of interest would be found in
            the frequency dimension of the results.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        RAISES
        ------
        TypeError
        -   Raised if 'find_in_dimension' is not a str.
        """
        if self._verbose:
            if eligible_entries is None:
                eligible_entries_msg = "all entries"
            print(
                f"Finding the index of the {value_method} values in the "
                f"{find_in_dimension} of the results in {data_keys}.\n- "
                f"Eligible entries: {eligible_entries_msg}\n"
            )

        if not isinstance(find_in_dimension, str):
            raise TypeError(
                "'find_in_dimension' must be a str, not "
                f"{type(find_in_dimension)}."
            )

        find_index_of_values = True

        process_idcs = self._prepare_for_nongroup_method(
            eligible_entries=eligible_entries
        )

        result_names = self._prepare_values_of_interest_results(
            value_method=value_method,
            data_keys=data_keys,
            find_in_dimension=find_in_dimension,
            find_index_of_values=find_index_of_values,
        )

        find_in_axis = self._find_dimension_axis(
            idcs=process_idcs, keys=data_keys, dimension=find_in_dimension
        )

        self._find_index_of_values_of_interest(
            value_method=value_method,
            process_idcs=process_idcs,
            data_keys=data_keys,
            find_in_dimension=find_in_dimension,
            find_in_axis=find_in_axis,
            result_names=result_names,
        )

    def _prepare_values_of_interest_results(
        self,
        value_method: str,
        data_keys: list[str],
        find_in_dimension: str,
        find_index_of_values: bool,
    ) -> list[str]:
        """Get the name of the attributes under which the results for the values
        of interest/the indices of these values should be stored, creating
        columns in the results for these attributes if they do not already
        exist.

        PARAMETERS
        ----------
        value_method : str
        -   The type of value which should be found. Accepted methods are: "max"
            for the maximum value; and "min" for the minimum value.

        data_keys : list of str
        -   Names of the attributes in the results containing data whose maximum
            values should be found.

        find_in_dimension : str
        -   Name of the dimension of the attributes in 'data_keys' whose values
            should found in. E.g. if "frequencies", values of interest would be
            found in the frequency dimension of the results.

        find_index_of_values : bool
        -   Whether or not to find the index of the values of interest in
            'find_in_dimension'. E.g. if True, and 'find_in_dimension' ==
            "frequencies", the frequency of the value of interest would be saved
            in the results rather than the value of interest itself.

        RETURNS
        -------
        result_names : list of str
        -   Names of the attributes under which the values of interest results
            should be stored, corresponding to the entries of 'data_keys'.
        """
        result_names = self._get_values_of_interest_result_names(
            value_method=value_method,
            data_keys=data_keys,
            find_in_dimension=find_in_dimension,
            find_index_of_values=find_index_of_values,
        )

        for name in result_names:
            if name not in self._results.keys():
                self._populate_columns([name])

        return result_names

    def _get_values_of_interest_result_names(
        self,
        value_method: str,
        data_keys: list[str],
        find_in_dimension: str,
        find_index_of_values: bool,
    ) -> list[str]:
        """Get the name of the attributes under which the results for the values
        of interest/the indices of these values should be stored.

        PARAMETERS
        ----------
        value_method : str
        -   The type of value which should be found. Accepted methods are: "max"
            for the maximum value; and "min" for the minimum value.

        data_keys : list of str
        -   Names of the attributes in the results containing data whose maximum
            values should be found.

        find_in_dimension : str
        -   Name of the dimension of the attributes in 'data_keys' whose values
            should found in. E.g. if "frequencies", values of interest would be
            found in the frequency dimension of the results.

        find_index_of_values : bool
        -   Whether or not to find the index of the values of interest in
            'find_in_dimension'. E.g. if True, and 'find_in_dimension' ==
            "frequencies", the frequency of the value of interest would be saved
            in the results rather than the value of interest itself.

        RETURNS
        -------
        result_names : list of str
        -   Names of the attributes under which the values of interest results
            should be stored, corresponding to the entries of 'data_keys'.
        """
        if find_index_of_values:
            return [
                f"{key}_index_of_{value_method}({find_in_dimension})"
                for key in data_keys
            ]
        return [
            f"{key}_{value_method}({find_in_dimension})" for key in data_keys
        ]

    def _find_values_of_interest(
        self,
        value_method: str,
        process_idcs: list[int],
        data_keys: list[str],
        find_in_axis: np.ndarray,
        result_names: list[str],
    ) -> None:
        """Finds a value in a dimension of the results.

        PARAMETERS
        ----------
        value_method : str
        -   The type of value which should be found. Accepted methods are: "max"
            for the maximum value; and "min" for the minimum value.

        process_idcs : list[int]
        -   Indices of the rows of results to process.

        data_keys : list[str]
        -   Names of the attributes in the results containing data whose maximum
            values should be found.

        find_in_dimension : str
        -   Name of the dimension of the attributes in 'data_keys' in which the
            values of interest, and then the index of these values, should be
            found. E.g. if "frequencies", values of interest would be found in
            the frequency dimension of the results.

        find_in_axis : numpy ndarray
        -   The axis of the data in which to find the values of interest, with
            shape [len('process_idcs') x len('data_keys')], giving an axis to
            index the value of interest in for each entry being processed.

        result_names : list[str]
        -   Names of the attributes under which the results should be stored,
            corresponding to the values of interest for the entries of
            'data_keys'.
        """
        find_value_function = self._get_find_value_function(value_method)

        for idx_i, idx in enumerate(process_idcs):
            for key_i, key in enumerate(data_keys):
                axis = find_in_axis[idx_i, key_i]

                entry = np.array(self._results.at[idx, key])

                self._results.at[idx, result_names[key_i]] = (
                    find_value_function(entry, axis=axis)
                )

    def _find_index_of_values_of_interest(
        self,
        value_method: str,
        process_idcs: list[int],
        data_keys: list[str],
        find_in_dimension: str,
        find_in_axis: np.ndarray,
        result_names: list[str],
    ) -> None:
        """Finds the index of a value in a dimension of the results.

        PARAMETERS
        ----------
        value_method : str
        -   The type of value which should be found. Accepted methods are: "max"
            for the maximum value; and "min" for the minimum value.

        process_idcs : list[int]
        -   Indices of the rows of results to process.

        data_keys : list[str]
        -   Names of the attributes in the results containing data whose maximum
            values should be found.

        find_in_dimension : str
        -   Name of the dimension of the attributes in 'data_keys' in which the
            values of interest, and then the index of these values, should be
            found. E.g. if "frequencies", values of interest would be found in
            the frequency dimension of the results.

        find_in_axis : numpy ndarray
        -   The axis of the data in which to find the values of interest, with
            shape [len('process_idcs') x len('data_keys')], giving an axis to
            index the value of interest in for each entry being processed.

        result_names : list[str]
        -   Names of the attributes under which the results should be stored,
            corresponding to the values of interest for the entries of
            'data_keys'.
        """
        find_value_function = self._get_find_value_function(value_method)

        for idx_i, idx in enumerate(process_idcs):
            for key_i, key in enumerate(data_keys):
                axis = find_in_axis[idx_i, key_i]

                entry = np.array(self._results.at[idx, key])
                values_of_interest = find_value_function(entry, axis=axis)

                data_index = self._find_index_in_nd_array(
                    values=entry,
                    values_of_interest=values_of_interest,
                    axis_of_interest=axis,
                    index_from=np.array(
                        self._results.loc[idx, find_in_dimension]
                    ),
                )

                self._results.at[idx, result_names[key_i]] = data_index.tolist()

    def _find_index_in_nd_array(
        self,
        values: np.ndarray,
        values_of_interest: np.ndarray,
        axis_of_interest: int,
        index_from: np.ndarray | None = None,
    ) -> np.ndarray:
        """Finds the index of values in an N-dimensional array.

        Given an array of values and a specific set of values you want to
        find, the (first) indices at which these desired values occur will be
        taken and is optionally converted into an interpretable value according
        to a set of values to index from, e.g. a set of frequencies or
        timepoints.

        PARAMETERS
        ----------
        values : numpy ndarray
        -   The array in which 'values_of_interest' should be found.

        values_of_interest : numpy ndarray
        -   The values of interest to find in 'values'.

        axis_of_interest : int
        -   The axis of 'values' in which 'values_of_interest' should be found.

        index_from : numpy ndarray | None; default None
        -   The values which the indices of 'values_of_interest' correspond to.
            If None, the indices themselves are returned without being converted
            to the corresponding values in 'index_from'.

        RETURNS
        -------
        indices : numpy ndarray
        -   The indices of 'values_of_interest' in 'values', taken from
            'index_from'.

        RAISES
        ------
        ValueError
        -   Raised if not all 'values_of_interest' entries are present in
            'values'.
        """
        indices = np.zeros_like(values_of_interest)

        search_values = np.moveaxis(values, axis_of_interest, -1)

        if isinstance(search_values[0], np.ndarray):
            for array_idx, array in enumerate(search_values):
                indices[array_idx] = self._find_index_in_nd_array(
                    values=array,
                    values_of_interest=values_of_interest[array_idx],
                    axis_of_interest=array.ndim - 1,
                    index_from=index_from,
                )

        else:
            try:
                first_index = np.nonzero(values == values_of_interest)[0][0]
            except IndexError as missing_values_of_interest_msg:
                missing_values_of_interest_msg = (
                    "Not all 'values_of_interest' entries are present in "
                    "'values'."
                )
                raise ValueError(missing_values_of_interest_msg) from None

            if index_from is not None:
                indices = index_from[first_index]
            else:
                indices = first_index

        if index_from is None:
            indices = np.array(indices, dtype=int)
        else:
            indices = np.array(indices, dtype=index_from.dtype)

        return indices

    def _get_find_value_function(self, method: str) -> Any:
        """Gets the function to use for finding the value of interest.

        PARAMETERS
        ----------
        method : str
        -   The type of value whose index should be found. Accepted methods are:
            "max" for the maximum value; and "min" for the minimum value.

        RETURNS
        -------
        function : Any
        -   The function to use to find the value of interest in the results.

        RAISES
        ------
        ValueError
        -   Raised if 'index_method' is not supported.
        """
        method_function_mapping = {"max": np.max, "min": np.min}

        try:
            return method_function_mapping[method]
        except KeyError as unsupported_method_error_msg:
            unsupported_method_error_msg = (
                f"Finding the {method} of results is not supported. Accepted "
                "forms of indexing are "
                f"{list[method_function_mapping.keys()]}."
            )
            raise ValueError(unsupported_method_error_msg) from None

    def isolate_bands(
        self,
        data_keys: list[str],
        isolate_dimension: str,
        bands: dict,
        eligible_entries: Union[dict, None] = None,
    ) -> None:
        """Isolates data from bands (i.e. portions) of the results (e.g
        frequency bands) into a new DataFrame.

        PARAMETERS
        ----------
        data_keys : list[str]
        -   Names of the attributes in the results containing data that should
            be isolated into bands.

        isolate_dimension : str
        -   Name of the attribute in the results which should be used to
            isolate the results into bands, e.g. "frequencies", "timepoints".

        bands : dict
        -   Dictionary where the keys are the labels of the bands in which the
            results should be isolated, and the values the entries in
            'band_index_key' which should be used to isolate the results into
            bands.
        -   The values for each key should consist of a lower and upper bound
            from which results should be taken, respectively.
        -   E.g. {"beta": [12, 35]} with a 'band_index_key' corresponding to a
            set of frequencies in Hz would mean a band with the label 'beta'
            would be created, and the values from 12-35 Hz stored.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.
        """
        if not isinstance(isolate_dimension, str):
            raise TypeError(
                "'isolate_dimension' must be a str, not "
                f"{type(isolate_dimension)}."
            )

        if self._verbose:
            if eligible_entries is None:
                eligible_entries_msg = "all entries"
            else:
                eligible_entries_msg = eligible_entries
            print(
                f"Finding the {isolate_dimension} bands of the results in "
                f"{data_keys}.\n- Bands: {bands}\n- Eligible entries: "
                f"{eligible_entries_msg}\n"
            )

        process_idcs = self._prepare_for_nongroup_method(
            eligible_entries=eligible_entries
        )

        self._prepare_band_results(
            bands=bands,
            data_keys=data_keys,
            isolate_dimension=isolate_dimension,
            process_idcs=process_idcs,
        )

        self._get_band_results(
            bands=bands,
            process_idcs=process_idcs,
            isolate_dimension=isolate_dimension,
            data_keys=data_keys,
        )

        self._results = deepcopy(self._band_results)
        self._band_results = None

    def _prepare_band_results(
        self,
        bands: dict,
        data_keys: list[str],
        isolate_dimension: str,
        process_idcs: list[int],
    ) -> None:
        """Checks that the inputs for isolating bands of results are
        appropriate, finds the indices of the band bounds in the results, and
        creates a new DataFrame in which the band results will be stored.

        PARAMETERS
        ----------
        bands : dict

        data_keys : list[str]

        isolate_dimension : str

        process_idcs : list[int]
        """
        for name, bounds in bands.items():
            if len(bounds) != 2:
                raise EntryLengthError(
                    "Error when trying to compute the band results:\nThe "
                    f"band '{name}' does not have the required lower- and "
                    f"upper-bound values (is instead {bounds})."
                )

        self._initialise_band_results_dict(
            bands=bands,
            data_keys=data_keys,
            isolate_dimension=isolate_dimension,
            process_idcs=process_idcs,
        )

    def _initialise_band_results_dict(
        self,
        bands: dict,
        data_keys: list[str],
        isolate_dimension: str,
        process_idcs: list[int],
    ) -> None:
        """Creates a dictionary for the band results to be stored in.

        PARAMETERS
        ----------
        bands : dict

        data_keys : list[str]

        isolate_dimension : str

        process_idcs : list[int]
        """
        band_results = self._fill_band_results_from_existing(
            process_idcs=process_idcs,
            n_bands=len(bands.keys()),
            isolate_dimension=isolate_dimension,
            data_keys=data_keys,
        )

        band_results.update(
            self._fill_band_results_info(
                isolate_dimension=isolate_dimension,
                bands=bands,
                process_idcs=process_idcs,
            )
        )

        n_entries = len(bands.keys()) * len(process_idcs)
        band_results.update(
            {key: [None for _ in range(n_entries)] for key in data_keys}
        )

        self._band_results = dict_to_df(band_results)

    def _fill_band_results_from_existing(
        self,
        process_idcs: list[int],
        n_bands: int,
        isolate_dimension: str,
        data_keys: list[str],
    ) -> dict:
        """Creates a dictionary for the band results based on the existing
        results, containing information about the conditions from which the
        results belong (e.g. subject ID, experimental condition states, channel
        properties, etc...).

        PARAMETERS
        ----------
        process_idcs : list[int]

        n_bands : int
        -   The number of bands being created.

        isolate_dimension : str

        data_keys : list[str]

        RETURNS
        -------
        band_results : dict
        -   Dictionary for the band results, containing information from the
            existing results.
        """
        band_results_keys = [
            key
            for key in self._results.keys()
            if key not in self._var_measures and key not in data_keys
        ]
        band_results = {}
        for key in band_results_keys:
            if key != isolate_dimension:
                band_results[key] = []
                for idx in process_idcs:
                    band_results[key].extend(
                        [
                            deepcopy(self._results[key][idx])
                            for _ in range(n_bands)
                        ]
                    )
            else:
                band_results[key] = [
                    None for _ in range(len(process_idcs) * n_bands)
                ]

        return band_results

    def _fill_band_results_info(
        self, isolate_dimension: str, bands: dict, process_idcs: list[int]
    ) -> dict:
        """Creates a dictionary for the band results containing information
        about the bands being isolated, including the band labels, and the lower
        and upper bounds of 'isolate_dimension'.

        PARAMETERS
        ----------
        isolate_dimension : str

        bands : dict

        process_idcs : list[int]

        RETURNS
        -------
        band_results_info : dict
        -   Dictionary for the band results, containing information about the
            bands being isolated.
        """
        n_entries = len(process_idcs)
        band_results_info = {}
        band_results_info[f"{isolate_dimension}_band_labels"] = (
            deepcopy(list(bands.keys())) * n_entries
        )

        dim_bounds = []
        for bounds in bands.values():
            dim_bounds.append([bounds[0], bounds[1]])
        band_results_info[f"{isolate_dimension}_band_bounds"] = []
        for _ in range(n_entries):
            band_results_info[f"{isolate_dimension}_band_bounds"].extend(
                deepcopy(dim_bounds)
            )

        return band_results_info

    def _get_band_results(
        self,
        bands: dict,
        process_idcs: list[int],
        isolate_dimension: str,
        data_keys: list[str],
    ) -> None:
        """Gets the results for the requested bands.

        PARAMETERS
        ----------
        bands : dict

        process_idcs : list[int]

        isolate_dimension : str

        data_keys : list[str]
        """
        isolate_axis = self._find_dimension_axis(
            idcs=process_idcs, keys=data_keys, dimension=isolate_dimension
        )

        band_bound_idcs = self._get_band_bound_indices(
            bands=bands,
            process_idcs=process_idcs,
            isolate_dimension=isolate_dimension,
        )

        self._set_isolate_dimension_bounds(
            process_idcs=process_idcs,
            band_bound_idcs=band_bound_idcs,
            isolate_dimension=isolate_dimension,
        )

        self._compute_band_results(
            process_idcs=process_idcs,
            band_bound_idcs=band_bound_idcs,
            data_keys=data_keys,
            isolate_axis=isolate_axis,
        )

    def _get_band_bound_indices(
        self, bands: dict, process_idcs: list[int], isolate_dimension: str
    ) -> list[list[int]]:
        """Finds the indices of the bounds to use to isolate the results into
        bands.

        PARAMETERS
        ----------
        bands : dict

        process_idcs : list[int]

        isolate_dimension : str

        RETURNS
        -------
        band_bound_idcs : list[list[int]]
        -   List of the boundary indices to isolate the results, with an entry
            for each band and each of the indices of results being processed,
            corresponding to the entries in 'self._band_results'.
        """
        band_bound_idcs = np.zeros(
            (len(process_idcs), len(bands.keys()), 2), dtype=int
        )
        for band_i, bounds in enumerate(bands.values()):
            for bound_val_i, bound_val in enumerate(bounds):
                band_bound_idcs[:, band_i, bound_val_i] = (
                    self._find_index_in_nd_array(
                        values=np.array(
                            self._results.loc[
                                process_idcs, isolate_dimension
                            ].to_list()
                        ),
                        values_of_interest=np.full(
                            (len(process_idcs)), bound_val
                        ),
                        axis_of_interest=1,
                    )
                )

        return band_bound_idcs

    def _set_isolate_dimension_bounds(
        self,
        process_idcs: list[int],
        band_bound_idcs: list[list[int]],
        isolate_dimension: str,
    ) -> None:
        """"""
        band_results_idx = 0
        for process_idx, row_idx in enumerate(process_idcs):
            for band_idcs in band_bound_idcs[process_idx]:
                self._band_results.at[process_idx, isolate_dimension] = (
                    self._results.at[row_idx, isolate_dimension][
                        band_idcs[0] : band_idcs[1] + 1
                    ]
                )
                band_results_idx += 1

    def _compute_band_results(
        self,
        process_idcs: list[int],
        band_bound_idcs: list[list[int]],
        data_keys: list[str],
        isolate_axis: np.ndarray,
    ) -> None:
        """Isolates the band results.

        PARAMETERS
        ----------
        process_idcs : list[int]
        -   Indices of the results being isolated.

        n_bands : int
        -   Number of bands being isolated.

        band_bound_idcs : list[list[int]]
        -   List of the boundary indices to isolate the results, with an entry
            for each band and each of the indices of results being processed,
            corresponding to the entries in 'self._band_results'.

        attributes : list[str]
        -   Names of the attributes in the results containing data that should
            be isolated into bands.

        band_index_key : str
        -   Name of the attribute in the results which should be used to
            isolate the results into bands.

        ignore_nan : bool
        -   Whether or not to ignore NaN values when applying measures to the
            isolated results.
        """
        n_bands = band_bound_idcs.shape[1]
        band_results_idx = 0
        for process_idx, row_idx in enumerate(process_idcs):
            for band_i in range(n_bands):
                bound_idcs = band_bound_idcs[process_idx, band_i]

                for key_i, key in enumerate(data_keys):
                    band_value_idcs = []
                    entry = np.array(self._results.at[row_idx, key])
                    for axis_i in range(entry.ndim):
                        if axis_i == isolate_axis[process_idx, key_i]:
                            band_value_idcs.extend(
                                np.arange(bound_idcs[0], bound_idcs[1] + 1)
                            )
                        else:
                            band_value_idcs = np.arange(entry.shape[axis_i])

                    self._band_results.at[band_results_idx, key] = np.array(
                        self._results.at[row_idx, key]
                    )[band_value_idcs]

                band_results_idx += 1

    def zscore_within_nodes(
        self,
        data_keys: list[str],
        zscore_dimension: str,
        eligible_entries: Union[dict, None] = None,
        ignore_nan: bool = True,
    ) -> None:
        """Z-scores results within nodes.

        PARAMETERS
        ----------
        data_keys : list[str]
        -   Names of the attributes in the results which should be z-scored.

        zscore_dimension : str
            Name of the dimension of the results to compute the mean and
            standard deviation of for the z-score.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        ignore_nan : bool; default True
        -   Whether or not to ignore NaN values when calculating the z-score.
        """
        if not isinstance(zscore_dimension, str):
            raise TypeError(
                "The dimension to z-score over should be of type str, not "
                f"{type(zscore_dimension)}."
            )

        if self._verbose:
            if eligible_entries is None:
                eligible_entries_msg = "all entries"
            print(
                f"Z-scoring results over the {zscore_dimension} dimension.\n- "
                f"Eligible entries: {eligible_entries_msg}\n"
            )

        process_idcs = self._prepare_for_nongroup_method(
            eligible_entries=eligible_entries
        )

        process_axis = self._find_dimension_axis(
            idcs=process_idcs, keys=data_keys, dimension=zscore_dimension
        )

        self._compute_zscore_within_nodes(
            idcs=process_idcs,
            keys=data_keys,
            ignore_nan=ignore_nan,
            axis_of_dimension=process_axis,
        )

    def _compute_zscore_within_nodes(
        self,
        idcs: list[int],
        keys: list[str],
        ignore_nan: bool,
        axis_of_dimension: np.ndarray,
    ) -> None:
        """Computes the z-scored results across a dimension within each node.

        PARAMETERS
        ----------
        indices : list[int]
        -   Unique indices of nodes in the results that should be processed.

        keys : list[str]
        -   Attributes of the results to z-score.

        ignore_nan : bool
        -   Whether or not to ignore NaN values when z-scoring.

        axis_of_dimension : numpy ndarray
        -   The axis of the data to z-score with shape [len('idcs') x
            len('data_keys')], giving an axis to compute the mean and standard
            deviation of for each entry being processed.
        """
        if ignore_nan:
            mean_func = np.nanmean
            std_func = np.nanstd
        else:
            mean_func = np.mean
            std_func = np.std

        for idx_i, idx in enumerate(idcs):
            for key_i, key in enumerate(keys):
                axis = axis_of_dimension[idx_i, key_i]
                entry = np.array(self._results.loc[idx, key])
                self._results.at[idx, key] = (
                    entry - mean_func(entry, axis=axis)
                ) / std_func(entry, axis=axis)

    def zscore_over_nodes(
        self,
        data_keys: list[str],
        group_keys: list[str],
        eligible_entries: Union[dict, None] = None,
        identical_keys: Union[list[str], None] = None,
        ignore_nan: bool = True,
    ) -> None:
        """Z-score results over nodes in the data.

        PARAMETERS
        ----------
        data_keys : list[str]
        -   Names of the attributes in the results containing data that should
            be z-scored.

        group_keys : [list[str]]
        -   Names of the attributes in the results to use to group results that
            will be z-scored together.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        identical_keys : list[str] | None; default None
        -   The names of the attributes in the results that will be checked if
            they are identical across the results being z-scored. If they are
            not identical, an error will be raised.

        ignore_nan : bool; default True
        -   Whether or not to ignore NaN values when z-scoring.
        """
        group_idcs = self._prepare_for_group_method(
            method="zscore_over_nodes",
            over_key=None,
            data_keys=data_keys,
            eligible_entries=eligible_entries,
            group_keys=group_keys,
            identical_keys=identical_keys,
            var_measures=None,
        )

        self._compute_zscore_over_nodes(
            group_idcs=group_idcs,
            data_keys=data_keys,
            ignore_nan=ignore_nan,
        )

    def _compute_zscore_over_nodes(
        self,
        group_idcs: list[int],
        data_keys: list[str],
        ignore_nan: bool,
    ) -> None:
        """Computes the z-scored results across nodes.

        PARAMETERS
        ----------
        indices : list[int]
        -   Unique indices of nodes in the results that should be processed.

        data_keys : list[str]
        -   Attributes of the results to z-score.

        ignore_nan : bool
        -   Whether or not to ignore NaN values when z-scoring.
        """
        if ignore_nan:
            mean_func = np.nanmean
            std_func = np.nanstd
        else:
            mean_func = np.mean
            std_func = np.std

        for idcs in group_idcs:
            for key in data_keys:
                entries = np.array(self._results.loc[idcs, key].tolist())
                entries = (entries - mean_func(entries, axis=0)) / std_func(
                    entries, axis=0
                )
                for entry_i, row_idx in enumerate(idcs):
                    self._results.at[row_idx, key] = entries[entry_i]

    def percentile(
        self,
        over_key: str,
        data_keys: list[str],
        group_keys: list[str],
        percentile_interval: float = 10.0,
        eligible_entries: Union[dict, None] = None,
        identical_keys: Union[list[str], None] = None,
        ignore_nan: bool = True,
    ) -> None:
        """Calculates the percentile of results.

        PARAMETERS
        ----------
        over_key : str
        -   Name of the attribute in the results to find the percentile over.

        data_keys : list[str]
        -   Names of the attributes in the results containing data whose
            percentile should be found.

        group_keys : [list[str]]
        -   Names of the attributes in the results to use to group results whose
            percentile will be calculated together.

        percentile_interval : float; default 10.0
        -   The intervals into which percentiles will be grouped between 0 and
            100. E.g. an interval of 10.0 means a values at the 51st, 55th, and
            59th percentiles would be converted to the 50th percentile.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        identical_keys : list[str] | None; default None
        -   The names of the attributes in the results that will be checked if
            they are identical across the results whose percentile is being
            calculated. If they are not identical, an error will be raised.

        ignore_nan : bool; default True
        -   Whether or not to ignore NaN values when calculating the percentile.
        """
        group_idcs = self._prepare_for_group_method(
            method="percentile",
            over_key=over_key,
            data_keys=data_keys,
            eligible_entries=eligible_entries,
            group_keys=group_keys,
            identical_keys=identical_keys,
            var_measures=None,
        )

        self._prepare_percentiles(data_keys)

        self._compute_percentile(
            percentile_entry_idcs=group_idcs,
            percentile_keys=data_keys,
            ignore_nan=ignore_nan,
            percentile_interval=percentile_interval,
        )

    def _prepare_percentiles(self, data_keys: list[str]) -> None:
        """Adds columns for storing percentiles to the results DataFrame, if
        these columns are not yet present.

        PARAMETERS
        ----------
        data_keys : list of str
        -   Names of the attributes in the results containing data whose
            percentile should be found. Columns for storing results will have
            the name "{key}_percentiles".
        """
        percentile_keys = [f"{key}_percentiles" for key in data_keys]
        missing_keys = [
            key for key in percentile_keys if key not in self._results.keys()
        ]
        if missing_keys != []:
            self._populate_columns(attributes=missing_keys)

    def _compute_percentile(
        self,
        percentile_entry_idcs: list[list[int]],
        percentile_keys: list[str],
        ignore_nan: bool,
        percentile_interval: float,
    ) -> None:
        """Computes the percentile of results over the unique node indices.

        PARAMETERS
        ----------
        percentile_entry_indices : list[list[int]]
        -   Unique indices of nodes in the results that should be processed.

        percentile_keys : list[str]
        -   Attributes of the results whose percentile should be computed.

        ignore_nan : bool
        -   Whether or not to ignore NaN values when computing the percentiles.

        percentile_interval : float
        -   The intervals into which percentiles will be grouped between 0 and
            100. E.g. an interval of 10.0 means a values at the 51st, 55th, and
            59th percentiles would be converted to the 50th percentile.
        """
        percentile_boundaries = np.arange(
            0.0, 100.0, percentile_interval
        ).tolist()

        for idcs in percentile_entry_idcs:
            for key in percentile_keys:
                entries = self._results.loc[idcs, key].tonumpy()

                percentiles = self._percentile_of_score(
                    entries, percentile_boundaries, ignore_nan
                )

                for entry_i, row_i in enumerate(idcs):
                    self._results.at[row_i, f"{key}_percentiles"] = percentiles[
                        entry_i
                    ]

    def _percentile_of_score(
        self,
        data: np.ndarray,
        find_percentiles: list[float],
        ignore_nan: bool,
    ) -> list[float]:
        """Finds what percentile group data belongs to, similar to SciPy's
        percentofscore function.

        PARAMETERS
        ----------
        data : numpy ndarray
        -   The data whose percentiles should be found, where the values
            belonging to different entries in the results corresponds to the 0th
            axis.

        find_percentiles : list of float
        -   The percentile groups to find, e.g. [0.0, 10.0, 20.0, ..., 90.0].

        ignore_nan : bool
        -   How to treat NaN values in 'data'. If True, NaN values' percentiles
            will be set to NaN. If False, an error will be raised if a NaN is
            encountered.

        RETURNS
        -------
        percentiles : list of float
        -   The percentile groups for each entry of 'data'.

        RAISES
        ------
        ValueError
        -   Raised if 'ignore_nan' is False and a NaN value is present in
            'data'.
        """
        boundary_values = np.percentile(data, find_percentiles, axis=0)
        percentiles = []

        for value in data:
            if any(np.isnan(value)):
                if not ignore_nan:
                    raise ValueError(
                        "A NaN value is present in the data, but NaNs have "
                        "been specified to not be ignored."
                    )
                percentiles.append(np.nan)
            else:
                keep_searching = True
                while keep_searching:
                    for boundary_idx, boundary in enumerate(boundary_values):
                        if value < boundary:
                            if boundary_idx == 0:
                                percentiles.append(find_percentiles[0])
                            else:
                                percentiles.append(
                                    find_percentiles[boundary_idx - 1]
                                )
                                keep_searching = False
                        elif boundary_idx == len(boundary_values) - 1:
                            percentiles.append(find_percentiles[-1])
                            keep_searching = False
                        if not keep_searching:
                            break

        return percentiles

    def interpolate(
        self,
        over_key: str,
        data_keys: list[str],
        group_keys: list[str],
        coords_key: str,
        interpolation_coords: np.ndarray,
        interpolation_settings: dict,
        pin_to_hemisphere: str | None = None,
        eligible_entries: dict | None = None,
        identical_keys: list[str] | None = None,
    ) -> None:
        """Interpolate data to a set of coordinates using SciPy's
        RBFInterpolator.

        PARAMETERS
        ----------
        over_key : str
        -   Name of the attribute in the results to interpolate over.

        data_keys : list[str]
        -   Names of the attributes in the results containing data which should
            be interpolated.

        group_keys : [list[str]]
        -   Names of the attributes in the results to use to group results whose
            interpolation will be calculated together.

        coords_key : str
        -   Name of the key for the coordinates of the original data points in
            the DataFrame.

        interpolation_coords : numpy ndarray
        -   The coordinates to interpolate the data to. Should be an [n x 3]
            array, where n is the number of points to interpolate to, and 3 are
            the x-, y-, and z-coordinates, respectively.

        interpolation_settings : dict
        -   The settings to use to interpolate the data, given as keyword
            arguments. See the documentation of SciPy's
            scipy.interpolate.RBFInterpolator class for accepted arguments.

        pin_to_hemispheres : str | None; default None
        -   Which hemisphere to pin the coordinates of the original data to. If
            "left", all data is treated as belonging to the left hemisphere. If
            "right", all data is treated as belonging to the right hemisphere.
            If None, no changes to the data coordinates are made.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        identical_keys : list[str] | None; default None
        -   The names of the attributes in the results that will be checked if
            they are identical across the results being interpolated. If they
            are not identical, an error will be raised.
        """
        self._prepare_interpolation(
            coords_key=coords_key,
            pin_to_hemisphere=pin_to_hemisphere,
            interpolation_coords=interpolation_coords,
        )

        group_idcs = self._prepare_for_group_method(
            method="interpolate",
            over_key=over_key,
            data_keys=data_keys,
            eligible_entries=eligible_entries,
            group_keys=group_keys,
            identical_keys=identical_keys,
            var_measures=None,
        )

        self._compute_interpolation(
            group_idcs=group_idcs,
            data_keys=data_keys,
            coords_key=coords_key,
            interpolation_coords=interpolation_coords,
            interpolation_settings=interpolation_settings,
            pin_to_hemisphere=pin_to_hemisphere,
            over_key=over_key,
            group_keys=group_keys,
            identical_keys=identical_keys,
        )

    def _prepare_interpolation(
        self,
        coords_key: str,
        interpolation_coords: np.ndarray,
        pin_to_hemisphere: str | None = None,
    ) -> None:
        """Checks that the inputs for interpolation are appropriate.

        PARAMETERS
        ----------
        coords_key : str

        interpolation_coords : numpy ndarray

        pin_to_hemispheres : str | None; default None
        """
        if coords_key not in self._results.keys():
            raise ValueError("'coords_key' is not in the results.")

        accepted_pin_args = ["left", "right", None]
        if pin_to_hemisphere not in accepted_pin_args:
            raise ValueError(
                f"pin_to_hemisphere must be one of {accepted_pin_args}."
            )

        coords_shape = np.shape(interpolation_coords)
        if len(coords_shape) != 2 or coords_shape[1] != 3:
            raise ValueError(
                "'interpolation_coords' must be an [n x 3] array (i.e. an x-, "
                "y-, and z-axis coordinate for each of the n points to "
                "interpolate to)."
            )

    def _compute_interpolation(
        self,
        group_idcs: list[list[int]],
        data_keys: list[str],
        coords_key: str,
        interpolation_coords: np.ndarray,
        interpolation_settings: dict,
        pin_to_hemisphere: str | None,
        over_key: str,
        group_keys: list[str],
        identical_keys: list[str] | None,
    ) -> None:
        """Computes the interpolation of the data.

        PARAMETERS
        ----------
        group_idcs : list[list[int]]

        data_keys : list[str]

        coords_key : str

        interpolation_coords : numpy ndarray

        interpolation_settings : dict

        pin_to_hemispheres : str | None

        over_key : str

        group_keys : [list[str]]

        identical_keys : list[str] | None
        """
        drop_idcs = []
        for idcs in group_idcs:
            results = []
            for key in data_keys:
                data = np.array(self._results.loc[idcs, key].tolist())
                if not np.all(np.isfinite(data)):
                    raise ValueError(
                        f"{key} data in the results contain NaN or infinity "
                        "values."
                    )

                data_coords = self._prepare_coords_for_interpolation(
                    data_coords=np.array(
                        self._results.loc[idcs, coords_key].tolist()
                    ),
                    pin_to_hemisphere=pin_to_hemisphere,
                )

                results.append(
                    RBFInterpolator(
                        data_coords, data, **interpolation_settings
                    )(interpolation_coords)
                )

            self._add_interpolated_results(
                data=results,
                row_idcs=idcs,
                over_key=over_key,
                coords_key=coords_key,
                data_keys=data_keys,
                interpolation_coords=interpolation_coords,
                group_keys=group_keys,
                identical_keys=identical_keys,
            )

            drop_idcs.extend(idcs)

        self._results = self._results.drop(index=drop_idcs)
        self._results = self._results.reset_index(drop=True)

    def _prepare_coords_for_interpolation(
        self, data_coords: np.ndarray, pin_to_hemisphere: str | None
    ) -> np.ndarray:
        """Prepares coordinates of the data for interpolation.

        PARAMETERS
        ----------
        data_coords : numpy ndarray
        -   Coordinates of the data being interpolated.

        pin_to_hemispheres : str | None

        RETURNS
        -------
        coords : numpy ndarray
        -   The coordinates of the data for interpolation, appropriately
            modified.
        """
        if not np.all(np.isfinite(data_coords)):
            raise ValueError(
                "Coordinates in the results contain NaN or infinity values."
            )

        return self._pin_to_hemisphere(data_coords, pin_to_hemisphere) * 1000

    def _pin_to_hemisphere(
        self, coords: np.ndarray, hemisphere: str
    ) -> np.ndarray:
        """Pin coordinates to a hemisphere.

        Parameters
        ----------
        coords : numpy ndarray, shape of (N, 3)
            Coordinates to pin to a hemisphere.

        hemisphere : str
            Hemisphere to pin coordinates to ("right" or "left").

        Returns
        -------
        pinned_coords : numpy ndarray, shape of (N, 3)
            Coordinates pinned to the requested hemisphere.
        """
        pinned_coords = coords.copy()
        if hemisphere == "left":
            pinned_coords[:, 0] = np.abs(pinned_coords[:, 0]) * -1
        elif hemisphere == "right":
            pinned_coords[:, 0] = np.abs(pinned_coords[:, 0])
        else:
            raise ValueError(
                "`hemisphere` to pin coordinates to must be 'left' or 'right'."
            )

        return pinned_coords

    def _add_interpolated_results(
        self,
        data: list[np.ndarray],
        row_idcs: list[int],
        over_key: str,
        coords_key: str,
        data_keys: list[str],
        interpolation_coords: np.ndarray,
        group_keys: list[str],
        identical_keys: list[str] | None,
    ) -> None:
        """Adds interpolated results to the existing results.

        Descriptive measures are set to None, identical and group keys share a
        common value across indices being interpolated, so the value of the
        first index being interpolated can be used, and all other keys
        (excluding 'over_key', 'coords_key', and 'data_keys') have their
        values set as a combination of the data being interpolated in the format
        "interpolated[{combined values}]".

        PARAMETERS
        ----------
        data : list of numpy ndarray
        -   List containing the results arrays for each of 'interpolate_keys',
            respectively.

        row_idcs : list[int]
        -   Indices of the rows in the results from which the interpolated data
            has been derived.

        over_key : str

        coords_key : str

        data_keys : list[str]

        interpolation_coords : numpy ndarray

        group_keys : [list[str]]

        identical_keys : list[str] | None
        """
        results = {key: None for key in self._results.keys()}
        for idx, key in enumerate(data_keys):
            results[key] = data[idx]

        n_rows = np.shape(interpolation_coords)[0]
        for key in results.keys():
            if key in self._desc_measures:
                results[key] = [None for _ in range(n_rows)]

            elif key == coords_key:
                results[coords_key] = interpolation_coords.tolist()

            elif key not in data_keys and key != coords_key:
                if identical_keys is not None and key in identical_keys:
                    new_value = self._results.loc[row_idcs[0], key]
                else:
                    unique_values = unique(self._results.loc[row_idcs, key])
                    if len(unique_values) > 1 or key == over_key:
                        new_value = "interpolated["
                        for entry in unique_values:
                            new_value += f"{entry}, "
                        new_value = f"{new_value[:-2]}]"
                    else:
                        new_value = unique_values[0]
                results[key] = [new_value for _ in range(n_rows)]

        self.append_from_dict(results)

    def gaussianise(
        self,
        over_key: str,
        data_keys: list[str],
        gaussianise_dimension: str | None,
        group_keys: list[str],
        eligible_entries: dict | None = None,
        identical_keys: list[str] | None = None,
    ) -> None:
        """Gaussianises results to have mean = 0 and standard deviation = 1.

        PARAMETERS
        ----------
        over_key : str
        -   Name of the attribute in the results to gaussianise over.

        data_keys : list[str]
        -   Names of the attributes in the results containing data whose
            percentile should be found.

        gaussianise_dimension : str | None
        -   The dimension of 'data_keys' to Gaussianise across. If None, all
            dimensions are Gaussianised across.

        group_keys : [list[str]]
        -   Names of the attributes in the results to use to group results whose
            percentile will be calculated together.

        coords_key : str
        -   Name of the key for the coordinates of the original data points in
            the DataFrame.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.

        identical_keys : list[str] | None; default None
        -   The names of the attributes in the results that will be checked if
            they are identical across the results whose percentile is being
            calculated. If they are not identical, an error will be raised.
        """
        group_idcs = self._prepare_for_group_method(
            method="gaussianise",
            over_key=over_key,
            data_keys=data_keys,
            eligible_entries=eligible_entries,
            group_keys=group_keys,
            identical_keys=identical_keys,
            var_measures=None,
        )

        self._compute_gaussianisation(
            group_idcs=group_idcs,
            data_keys=data_keys,
            gaussianise_dimension=gaussianise_dimension,
        )

    def _compute_gaussianisation(
        self,
        group_idcs: list[list[int]],
        data_keys: list[str],
        gaussianise_dimension: str | None,
    ) -> None:
        """Gaussianises the data.

        PARAMETERS
        ----------
        group_idcs : list[list[int]]

        data_keys : list[str]

        gaussianise_dimension : str | None
        """
        for idcs in group_idcs:
            for key in data_keys:
                if gaussianise_dimension is not None:
                    gaussianise_axis = np.unique(
                        self._find_dimension_axis(
                            idcs=idcs,
                            keys=[key],
                            dimension=gaussianise_dimension,
                        )
                    )
                    if len(gaussianise_axis) != 1:
                        raise ValueError(
                            "When Gaussianising multiple rows of results, "
                            "each row must have the dimension being "
                            "Gaussianised across in the same axis position."
                        )
                    gaussianise_axis = gaussianise_axis[0]
                else:
                    gaussianise_axis = None

                data = np.array(self._results.loc[idcs, key].tolist())
                if not np.all(np.isfinite(data)):
                    raise ValueError(
                        f"{key} data in the results contain NaN or infinity "
                        "values."
                    )

                self._results.loc[idcs, key] = gaussian_transform(
                    data, axis=gaussianise_axis
                ).tolist()

    def project_to_mesh(
        self,
        mesh: str,
        coords_key: str,
        pin_to_hemisphere: str | None = None,
        eligible_entries: dict | None = None,
    ) -> None:
        r"""Project channel coordinates to the closest points on a mesh.

        Parameters
        ----------
        mesh : str
            Name of the mesh to use for projection. Mesh must be present in the
            MNE data folder, e.g.
            'C:\Users\user\mne_data\MNE-sample-data\subjects'.

        coords_key : str
            Name of the key for the coordinates to project.

        pin_to_hemisphere : str | None; default None
            Which hemisphere to pin the coordinates to before projecting. If
            "left", coordinates are pinned to the left hemisphere. If "right",
            coordinates are pinned to the right hemisphere. If None, no
            changes to the coordinates are made.

        eligible_entries : dict | None; default None
            Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.
        """
        accepted_pin_args = ["left", "right", None]
        if pin_to_hemisphere not in accepted_pin_args:
            raise ValueError(
                f"`pin_to_hemisphere` must be one of {accepted_pin_args}."
            )

        process_idcs = self._prepare_for_nongroup_method(
            eligible_entries=eligible_entries
        )

        self._project_to_mesh(
            process_idcs=process_idcs,
            mesh_name=mesh,
            coords_key=coords_key,
            pin_to_hemisphere=pin_to_hemisphere,
        )

    def _project_to_mesh(
        self,
        process_idcs: list[int],
        mesh_name: str,
        coords_key: str,
        pin_to_hemisphere: str | None,
    ) -> None:
        """Project channel coordinates to the closest points on a mesh."""
        coords = np.array(self._results.loc[process_idcs, coords_key].tolist())
        if pin_to_hemisphere is not None:
            coords = self._pin_to_hemisphere(coords, pin_to_hemisphere)

        sample_path = mne.datasets.sample.data_path()
        subjects_dir = sample_path / "subjects"

        # transform coords into proper space for projection
        mri_mni_trans = mne.read_talxfm(mesh_name, subjects_dir)
        mri_mni_inv = np.linalg.inv(mri_mni_trans["trans"])
        coords = mne.transforms.apply_trans(mri_mni_inv, coords)

        path_mesh = f"{subjects_dir}\\{mesh_name}\\surf\\{mesh_name}.glb"
        with open(path_mesh, mode="rb") as f:
            scene = trimesh.exchange.gltf.load_glb(f)
        mesh: trimesh.Trimesh = trimesh.Trimesh(
            **scene["geometry"]["geometry_0"]
        )
        coords = mesh.nearest.on_surface(coords)[0]
        if mesh_name.lower() == "mni_icbm152_nlin_asym_09b":
            coords *= 1.05
        # transforms coords back into MNI space
        coords = mne.transforms.apply_trans(mri_mni_trans, coords)

        self._results.loc[process_idcs, coords_key] = pd.Series(
            coords.tolist(), index=self._results.index[process_idcs]
        )

    def track_fibres_within_radius(
        self,
        atlas: str,
        seeds_key: str,
        seeds_coords_key: str,
        seeds_types_key: str,
        sphere_radii: dict,
        targets_key: str | list[str] | None = None,
        targets_coords_key: str | list[str] = None,
        targets_types_key: str | list[str] = None,
        allow_bypassing_fibres: bool = True,
        pin_to_hemisphere: str | None = None,
        eligible_entries: dict | None = None,
    ) -> None:
        """Find fibres within a specified radius of seeds (and targets).

        PARAMETERS
        ----------
        atlas : str
            Name of the atlas to use for fibre tracking. Accepted entries are:
            "holographic_hyperdirect_filtered";
            "holographic_pallidosubthalamic_filtered".

        seeds_key : str
            Name of the key for the seed channel names in the DataFrame.

        seeds_coords_key : str
            Name of the key for the coordinates of the seed channels in the
            DataFrame.

        seeds_types_key : str | list[str]
            Name of the key for the types of the seed channels in the
            DataFrame.

        sphere_radii : dict
            Radii of the sphere around each type of channel (in units of the
            atlas) to use when determining whether a fibre belongs to that
            channel. Keys should be the types of channels in the DataFrame and
            the values the corresponding radii for that channel type.

        targets_key : str | list[str] | None
            Name of the key for the target channel names in the DataFrame. If a
            list of str, must have length 2, in which case for each row of the
            DataFrame, the target key which does not contain the name of the
            seed channel will be used as the key for the target channels. If
            ``None``, all fibres close to seeds are taken.

        targets_coords_key : str | list[str]
            Name of the key for the coordinates of the target channels in the
            DataFrame. Must correspond to the entries of `targets_key`.

        targets_types_key : str | list[str]
            Name of the key for the types of the target channels in the
            DataFrame. Must correspond to the entries of `targets_key`.

        allow_bypassing_fibres : bool; default True
            Whether or not to allow the identified fibres to pass through the
            sphere radii. If ``False``, only fibres terminating in the spheres
            will be considered.

        pin_to_hemispheres : str | None; default None
        -   Which hemisphere to pin the coordinates of the original data to. If
            "left", all data is treated as belonging to the left hemisphere. If
            "right", all data is treated as belonging to the right hemisphere.
            If None, no changes to the data coordinates are made.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.
        """
        atlas_fpath, _ = self._prepare_fibre_tracking(
            atlas=atlas,
            targets_key=targets_key,
            targets_coords_key=targets_coords_key,
            targets_types_key=targets_types_key,
            normalise_distance=None,
            pin_to_hemisphere=pin_to_hemisphere,
        )

        process_idcs = self._prepare_for_nongroup_method(
            eligible_entries=eligible_entries
        )

        self._track_fibres_within_radius(
            process_idcs=process_idcs,
            atlas_fpath=atlas_fpath,
            seeds_key=seeds_key,
            targets_key=targets_key,
            seeds_coords_key=seeds_coords_key,
            targets_coords_key=targets_coords_key,
            seeds_types_key=seeds_types_key,
            targets_types_key=targets_types_key,
            sphere_radii=sphere_radii,
            allow_bypassing_fibres=allow_bypassing_fibres,
            pin_to_hemisphere=pin_to_hemisphere,
        )

    def _prepare_fibre_tracking(
        self,
        atlas: str,
        targets_key: str | list[str] | None,
        targets_coords_key: str | list[str] | None,
        targets_types_key: str | list[str] | None,
        normalise_distance: str | None,
        pin_to_hemisphere: str | None,
    ) -> tuple[str, Callable]:
        """Checks that the inputs for fibre tracking are appropriate.

        Returns
        -------
        atlas_fpath : str
            Filepath to the atlas.

        normalise_distance_func : Callable | None
            Lambda function to apply to the distance between fibres and seeds.
        """
        accepted_atlases = [
            "holographic_hyperdirect_filtered",
            "holographic_pallidosubthalamic_filtered",
        ]
        if atlas not in accepted_atlases:
            raise ValueError(
                f"The atlas {atlas} is not supported. Supported atlases are: "
                f"{accepted_atlases}."
            )
        atlas_fpath = os.path.join(
            os.getcwd(), "coherence", "fibre_atlases", f"{atlas}.mat"
        )

        targets = True
        if targets_key is None:
            targets_coords_key = None
            targets_types_key = None
            targets = False

        if targets:
            acceptable_key_types = (list, str, None)
            if (
                (not isinstance(targets_key, acceptable_key_types))
                ^ (not isinstance(targets_coords_key, acceptable_key_types))
                ^ (not isinstance(targets_types_key, acceptable_key_types))
            ):
                raise ValueError(
                    "`targets_key`, `targets_coords_key`, and "
                    "`targets_types_key` must all be lists, strings, or None."
                )
            if isinstance(targets_key, list) and len(targets_key) != 2:
                raise ValueError(
                    "If `targets_key` is a list, it must have length 2."
                )
            if (
                isinstance(targets_coords_key, list)
                and len(targets_coords_key) != 2
            ):
                raise ValueError(
                    "If `targets_coords_key` is a list, it must have length 2."
                )
            if (
                isinstance(targets_types_key, list)
                and len(targets_types_key) != 2
            ):
                raise ValueError(
                    "If `targets_types_key` is a list, it must have length 2."
                )
        else:
            if targets_coords_key is not None or targets_types_key is not None:
                raise TypeError(
                    "`targets_coords_key` and `targets_types_key` must be "
                    "None if `targets_key` is None."
                )

        accepted_normalise_args = ["inv_sqrd", None]
        if normalise_distance not in accepted_normalise_args:
            raise ValueError(
                "`normalise_distance` must be one of "
                f"{accepted_normalise_args}."
            )
        if normalise_distance == "inv_sqrd":
            normalise_distance_func = lambda x: (1 / x) ** 2
        else:
            normalise_distance_func = None

        accepted_pin_args = ["left", "right", None]
        if pin_to_hemisphere not in accepted_pin_args:
            raise ValueError(
                f"`pin_to_hemisphere` must be one of {accepted_pin_args}."
            )

        return atlas_fpath, normalise_distance_func

    def _track_fibres_within_radius(
        self,
        process_idcs: list[int],
        atlas_fpath: str,
        seeds_key: str,
        targets_key: str | list[str] | None,
        seeds_coords_key: str,
        targets_coords_key: str | list[str] | None,
        seeds_types_key: str,
        targets_types_key: str | list[str] | None,
        sphere_radii: dict,
        allow_bypassing_fibres: bool,
        pin_to_hemisphere: str | None,
    ) -> None:
        """Performs fibre tracking."""
        column_name_end = seeds_key
        search_opposing = False
        if isinstance(targets_key, list):
            column_name_end += "-opposing_channels"
            search_opposing = True
        elif targets_key is not None:
            column_name_end += f"-{targets_key}"
        fibre_ids_column = f"fibre_ids_{column_name_end}"
        n_fibres_column = f"n_fibres_{column_name_end}"
        for column_name in [fibre_ids_column, n_fibres_column]:
            if column_name not in self._results.keys():
                self._populate_columns([column_name])

        targets = True
        if targets_key is None:
            targets = False

        fibre_tracking = TrackFibres(atlas_fpath)
        for idx in process_idcs:
            seed_coords = (
                np.array(self._results[seeds_coords_key].iloc[idx]) * 1000
            )  # convert to mm
            if seed_coords.ndim == 1:
                seed_coords = seed_coords[np.newaxis, :]
            if pin_to_hemisphere is not None:
                seed_coords = self._pin_to_hemisphere(
                    seed_coords, pin_to_hemisphere
                )
            seeds_types = self._results[seeds_types_key].iloc[idx]

            if search_opposing:
                ch_in_key = [0, 0]
                for key_idx, key in enumerate(targets_key):
                    if (
                        self._results[seeds_key].iloc[idx]
                        in self._results[key].iloc[idx]
                    ):
                        ch_in_key[key_idx] = 1
                if sum(ch_in_key) == 0:
                    raise ValueError(
                        "The set of target channels cannot be identified for "
                        f"the seed channel in index {idx}."
                    )
                elif sum(ch_in_key) == 2:
                    raise ValueError(
                        f"The seed channel in index {idx} is in both possible "
                        "sets of target channels."
                    )
                else:
                    target_coords_key = targets_coords_key[ch_in_key.index(0)]
                    target_types_key = targets_types_key[ch_in_key.index(0)]
            elif targets:
                target_coords_key = targets_coords_key
                target_types_key = targets_types_key

            if targets:
                target_coords = (
                    np.array(self._results[target_coords_key].iloc[idx]) * 1000
                )  # convert to mm
                if target_coords.ndim == 1:
                    target_coords = target_coords[np.newaxis, :]
                if pin_to_hemisphere is not None:
                    target_coords = self._pin_to_hemisphere(
                        target_coords, pin_to_hemisphere
                    )
                targets_types = self._results[target_types_key].iloc[idx]

            fibre_ids, n_fibres = fibre_tracking.find_within_radius(
                np.array(seed_coords)[np.newaxis],
                sphere_radii[seeds_types],
                np.array(target_coords)[np.newaxis] if targets else None,
                sphere_radii[targets_types] if targets else None,
                allow_bypassing_fibres,
            )
            self._results.at[idx, n_fibres_column] = n_fibres[0]
            self._results.at[idx, fibre_ids_column] = fibre_ids[0]

    def track_closest_fibres(
        self,
        atlas: str,
        seeds_key: str,
        seeds_coords_key: str,
        normalise_distance: str | None = None,
        pin_to_hemisphere: str | None = None,
        eligible_entries: dict | None = None,
    ) -> None:
        """Find fibres closest to seeds.

        PARAMETERS
        ----------
        atlas : str
            Name of the atlas to use for fibre tracking. Accepted entries are:
            "holographic_hyperdirect_filtered";
            "holographic_pallidosubthalamic_filtered".

        seeds_key : str
            Name of the key for the seed channel names in the DataFrame.

        seeds_coords_key : str
            Name of the key for the coordinates of the seed channels in the
            DataFrame.

        normalise_distance : str | None; default None
            Normalisation to apply to the distance between the closest fibres
            and the seeds before storing in the DataFrame. Accepts: "inv_sqrd".
            If ``None``, no normalisation is applied.

        pin_to_hemispheres : str | None; default None
        -   Which hemisphere to pin the coordinates of the original data to. If
            "left", all data is treated as belonging to the left hemisphere. If
            "right", all data is treated as belonging to the right hemisphere.
            If None, no changes to the data coordinates are made.

        eligible_entries : dict | None; default None
        -   Dictionary where the keys are attributes in the data and the values
            are the values of the attributes which are considered eligible for
            processing. If None, all entries are processed.
        """
        atlas_fpath, normalise_distance_func = self._prepare_fibre_tracking(
            atlas=atlas,
            targets_key=None,
            targets_coords_key=None,
            targets_types_key=None,
            normalise_distance=normalise_distance,
            pin_to_hemisphere=pin_to_hemisphere,
        )

        process_idcs = self._prepare_for_nongroup_method(
            eligible_entries=eligible_entries
        )

        self._track_closest_fibres(
            process_idcs=process_idcs,
            atlas_fpath=atlas_fpath,
            seeds_key=seeds_key,
            seeds_coords_key=seeds_coords_key,
            normalise_distance=normalise_distance,
            normalise_distance_func=normalise_distance_func,
            pin_to_hemisphere=pin_to_hemisphere,
        )

    def _track_closest_fibres(
        self,
        process_idcs: list[int],
        atlas_fpath: str,
        seeds_key: str,
        seeds_coords_key: str,
        normalise_distance: str | None,
        normalise_distance_func: Callable | None,
        pin_to_hemisphere: str | None,
    ) -> None:
        """Finds closest fibres (IDs and distances) to seeds."""
        fibre_ids_column = f"closest_fibre_ids_{normalise_distance}_{seeds_key}"
        fibre_distances_column = (
            f"smallest_fibre_distances_{normalise_distance}_{seeds_key}"
        )
        for column_name in [fibre_ids_column, fibre_distances_column]:
            if column_name not in self._results.keys():
                self._populate_columns([column_name])

        fibre_tracking = TrackFibres(atlas_fpath)
        seed_coords = (
            np.array(
                self._results.loc[process_idcs, seeds_coords_key].to_list()
            )
            * 1000
        )  # convert to mm
        if seed_coords.ndim == 1:
            seed_coords = seed_coords[np.newaxis, :]
        if pin_to_hemisphere is not None:
            seed_coords = self._pin_to_hemisphere(
                seed_coords, pin_to_hemisphere
            )

        fibre_ids, distances = fibre_tracking.find_closest(
            seed_coords, normalise_distance_func
        )
        self._results.loc[process_idcs, fibre_distances_column] = distances
        self._results.loc[process_idcs, fibre_ids_column] = fibre_ids

    def results_as_df(self) -> pd.DataFrame:
        """Returns the results as a pandas DataFrame.

        RETURNS
        -------
        results : pandas DataFrame
        -   The results as a pandas DataFrame.
        """
        return self._results

    def _check_attrs_identical(self, attrs: list[str]) -> None:
        """Checks that the values of attributes in the results are identical
        across nodes.

        PARAMETERS
        ----------
        attrs : list[str]
        -   Names of the attributes to check in the results.

        RAISES
        ------
        UnidenticalEntryError
        -   Raised if the values of an attribute are not identical across the
            nodes in the results.
        """
        for attr in attrs:
            attr_vals = self._results[attr].tolist()
            if len(np.unique(attr_vals)) != 1:
                raise UnidenticalEntryError(
                    "Error when checking whether values belonging to an "
                    "attribute of the results are identical:\nValues of the "
                    f"attribute '{attr}' are not identical."
                )

    def _sequester_to_dicts(
        self, sequester: dict[list[str]]
    ) -> tuple[dict[dict], list[str]]:
        """Sequesters attributes of the results into dictionaries within a
        parent dictionary

        PARAMETERS
        ----------
        sequester : dict[list[str]]
        -   Attributes of the results to sequester into dictionaries within the
            returned results dictionary.
        -   Each key is the name of the dictionary that the attributes (the
            values for this key given as strings within a list corresponding to
            the names of the attributes) will be included in.
        -   E.g. an 'extract_to_dicts' of {"metadata": ["subject", "session"]}
            would create a dictionary within the returned dictionary of results
            of {"metadata": {"subject": VALUE, "session": VALUE}}.
        -   Values of extracted attributes must be identical for each node in
            the results, so that the dictionary the value is sequestered into
            contains only a single value for each attribute.

        RETURNS
        -------
        results : dict[dict]
        -   Dictionary with the requested attributes sequestered into the
            requested dictionaries.

        attrs_to_sequester : list[str]
        -   Names of attributes in the results that have been sequestered into
            dictionaries.
        """
        attrs_to_sequester = []
        for values in sequester.values():
            attrs_to_sequester.extend(values)
        self._check_attrs_identical(attrs=attrs_to_sequester)

        results = {}
        for dict_name, attrs in sequester.items():
            results[dict_name] = {}
            for attr in attrs:
                results[dict_name][attr] = self._results[attr][0]

        return results, attrs_to_sequester

    def results_as_dict(
        self, sequester_to_dicts: Union[dict[list[str]], None] = None
    ) -> dict:
        """Converts the results from a pandas DataFrame to a dictionary and
        returns the results.

        PARAMETERS
        ----------
        sequester_to_dicts : dict[list[str]] | None; default None
        -   Attributes of the results to sequester into dictionaries within the
            returned results dictionary.
        -   Each key is the name of the dictionary that the attributes (the
            values for this key given as strings within a list corresponding to
            the names of the attributes) will be included in.
        -   E.g. an 'extract_to_dicts' of {"metadata": ["subject", "session"]}
            would create a dictionary within the returned dictionary of results
            of {"metadata": {"subject": VALUE, "session": VALUE}}.
        -   Values of extracted attributes must be identical for each node in
            the results, so that the dictionary the value is sequestered into
            contains only a single value for each attribute.

        RETURNS
        -------
        results : dict
        -   The results as a dictionary.
        """
        if sequester_to_dicts is not None:
            results, ignore_attrs = self._sequester_to_dicts(
                sequester=sequester_to_dicts
            )
        else:
            results = {}
            ignore_attrs = []

        for attr in self._results.keys():
            if attr not in ignore_attrs:
                results[attr] = self._results[attr].copy().tolist()

        return results

    def save_object(
        self,
        fpath: str,
        ask_before_overwrite: Optional[bool] = None,
    ) -> None:
        """Saves the PostProcess object as a .pkl file.

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
            ask_before_overwrite = self._verbose

        save_object(
            to_save=self,
            fpath=fpath,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self._verbose,
        )

    def save_results(
        self,
        fpath: str,
        ftype: Union[str, None] = None,
        sequester_to_dicts: Union[dict[list[str]], None] = None,
        ask_before_overwrite: Union[bool, None] = None,
    ) -> None:
        """Saves the results and additional information as a file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the data should be saved.

        ftype : str | None; default None
        -   The filetype of the data that will be saved, without the leading
            period. E.g. for saving the file in the json format, this would be
            "json", not ".json".
        -   The information being saved must be an appropriate type for saving
            in this format.
        -   If None, the filetype is determined based on 'fpath', and so the
            extension must be included in the path.

        sequester_to_dicts : dict[list[str]] | None; default None
        -   Attributes of the results to sequester into dictionaries within the
            returned results dictionary.
        -   Each key is the name of the dictionary that the attributes (the
            values for this key given as strings within a list corresponding to
            the names of the attributes) will be included in.
        -   E.g. an 'extract_to_dicts' of {"metadata": ["subject", "session"]}
            would create a dictionary within the returned dictionary of results
            of {"metadata": {"subject": VALUE, "session": VALUE}}.
        -   Values of extracted attributes must be identical for each node in
            the results, so that the dictionary the value is sequestered into
            contains only a single value for each attribute.

        ask_before_overwrite : bool | None; default the object's verbosity
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.
        -   By default, this is set to None, in which case the value of the
            verbosity when the Signal object was instantiated is used.
        """
        if ask_before_overwrite is None:
            ask_before_overwrite = self._verbose

        save_dict(
            to_save=self.results_as_dict(sequester_to_dicts=sequester_to_dicts),
            fpath=fpath,
            ftype=ftype,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self._verbose,
        )


def load_results_of_types(
    folderpath_processing: str,
    to_analyse: dict[str],
    result_types: dict[Union[list[str], list[list[str]]]],
    extract_from_dicts: Optional[dict[list[str]]] = None,
    identical_keys: Optional[list[str]] = None,
    discard_keys: Optional[list[str]] = None,
    allow_missing: bool = False,
    result_ftype: str = ".json",
) -> PostProcess:
    """Loads results of a multiple types and merges them into a single
    PostProcess object.

    PARAMETERS
    ----------
    folderpath_processing : str
    -   Folderpath to where the processed results are located.

    to_analyse : dict[str]
    -   Dictionary in which each entry represents a different piece of results.
    -   Contains the keys: 'sub' (subject ID); 'ses' (session name); 'task'
        (task name); 'acq' (acquisition type); and 'run' (run number).

    result_types : dict of list of str | dict of list of list of str
    -   Dictionary with a single key ("merge" or "append"), indicating how to
        combined the results, and whose value specifies the results to combine.
    -   If the key is "merge", the value should be a list of str, where the str
        are the names of the types of results to merge, resulting in new
        columns being added to the results, with entries being aligned according
        to matching row entries.
    -   If the key is "append", the value should be a list of list of str, where
        each list of str indicates the results to append together (i.e. add new
        rows for shared columns), before all results are merged.

    extract_from_dicts : dict[list[str]] | None; default None
    -   The entries of dictionaries within 'results' to include in the
        processing.
    -   Entries which are extracted are treated as being identical for all
        values in the 'results' dictionary.

    identical_keys : list[str] | None; default None
    -   The keys in 'results' which are identical across channels and for
        which only one copy is present.

    discard_keys : list[str] | None; default None
    -   The keys which should be discarded immediately without
        processing.

    allow_missing : bool; default False
    -   Whether or not to allow new rows to be present in the merged results
        with NaN values for columns not shared between the results being
        merged if the shared columns do not have matching values.
    -   I.e. if you want to make sure you are merging results from the same
        channels, set this to False, otherwise results from different
        channels will be merged and any missing information will be set to
        NaN.

    result_ftype : str; default ".json"
    -   The filetype of the results, with the leading period, e.g. JSON would be
        specified as ".json".

    RETURNS
    -------
    merged_results : PostProcess
    -   The results appended and/or merged across the specified result types.
    """
    accepted_results_combs = ["merge", "append"]
    if len(result_types.keys()) != 1:
        raise ValueError(
            "Only one method for combining results can be given ('merge' or "
            f"'append'), but {accepted_results_combs} methods were given."
        )
    results_comb_method = list(result_types.keys())[0]
    results_comb_order = result_types[results_comb_method]
    if results_comb_method not in accepted_results_combs:
        raise ValueError(
            f"The result combination method '{results_comb_method}' is not "
            f"supported. Supported methods are: {accepted_results_combs}."
        )

    if results_comb_method == "append":
        results_to_merge = []
        for append_results in results_comb_order:
            first_append = True
            for result_type in append_results:
                results = load_results_of_type(
                    folderpath_processing=folderpath_processing,
                    to_analyse=to_analyse,
                    result_type=result_type,
                    extract_from_dicts=extract_from_dicts,
                    identical_keys=identical_keys,
                    discard_keys=discard_keys,
                    result_ftype=result_ftype,
                )
                if first_append:
                    appended_results = deepcopy(results)
                    first_append = False
                else:
                    appended_results.append_from_df(
                        new_results=deepcopy(results.results_as_df()),
                    )
            results_to_merge.append(appended_results)
    else:
        results_to_merge = results_comb_method

    first_merge = True
    for to_merge in results_to_merge:
        if results_comb_method == "merge":
            results = load_results_of_type(
                folderpath_processing=folderpath_processing,
                to_analyse=to_analyse,
                result_type=to_merge,
                extract_from_dicts=extract_from_dicts,
                identical_keys=identical_keys,
                discard_keys=discard_keys,
                result_ftype=result_ftype,
            )
        else:
            results = to_merge

        if first_merge:
            merged_results = deepcopy(results)
            first_merge = False
        else:
            merged_results.merge_from_df(
                new_results=deepcopy(results.results_as_df()),
                allow_missing=allow_missing,
            )

    return merged_results


def load_results_of_type(
    folderpath_processing: str,
    to_analyse: list[dict[str]],
    result_type: str,
    extract_from_dicts: Optional[dict[list[str]]] = None,
    identical_keys: Optional[list[str]] = None,
    discard_keys: Optional[list[str]] = None,
    result_ftype: str = ".json",
) -> PostProcess:
    """Loads results of a single type and appends them into a single PostProcess
    object.

    PARAMETERS
    ----------
    folderpath_processing : str
    -   Folderpath to where the processed results are located.

    to_analyse : list[dict[str]]
    -   Dictionary in which each entry represents a different piece of results.
    -   Contains the keys: 'sub' (subject ID); 'ses' (session name); 'task'
        (task name); 'acq' (acquisition type); and 'run' (run number).

    result_type : str
    -   The type of results to analyse.

    extract_from_dicts : dict[list[str]] | None; default None
    -   The entries of dictionaries within 'results' to include in the
        processing.
    -   Entries which are extracted are treated as being identical for all
        values in the 'results' dictionary.

    identical_keys : list[str] | None; default None
    -   The keys in 'results' which are identical across channels and for
        which only one copy is present.

    discard_keys : list[str] | None; default None
    -   The keys which should be discarded immediately without
        processing.

    result_ftype : str; default ".json"
    -   The filetype of the results, with the leading period, e.g. JSON would be
        specified as ".json".

    RETURNS
    -------
    results : PostProcess
    -   The appended results for a single type of data.
    """
    first_result = True
    for result_info in to_analyse:
        result_fpath = generate_sessionwise_fpath(
            folderpath=folderpath_processing,
            dataset=result_info["cohort"],
            subject=result_info["sub"],
            session=result_info["ses"],
            task=result_info["task"],
            acquisition=result_info["acq"],
            run=result_info["run"],
            group_type=result_type,
            filetype=result_ftype,
        )
        result = load_file(fpath=result_fpath)
        if first_result:
            results = PostProcess(
                results=result,
                extract_from_dicts=extract_from_dicts,
                identical_keys=identical_keys,
                discard_keys=discard_keys,
            )
            first_result = False
        else:
            results.append_from_dict(
                new_results=result,
                extract_from_dicts=extract_from_dicts,
                identical_keys=identical_keys,
                discard_keys=discard_keys,
            )

    return results
