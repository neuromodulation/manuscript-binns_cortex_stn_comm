"""Contains functions and classes for handling entries within objects.

CLASSES
-------
FillerObject
-   Empty object that can be filled with attributes that would otherwise not be
    accessible as a class attribute.

METHODS
-------
check_lengths_dict_identical
-   Checks whether the lengths of entries within a dictionary are identical.

check_lengths_dict_equals_n
-   Checks whether the lengths of entries within a dictionary is equal to a
    given number.

check_lengths_list_identical
-   Checks whether the lengths of entries within a list are identical.

check_lengths_list_equals_n
-   Checks whether the lengths of entries within a list is equal to a given
    number.

check_repeated_vals
-   Checks whether duplicates exist within an input list.

check_matching_entries
-   Checks whether the entries of objects match one another.

check_master_entries_in_sublists
-   Checks whether all values in a master list are present in a set of sublists.

check_sublist_entries_in_master
-   Checks whether all values in a set of sublists are present in a master list.

ordered_list_from_dict
-   Creates a list from entries in a dictionary, sorted based on a given order.

ordered_dict_from_list
-   Creates a dictionary with keys occurring in a given order.

ragged_array_to_list
-   Converts a ragged numpy array of nested arrays to a ragged list of nested
    lists.

drop_from_list
-   Drops specified entries from a list.

drop_from_dict
-   Removes specified entries from a dictionary.

sort_inputs_results
-   Checks that the values in 'results' are in the appropriate format for
    processing with PostProcess or Plotting class objects.

dict_to_df
-   Converts a dictionary into a pandas DataFrame.
"""

from copy import deepcopy
from itertools import chain
from typing import Any, Optional, Union
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from coh_exceptions import (
    DuplicateEntryError,
    EntryLengthError,
    MissingEntryError,
    PreexistingAttributeError,
    UnidenticalEntryError,
)


class FillerObject:
    """Creates an empty object that can be filled with attributes that would
    otherwise not be accessible as a class attribute."""


def _find_lengths_dict(
    to_check: dict,
    ignore_values: Optional[list] = None,
    ignore_keys: Optional[list] = None,
) -> list[int]:
    """Finds the lengths of entries within a dictionary.

    PARAMETERS
    ----------
    to_check : dict
    -   The dictionary for which the lengths of the entries should be checked.

    ignore_values : list | None; default None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no values are ignored.

    ignore_keys : list | None; default None
    -   The keys of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no keys are ignored.

    RETURNS
    -------
    entry_lengths : list[int]
    -   The lengths of entries in the list.
    """

    if ignore_values is None:
        ignore_values = []
    if ignore_keys is None:
        ignore_keys = []

    entry_lengths = []
    for key, value in to_check.items():
        if key not in ignore_keys or value not in ignore_values:
            entry_lengths.append(len(value))

    return entry_lengths


def check_lengths_dict_identical(
    to_check: dict,
    ignore_values: Optional[list] = None,
    ignore_keys: Optional[list] = None,
) -> tuple[bool, Union[int, list[int]]]:
    """Checks whether the lengths of entries in the input dictionary are
    identical.

    PARAMETERS
    ----------
    to_check : dict
    -   The dictionary for which the lengths of the entries should be checked.

    ignore_values : list | None; default None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no values are ignored.

    ignore_keys : list | None; default None
    -   The keys of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no keys are ignored.

    RETURNS
    -------
    identical : bool
    -   Whether or not the lengths of the entries are identical.

    lengths : int | list
    -   The length(s) of the entries. If the lengths are identical,
        'lengths' is an int representing the length of all items.
    -   If the lengths are not identical, 'lengths' is a list containing the
        lengths of the individual entries (i.e. 'entry_lengths').
    """

    entry_lengths = _find_lengths_dict(
        to_check=to_check, ignore_values=ignore_values, ignore_keys=ignore_keys
    )

    if entry_lengths.count(entry_lengths[0]) == len(entry_lengths):
        identical = True
        lengths = entry_lengths[0]
    else:
        identical = False
        lengths = entry_lengths

    return identical, lengths


def check_lengths_dict_equals_n(
    to_check: dict,
    n: int,
    ignore_values: Optional[list] = None,
    ignore_keys: Optional[list] = None,
) -> bool:
    """Checks whether the lengths of entries in the input dictionary are equal
    to a given number.

    PARAMETERS
    ----------
    to_check : list
    -   The list for which the lengths of the entries should be checked.

    ignore_values : list | None; default None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no values are ignored.

    n : int
    -   The integer which the lengths of the entries should be equal to.

    RETURNS
    -------
    all_n : bool
    -   Whether or not the lengths of the entries are equal to 'n'.
    """

    entry_lengths = _find_lengths_dict(
        to_check=to_check, ignore_values=ignore_values, ignore_keys=ignore_keys
    )

    if entry_lengths.count(n) == len(entry_lengths):
        all_n = True
    else:
        all_n = False

    return all_n


def _find_lengths_list(
    to_check: list, ignore_values: Optional[list], axis: int
) -> list[int]:
    """Finds the lengths of entries within a list.

    PARAMETERS
    ----------
    to_check : list
    -   The list for which the lengths of the entries should be checked.

    ignore_values : list | None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None, no values are ignored.

    axis : int
    -   The axis of the list whose lengths should be checked.

    RETURNS
    -------
    entry_lengths : list[int]
    -   The lengths of entries in the list.
    """

    if ignore_values is None:
        ignore_values = []
    entry_lengths = []
    for value in to_check:
        if value not in ignore_values:
            value = np.asarray(value, dtype=object)
            entry_lengths.append(np.shape(value)[axis])

    return entry_lengths


def check_lengths_list_identical(
    to_check: list, ignore_values: Optional[list] = None, axis: int = 0
) -> tuple[bool, Union[int, list[int]]]:
    """Checks whether the lengths of entries in the input list are identical.

    PARAMETERS
    ----------
    to_check : list
    -   The list for which the lengths of the entries should be checked.

    ignore_values : list | None; default None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no values are ignored.

    axis : int | default 0
    -   The axis of the list whose length should be checked.

    RETURNS
    -------
    identical : bool
    -   Whether or not the lengths of the entries are identical.

    lengths : int | list
    -   The length(s) of the entries. If the lengths are identical,
        'lengths' is an int representing the length of all items.
    -   If the lengths are not identical, 'lengths' is a list containing the
        lengths of the individual entries (i.e. 'entry_lengths').
    """

    entry_lengths = _find_lengths_list(
        to_check=to_check, ignore_values=ignore_values, axis=axis
    )

    if entry_lengths.count(entry_lengths[0]) == len(entry_lengths):
        identical = True
        lengths = entry_lengths[0]
    else:
        identical = False
        lengths = entry_lengths

    return identical, lengths


def check_lengths_list_equals_n(
    to_check: list, n: int, ignore_values: Optional[list] = None, axis: int = 0
) -> bool:
    """Checks whether the lengths of entries in the input dictionary are equal
    to a given number.

    PARAMETERS
    ----------
    to_check : list
    -   The list for which the lengths of the entries should be checked.

    n : int
        -   The integer which the lengths of the entries should be equal to.

    ignore_values : list | None; default None
    -   The values of entries within 'to_check' to ignore when checking the
        lengths of entries.
    -   If None (default), no values are ignored.

    axis : int | default 0
    -   The axis of the list whose lengths should be checked.

    RETURNS
    -------
    all_n : bool
    -   Whether or not the lengths of the entries are equal to 'n'.
    """

    entry_lengths = _find_lengths_list(
        to_check=to_check, ignore_values=ignore_values, axis=axis
    )

    if entry_lengths.count(n) == len(entry_lengths):
        all_n = True
    else:
        all_n = False

    return all_n


def unique(values: list) -> list:
    """Finds the unique values in a list in the original order in which the
    entries occur in the the original list.
    -   Similar to calling numpy's 'unique' with 'return_index' set to 'True'
        and then reordering the result of numpy's 'unique' to restore the order
        in which the unique values occurred in the original list.

    PARAMETERS
    ----------
    values : list
    -   The values whose unique entries should be found.

    RETURNS
    -------
    unique_values : list
    -   The unique entries in 'values'.
    """
    unique_values = []
    for entry in values:
        if entry not in unique_values:
            unique_values.append(entry)

    return unique_values


def check_vals_identical_list(
    to_check: list,
) -> tuple[bool, Union[list, None]]:
    """Checks whether all values within a list are identical.

    PARAMETERS
    ----------
    to_check : list
    -   The list whose values should be checked.

    RETURNS
    -------
    is_identical : bool
    -   Whether or not all values within the list are identical.

    unique_vals : list | None
    -   The unique values in the list. If all values are identical, this is
        'None'.
    """

    is_identical = True
    compare_against = to_check[0]
    for val in to_check[1:]:
        if val != compare_against:
            is_identical = False

    if is_identical:
        unique_vals = None
    else:
        unique_vals = unique(to_check)

    return is_identical, unique_vals


def check_vals_identical_df(
    dataframe: pd.DataFrame, keys: list[str], idcs: list[list[int]]
) -> None:
    """Checks that a DataFrame attribute's values at specific indices are
    identical.

    PARAMETERS
    ----------
    dataframe : pandas DataFrame
    -   DataFrame containing the values to check.

    keys : list[str]
    -   Names of the attributes in the DataFrame whose values should be checked.

    idcs : list[list[int]]
    -   The indices of the entries in the attributes whose values should be
        checked.
    -   Each entry is a list of integers corresponding to the indices of
        the results to compare together.

    RAISES
    ------
    UnidenticalEntryError
    -   Raised if any of the groups of values being compared are not identical.
    """

    for key in keys:
        for group_idcs in idcs:
            if len(group_idcs) > 1:
                is_identical, unique_vals = check_vals_identical_list(
                    to_check=dataframe[key].iloc[group_idcs].tolist()
                )
                if not is_identical:
                    raise UnidenticalEntryError(
                        "Error when checking that the attributes of "
                        "results belonging to the same group share the "
                        f"same values:\nThe values of '{key}' in rows "
                        f"{group_idcs} do not match.\nValues:{unique_vals}\n"
                    )


def get_eligible_idcs_lists(
    to_check: dict[list],
    eligible_vals: dict[list],
    idcs: Union[list[int], None] = None,
) -> list[int]:
    """Finds indices of items in multiple lists that have a certain value.
    -   Indices are found in turn for each list, such that the number of
        eligible indices can decrease for every list being checked.

    PARAMETERS
    ----------
    to_check : dict[list]
    -   Lists whose values should be checked, stored in a dictionary.
    -   The keys of the dictionary should be the same as those in
        'eligible_vals' for the corresponding eligible values.

    eligible_vals : dict[list]
    -   Lists containing values that are considered 'eligible', and whose
        indices will be recorded, stored in a dictionary.
    -   The keys of the dictionary should be the same as those in 'to_check' for
        the corresponding values to be checked.

    idcs : list[int] | None; default None
    -   Indices of the items in the first list of 'to_check' to check.
    -   If 'None', all items in the first list are checked.

    RETURNS
    -------
    idcs : list[int]
    -   List containing the indices of 'eligible' entries across all lists.
    """

    idcs = deepcopy(idcs)

    if idcs is None:
        _, length = check_lengths_dict_identical(to_check=to_check)
        idcs = np.arange(length).tolist()

    for key, value in to_check.items():
        if key in eligible_vals.keys():
            idcs = get_eligible_idcs_list(
                vals=value, eligible_vals=eligible_vals[key], idcs=idcs
            )

    return idcs


def get_eligible_idcs_list(
    vals: list,
    eligible_vals: list,
    idcs: Union[list[int], None] = None,
) -> list[int]:
    """Finds indices of items in a list that have a certain value.

    PARAMETERS
    ----------
    vals : list
    -   List whose values should be checked.

    eligible_vals : list
    -   List containing values that are considered 'eligible', and whose indices
        will be recorded.

    idcs : list[int] | None; default None
    -   Indices of the items in 'to_check' to check.
    -   If 'None', all items are checked.

    RETURNS
    -------
    list[int]
    -   List containing the indices of items in 'to_check' with 'eligible'
        values.
    """
    if idcs is None:
        idcs = range(len(vals))

    return [idx for idx in idcs if vals[idx] in eligible_vals]


def get_group_names_idcs(
    dataframe: pd.DataFrame,
    keys: Union[list[str], None] = None,
    eligible_idcs: Union[list[int], None] = None,
    replacement_idcs: Union[list[int], None] = None,
    special_vals: Union[dict[str], None] = None,
    keys_in_names: bool = True,
) -> dict[int]:
    """Combines the values of DataFrame columns into a string on a row-by-row
    basis (i.e. one string for each row) and finds groups of items containing
    these values, returning the names of the groups and their indices.

    PARAMETERS
    ----------
    dataframe : pandas DataFrame
    -   DataFrame whose values should be combined across columns.

    keys : list[str] | None
    -   Names of the columns in the DataFrame whose values should be combined.
    -   If 'None', all columns are used.

    eligible_idcs : list[int] | None
    -   Indices of the rows in the DataFrame whose values should be combined.
    -   If 'None', all rows are used.

    replacement_idcs : list[int] | None
    -   List containing indices that the indices of items in 'vals' should be
        replaced with.
    -   Must have the same length as 'vals'.
    -   E.g. if items in positions 0, 1, and 2 of 'vals' were grouped together
        and the values of 'replacement_idcs' in positions 0 to 2 were [2, 6, 9],
        respectively, the resulting indices for this group would be [2, 6, 9].
    -   If None, the original indices are used.

    special_vals : dict[str] | None
    -   Instructions for how to treat specific values in the DataFrame.
    -   Keys are the special values that the values should begin with, whilst
        values are the values that the special values should be replaced with.
    -   E.g. {"avg[": "avg_"} would mean values in the DataFrame beginning with
        'avg[' would have this beginning replaced with 'avg_', followed by the
        column name, so a value beginning with 'avg[' in the 'channels' column
        would become 'avg_channels'.

    keys_in_names : bool; default True
    -   Whether or not the names of groups should contain the keys to which the
        value belong.

    RETURNS
    -------
    group_names_idcs : dict[int]
    -   Dictionary where each key is the name of the group, and each value the
        indices of rows in 'dataframe' corresponding to this group.
    """

    combined_values = combine_col_vals_df(
        dataframe=dataframe,
        keys=keys,
        idcs=eligible_idcs,
        special_vals=special_vals,
        include_keys=keys_in_names,
    )
    group_idcs, group_names = get_group_idcs(
        vals=combined_values, replacement_idcs=replacement_idcs
    )

    group_names_idcs = {}
    for idx, name in enumerate(group_names):
        group_names_idcs[name] = group_idcs[idx]

    return group_names_idcs


def get_group_idcs(
    vals: list, replacement_idcs: Union[list[int], None] = None
) -> tuple[list[list[int]], list]:
    """Finds groups of items in a list containing the same values, and returns
    their indices.

    PARAMETERS
    ----------
    vals : list
    -   List containing the items that should be compared.

    replacement_idcs : list[int] | None
    -   List containing indices that the indices of items in 'vals' should be
        replaced with.
    -   Must have the same length as 'vals'.
    -   E.g. if items in positions 0, 1, and 2 of 'vals' were grouped together
        and the values of 'replacement_idcs' in positions 0 to 2 were [2, 6, 9],
        respectively, the resulting indices for this group would be [2, 6, 9].
    -   If None, the original indices are used.

    RETURNS
    -------
    group_idcs : list[list[int]]
    -   List of lists where each list contains the indices for a group of items
        in 'vals' that share the same value.

    unique_vals : list
    -   List of the unique values, corresponding to the groups in 'group_idcs'.

    RAISES
    ------
    EntryLengthError
    -   Raised if 'vals' and 'replacement_idcs' do not have the same length.
    """

    if replacement_idcs is None:
        replacement_idcs = range(len(vals))
    else:
        if len(replacement_idcs) != len(vals):
            raise EntryLengthError(
                "Error when trying to find the group indices of items:\nThe "
                "values and replacement indices do not have the same lengths "
                f"({len(vals)} and {len(replacement_idcs)}, respectively).\n"
            )

    unique_vals = unique(vals)
    group_idcs = []
    for unique_val in unique_vals:
        group_idcs.append([])
        for idx, val in enumerate(vals):
            if unique_val == val:
                group_idcs[-1].append(replacement_idcs[idx])

    return group_idcs, unique_vals


def reorder_rows_dataframe(
    dataframe: pd.DataFrame, key: str, values_order: list
) -> pd.DataFrame:
    """Reorders the rows of a pandas DataFrame based on the order in which
    values occur in a given column.

    If certain values are not present in the data, they are not included in the
    ordering. If no values are present in the data, no ordering is performed.

    PARAMETERS
    ----------
    dataframe : pandas DataFrame
    -   DataFrame to reorder.

    key : str
    -   Name of the column of the DataFrame to use for the reordering.

    values_order : list
    -   Values in the column of the DataFrame used for the reordering. The order
        in which values occur in 'values_order' determines the order in which
        the DataFrame rows are reordered.

    RETURNS
    -------
    dataframe : pandas DataFrame
    -   Reordered DataFrame.
    """
    unique_values = np.unique(dataframe[key].tolist())

    remove_values = []
    for value in values_order:
        if value not in unique_values:
            remove_values.append(value)
    values_order = [val for val in values_order if val not in remove_values]

    for value in unique_values:
        if value not in values_order:
            values_order.append(value)

    if values_order != []:
        dataframe = deepcopy(dataframe)
        dataframe = dataframe.set_index(key).loc[values_order].reset_index()

    return dataframe


def combine_col_vals_df(
    dataframe: pd.DataFrame,
    keys: Union[list[str], None] = None,
    idcs: Union[list[int], None] = None,
    special_vals: Union[dict[str], None] = None,
    joiner: str = " & ",
    include_keys: bool = True,
) -> list[str]:
    """Combines the values of DataFrame columns into a string on a row-by-row
    basis (i.e. one string for each row).

    PARAMETERS
    ----------
    dataframe : pandas DataFrame
    -   DataFrame whose values should be combined across columns.

    keys : list[str] | None
    -   Names of the columns in the DataFrame whose values should be combined.
    -   If 'None', all columns are used.

    idcs : list[int] | None
    -   Indices of the rows in the DataFrame whose values should be combined.
    -   If 'None', all rows are used.

    special_vals : dict[str] | None
    -   Instructions for how to treat specific values in the DataFrame.
    -   Keys are the special values that the values should begin with, whilst
        values are the values that the special values should be replaced with.
    -   E.g. {"avg[": "avg_"} would mean values in the DataFrame beginning with
        'avg[' would have this beginning replaced with 'avg_', followed by the
        column name, so a value beginning with 'avg[' in the 'channels' column
        would become 'avg_channels'.

    joiner : str; default " & "
    -   String to join each entry for a given row with.

    include_keys : bool; default True
    -   Whether or not the names of groups should contain the keys to which the
        values belong.

    RETURNS
    -------
    combined_vals : list[str]
    -   The values of the DataFrame columns combined on a row-by-row basis, with
        length equal to that of 'idcs'.
    """

    if keys is None:
        keys = dataframe.keys().tolist()
    if idcs is None:
        idcs = dataframe.index.tolist()
    if special_vals is None:
        special_vals = {}

    combined_vals = []
    for entry_i, row_i in enumerate(idcs):
        combined_vals.append("")
        for key in keys:
            value = str(dataframe[key].iloc[row_i])
            if include_keys:
                value = f"{key}-{value}"
            for to_replace, replacement in special_vals.items():
                start_i = len(f"{key}-")
                end_i = start_i + len(to_replace)
                if value[start_i:end_i] == to_replace:
                    value = f"{replacement}{key}"
            combined_vals[entry_i] += f"{value}{joiner}"
        combined_vals[entry_i] = combined_vals[entry_i][: -len(joiner)]

    return combined_vals


def combine_vals_list(
    obj: list,
    idcs: Union[list[int], None] = None,
    special_vals: Union[dict[str], None] = None,
    joiner: str = " & ",
) -> str:
    """Combines the values of DataFrame columns into a string on a row-by-row
    basis (i.e. one string for each row).

    PARAMETERS
    ----------
    obj : list
    -   List whose values should be combined.

    idcs : list[int] | None
    -   Indices of the values that should be combined.
    -   If 'None', all values are used.

    special_vals : dict[str] | None
    -   Instructions for how to treat specific values in the list.
    -   Keys are the special values that the values should begin with, whilst
        values are the values that the special values should be replaced with.
    -   E.g. {"avg[": "avg_"} would mean values in the list beginning with
        'avg[' would have this beginning replaced with 'avg_', followed by the
        column name, so a value beginning with 'avg[' in the 'channels' column
        would become 'avg_channels'.

    joiner : str; default " & "
    -   String to join each value with.

    RETURNS
    -------
    combined_vals : str
    -   The values of the list combined into a single string.
    """

    if idcs is None:
        idcs = np.arange(len(obj))
    if special_vals is None:
        special_vals = {}

    combined_vals = ""
    for val_i, val in enumerate(obj):
        if val_i in idcs:
            for to_replace, replacement in special_vals.items():
                if val[: len(to_replace)] == to_replace:
                    val = replacement
            combined_vals += f"{val}{joiner}"

    return combined_vals[: -len(joiner)]


def separate_vals_string(obj: str, separate_at: str) -> list[str]:
    """Splits a string into substrings based on the occurrence of characters
    within the string, printing a warning if no separation takes place.

    PARAMETERS
    ----------
    obj : str
    -   The string to separate.

    separate_at : str
    -   Characters in the string to split the data at.
    -   E.g. if 'separate_at' were " & ", the string "one & two" would be split
        into two strings: "one" and "two".

    RETURNS
    -------
    separated_vals : list[str]
    -   The string split into substrings.
    """

    separated_vals = []
    start_i = 0
    while True:
        separate_i = obj.find(separate_at, start_i)
        if separate_i == -1:
            separated_vals.append(obj[start_i:])
            break
        separated_vals.append(obj[start_i:separate_i])
        start_i = separate_i + len(separate_at)

    if len(separated_vals) == 1:
        print(
            "Warning: The string was not split into multiple substrings, as no "
            f"occurrence of '{separate_at}' within the input string was found."
        )

    return separated_vals


def rearrange_axes(
    obj: Union[list, NDArray], old_order: list[str], new_order: list[str]
) -> Union[list, NDArray]:
    """Rearranges the axes of an object.

    PARAMETERS
    ----------
    obj : list | numpy array
    -   The object whose axes should be rearranged.

    old_order : list[str]
    -   Names of the axes in 'obj' in their current positions.

    new_axes : list[str]
    -   Names of the axes in 'obj' in their desired positions.

    RETURNS
    -------
    list | numpy array
    -   The object with the rearranged axis order.
    """

    return np.transpose(obj, [old_order.index(dim) for dim in new_order])


def check_repeated_vals(
    to_check: list,
) -> tuple[bool, Optional[list]]:
    """Checks whether repeated values exist within an input list.

    PARAMETERS
    ----------
    to_check : list
    -   The list of values whose entries should be checked for repeats.

    RETURNS
    -------
    repeats : bool
    -   Whether or not repeats are present.

    repeated_vals : list | None
    -   The list of repeated values, or 'None' if no repeats are present.
    """

    seen = set()
    seen_add = seen.add
    repeated_vals = list(
        set(val for val in to_check if val in seen or seen_add(val))
    )
    if not repeated_vals:
        repeats = False
        repeated_vals = None
    else:
        repeats = True

    return repeats, repeated_vals


def check_non_repeated_vals_lists(
    lists: list[list], allow_non_repeated: bool = True
) -> bool:
    """Checks that each list in a list of lists contains values which also
    occur in each and every other list.

    PARAMETERS
    ----------
    lists : list[lists]
    -   Master list containing the lists whose values should be checked for
        non-repeating values.

    allow_non_repeated : bool; default True
    -   Whether or not to allow non-repeated values to be present. If not, an
        error is raised if a non-repeated value is detected.

    RETURNS
    -------
    all_repeated : bool
    -   Whether or not all values of the lists are present in each and every
        other list.

    RAISES
    ------
    MissingEntryError
    -   Raised if a list contains a value that does not occur in each and every
        other list and 'allow_non_repeated' is 'False'.
    """

    compare_list = lists[0]
    all_repeated = True
    checking = True
    while checking:
        for check_list in lists[1:]:
            non_repeated_vals = [
                val for val in compare_list if val not in check_list
            ]
            non_repeated_vals.extend(
                [val for val in check_list if val not in compare_list]
            )
            if non_repeated_vals:
                if not allow_non_repeated:
                    raise MissingEntryError(
                        "Error when checking whether all values of a list are "
                        "repeated in another list:\nThe value(s) "
                        f"{non_repeated_vals} is(are) not present in all "
                        "lists.\n"
                    )
                else:
                    all_repeated = False
                    checking = False
        checking = False

    return all_repeated


def check_matching_entries(objects: list) -> bool:
    """Checks whether the entries of objects match one another.

    PARAMETERS
    ----------
    objects : list
    -   The objects whose entries should be compared.

    RETURNS
    -------
    matching : bool
    -   If True, the entries of the objects match. If False, the entries do not
        match.

    RAISES
    ------
    EntryLengthError
    -   Raised if the objects do not have equal lengths.
    """

    equal, length = check_lengths_list_identical(objects)
    if not equal:
        raise EntryLengthError(
            "Error when checking whether the entries of objects are "
            f"identical:\nThe lengths of the objects ({length}) do not "
            "match."
        )

    checking = True
    matching = True
    while checking and matching:
        object_i = 1
        for entry_i, base_value in enumerate(objects[0]):
            for object_values in objects[1:]:
                object_i += 1
                if object_values[entry_i] != base_value:
                    matching = False
                    checking = False
        checking = False

    return matching


def check_master_entries_in_sublists(
    master_list: list,
    sublists: list[list],
    allow_duplicates: bool = True,
) -> tuple[bool, Optional[list]]:
    """Checks whether all values in a master list are present in a set of
    sublists.

    PARAMETERS
    ----------
    master_list : list
    -   A master list of values.

    sublists : list[list]
    -   A list of sublists of values.

    allow_duplicates : bool; default True
    -   Whether or not to allow duplicate values to be present in the sublists.

    RETURNS
    -------
    all_present : bool
    -   Whether all values in the master list were present in the sublists.

    absent_entries : list | None
    -   The entry/ies of the master list missing from the sublists. If no
        entries are missing, this is None.
    """

    combined_sublists = list(chain(*sublists))

    if not allow_duplicates:
        duplicates, duplicate_entries = check_repeated_vals(combined_sublists)
        if duplicates:
            raise DuplicateEntryError(
                "Error when checking the presence of master list entries "
                f"within a sublist:\nThe entries {duplicate_entries} are "
                "repeated within the sublists.\nTo ignore this error, set "
                "'allow_duplicates' to True."
            )

    all_present = True
    absent_entries = []
    for entry in master_list:
        if entry not in combined_sublists:
            all_present = False
            absent_entries.append(entry)
    if not absent_entries:
        absent_entries = None

    return all_present, absent_entries


def check_sublist_entries_in_master(
    master_list: list,
    sublists: list[list],
    allow_duplicates: bool = True,
) -> tuple[bool, Optional[list]]:
    """Checks whether all values in a set of sublists are present in a master
    list.

    PARAMETERS
    ----------
    master_list : list
    -   A master list of values.

    sublists : list[list]
    -   A list of sublists of values.

    allow_duplicates : bool; default True
    -   Whether or not to allow duplicate values to be present in the sublists.

    RETURNS
    -------
    all_present : bool
    -   Whether all values in the sublists were present in the master list.

    absent_entries : list | None
    -   The entry/ies of the sublists missing from the master list. If no
        entries are missing, this is None.
    """

    combined_sublists = list(chain(*sublists))

    if not allow_duplicates:
        duplicates, duplicate_entries = check_repeated_vals(combined_sublists)
        if duplicates:
            raise DuplicateEntryError(
                "Error when checking the presence of master list entries "
                f"within a sublist:\nThe entries {duplicate_entries} are "
                "repeated within the sublists.\nTo ignore this error, set "
                "'allow_duplicates' to True."
            )

    all_present = True
    absent_entries = []
    for entry in combined_sublists:
        if entry not in master_list:
            all_present = False
            absent_entries.append(entry)
    if not absent_entries:
        absent_entries = None

    return all_present, absent_entries


def ordered_list_from_dict(list_order: list[str], dict_to_order: dict) -> list:
    """Creates a list from entries in a dictionary, sorted based on a given
    order.

    PARAMETERS
    ----------
    list_order : list[str]
    -   The names of keys in the dictionary, in the order that
        the values will occur in the list.

    dict_to_order : dict
    -   The dictionary whose entries will be added to the list.

    RETURNS
    -------
    list
    -   The ordered list.
    """

    return [dict_to_order[key] for key in list_order]


def ordered_dict_keys_from_list(
    dict_to_order: dict, keys_order: list[str]
) -> dict:
    """Reorders a dictionary so that the keys occur in a given order.

    PARAMETERS
    ----------
    dict_to_order : dict
    -   The dictionary to be ordered.

    keys_order : list[str]
    -   The order in which the keys should occur in the ordered dictionary.

    RETURNS
    -------
    ordered_dict : dict
    -   The dictionary with keys in a given order.
    """

    ordered_dict = {}
    for key in keys_order:
        ordered_dict[key] = dict_to_order[key]

    return ordered_dict


def check_if_ragged(
    to_check: Union[
        list[Union[list, NDArray]],
        NDArray,
    ]
) -> bool:
    """Checks whether a list or array of sublists or subarrays is 'ragged' (i.e.
    has sublists or subarrays with different lengths).

    PARAMETERS
    ----------
    to_check : list[list | numpy array] | numpy array[list | numpy array]
    -   The list or array to check.

    RETURNS
    -------
    ragged : bool
    -   Whether or not 'to_check' is ragged.
    """

    identical, _ = check_lengths_list_identical(to_check=to_check)
    if identical:
        ragged = False
    else:
        ragged = True

    return ragged


def ragged_array_to_list(
    ragged_array: NDArray,
) -> list[list]:
    """Converts a ragged numpy array of nested arrays to a ragged list of nested
    lists.

    PARAMETERS
    ----------
    ragged_array : numpy array[numpy array]
    -   The ragged array to convert to a list.

    RETURNS
    -------
    ragged_list : list[list]
    -   The ragged array as a list.
    """

    ragged_list = []
    for array in ragged_array:
        ragged_list.append(array.tolist())

    return ragged_list


def drop_from_list(obj: list, drop: list[str]) -> list:
    """Drops specified entries from a list.

    PARAMETERS
    ----------
    obj : list
    -   List with entries that should be dropped.

    drop : list
    -   List of entries to drop.

    RETURNS
    -------
    new_obj : list
    -   List with specified entries dropped.
    """

    new_obj = []
    for item in obj:
        if item not in drop:
            new_obj.append(item)

    return new_obj


def drop_from_dict(obj: dict, drop: list[str], copy: bool = True) -> dict:
    """Removes specified entries from a dictionary.

    PARAMETERS
    ----------
    obj : dict
    -   Dictionary with entries to remove.

    drop : list[str]
    -   Names of the entries to remove.

    copy : bool; default True
    -   Whether or not to create a copy of the object from which the entries
        are dropped.

    RETURNS
    -------
    new_obj : dict
    -   Dictionary with entries removed.
    """
    if copy:
        new_obj = deepcopy(obj)
    else:
        new_obj = obj

    for item in drop:
        del new_obj[item]

    return new_obj


def combine_dicts(dicts: list[dict]) -> dict:
    """Combines a list of dictionaries into a single dictionary.

    PARAMETERS
    ----------
    dicts : list of dict
    -   The dictionaries to combine.

    RAISES
    ------
    KeyError
    -   Raised if not all keys in the dictionaries are unique.
    """
    all_keys = [
        key for single_dict in dicts for key in list(single_dict.keys())
    ]
    repeats_present, repeated_keys = check_repeated_vals(all_keys)
    if repeats_present:
        raise KeyError(
            "The dictionaries being combined must have unique keys, but the "
            f"following keys are shared: {repeated_keys}."
        )

    combined_dict = {}
    [combined_dict.update(single_dict) for single_dict in dicts]

    return combined_dict


def _check_dimensions_results(
    dimensions: list[Union[str, list[str]]],
    results_key: str,
) -> list[Union[str, list[str]]]:
    """Checks whether dimensions of results are in the correct format.

    PARAMETERS
    ----------
    dimensions : list[str] | list[list[str]]
    -   Dimensions of results, either a list of strings corresponding to the
        dimensions of all nodes/channels in the results, or a list of lists of
        strings, where each dimension corresponds to an individual node/channel.
    -   In the latter case, the dimensions of individual nodes/channels should
        not contain the axis "channel", as this is already the case. The
        "channels" axis will be set to the 0th axis, followed by the
        "frequencies" axis, followed by any other axes.
    -   E.g. if two channels were present, dimensions could be ["channels",
        "frequencies", "epochs", "timepoints"] or [["frequencies", "epochs",
        "timepoints"], ["timepoints", "frequencies", "epochs"]]. In the former
        case, the dimensions would be taken as-is. In the latter case,
        "channels" would be set to the 0th axis, and "frequencies" to the 1st
        axis, followed by any additional axes based on the order in which they
        occur in the first sublist, resulting in dimensions for all
        nodes/channels of ["channels", "frequencies", "epochs", "timepoints"].

    results_key : str
    -   Name of the entry in the results the dimensions are for.

    RETURNS
    -------
    dimensions : list[str] | list[list[str]]
    -   Dimensions of the results.
    -   If 'dimensions' was a list of strings, 'dimensions' is unchanged.
    -   If 'dimensions' was a list of sublists of strings in which each sublist
        of strings was the same, 'dimensions' is reduced to a single list of
        strings in which "channels" is set to the 0th axis.
    -   If 'dimensions' was a list of sublists of strings and each sublist of
        strings was not the same, 'dimensions' remains as a list of sublists.

    dims_to_find : list[str]
    -   Names of the dimensions and the order in which they should occur. If
        dimensions are given for each individual channel/node, the 0th axis is
        set to "frequencies", with the following axes set to the order in which
        they occur in the first node/channel.

    RAISES
    ------
    ValueError
    -   Raised if both "channels" and "connections" are present in the
        dimensions.
    -   Raised if the dimensions are a list of sublists corresponding to the
        dimensions of individual channels/connections, but with "channels"
        already being included in the dimensions of these individual
        channels/connections which is, by their very nature, incorrect.
    """
    if "connections" in dimensions:
        signal_dim = "connections"
    elif "channels" in dimensions:
        signal_dim = "channels"
    elif "connections" in dimensions and "channels" in dimensions:
        raise NotImplementedError(
            "Processing results containing data for both 'channels' and "
            "'connections' is not supported."
        )
    else:
        signal_dim = "channels"

    identical_dimensions = True
    if all(isinstance(entry, list) for entry in dimensions):
        for dims in dimensions:
            if signal_dim in dims:
                raise ValueError(
                    "Error when trying to sort the dimensions of the results:\n"
                    "Multiple dimensions for the results entry "
                    f"'{results_key}' are present. In this case, it is assumed "
                    "that each entry in the dimensions corresponds to each "
                    f"channel/node in the results. As a result, a {signal_dim} "
                    "axis should not be present in the dimensions, but it is "
                    f"{dims}.\nEither provide dimensions which are only a "
                    "single list of strings that applies to all "
                    "channels/nodes, or give dimensions for each individual "
                    f"channel/node (in which case no {signal_dim}) axis should "
                    "be present in the dimensions)."
                )
        identical_dimensions, _ = check_vals_identical_list(to_check=dimensions)
        if identical_dimensions:
            dimensions = [signal_dim, *dimensions[0]]
        else:
            check_non_repeated_vals_lists(
                lists=dimensions, allow_non_repeated=False
            )

    if identical_dimensions:
        dims_to_find = [signal_dim]
        [
            dims_to_find.append(dim)
            for dim in dimensions
            if dim not in dims_to_find
        ]
    else:
        dims_to_find = dimensions[0]

    return dimensions, dims_to_find


def _sort_dimensions_results(results: dict, verbose: bool) -> tuple[dict, list]:
    """Rearranges the dimensions of attributes in a results dictionary so that
    the 0th axis corresponds to results from different channels/nodes, and the
    1st dimension to different frequencies. If no dimensions, are given, the 0th
    axis is assumed to correspond to channels/nodes and the 1st axis to
    frequencies.
    -   Dimensions for an attribute, say 'X', would be contained in an
        attribute of the results dictionary under the name 'X_dimensions'.
    -   The dimensions should be provided as a list of strings containing the
        values "channels" or "nodes" and "frequencies" in the positions whose
        index corresponds to these axes in the values of 'X'. A single list
        should be given, i.e. 'X_dimensions' should hold for all entries of 'X'.
    -   E.g. if 'X' has shape [25, 10, 50, 300] with an 'X_dimensions' of
        [epochs x channels/nodes x frequencies x timepoints], the shape of 'X'
        would be rearranged to [10, 50, 25, 300], corresponding to the
        dimensions [channels/nodes x frequencies x epochs x timepoints].
    -   The axis for channels/nodes should be indicated as "channels" or "nodes,
        respectively, and the axis for frequencies should be marked as
        "frequencies".
    -   If the dimensions is a list of lists of strings, there should be a
        sublist for each channel/node in the results. Dimensions in the sublists
        should correspond to the results of each individual channel/node (i.e.
        no "channel" or "node" axis should be present in the dimensions of an
        individual node/channel as this is agiven).

    PARAMETERS
    ----------
    results : dict
    -   The results with dimensions of attributes to rearrange.

    verbose : bool
    -   Whether or not to report changes to the dimensions.

    RETURNS
    -------
    results : dict
    -   The results with dimensions of attributes in the appropriate order.

    dims_keys : list[str] | empty list
    -   Names of the dimension attributes in the results dictionary, or an empty
        list if no attributes are given.
    """
    dims_keys = []
    for key in results.keys():
        dims_key = f"{key}_dimensions"
        new_dims_set = False
        if dims_key in results.keys():
            dimensions = results[dims_key]
            if "connections" in dimensions:
                signal_dim = "connections"
            elif "channels" in dimensions:
                signal_dim = "channels"
            elif "nodes" in dimensions and "channels" in dimensions:
                raise NotImplementedError(
                    "Processing results containing data for both 'channels' "
                    "and 'connections' is not supported."
                )
            else:
                signal_dim = "channels"
            dimensions, dims_to_find = _check_dimensions_results(
                dimensions=results[dims_key], results_key=key
            )
            if all(isinstance(entry, list) for entry in dimensions):
                for node_i, dims in enumerate(dimensions):
                    curr_axes_order = np.arange(len(dims)).tolist()
                    new_axes_order = [dims.index(dim) for dim in dims_to_find]
                    if new_axes_order != curr_axes_order:
                        results.at[node_i, key] = np.transpose(
                            results[key][node_i],
                            new_axes_order,
                        ).tolist()
                new_dims = [signal_dim, *dims_to_find]
                if verbose:
                    print(
                        f"Changing the dimensions of '{key}' which were "
                        "variable across the nodes/channels to a single "
                        f"dimension {new_dims}.\n"
                    )
            else:
                curr_axes_order = np.arange(len(dimensions)).tolist()
                new_axes_order = [
                    dimensions.index(dim)
                    for dim in dims_to_find
                    if dim in dimensions
                ]
                if new_axes_order != curr_axes_order:
                    results[key] = np.transpose(
                        results[key],
                        new_axes_order,
                    ).tolist()
                    new_dims_set = True
                old_dims = deepcopy(dimensions)
                new_dims = [dimensions[i] for i in new_axes_order]
                if verbose and new_dims_set:
                    print(
                        f"Rearranging the dimensions of '{key}' from "
                        f"{old_dims} to {new_dims}.\n"
                    )
            results[dims_key] = new_dims[1:]
            dims_keys.append(dims_key)

    return results, dims_keys


def _check_entry_lengths_results(
    results: dict, ignore: Union[list[str], None]
) -> int:
    """Checks that the lengths of list and numpy array entries in 'results' have
    the same length of axis 0.

    PARAMETERS
    ----------
    results : dict
    -   The results whose entries will be checked.

    ignore : list[str] | None
    -   The entries in 'results' which should be ignored, such as those which
        are identical across channels and for which only one copy is present.
        These entries are not included when checking the lengths, as these will
        be handled later.

    RETURNS
    -------
    length : int
    -   The lengths of the 0th axis of lists and numpy arrays in 'results'.

    RAISES
    ------
    TypeError
    -   Raised if the 'results' contain an entry that is neither a list, numpy
        array, or dictionary.

    EntryLengthError
    -   Raised if the list or numpy array entries in 'results' do not all have
        the same length along axis 0.
    """

    if ignore is None:
        ignore = []

    supported_dtypes = [list, np.ndarray, dict]
    check_len_dtypes = [list, np.ndarray]

    to_check = []

    for key, value in results.items():
        if key not in ignore:
            dtype = type(value)
            if dtype in supported_dtypes:
                if dtype in check_len_dtypes:
                    to_check.append(value)
            else:
                raise TypeError(
                    "Error when trying to process the results:\nThe results "
                    f"dictionary contains an entry ('{key}') that is not of a "
                    f"supported data type ({supported_dtypes}).\n"
                )

    identical, length = check_lengths_list_identical(to_check=to_check, axis=0)
    if not identical:
        raise EntryLengthError(
            "Error when trying to process the results:\nThe length of "
            "entries in the results along axis 0 is not identical, but "
            "should be.\n"
        )

    return length


def _sort_identical_entries_results(
    results: dict,
    identical_entries: list[str],
    entry_length: int,
    verbose: bool,
) -> dict:
    """Creates a list equal to the length of other entries in 'results' for all
    entries specified in 'identical_entries', where each element of the list is
    a copy of the specified entries.

    PARAMETERS
    ----------
    results : dict
    -   The results dictionary with identical entries to sort.

    identical_entries : list[str]
    -   The entries in 'results' to convert to a list with length of axis 0
        equal to that of the 0th axis of other entries.

    entry_length : int
    -   The length of the 0th axis of entries in 'results'.

    verbose : bool
    -   Whether or not to print a description of the sorting process.

    RETURNS
    -------
    results : dict
    -   The results dictionary with identical entries sorted.
    """
    for entry in identical_entries:
        results[entry] = [deepcopy(results[entry]) for _ in range(entry_length)]

    if verbose:
        print(
            f"Creating lists of the entries {identical_entries} in the results "
            f"with length {entry_length}.\n"
        )

    return results


def _add_dict_entries_to_results(
    results: dict,
    extract: dict,
    entry_length: int,
    verbose: bool,
    extract_from: Union[dict, None] = None,
) -> dict:
    """Extracts entries from dictionaries in 'results' and adds them to the
    results as a list whose length matches that of the other 'results' entries
    which are lists or numpy arrays.

    PARAMETERS
    ----------
    results : dict
    -   The results containing the dictionaries whose values should be
        extracted.

    extract : dict
    -   Dictionary whose keys are the names of dictionaries in 'results', and
        whose values correspond to the entries in the dictionaries in 'results'
        to extract. Subdictionaries in 'results' can be accessed using
        subdictionaries in 'extract', and the values of dictionary keys accessed
        using lists of strings.

    entry_length : int
    -   The length of the 0th axis of entries in 'results'.

    verbose : bool
    -   Whether or not to print a description of the sorting process.

    extract_from : dict | None; default None
    -   The dictionary from which the information should be extracted. If None,
        'results' is used.

    RETURNS
    -------
    results : dict
    -   The results with the desired dictionary entries extracted.
    """
    if extract_from is None:
        extract_from = results

    for dict_name, dict_entries in extract.items():
        for entry in dict_entries:
            if entry in results.keys():
                raise PreexistingAttributeError(
                    f"Error when processing the results:\nThe entry '{entry}' "
                    f"from the dictionary '{dict_name}' is being extracted and "
                    "added to the results, however an attribute named "
                    f"'{entry}' is already present in the results.\n"
                )
            repeat_val = extract_from[dict_name][entry]
            if isinstance(repeat_val, dict):
                results = _add_dict_entries_to_results(
                    results=results,
                    extract=dict_entries,
                    entry_length=entry_length,
                    verbose=False,
                    extract_from=results[dict_name],
                )
            else:
                results[entry] = [deepcopy(repeat_val)] * entry_length

        if verbose:
            print(
                f"Extracting the entries {dict_entries} from the dictionary "
                f"'{dict_name}' into the results with length {entry_length}.\n"
            )

    return results


def _drop_dicts_from_results(results: dict) -> dict:
    """Removes dictionaries from 'results' after the requested entries, if
    applicable, have been extracted.

    PARAMETERS
    ----------
    results : dict
    -   The results with dictionaries entries to drop.

    RETURNS
    -------
    results : dict
    -   The results with dictionary entries dropped.
    """

    to_drop = []
    for key, value in results.items():
        if isinstance(value, dict):
            to_drop.append(key)

    for key in to_drop:
        del results[key]

    return results


def _sort_dicts_results(
    results: dict,
    extract_from_dicts: Union[dict, None],
    entry_length: int,
    verbose: bool,
) -> dict:
    """Handles the presence of dictionaries within 'results', extracting the
    requested entries, if applicable, before discarding the dictionaries.

    PARAMETERS
    ----------
    results : dict
    -   The results to sort.

    extract_from_dicts : dict | None
    -   The entries of dictionaries within 'results' to include in the
        processing. Subdictionaries of dictionaries can be accessed.
    -   Entries which are extracted are treated as being identical for all
        values in the 'results' dictionary.

    entry_length : int
    -   The length of the 0th axis of entries in 'results'.

    verbose : bool
    -   Whether or not to print a description of the sorting process.

    RETURNS
    -------
    dict
    -   The sorted results, with the desired dictionary entries extracted, if
        applicable, and the dictionaries discarded.
    """

    if extract_from_dicts is not None:
        results = _add_dict_entries_to_results(
            results=results,
            extract=extract_from_dicts,
            entry_length=entry_length,
            verbose=verbose,
        )

    return _drop_dicts_from_results(results=results)


def sort_inputs_results(
    results: dict,
    extract_from_dicts: Union[dict, None],
    identical_keys: Union[list[str], None],
    discard_keys: Union[list[str], None],
    verbose: bool = True,
) -> None:
    """Checks that the values in 'results' are in the appropriate format for
    processing with PostProcess or Plotting class objects.

    PARAMETERS
    ----------
    results : dict
    -   The results which will be checked.
    -   Entries which are lists or numpy arrays should have the same length
        of axis 0.

    extract_from_dicts : dict | None
    -   The entries of dictionaries within 'results' to include in the
        processing. Subdictionaries of dictionaries can be accessed.
    -   Entries which are extracted are treated as being identical for all
        values in the 'results' dictionary.

    identical_keys : list[str] | None
    -   The keys in 'results' which are identical across channels and for
        which only one copy is present.

    discard_keys : list[str] | None
    -   The keys which should be discarded immediately without
        processing.

    RETURNS
    -------
    dict
    -   The results with requested dictionary keys extracted to the results, if
        applicable, and the dictionaries subsequently removed.
    """
    if discard_keys is not None:
        results = drop_from_dict(obj=results, drop=discard_keys, copy=False)

    results, dims_keys = _sort_dimensions_results(
        results=results, verbose=verbose
    )

    if identical_keys is None:
        identical_keys = []
    identical_keys = [*identical_keys, *dims_keys]

    entry_length = _check_entry_lengths_results(
        results=results, ignore=identical_keys
    )

    results = _sort_dicts_results(
        results=results,
        extract_from_dicts=extract_from_dicts,
        entry_length=entry_length,
        verbose=verbose,
    )

    if identical_keys is not None:
        results = _sort_identical_entries_results(
            results=results,
            identical_entries=identical_keys,
            entry_length=entry_length,
            verbose=verbose,
        )

    return results


def dict_to_df(obj: dict) -> pd.DataFrame:
    """Converts a dictionary into a pandas DataFrame.

    PARAMETERS
    ----------
    obj : dict
    -   Dictionary to convert.

    RETURNS
    -------
    pandas DataFrame
    -   The converted dictionary.
    """

    return pd.DataFrame.from_dict(data=obj, orient="columns")


def check_posdef(A: NDArray) -> bool:
    """Checks whether a matrix is positive-definite.

    PARAMETERS
    ----------
    A : numpy array
    -   The matrix to check the positive-definite nature of.

    RETURNS
    -------
    is_posdef : bool
    -   Whether or not the matrix is positive-definite.

    NOTES
    -----
    -   First checks if the matrix is symmetric, and then performs a Cholesky
        decomposition.
    -   If the matrix is not symmetric, the positive-definite nature is
        determined to be false.
    -   If the matrix is symmetric and the Cholesky decomposition fails, the
        positive-definite nature is determined to be false, otherwise the matrix
        is said to be positive-definite.
    """

    is_posdef = True
    if np.allclose(A, A.T):  # For differences due to floating-point errors
        try:
            np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            is_posdef = False
    else:
        is_posdef = False

    return is_posdef


def create_lambda(obj: Any) -> Any:
    """Creates a lambda from an object, useful for when the object has been
    created in a for loop."""
    return lambda: obj


def check_svd_params(n_signals: int, take_n_components: int) -> None:
    """Checks that the parameters used for a singular value decomposition (SVD)
    are compatible with the data being used.

    PARAMETERS
    ----------
    n_signals : int
    -   The number of signals in the data the SVD is being performed on. This is
        the maximum number of components that can be taken from the SVD.

    take_n_components : int
    -   The number of components being taken from the SVD.

    RAISES
    ------
    ValueError
    -   Raised if 0 components are being taken from the SVD, or the number of
        components being taken are greater than the number of signals (i.e. the
        maximum number of components available).
    """
    if take_n_components == 0:
        raise ValueError(
            "0 components are being taken from the singular value "
            "decomposition, but this must be at least 1."
        )
    if take_n_components > n_signals:
        raise ValueError(
            f"At most {n_signals} components can be taken from the singular "
            f"value decomposition, but {take_n_components} are being taken."
        )


def loop_index(reset_at: int, idx: int) -> int:
    """Resets an index value to 0 if a value is reached or exceeded.

    PARAMETERS
    ----------
    reset_at : int
    -   The value to reset the index to if it is reached or exceeded.

    idx : int
    -   The index value.

    RETURNS
    -------
    int
    -   The new index value, reset to 0 if conditions are met.
    """
    if idx >= reset_at:
        return 0
    else:
        return idx
