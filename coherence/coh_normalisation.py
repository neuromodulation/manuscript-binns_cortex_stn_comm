"""Functions for normalising data.

METHODS
-------
find_exclusion_indices
-   Finds the indices of the frequencies to exclude from the normalisation
    calculation.

find_inclusion_indices
-   Finds the indices of the frequencies to include in the normalisation
    calculation.

sort_data_dims
-   Sorts the dimensions of the data being normalised so that the dimension
    being normalised is dimension 0.

restore_data_dims
-   Restores the dimensions of the data to their original format before
    normalisation.

norm_percentage_total
-   Applies percentage total normalisation to the data.
"""

from typing import Union
import numpy as np
from scipy.special import erfinv
from coh_exceptions import EntryLengthError, UnavailableProcessingError
from coh_handle_entries import check_lengths_list_identical


def find_exclusion_indices(
    freqs: list[Union[int, float]],
    line_noise_freq: Union[int, float],
    exclusion_window: Union[int, float],
    freq_range: list[int | float],
) -> list[int]:
    """Finds the indices of the frequencies to exclude from the normalisation
    calculation.

    PARAMETERS
    ----------
    freqs : list[int | float]
    -   The frequencies (in Hz) corresponding to the values in 'data'.

    line_noise_freq : int | float
    -   The frequency (in Hz) of the line noise in the data.

    exclusion_window : int | float
    -   The size of the windows (in Hz) to exclude frequencies around the line
        noise and harmonic frequencies from the calculations of what to
        normalise the data by.
    -   If 0, no frequencies are excluded.
    -   E.g. if the line noise is 50 Hz and 'exclusion_line_noise_window' is 10,
        the results from 45 - 55 Hz would be omitted.

    freq_range : list of int or float
        Frequency range (in Hz) to use for computing the normalisation,
        consisting of the lower and upper frequency, respectively.

    RETURNS
    -------
    exclusion_indices : list[int]
    -   The indices of the entries in 'freqs' to exclude from the normalisation
        calculation.

    RAISES
    ------
    ValueError
    -   Raised if the 'exclusion_window' or `exclude_low_freq` is less than 0.
    """
    if freq_range[0] >= freq_range[1]:
        raise ValueError(
            "First entry of `freq_range` must be less than the second entry."
        )
    if (
        exclusion_window < 0
        or freq_range[0] < 0
        or freq_range[1] > np.max(freqs)
    ):
        raise ValueError(
            "Error when finding indices of data to exclude:\nThe exclusion "
            "window and frequency bounds to exclude must be >= 0 and <= "
            "max(freqs)."
        )

    half_window = exclusion_window / 2
    exclusion_indices = []
    bad_freqs = np.arange(
        start=line_noise_freq,
        stop=freqs[-1] + line_noise_freq,
        step=line_noise_freq,
    )
    for freq_i, freq in enumerate(freqs):
        if freq <= freq_range[0] or freq >= freq_range[1]:
            exclusion_indices.append(freq_i)
        else:
            for bad_freq in bad_freqs:
                if (
                    freq >= bad_freq - half_window
                    and freq <= bad_freq + half_window
                ):
                    exclusion_indices.append(freq_i)

    return np.unique(exclusion_indices)  # to be safe in case of overlaps


def find_inclusion_indices(
    freqs: list[Union[int, float]],
    line_noise_freq: Union[int, float],
    exclusion_window: Union[int, float],
    freq_range=list[int, float],
) -> list[int]:
    """Finds the indices of the frequencies to include in the normalisation
    calculation.

    PARAMETERS
    ----------
    freqs : list[int | float]
    -   The frequencies (in Hz) corresponding to the values in 'data'.

    line_noise_freq : int | float
    -   The frequency (in Hz) of the line noise in the data.

    exclusion_window : int | float
    -   The size of the windows (in Hz) to exclude frequencies around the line
        noise and harmonic frequencies from the calculations of what to
        normalise the data by.
    -   If 0, no frequencies are excluded.
    -   E.g. if the line noise is 50 Hz and 'exclusion_line_noise_window' is 10,
        the results from 45 - 55 Hz would be omitted.

    freq_range : list of int or float | None; default None
        Frequency range (in Hz) to use for computing the normalisation,
        consisting of the lower and upper frequency, respectively.

    RETURNS
    -------
    inclusion_indices : list[int]
    -   The indices of the entries in 'freqs' to include in the normalisation
        calculation.
    """
    if (
        exclusion_window is not None or exclusion_window != 0
    ) and freq_range is not None:
        exclusion_indices = find_exclusion_indices(
            freqs=freqs,
            line_noise_freq=line_noise_freq,
            exclusion_window=exclusion_window,
            freq_range=freq_range,
        )
    else:
        exclusion_indices = []

    return [i for i in range(len(freqs)) if i not in exclusion_indices]


def sort_data_dims(
    data: np.ndarray,
    data_dims: list[str],
    within_dim: str,
) -> tuple[np.ndarray, list[str]]:
    """Sorts the dimensions of the data being normalised so that the dimension
    being normalised is dimension 0.

    PARAMETERS
    ----------
    data : numpy ndarray
    -   The data to normalise.

    data_dims : list[str]
    -   Descriptions of the data dimensions.

    within_dim : str
    -   The dimension to apply the normalisation within.
    -   E.g. if the data has dimensions "channels" and "frequencies", setting
        'within_dims' to "channels" would normalise the data across the
        frequencies within each channel.

    RETURNS
    -------
    data : numpy ndarray
    -   The data with the dimension being normalised as the 0th dimension.

    sorted_data_dims : list[str]
    -   The dimensions of the sorted data.
    """
    within_dim_i = data_dims.index(within_dim)
    if within_dim_i != 0:
        sorted_data_dims = [within_dim].extend(
            [dim for dim in data_dims if dim != within_dim]
        )
        transposition_indices = [
            data_dims.index(dim) for dim in sorted_data_dims
        ]
        data = np.transpose(data, transposition_indices)
    else:
        sorted_data_dims = data_dims

    return data, sorted_data_dims


def restore_data_dims(
    data: np.ndarray, current_dims: list[str], restore_dims: list[str]
) -> np.ndarray:
    """Restores the dimensions of the data to their original format before
    normalisation.

    PARAMETERS
    ----------
    data : numpy ndarray
    -   The data whose dimensions will be restored.

    current_dims : list[str]
    -   The dimensions of 'data'.

    restore_dims : list[str]
    -   The dimensions of 'data' to restore.

    RETURNS
    -------
    data : numpy ndarray
    -   The data with restored dimensions.
    """
    identical, lengths = check_lengths_list_identical(
        to_check=[data.shape, current_dims, restore_dims]
    )
    if not identical:
        raise EntryLengthError(
            "Error when restoring the dimensions of the data after "
            "normalisation:\nThe lengths of the actual data dimensions "
            f"({lengths[0]}), specified data dimensions ({lengths[1]}), and "
            f"desired data dimensions ({lengths[2]}) must match."
        )

    return np.transpose(data, [current_dims.index(dim) for dim in restore_dims])


def norm_percentage_total(
    data: np.ndarray,
    freqs: list[Union[int, float]],
    data_dims: list[str],
    within_dim: str,
    line_noise_freq: Union[int, float],
    exclusion_window: Union[int, float],
    freq_range: list[int | float],
) -> np.ndarray:
    """Applies percentage total normalisation to the data.

    PARAMETERS
    ----------
    data : numpy ndarray
    -   The data to normalise.

    freqs : list[int | float]
    -   The frequencies (in Hz) corresponding to the values in 'data'.

    data_dims : list[str]
    -   Descriptions of the data dimensions.

    within_dim : str
    -   The dimension to apply the normalisation within.
    -   E.g. if the data has dimensions "channels" and "frequencies", setting
        'within_dims' to "channels" would normalise the data across the
        frequencies within each channel.
    -   Currently, normalising only two-dimensional data is supported.

    line_noise_freq : int | float
    -   The frequency (in Hz) of the line noise in the data.

    exclusion_window : int | float
    -   The size of the windows (in Hz) to exclude frequencies around the line
        noise and harmonic frequencies from the calculations of what to
        normalise the data by.
    -   If 0, no frequencies are excluded.
    -   E.g. if the line noise is 50 Hz and 'exclusion_line_noise_window' is 10,
        the results from 45 - 55 Hz would be omitted.

    freq_range : list of int or float | None; default None
        Frequency range (in Hz) to use for computing the normalisation,
        consisting of the lower and upper frequency, respectively.
    """
    if len(data_dims) > 2 or len(data.shape) > 2:
        raise UnavailableProcessingError(
            "Error when percentage-total normalising the data:\nOnly "
            "two-dimensional data can be normalised."
        )

    data, new_data_dims = sort_data_dims(
        data=data, data_dims=data_dims, within_dim=within_dim
    )

    inclusion_idcs = find_inclusion_indices(
        freqs=freqs,
        line_noise_freq=line_noise_freq,
        exclusion_window=exclusion_window,
        freq_range=freq_range,
    )

    for data_i in range(data.shape[0]):
        data[data_i] = (
            data[data_i] / np.sum(data[data_i][inclusion_idcs])
        ) * 100

    return restore_data_dims(
        data=data, current_dims=new_data_dims, restore_dims=data_dims
    )


def gaussian_transform(data: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Gaussianises data to have mean = 0 and standard deviation = 1.

    PARAMETERS
    ----------
    data : numpy ndarray
    -   Array containing the data to Gaussianise.

    axis : int | None (default None)
    -   The axis to perform Gaussianisation over. If None, the entire array is
        Gaussianised across at once.

    RETURNS
    -------
    gaussianised_data : numpy ndarray
    -   The Gaussianised data.

    REFERENCES
    ----------
    [1] Van Albada & Robinson (2007). Journal of Neuroscience Methods. DOI:
    10.1016/j.jneumeth.2006.11.004.
    """
    if axis is None:
        data_shape = data.shape
        data = data.flatten()
    else:
        data = data.transpose(
            (axis, *[i for i in range(data.ndim) if i != axis])
        )

    gaussianised_data = _compute_gaussianisation(data)

    if axis is None:
        gaussianised_data = gaussianised_data.reshape(data_shape)
    else:
        reverse_transposition = []
        for i in range(data.ndim):
            if i >= axis:
                reverse_transposition.append(i)
            else:
                reverse_transposition.append(i + 1)
        gaussianised_data = np.transpose(
            gaussianised_data, reverse_transposition
        )

    return gaussianised_data


def _compute_gaussianisation(
    data: np.ndarray, axis: int | None = None
) -> np.ndarray:
    """Gaussianises data to have mean = 0 and standard deviation = 1.

    PARAMETERS
    ----------
    data : numpy ndarray
    -   Array containing the data to Gaussianise.

    axis : int | None; default None
    -   Axis of the array to Gaussianise across. If None, the whole array is
        Gaussianised together.

    RETURNS
    -------
    gaussianised_data : numpy ndarray
    -   The Gaussianised data.

    REFERENCES
    ----------
    [1] Van Albada & Robinson (2007). Journal of Neuroscience Methods. DOI:
    10.1016/j.jneumeth.2006.11.004.
    """
    if axis is None:
        data_shape = data.shape
        data = data.flatten()

    if data.ndim == 1:
        n = np.unique(data, return_inverse=True)[1]
        sorted_n = np.sort(n)
        new_sorted = sorted_n.copy()
        indices = np.argsort(np.argsort(n))

        ties = 0
        for idx, val in enumerate(sorted_n[:-1]):
            if val == sorted_n[idx + 1]:
                ties += 1
            else:
                new_sorted[idx + 1 :] = new_sorted[idx + 1 :] + ties

        rank = new_sorted[indices] + 1

        cdf = rank / len(data) - 1 / (2 * len(data))

        gaussianised_data = np.sqrt(2) * erfinv(2 * cdf - 1)
    else:
        gaussianised_data = np.zeros(data.shape)
        dim_shapes = data.shape[:-1]
        for idx in range(data.shape[-1]):
            idcs = (*[np.arange(shape) for shape in dim_shapes], idx)
            gaussianised_data[idcs] = _compute_gaussianisation(data=data[idcs])

    if axis is None:
        gaussianised_data.reshape(data_shape)

    return gaussianised_data
