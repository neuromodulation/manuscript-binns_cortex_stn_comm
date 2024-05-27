"""Class for performing SSD."""

from copy import deepcopy
from typing import Any
from matplotlib import pyplot as plt
from mne import time_frequency
from mne.decoding import SSD
import numpy as np
from coh_exceptions import ProcessingOrderError
from coh_handle_entries import (
    combine_dicts,
    combine_vals_list,
    get_eligible_idcs_lists,
    get_group_names_idcs,
    ordered_list_from_dict,
    rearrange_axes,
    unique,
)
from coh_processing_methods import ProcMethod
from coh_saving import save_dict
from coh_signal import Signal


class PowerSSD(ProcMethod):
    """Perform SSD analysis"""

    channels = None
    transformed_ch_names = None
    _original_ch_names = None
    _channels_str = None
    _n_groups = None
    _group_ch_types = None
    _group_ch_names = None

    filt_params_signal = None
    _filt_params_signal = None
    filt_params_noise = None
    _filt_params_noise = None
    _filt_params_names = None
    _n_bands = None
    group_ranks = None

    regularisation = None
    covariance_params = None

    _plot = None

    _ssd_results = None

    _windows_averaged = False

    _power_computed = False

    def __init__(self, signal: Signal, verbose: bool = True) -> None:
        super().__init__(signal=signal, verbose=verbose)
        self._sort_inputs()

    def _sort_inputs(self) -> None:
        super()._sort_inputs()

        if not self.signal._epoched:
            raise ProcessingOrderError(
                "SSD can only be computed on epoched data."
            )

    def fit_transform(
        self,
        channels: list[list[str]] | dict,
        filt_params_signal: dict[dict],
        filt_params_noise: dict[dict],
        regularisation: str | float | None = None,
        covariance_params: dict | None = None,
        group_ranks: list[int] | None = None,
        transformed_ch_names: list[str] | None = None,
        plot: bool = False,
        power_kwargs: dict = None,
    ) -> None:
        """Fits SSD models to the data and transforms it.

        PARAMETERS
        ----------
        channels : list of list of str | dict
        -   The channels to perform SSD on together (Note: MNE's SSD
            implementation only supports computing SSD on a single type of
            channels at a time). If a list of list of str, the str should be
            channel names grouped into lists of channels that will be processed
            together. If a dict, ...

        filt_params_signal : dict of dict
        -   Dictionary where the keys are the names of the signal bands, and the
            values as in `mne.decoding.ssd`. The names of the bands must match
            those in `filt_params_noise`.

        filt_params_noise : dict of dict
        -   Dictionary where the keys are the names of the noise bands, and the
            values as in `mne.decoding.ssd`. The names of the bands must match
            those in `filt_params_signal`.

        regularisation : str | float | None; default None
        -   As in `mne.decoding.SSD`.

        covariance_params : dict | None; default None
        -   As in `mne.decoding.SSD`.

        group_ranks : list of int | None; default None
        -   Rank subspace that the data of each group should be projected to
            before performing SSD.

        transformed_ch_names : list of str | None; default None
        -   Names to store the data for the transformed channels under. One name
            should be given for each group in `channels`. If None and `channels`
            is a dict, information from the channel grouping will be used to
            generate channel names, otherwise a generic "group-X" name will be
            used.

        plot : bool; default False
        -   Whether or not to plot information about the SSD.

        power_kwargs : dict
        -   Keyword arguments for the multitaper computation of MNE's
            psd_array_multitaper function.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the data in the object has already been processed.
        """
        if self._processed:
            raise ProcessingOrderError(
                "The data in the object has already been processed."
            )

        self.channels = deepcopy(channels)
        self.filt_params_signal = deepcopy(filt_params_signal)
        self.filt_params_noise = deepcopy(filt_params_noise)
        self.regularisation = deepcopy(regularisation)
        self.covariance_params = deepcopy(covariance_params)
        self.group_ranks = deepcopy(group_ranks)
        self.transformed_ch_names = deepcopy(transformed_ch_names)
        self._plot = deepcopy(plot)

        self._sort_processing_inputs()

        self._compute_ssd()
        if self._plot:
            self._plot_ssd_info()

        self._compute_power(**power_kwargs)

        self._sort_dimensions()

    def _sort_processing_inputs(self) -> None:
        """Sorts processing inputs for performing SSD."""
        self._sort_filt_params()
        self._sort_channels()
        self._sort_channel_names()
        self._sort_ranks()

    def _sort_channels(self) -> None:
        """Find the channels to perform SSD on together (if channels is a dict),
        and assigns these channel groups."""
        if isinstance(self.channels, dict):
            features = self._features_to_df()
            eligible_idcs = get_eligible_idcs_lists(
                features, self.channels["eligible_entries"]
            )
            group_idcs = get_group_names_idcs(
                features,
                self.channels["grouping"],
                eligible_idcs=eligible_idcs,
                replacement_idcs=eligible_idcs,
                keys_in_names=False,
            )
            self._group_ch_names = [
                key.replace(" & ", "_") for key in group_idcs.keys()
            ]
            channels_list = []
            channels_str = []
            for idcs in group_idcs.values():
                channels_list.append(
                    [self.signal.data[0].ch_names[idx] for idx in idcs]
                )
                channels_str.append(
                    combine_vals_list(
                        [self.signal.data[0].ch_names[idx] for idx in idcs]
                    )
                )
            self.channels = channels_list
            self._channels_str = channels_str
        else:
            if not isinstance(self.channels, list) or not isinstance(
                self.channels[0], list
            ):
                raise TypeError(
                    "'channels' should be a list of list of str, or a dict."
                )
            self._channels_str = [
                combine_vals_list(names) for names in self.channels
            ]

        self._n_groups = len(self.channels)

        group_ch_types = []
        for group in self.channels:
            ch_types = self.signal.data[0].get_channel_types(picks=group)
            if len(np.unique(ch_types)) != 1:
                raise ValueError(
                    "Channels being processed together with SSD must be of the "
                    "same type."
                )
            group_ch_types.append(ch_types[0])
        self._group_ch_types = group_ch_types

    def _sort_channel_names(self) -> None:
        """Sorts the names for the transformed channels."""
        if self.transformed_ch_names is None:
            if self._group_ch_names is not None:
                self.transformed_ch_names = deepcopy(self._group_ch_names)
            else:
                self.transformed_ch_names = [
                    f"ch_group-{group_i}"
                    for group_i in range(len(self.channels))
                ]
        else:
            if len(self.transformed_ch_names) != len(self.channels):
                raise ValueError(
                    "`transformed_ch_names` must equal the number of channel "
                    "groups."
                )

    def _sort_filt_params(self) -> None:
        """Ensures filter parameters are in an appropriate format."""
        if not isinstance(self.filt_params_signal, dict) or not isinstance(
            self.filt_params_noise, dict
        ):
            raise TypeError(
                "Filter parameters must be given as lists of dicts."
            )

        if self.filt_params_signal.keys() != self.filt_params_noise.keys():
            raise ValueError(
                "Filter parameters for signal and noise must contain "
                "information for the same bands."
            )

        self._filt_params_signal = list(self.filt_params_signal.values())
        self._filt_params_noise = list(self.filt_params_noise.values())
        self._filt_params_names = list(self.filt_params_signal.keys())

        self._n_bands = len(self._filt_params_signal)

    def _sort_ranks(self) -> None:
        """Check that the correct number of rank groups is given."""
        if self.group_ranks is None:
            self.group_ranks = [None for _ in range(len(self.channels))]
        if not isinstance(self.group_ranks, list):
            raise TypeError("`group_ranks` must be a list.")
        if len(self.group_ranks) != len(self.channels):
            raise ValueError(
                "`group_ranks` must be given for each channel group."
            )

    def _compute_ssd(self) -> None:
        """Fits data to SSD for each band and stores information about the
        transformed data, the spatial filters/patterns, and spectral ratios."""
        ssd_info = []
        for group_i in range(self._n_groups):
            ssd_info.append([])
            group_data = [
                deepcopy(data).pick(self.channels[group_i])
                for data in self.signal.data
            ]
            if self.group_ranks[group_i] is not None:
                rank = {
                    self._group_ch_types[group_i]: self.group_ranks[group_i]
                }
            else:
                rank = None
            for band_i in range(self._n_bands):
                ssd_info[group_i].append({})
                group_ssd = [
                    SSD(
                        info=group_data[0].info,
                        filt_params_signal=self._filt_params_signal[band_i],
                        filt_params_noise=self._filt_params_noise[band_i],
                        reg=self.regularisation,
                        n_components=None,
                        picks=group_data[0].ch_names,
                        sort_by_spectral_ratio=True,
                        return_filtered=True,
                        n_fft=group_data[0].get_data().shape[-1],
                        cov_method_params=self.covariance_params,
                        rank=rank,
                    ).fit(data.get_data())
                    for data in group_data
                ]

                if (
                    len(unique([ssd.patterns_.shape[0] for ssd in group_ssd]))
                    != 1
                ):
                    raise ValueError(
                        "The rank of the data is not identical across windows, "
                        "meaning there is a different number of SSD components "
                        "across windows, but this is not supported."
                    )

                ssd_info[group_i][band_i]["data"] = np.array(
                    [
                        ssd.transform(data.get_data())
                        for ssd, data in zip(group_ssd, group_data)
                    ]
                )
                ssd_info[group_i][band_i]["filters"] = np.array(
                    [ssd.filters_ for ssd in group_ssd]
                )
                ssd_info[group_i][band_i]["patterns"] = np.array(
                    [ssd.patterns_.T for ssd in group_ssd]
                )
                ssd_info[group_i][band_i]["eigenvalues"] = np.array(
                    [ssd.eigvals_ for ssd in group_ssd]
                )
                ssd_info[group_i][band_i]["spectral_ratios"] = np.array(
                    [
                        ssd.get_spectral_ratio(data)[0]
                        for ssd, data in zip(
                            group_ssd, ssd_info[group_i][band_i]["data"]
                        )
                    ]
                )
                ssd_info[group_i][band_i]["n_comps"] = group_ssd[
                    0
                ].patterns_.shape[0]

        self._results = ssd_info
        self._sort_transformed_ch_names()
        self._generate_extra_info()

    def _sort_transformed_ch_names(self) -> None:
        """Gets the names of the transformed channels for each band and
        component."""
        for group_i, group_name in enumerate(self.transformed_ch_names):
            for band_i, band_name in enumerate(self._filt_params_names):
                n_comps = self._results[group_i][band_i]["n_comps"]
                self._results[group_i][band_i]["transformed_ch_names"] = []
                for comp_i in range(n_comps):
                    self._results[group_i][band_i][
                        "transformed_ch_names"
                    ].append(f"{group_name}_{band_name}_comp-{comp_i+1}")

        self._transformed_ch_names = self._get_band_feature_info(
            copy_from="transformed_ch_names",
            repeat_for_channels=False,
            copy_mode="repeat",
        )

    def _generate_extra_info(self) -> None:
        """Generates additional information related to the SSD analysis."""
        self._generate_transformed_ch_types()
        self._generate_transformed_ch_coords()

        attributes = [
            "ch_reref_types",
            "ch_regions",
            "ch_subregions",
            "ch_hemispheres",
            "ch_epoch_orders",
        ]
        for attr in attributes:
            if (
                attr in self.signal.extra_info.keys()
                and self.signal.extra_info[attr] is not None
            ):
                self._generate_transformed_attribute(attr)

    def _generate_transformed_ch_types(self) -> None:
        """Gets the types of channels in the results.

        If the types of each channel in a group are identical, this type is
        given as a string, otherwise the unique types are taken and joined into
        a single string by the " & " characters.
        """
        ch_types = {}
        for group_i, transformed_name in enumerate(self.transformed_ch_names):
            ch_types[transformed_name] = combine_vals_list(
                unique(
                    self.signal.data[0].get_channel_types(
                        picks=self.channels[group_i]
                    )
                )
            )

        self.extra_info["transformed_ch_types"] = {}
        for name in self._transformed_ch_names:
            for group_name in self.transformed_ch_names:
                if name.startswith(group_name):
                    self.extra_info["transformed_ch_types"][name] = ch_types[
                        group_name
                    ]
                    break

    def _generate_transformed_ch_coords(self) -> None:
        """Gets the coordinates of channels in the results for each channel."""
        ch_coords = {}
        for group_i, transformed_name in enumerate(self.transformed_ch_names):
            ch_coords[transformed_name] = self.signal.get_coordinates(
                picks=self.channels[group_i]
            )

        self.extra_info["transformed_ch_coords"] = {}
        for name in self._transformed_ch_names:
            for group_name in self.transformed_ch_names:
                if name.startswith(group_name):
                    self.extra_info["transformed_ch_coords"][name] = ch_coords[
                        group_name
                    ]
                    break

    def _generate_transformed_attribute(self, attribute: str) -> None:
        """Gets the information of transformed channels in the results.

        If the information of an attribute for each channel in a group are
        identical, this information is given as a string, otherwise the unique
        information is taken and joined into a single string by the " & "
        characters.

        PARAMETERS
        ----------
        attribute : str
        -   The name of the attribute in `extra_info` to transform.
        """
        info = {}
        for group_i, transformed_name in enumerate(self.transformed_ch_names):
            info[transformed_name] = combine_vals_list(
                unique(
                    ordered_list_from_dict(
                        list_order=self.channels[group_i],
                        dict_to_order=self.extra_info[attribute],
                    )
                )
            )

        transformed_info = {}
        for name in self._transformed_ch_names:
            for group_name in self.transformed_ch_names:
                if name.startswith(group_name):
                    transformed_info[name] = info[group_name]
                    break

        self.extra_info[f"transformed_{attribute}"] = transformed_info

    def _plot_ssd_info(self):
        """Plots the PSD of the SSD-filtered data and the spectral ratios of
        each SSD component for each channel group, frequency band, and
        window."""
        channel_names = deepcopy(self._channels_str)
        for group_i in range(self._n_groups):
            if len(channel_names[group_i]) > 100:
                channel_names[group_i] = f"{channel_names[group_i][:100]}..."
            for band_i in range(self._n_bands):
                ssd_info = self._results[group_i][band_i]

                signal_freqs = [
                    self._filt_params_signal[band_i]["l_freq"],
                    self._filt_params_signal[band_i]["h_freq"],
                ]
                noise_freqs = [
                    self._filt_params_noise[band_i]["l_freq"],
                    self._filt_params_noise[band_i]["h_freq"],
                ]
                for window_i in range(self._n_windows):
                    psd, freqs = time_frequency.psd_array_welch(
                        ssd_info["data"][window_i],
                        sfreq=self.signal.data[0].info["sfreq"],
                        n_fft=int(self.signal.data[0].info["sfreq"]),
                    )

                    plot_idcs = self._get_freq_idcs_for_plot(freqs)

                    fig, axis = plt.subplots(1, 1)
                    self._plot_ssd_psd(
                        axis, psd, freqs, plot_idcs, signal_freqs, noise_freqs
                    )
                    self._plot_ssd_inset(axis, ssd_info, window_i)

                    title = (
                        f"Channels: {channel_names[group_i]}\n\nSignal "
                        f"frequencies: {signal_freqs[0]} - {signal_freqs[1]} "
                        f"Hz     Noise frequencies: {noise_freqs[0]} - "
                        f"{noise_freqs[1]} Hz"
                    )
                    if self.signal._windowed:
                        title += f"     Window {window_i} of {self._n_windows}"
                    fig.suptitle(title)

                    fig.legend(loc="upper left")
                    plt.show()

    def _get_freq_idcs_for_plot(self, freqs: np.ndarray) -> list[int]:
        """Gets the indices of the frequencies to plot for visualising the SSD
        models based on the bandpass filtering of the data.

        PARAMETERS
        ----------
        freqs : nunmpy ndarray
        -   The available frequencies to plot.

        RETURNS
        -------
        plot_freq_idcs : list[int]
        -   Indices of the lower and upper frequencies, respectively, to plot.
        """
        plot_freq_range = [freqs[0], freqs[-1]]
        if self.signal.data[0].info["highpass"]:
            plot_freq_range[0] = self.signal.data[0].info["highpass"]
        if self.signal.data[0].info["lowpass"]:
            plot_freq_range[-1] = self.signal.data[0].info["lowpass"]
        return [
            np.where(freqs == plot_freq_range[0])[0][0],
            np.where(freqs == plot_freq_range[1])[0][0],
        ]

    def _plot_ssd_psd(
        self,
        axis: plt.Axes,
        psd: np.ndarray,
        freqs: np.ndarray,
        plot_idcs: list[int],
        signal_freqs: list[float],
        noise_freqs: list[float],
    ) -> None:
        """Plot the PSD of the SSD-transformed data and highlight the signal
        and noise segments.

        PARAMETERS
        ----------
        axis : matplotlib pyplot Axes
        -   The subplot axis on which to plot the PSD.

        psd : numpy ndarray
        -   The PSD of the SSD-transformed data.

        freqs : numpy ndarray
        -   The frequencies corresponding to `psd`.

        signal_freqs: list[int]
        -   The lower and upper bound, respectively, of the signal frequencies.

        noise_freqs: list[int]
        -   The lower and upper bound, respectively, of the noise frequencies.
        """
        for comp_i in range(psd.shape[1]):
            axis.plot(
                freqs[plot_idcs[0] : plot_idcs[1]],
                psd[:, comp_i, plot_idcs[0] : plot_idcs[1]].mean(axis=0),
                label=f"SSD component {comp_i + 1}",
            )

        axis.axvspan(signal_freqs[0], signal_freqs[1], alpha=0.2, color="grey")
        axis.axvspan(
            noise_freqs[0],
            signal_freqs[0],
            fill=False,
            alpha=0.3,
            hatch="///",
            color="grey",
        )
        axis.axvspan(
            signal_freqs[1],
            noise_freqs[1],
            fill=False,
            alpha=0.3,
            hatch="///",
            color="grey",
        )

        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Power (dB)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    def _plot_ssd_inset(
        self, axis: plt.Axes, ssd_info: dict, window_i: int
    ) -> None:
        """Plot the eigenvalues and spectral ratios of the SSD components as an
        inset alongside the PSD.

        PARAMETERS
        ----------
        axis : matplotlib pyplot Axes
        -   The subplot axis on which to plot the PSD.

        ssd_info : dict
        -   Dictionary containing the eigenvalues and spectral ratios.

        window_i : int
        -   The window of the data being plotted.
        """
        n_comps = len(ssd_info["eigenvalues"][window_i])
        components = np.arange(n_comps, dtype=int) + 1

        inset = axis.inset_axes([0.7, 0.6, 0.3, 0.4])
        inset.plot(
            components,
            ssd_info["eigenvalues"][window_i],
            label="eigenvalues",
            color="blue",
        )
        inset.set_xlim(inset.get_xlim())
        inset.set_ylabel("Eigenvalues (A.U.)")
        inset.grid(True, axis="x")

        inset_y2 = inset.twinx()
        inset_y2.plot(
            components, ssd_info["spectral_ratios"][window_i], color="orange"
        )
        inset.plot(
            -1,
            np.mean(inset.get_ylim()),
            color="orange",
            label="spectral ratios",
        )
        inset_y2.plot(components, [1] * n_comps, color="k", linestyle="--")
        inset_y2.set_ylabel("Spectral ratio (dB)")

        inset.set_xlabel("Components")
        inset.set_xticks(components)
        inset.set_xticklabels(components)
        inset.legend()

    def _compute_power(
        self,
        fmin: int | float = 0,
        fmax: float = np.inf,
        bandwidth: int | float | None = None,
        adaptive: bool = False,
        low_bias: bool = True,
        normalization: str = "length",
        n_jobs: int = 1,
    ) -> None:
        """Compute PSD for the transformed data using multitapers.

        PARAMETERS
        ----------
        fmin : int | float; default 0
        -   The minimum frequency of interest.

        fmax : int | float; default infinite
        -   The maximum frequency of interest.

        bandwidth : float | None; default None
        -   The bandwidth of the multitaper windowing function, in Hz. If
            'None', this is set to a window half-bandwidth of 4.

        adaptive : bool; default False
        -   Whether or not to use adaptive weights to combine the tapered
            spectra into the power spectral density.

        low_bias : bool; default True.
        -   Whether or not to use only tapers with more than 90% spectral
            concentration within bandwidth.

        normalization : str; default "length"
        -   The normalisation strategy to use. If "length", the power spectra is
            normalised by the length of the signal. If "full", the power spectra
            is normalised by the sampling rate and the signal length.

        n_jobs : int; default 1
        -   The number of jobs to run in parallel. If '-1', this is set to the
            number of CPU cores. Requires the 'joblib' package.
        """
        for group_i, group_results in enumerate(self._results):
            print(
                f"\n---=== Computing power for group {group_i+1} of "
                f"{len(self._results)} ===---\n"
            )
            for band_i, band_results in enumerate(group_results):
                psds = []
                for window_data in band_results["data"]:
                    psd, freqs = time_frequency.psd_array_multitaper(
                        x=window_data,
                        sfreq=self.signal.data[0].info["sfreq"],
                        fmin=fmin,
                        fmax=fmax,
                        bandwidth=bandwidth,
                        adaptive=adaptive,
                        low_bias=low_bias,
                        normalization=normalization,
                        n_jobs=n_jobs,
                        verbose=self.verbose,
                    )
                    psds.append(psd)
                self._results[group_i][band_i]["power"] = np.mean(psds, axis=1)
        self._freqs = freqs

        self._results_dims = ["windows", "channels", "frequencies"]
        self._power_computed = True

    def _sort_dimensions(self) -> None:
        """Removes the window dimension of the results if the data has not been
        windowed."""
        results_keys = [
            "data",
            "filters",
            "patterns",
            "eigenvalues",
            "spectral_ratios",
            "power",
        ]
        if "windows" not in self.results_dims:
            for group_i in range(self._n_groups):
                for band_i in range(self._n_bands):
                    for key in results_keys:
                        self._results[group_i][band_i][key] = self._results[
                            group_i
                        ][band_i][key][0]

    def power_as_array(self, dimensions: list[str] | None = None) -> np.ndarray:
        """Extracts and returns power of transformed data as an array.

        PARAMETERS
        ----------
        dimensions : list[str] | None;  default None
        -   The dimensions of the power results that will be returned. If
            'None', the current dimensions are used.

        RETURNS
        -------
        power : numpy ndarray
        -   The power of the transformed data.
        """
        power = []

        if dimensions is None:
            dimensions = self.results_dims

        for group_i in range(self._n_groups):
            for band_i in range(self._n_bands):
                power.extend(self._results[group_i][band_i]["power"])
        power = np.array(power)

        return rearrange_axes(
            obj=power, old_order=self.results_dims, new_order=dimensions
        )

    def topographies_as_array(self) -> np.ndarray:
        """Extracts the topographies as an array.

        RETURNS
        -------
        topographies : numpy ndarray
        -   The topographies for each group, band, channel, and component as a
            vector.
        """
        return self._rearrange_topographies()[0]

    def results_as_dict(self) -> dict:
        """Returns the SSD results and additional information as a dictionary.

        RETURNS
        -------
        results_dict : dict
        -   The results and additional information stored as a dictionary.
        """
        core_info = self._core_info_for_results_dict()
        extra_info = self._extra_info_for_results_dict()

        return combine_dicts([core_info, extra_info])

    def _core_info_for_results_dict(self) -> dict:
        """Returns core information about the connectivity results which is
        always present.

        RETURNS
        -------
        core_info : dict
        -   The core information about the connectivity results.
        """
        dimensions = self._get_optimal_dims()

        if "windows" in dimensions:
            side_result_dims = ["windows", "channels"]
        else:
            side_result_dims = ["channels"]

        power = self.power_as_array(dimensions=dimensions)

        core_info = {
            "power-ssd": power.tolist(),
            "frequencies": self._freqs.tolist(),
            "power-ssd_dimensions": dimensions,
            "component_eigenvalues": self._get_group_feature_info(
                copy_from="eigenvalues",
                repeat_for_channels=False,
                copy_mode="repeat",
            ),
            "component_eigenvalues_dimensions": side_result_dims,
            "component_spectral_ratios": self._get_group_feature_info(
                copy_from="spectral_ratios",
                repeat_for_channels=False,
                copy_mode="repeat",
            ),
            "component_spectral_ratios_dimensions": side_result_dims,
            "transformed_ch_names": self._transformed_ch_names,
            "band_names": self._get_band_feature_info(
                copy_from=self._filt_params_names,
                repeat_for_channels=False,
                copy_mode="repeat",
            ),
            "original_ch_names": self._get_group_feature_info(
                copy_from=self._channels_str,
                repeat_for_channels=False,
                copy_mode="repeat",
            ),
            "component_numbers": (
                np.array(
                    self._get_band_feature_info(
                        copy_from=None,
                        repeat_for_channels=False,
                        copy_mode="repeat",
                    )
                )
                + 1
            ).tolist(),
            "sampling_frequency": self.signal.data[0].info["sfreq"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.data[0].info["subject_info"],
        }
        core_info.update(self._dimensions_info_for_results_dict())

        return core_info

    def _dimensions_info_for_results_dict(self) -> dict:
        """Returns information about the dimensions of the SSD results.

        RETURNS
        -------
        dimensions_info : dict
        -   Information about the dimensions of the results
        """
        dimensions_info = {}
        if "windows" in self.results_dims:
            dimensions_info["windows"] = (
                np.arange(self._n_windows) + 1
            ).tolist()

        return dimensions_info

    def _extra_info_for_results_dict(self) -> dict:
        """Returns extra information about the results which is optionally
        present.

        RETURNS
        -------
        extra_info : dict
        -   Additional information about the results.
        """
        extra_info = {}
        extra_info_keys = [
            "transformed_ch_types",
            "transformed_ch_coords",
            "transformed_ch_regions",
            "transformed_ch_subregions",
            "transformed_ch_hemispheres",
            "transformed_ch_reref_types",
            "transformed_ch_epoch_orders",
            "metadata",
        ]
        for key in extra_info_keys:
            if (
                key in self.extra_info.keys()
                and self.extra_info[key] is not None
            ):
                if key == "metadata":
                    extra_info["metadata"] = self.signal.extra_info["metadata"]
                else:
                    extra_info[key] = [
                        self.extra_info[key][ch_name]
                        for ch_name in self._transformed_ch_names
                    ]

        return extra_info

    def topographies_as_dict(self) -> dict:
        """Returns the SSD topographies and additional information as a
        dictionary.

        RETURNS
        -------
        topographies_dict : dict
        -   The topographies and additional information stored as a dictionary.
        """
        core_info = self._core_info_for_topographies_dict()
        extra_info = self._extra_info_for_topographies_dict()

        return combine_dicts([core_info, extra_info])

    def _core_info_for_topographies_dict(self) -> dict:
        """Returns core information about the SSD topographies which is always
        present.

        RETURNS
        -------
        core_info : dict
        -   The core information about the topographies.
        """
        topos, topo_dimensions = self._rearrange_topographies()

        self._transformed_ch_names_topos = self._get_band_feature_info(
            copy_from="transformed_ch_names",
            repeat_for_channels=True,
            copy_mode="tile",
        )
        self._original_ch_names_topos = self._get_group_feature_info(
            copy_from=self.channels,
            repeat_for_channels=False,
            copy_mode="repeat",
        )

        core_info = {
            "ssd_topographies": topos,
            "ssd_topographies_dimensions": topo_dimensions,
            "ch_names": self._original_ch_names_topos,
            "transformed_ch_names": self._transformed_ch_names_topos,
            "component_numbers": (
                np.array(
                    self._get_band_feature_info(
                        copy_from=None,
                        repeat_for_channels=True,
                        copy_mode="tile",
                    )
                )
                + 1
            ).tolist(),
            "component_eigenvalues": self._get_group_feature_info(
                copy_from="eigenvalues",
                repeat_for_channels=True,
                copy_mode="tile",
            ),
            "component_spectral_ratios": self._get_group_feature_info(
                copy_from="spectral_ratios",
                repeat_for_channels=True,
                copy_mode="tile",
            ),
            "band_names": self._get_band_feature_info(
                self._filt_params_names,
                repeat_for_channels=True,
                copy_mode="repeat",
            ),
            "sampling_frequency": self.signal.data[0].info["sfreq"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.data[0].info["subject_info"],
        }

        dimensions_info = self._dimensions_info_for_results_dict()
        if "windows" in dimensions_info.keys():
            core_info.update(windows=dimensions_info["windows"])

        return core_info

    def _rearrange_topographies(self) -> tuple[np.ndarray, list[str]]:
        """Rearrange topography results into an array containing the
        concatenated topographies for each channel and each component within
        each group.

        RETURNS
        -------
        topgraphies : numpy ndarray
        -   The rearranged topography results.

        topography_dimensions : list of str
        -   Names of the dimensions of the topography results.
        """
        topos = []
        for group_i in range(self._n_groups):
            for band_i in range(self._n_bands):
                results = self._results[group_i][band_i]
                for ch_i in range(len(self.channels[group_i])):
                    for comp_i in range(results["n_comps"]):
                        if not self.signal._windowed:
                            topos.append(results["patterns"][ch_i, comp_i])
                        else:
                            topos.extend(
                                [
                                    results["patterns"][window_i][ch_i, comp_i]
                                    for window_i in range(self._n_windows)
                                ]
                            )
        topos = np.array(topos)

        topo_dimensions = ["channels", "windows"]
        if "windows" not in self.results_dims:
            topo_dimensions = [
                dim for dim in topo_dimensions if dim != "windows"
            ]

        return topos, topo_dimensions

    def _extra_info_for_topographies_dict(self) -> dict:
        """Returns extra information about the topographies which is optionally
        present.

        RETURNS
        -------
        extra_info : dict
        -   Additional information about the topographies.
        """
        extra_info = {}
        extra_info_keys = [
            "ch_types",
            "ch_regions",
            "ch_subregions",
            "ch_hemispheres",
            "ch_reref_types",
            "ch_epoch_orders",
            "metadata",
        ]

        for group, use_channels in zip(
            ["", "transformed_"],
            [self._original_ch_names_topos, self._transformed_ch_names_topos],
        ):
            for key in extra_info_keys:
                if (
                    key in self.extra_info.keys()
                    and self.extra_info[key] is not None
                ):
                    if key == "metadata":
                        extra_info["metadata"] = self.signal.extra_info[
                            "metadata"
                        ]
                    else:
                        extra_info[f"{group}{key}"] = [
                            self.extra_info[f"{group}{key}"][ch_name]
                            for ch_name in use_channels
                        ]

        extra_info["ch_coords"] = self.signal.get_coordinates(
            picks=self._original_ch_names_topos
        )

        return extra_info

    def _get_group_feature_info(
        self,
        copy_from: Any,
        repeat_for_channels: bool = False,
        copy_mode: str = "repeat",
    ) -> list:
        """Finds the information about a feature of the data for when the
        results are returned as a dict, where the feature is shared for a group
        of results, regardless of the band.

        PARAMETERS
        ----------
        copy_from : Any
        -   If not None and not a str, `copy_from`[group_i] will be appended to
            the list of features for each component in the group and band. If
            not None and a str, self.results[`copy_from`] for the group will be
            appended in the same manner. If None, the number of components in
            group and band will be enumerated across and these values appended.

        repeat_for_channels : bool; default False
        -   Whether or not to repeat each entry for the number of channels in
            that group and band.

        copy_mode : str; default "repeat"
        -   How to copy the entries. If "repeat", `numpy.repeat` is used. If
            "tile", `numpy.tile` is used.

        RETURNS
        -------
        feature : list
        -   The information about a feature of the data for when the results are
            returned as a dict.
        """
        supported_copy_modes = ["repeat", "tile"]
        assert copy_mode in supported_copy_modes, (
            "The requested `copy_mode` is not supported. Please contact the "
            "developers."
        )
        if copy_mode == "repeat":
            method = np.repeat
        else:
            method = np.tile

        feature = []
        n_chs = 1
        for group_i in range(self._n_groups):
            if repeat_for_channels:
                n_chs = len(self.channels[group_i])

            for band_i in range(self._n_bands):
                n_comps = self._results[group_i][band_i]["n_comps"]

                if copy_from is not None and not isinstance(copy_from, str):
                    values = method(
                        copy_from[group_i], n_comps * n_chs
                    ).tolist()

                elif isinstance(copy_from, str):
                    values = method(
                        [
                            var
                            for var in self._results[group_i][band_i][copy_from]
                        ],
                        n_chs,
                    ).tolist()

                else:
                    values = method(
                        [var for var in range(n_comps)], n_chs
                    ).tolist()

                feature.extend(values)

        return feature

    def _get_band_feature_info(
        self,
        copy_from: Any,
        repeat_for_channels: bool = False,
        copy_mode: str = "repeat",
    ) -> list:
        """Finds the information about a feature of the data for when the
        results are returned as a dict, where the feature is shared for a group
        of results, but not all bands.

        PARAMETERS
        ----------
        copy_from : Any
        -   If not None and not a str, `copy_from`[band_i] will be appended to
            the list of features for each component in the group. If not None
            and a str, self.results[`copy_from`] for the  band will be appended
            in the same manner. If None, the number of components in band will
            be enumerated across and these values appended.

        repeat_for_channels : bool; default False
        -   Whether or not to repeat each entry for the number of channels in
            that group and band.

        copy_mode : str; default "repeat"
        -   How to copy the entries. If "repeat", `numpy.tile` is used. If
            "tile", `numpy.tile` is used.

        RETURNS
        -------
        feature : list
        -   The information about a feature of the data for when the results are
            returned as a dict.
        """
        supported_copy_modes = ["repeat", "tile"]
        assert copy_mode in supported_copy_modes, (
            "The requested `copy_mode` is not supported. Please contact the "
            "developers."
        )
        if copy_mode == "repeat":
            method = np.repeat
        else:
            method = np.tile

        feature = []
        n_chs = 1
        for group_i in range(self._n_groups):
            if repeat_for_channels:
                n_chs = len(self.channels[group_i])

            for band_i in range(self._n_bands):
                n_comps = self._results[group_i][band_i]["n_comps"]

                if copy_from is not None and not isinstance(copy_from, str):
                    values = method(copy_from[band_i], n_comps * n_chs).tolist()

                elif isinstance(copy_from, str):
                    values = method(
                        [
                            var
                            for var in self._results[group_i][band_i][copy_from]
                        ],
                        n_chs,
                    ).tolist()

                else:
                    values = method(
                        [var for var in range(n_comps)], n_chs
                    ).tolist()

                feature.extend(values)

        return feature

    def save_results(
        self,
        fpath: str,
        ftype: str | None = None,
        ask_before_overwrite: bool | None = None,
    ) -> None:
        """Saves the power results and additional information as a file.

        PARAMETERS
        ----------
        fpath : str
        -   Location where the power results should be saved.

        ftype : str | None; default None
        -   The filetype of the power results that will be saved, without the
            leading period. E.g. for saving the file in the json format, this
            would be "json", not ".json".
        -   The information being saved must be an appropriate type for saving
            in this format.
        -   If None, the filetype is determined based on 'fpath', and so the
            extension must be included in the path.

        ask_before_overwrite : bool | None; default the object's verbosity
        -   If True, the user is asked to confirm whether or not to overwrite a
            pre-existing file if one exists.
        -   If False, the user is not asked to confirm this and it is done
            automatically.
        -   By default, this is set to None, in which case the value of the
            verbosity when the Signal object was instantiated is used.
        """
        if ask_before_overwrite is None:
            ask_before_overwrite = self.verbose

        save_dict(
            to_save=self.results_as_dict(),
            fpath=fpath,
            ftype=ftype,
            ask_before_overwrite=ask_before_overwrite,
            verbose=self.verbose,
        )

        save_dict(
            to_save=self.topographies_as_dict(),
            fpath=f"{fpath}_topographies",
            ftype=ftype,
            ask_before_overwrite=ask_before_overwrite,
            convert_numpy_to_python=True,
            verbose=self.verbose,
        )
