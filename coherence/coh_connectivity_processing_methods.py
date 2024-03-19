"""Abstract subclasses for implementing connectivity processing methods.

CLASSES
-------
ProcConnectivity
-   Class for processing connectivity results. A subclass of 'ProcMethod'.

ProcSingularConnectivity
-   Class for processing connectivity results between pairs of single channels.
    A subclass of 'ProcConnectivity'.

ProcMultivariateConnectivity
-   Class for processing multivariate connectivity results. A subclass of
    'ProcConnectivity'.
"""

from abc import abstractmethod
from copy import deepcopy
from typing import Union
import numpy as np
from numpy.typing import NDArray
from mne_connectivity import (
    seed_target_indices,
    multivariate_spectral_connectivity_epochs,
    spectral_connectivity_epochs,
    SpectralConnectivity,
    SpectroTemporalConnectivity,
    MultivariateSpectralConnectivity,
    MultivariateSpectroTemporalConnectivity,
)
from coh_exceptions import ProcessingOrderError, UnavailableProcessingError
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
from coh_progress_bar import ProgressBar
from coh_saving import save_dict
from coh_signal import Signal


class ProcConnectivity(ProcMethod):
    """Class for processing connectivity results. A subclass of 'ProcMethod'.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process (abstract)
    -   Processes the data.

    save_object (abstract)
    -   Saves the object as a .pkl file.

    save_results (abstract)
    -   Converts the results and additional information to a dictionary and
        saves them as a file.

    results_as_dict (abstract)
    -   Organises the results and additional information into a dictionary.
    """

    @abstractmethod
    def __init__(self, signal: Signal, verbose: bool) -> None:
        super().__init__(signal, verbose)

        self.con_methods = None
        self.power_method = None
        self.seeds = None
        self.targets = None
        self.indices = None
        self.fmin = None
        self.fmax = None
        self.fksip = None
        self.faverage = None
        self.tmin = None
        self.tmax = None
        self.mt_bandwidth = None
        self.mt_adaptive = None
        self.mt_low_bias = None
        self.cwt_freqs = None
        self.cwt_n_cycles = None
        self.average_windows = None
        self.average_timepoints = None
        self.absolute_connectivity = None
        self.block_size = None
        self.n_jobs = None

        self._n_cons = None
        self._n_freqs = None
        self._n_times = None

        self._progress_bar = None

        self._windows_averaged = False
        self._timepoints_averaged = False

    @abstractmethod
    def process(self) -> None:
        """Processes the data."""

    def _sort_inputs(self) -> None:
        """Checks the inputs to the object.

        RAISES
        ------
        ValueError
        -   Raised if the dimensions of the data in the Signal object is not
            supported.
        """
        supported_data_dims = ["windows", "epochs", "channels", "timepoints"]
        if self.signal._data_dimensions != supported_data_dims:
            raise ValueError(
                "Error when trying to perform coherence analysis on the "
                "data:\nData in the Signal object has the dimensions "
                f"{self.signal.data_dimensions}, but only data with dimensions "
                f"{supported_data_dims} is supported."
            )
        super()._sort_inputs()
        self.extra_info.update(
            node_ch_types=None,
            node_ch_reref_types=None,
            node_ch_coords=None,
            node_ch_regions=None,
            node_ch_subregions=None,
            node_ch_hemispheres=None,
            node_lateralisation=None,
            node_ch_epoch_orders=None,
        )

    def _sort_processing_inputs(self) -> None:
        """Sorts processing arguments."""
        self._n_cons = len(self.indices[0])

    def _take_absolute_connectivity(
        self,
        connectivity: list[Union[SpectralConnectivity, SpectroTemporalConnectivity]],
    ) -> list[Union[SpectralConnectivity, SpectroTemporalConnectivity]]:
        """Takes the absolute value of the connectivity and returns it.

        PARAMETERS
        ----------
        connectivity : MNE SpectralConnectivity or MNE
        SpectroTemporalConnectivity
        -   The connectivity object where the absolute connectivity values
            should be taken.

        RETURNS
        -------
        absolute_connectivity : list of MNE SpectralConnectivity or MNE
        SpectroTemporalConnectivity
        -   The connectivity objects with absolute connectivity values.
        """
        absolute_connectiviy = []
        for con_class in connectivity:
            if isinstance(con_class, SpectroTemporalConnectivity):
                absolute_connectiviy.append(
                    SpectroTemporalConnectivity(
                        data=np.abs(con_class.get_data()),
                        freqs=con_class.freqs,
                        times=con_class.times,
                        n_nodes=con_class.n_nodes,
                        names=con_class.names,
                        indices=con_class.indices,
                        method=con_class.method,
                        spec_method=con_class.spec_method,
                        n_epochs_used=con_class.n_epochs_used,
                    )
                )
            elif isinstance(con_class, SpectralConnectivity):
                SpectralConnectivity(
                    data=np.abs(con_class.get_data()),
                    freqs=con_class.freqs,
                    n_nodes=con_class.n_nodes,
                    names=con_class.names,
                    indices=con_class.indices,
                    method=con_class.method,
                    spec_method=con_class.spec_method,
                    n_epochs_used=con_class.n_epochs_used,
                )
            else:
                raise TypeError(
                    "The connectivity object provided is not of a supported " "type."
                )

        return absolute_connectiviy

    def _sort_dimensions(self) -> None:
        """Establishes dimensions of the connectivity results and averages
        across windows, if requested."""
        self._results_dims = ["windows", "connections", "frequencies"]
        if "times" in self.results[0][0].coords:
            self._results_dims.append("timepoints")

        if self.average_windows:
            self._average_windows_results()

        if self.average_timepoints:
            self._average_timepoints_results()

    def _average_windows_results(self) -> None:
        """Averages the connectivity results across windows.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the windows have already been averaged across.
        """
        if self._windows_averaged:
            raise ProcessingOrderError(
                "Error when averaging the connectivity results across "
                "windows: Results have already been averaged across windows."
            )

        n_windows = len(self.results[0])

        windowed_results = []
        for con_method in self.results:
            kwargs = dict(
                data=np.asarray([data.get_data() for data in con_method]).mean(axis=0),
                freqs=con_method[0].freqs,
                n_nodes=con_method[0].n_nodes,
                names=con_method[0].names,
                indices=con_method[0].indices,
                method=con_method[0].method,
                n_epochs_used=con_method[0].n_epochs_used,
            )

            if "timepoints" in self.results_dims:
                kwargs.update(
                    times=con_method[0].times,
                )
                connectivity_obj = SpectroTemporalConnectivity
            else:
                connectivity_obj = SpectralConnectivity

            if (
                hasattr(con_method, "topographies")
                and con_method.topographies is not None
            ):
                kwargs.update(
                    topograhies=np.asarray(
                        [data.topographies for data in con_method]
                    ).mean(axis=0)
                )

            windowed_results.append(connectivity_obj(**kwargs))

        self._windows_averaged = True
        if self.verbose:
            print(f"Averaging the data over {n_windows} windows.\n")

    def _average_timepoints_results(self) -> None:
        """Averages the connectivity results across timepoints.

        RAISES
        ------
        ProcessingOrderError
        -   Raised if the timepoints have already been averaged across.
        """
        if self._timepoints_averaged:
            raise ProcessingOrderError(
                "Error when averaging the connectivity results across "
                "timepoints: Results have already been averaged across "
                "timepoints."
            )

        if "timepoints" not in self.results_dims:
            raise UnavailableProcessingError(
                "Error when attempting to average the timepoints in the "
                "connectivity results: There is no timepoints axis present "
                f"in the data. The present axes are: {self.results_dims}."
            )

        n_times = self.results[0][0].shape[2]

        time_averaged_results = []
        for con_method in self.results:
            time_averaged_results.append([])
            for window_data in con_method:
                kwargs = dict(
                    data=window_data.get_data().mean(axis=2),
                    freqs=con_method[0].freqs,
                    n_nodes=con_method[0].n_nodes,
                    names=con_method[0].names,
                    indices=con_method[0].indices,
                    method=con_method[0].method,
                    n_epochs_used=con_method[0].n_epochs_used,
                )

                if (
                    hasattr(con_method, "topographies")
                    and con_method.topographies is not None
                ):
                    topographies = window_data.topographies.copy()
                    for group_i, group in enumerate(topographies):
                        for con_i, con in enumerate(group):
                            topographies[group_i][con_i] = con.mean(axis=2)
                    kwargs.update(topograhies=topographies)

                time_averaged_results[-1].append(SpectralConnectivity(**kwargs))

        self._results_dims.pop(self._results_dims.index("timepoints"))

        self._timepoints_averaged = True
        if self.verbose:
            print(f"Averaging the data over {n_times} timepoints.\n")

    @abstractmethod
    def _sort_seeds_targets(self) -> None:
        """Sorts the names of the seeds and targets for the connectivity
        analysis, and generates the corresponding channel indices."""

    @abstractmethod
    def _expand_seeds_targets(self) -> None:
        """Expands the channels in the seed and target groups such that
        connectivity is computed bwteen each seed and each target group."""

    @abstractmethod
    def _generate_extra_info(self) -> None:
        """Generates additional information related to the connectivity
        analysis."""

    @abstractmethod
    def _generate_node_ch_types(self) -> None:
        """Gets the types of channels in the connectivity results."""

    @abstractmethod
    def _generate_node_ch_reref_types(self) -> None:
        """Gets the rereferencing types of channels in the connectivity
        results."""

    @abstractmethod
    def _generate_node_ch_coords(self) -> None:
        """Gets the coordinates of channels in the connectivity results,
        averaged across for each channel in the seeds and targets."""

    @abstractmethod
    def _generate_node_ch_regions(self) -> None:
        """Gets the regions of channels in the connectivity results."""

    @abstractmethod
    def _generate_node_ch_subregions(self) -> None:
        """Gets the subregions of channels in the connectivity results."""

    @abstractmethod
    def _generate_node_ch_hemispheres(self) -> None:
        """Gets the hemispheres of channels in the connectivity results."""

    @abstractmethod
    def _generate_node_lateralisation(self) -> None:
        """Gets the lateralisation of the channels in the connectivity node."""

    @abstractmethod
    def _generate_node_ch_epoch_orders(self) -> None:
        """Gets the epoch orders of channels in the connectivity results."""

    def get_results_as_array(
        self, dimensions: Union[list[str], None] = None
    ) -> NDArray:
        """Extracts and returns results as an array.

        PARAMETERS
        ----------
        dimensions : list[str] | None;  default None
        -   The dimensions of the results that will be returned.
        -   If 'None', the current dimensions are used.

        RETURNS
        -------
        results : list of numpy array
        -   The results for each connectivity method
        """

        if dimensions is None:
            dimensions = self.results_dims

        if self._windows_averaged:
            results = [con_method[0].get_data() for con_method in self.results]
        else:
            results = [[] for _ in range(len(self.results))]
            for method_i, con_method in enumerate(self.results):
                results[method_i] = np.array(
                    [window_data.get_data() for window_data in con_method]
                )

        results = [
            rearrange_axes(
                obj=con_data, old_order=self.results_dims, new_order=dimensions
            )
            for con_data in results
        ]

        return deepcopy(results)

    def results_as_dict(self) -> dict:
        """Returns the connectivity results and additional information as a
        dictionary.

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
        results = self.get_results_as_array(dimensions=dimensions)

        core_info = {}
        for con_method, con_data in zip(self.con_methods, results):
            core_info[f"connectivity-{con_method}"] = con_data.tolist()
            core_info[f"connectivity-{con_method}_dimensions"] = dimensions
        core_info.update(
            seed_names=self.seeds,
            target_names=self.targets,
            sampling_frequency=self.signal.data[0].info["sfreq"],
            processing_steps=self.processing_steps,
            subject_info=self.signal.data[0].info["subject_info"],
        )
        core_info.update(self._dimensions_info_for_results_dict())

        return core_info

    def _dimensions_info_for_results_dict(self) -> dict:
        """Returns information about the dimensions of the connectivity results.

        RETURNS
        -------
        dimensions_info : dict
        -   Information about the dimensions of the connectivity results
        """
        dimensions_info = {}
        if "windows" in self.results_dims:
            dimensions_info["windows"] = (np.arange(self._n_windows) + 1).tolist()
        if "frequencies" in self.results_dims:
            dimensions_info["frequencies"] = list(self.results[0][0].freqs)
        if "timepoints" in self.results_dims:
            dimensions_info["timepoints"] = list(self.results[0][0].times)

        return dimensions_info

    def _extra_info_for_results_dict(self) -> dict:
        """Returns extra information about the connectivity results which is
        optionally present.

        RETURNS
        -------
        extra_info : dict
        -   Additional information about the connectivity results.
        """
        extra_info = {}
        group_names = ["seed", "target"]
        extra_info_key_mappings = {
            "node_ch_types": "_types",
            "node_ch_coords": "_coords",
            "node_ch_regions": "_regions",
            "node_ch_subregions": "_subregions",
            "node_ch_hemispheres": "_hemispheres",
            "node_ch_reref_types": "_reref_types",
            "node_lateralisation": "node_lateralisation",
            "node_ch_epoch_orders": "node_epoch_orders",
            "metadata": "metadata",
        }
        same_info_seeds_targets_keys = [
            "node_lateralisation",
            "node_ch_epoch_orders",
            "metadata",
        ]

        for key, value in extra_info_key_mappings.items():
            if self.extra_info[key] is not None:
                if key in same_info_seeds_targets_keys:
                    extra_info[value] = self.extra_info[key]
                else:
                    for group_idx, group_name in enumerate(group_names):
                        new_key = f"{group_name}{value}"
                        extra_info[new_key] = self.extra_info[key][group_idx]
        return extra_info

    def save_results(
        self,
        fpath: str,
        ftype: Union[str, None] = None,
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
            convert_numpy_to_python=True,
            verbose=self.verbose,
        )


class ProcBivariateConnectivity(ProcConnectivity):
    """Class for processing connectivity results between pairs of single
    channels. A subclass of 'ProcConnectivity'.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process (abstract)
    -   Processes the data.

    save_object (abstract)
    -   Saves the object as a .pkl file.

    save_results (abstract)
    -   Converts the results and additional information to a dictionary and
        saves them as a file.

    results_as_dict (abstract)
    -   Organises the results and additional information into a dictionary.
    """

    @abstractmethod
    def __init__(self, signal: Signal, verbose: bool) -> None:
        super().__init__(signal, verbose)

    @abstractmethod
    def process(self) -> None:
        """Processes the data."""

    def _sort_seeds_targets(self) -> None:
        """Sorts the names of the seeds and targets for the connectivity
        analysis, and generates the corresponding channel indices.

        If the seeds and/or targets are dictionaries, the names of the seeds and
        targets will be automatically generated based on the information in the
        dictionaries, and then expanded, such that connectivity is calculated
        between every seed and every target.

        If the seeds and targets are both lists, the channel names in these
        lists are taken as the seeds and targets and no expansion is performed.

        RAISES
        ------
        ValueError
        -   Raised if the seeds and/or targets contain multiple channels per
            node.
        """
        groups = ["seeds", "targets"]
        features = self._features_to_df()
        groups_vals = [getattr(self, group) for group in groups]
        expand_seeds_targets = False
        for group_i, group in enumerate(groups):
            group_vals = groups_vals[group_i]
            if isinstance(group_vals, dict):
                expand_seeds_targets = True
                eligible_idcs = get_eligible_idcs_lists(
                    features, group_vals["eligible_entries"]
                )
                group_idcs = get_group_names_idcs(
                    features,
                    group_vals["grouping"],
                    eligible_idcs=eligible_idcs,
                    replacement_idcs=eligible_idcs,
                )
                names = []
                for idx in group_idcs.values():
                    if len(idx) > 1:
                        raise ValueError(
                            "For singular connectivity, seeds and targets for "
                            "a node can only contain one channel each, however "
                            f"a node of the {group} contains {len(idx)} "
                            "channels.\nIf you wish to compute connectivity "
                            "between groups of channels, please use one of the "
                            "multivariate connectivity methods."
                        )
                    names.append(self.signal.data[0].ch_names[idx[0]])
                setattr(self, group, names)
            elif isinstance(group_vals, list):
                names = []
                for val in group_vals:
                    if not isinstance(val, str):
                        raise TypeError(
                            "Seeds and targets must be specified as strings, "
                            "as for singular connectivity, seeds and targets "
                            "can only contain one channel each.\nIf you wish "
                            "to compute connectivity between groups of "
                            "channels, please use one of the multivariate "
                            "connectivity methods."
                        )
                    names.append(val)
                setattr(self, group, names)
            else:
                raise TypeError(
                    "Seeds and targets must given as lists, or as dictionaries "
                    "with instructions for generating these lists, however the "
                    f"{group} are of type {type(group_vals)}."
                )

        if expand_seeds_targets:
            self._expand_seeds_targets()

        if len(self.seeds) != len(self.targets):
            raise ValueError(
                "Seeds and targets must contain the same number of entries, "
                f"but do not ({len(self.seeds)} and {len(self.targets)}, "
                "respectively)."
            )

        self._generate_indices()

    def _expand_seeds_targets(self) -> None:
        """Expands the channels in the seed and target groups such that
        connectivity is computed bwteen each seed and each target group.

        Should be used when seeds and/or targets have been automatically
        generated based on channel types.
        """
        seeds = []
        targets = []
        for seed in self.seeds:
            for target in self.targets:
                seeds.append(seed)
                targets.append(target)
        self.seeds = seeds
        self.targets = targets

    def _generate_indices(self) -> None:
        """Generates MNE-readable indices for calculating connectivity between
        signals."""
        self.indices = seed_target_indices(
            seeds=[
                i
                for i, name in enumerate(self.signal.data[0].ch_names)
                if name in self.seeds
            ],
            targets=[
                i
                for i, name in enumerate(self.signal.data[0].ch_names)
                if name in self.targets
            ],
        )

    def _sort_used_settings(self) -> None:
        """Collects the settings that are relevant for the processing being
        performed and adds only these settings to the 'processing_steps'
        dictionary."""
        used_settings = {
            "con_methods": self.con_methods,
            "power_method": self.power_method,
            "average_windows": self.average_windows,
            "average_timepoints": self.average_timepoints,
            "absolute_connectivity": self.absolute_connectivity,
            "t_min": self.tmin,
            "t_max": self.tmax,
        }

        if self.power_method == "multitaper":
            add_settings = {
                "mt_bandwidth": self.mt_bandwidth,
                "mt_adaptive": self.mt_adaptive,
                "mt_low_bias": self.mt_low_bias,
            }
        elif self.power_method == "cwt_morlet":
            add_settings = {"cwt_n_cycles": self.cwt_n_cycles}
        used_settings.update(add_settings)

        self.processing_steps["spectral_connectivity"] = used_settings

    def _get_results(self) -> None:
        """Performs the connectivity analysis."""
        if self.verbose:
            self._progress_bar = ProgressBar(
                n_steps=len(self.signal.data),
                title="Computing connectivity",
            )

        self.results = [[] for _ in range(len(self.con_methods))]
        for window_idx, window_data in enumerate(self.signal.data):
            if self.verbose:
                print(
                    f"Computing connectivity for window {window_idx+1} of "
                    f"{len(self.signal.data)}.\n"
                )
            window_connectivity = spectral_connectivity_epochs(
                data=window_data,
                method=self.con_methods,
                indices=self.indices,
                sfreq=window_data.info["sfreq"],
                mode=self.power_method,
                fmin=self.fmin,
                fmax=self.fmax,
                fskip=self.fskip,
                faverage=self.faverage,
                tmin=self.tmin,
                tmax=self.tmax,
                mt_bandwidth=self.mt_bandwidth,
                mt_adaptive=self.mt_adaptive,
                mt_low_bias=self.mt_low_bias,
                cwt_freqs=self.cwt_freqs,
                cwt_n_cycles=self.cwt_n_cycles,
                block_size=self.block_size,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )

            if not isinstance(window_connectivity, list):
                window_connectivity = [window_connectivity]

            if self.absolute_connectivity:
                window_connectivity = self._take_absolute_connectivity(
                    window_connectivity
                )

            for method_i, con_method in enumerate(window_connectivity):
                self.results[method_i].append(con_method)

            if self._progress_bar is not None:
                self._progress_bar.update_progress()

        if self._progress_bar is not None:
            self._progress_bar.close()

        self._sort_dimensions()
        self._generate_extra_info()
        self._processed = True

    def _check_connectivity_class(
        self,
        connectivity: list[Union[SpectralConnectivity, SpectroTemporalConnectivity]],
    ) -> None:
        """Checks that the connectivity object returned from MNE is of the
        expected type.

        PARAMETERS
        ----------
        connectivity : list of MNE SpectralConnectivity or MNE
            SpectroTemporalConnectivity
        -   The MNE connectivity objects.
        """
        for con_method in connectivity:
            correct_class = True
            if not isinstance(con_method, SpectralConnectivity) and not isinstance(
                con_method, SpectroTemporalConnectivity
            ):
                correct_class = False
            assert correct_class, (
                "The MNE connectivity object is expected to be of type "
                "SpectralConnectivity or SpectroTemporalConnectivity."
            )

    def _generate_extra_info(self) -> None:
        """Generates additional information related to the connectivity
        analysis."""
        self._generate_node_ch_types()
        self._generate_node_ch_reref_types()
        self._generate_node_ch_coords()
        self._generate_node_ch_regions()
        self._generate_node_ch_subregions()
        self._generate_node_ch_hemispheres()
        self._generate_node_lateralisation()
        self._generate_node_ch_epoch_orders()

    def _generate_node_ch_types(self) -> None:
        """Gets the types of channels in the connectivity results."""
        node_ch_types = [[], []]
        groups = ["seeds", "targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_types[group_i].append(
                    self.signal.data[0].get_channel_types(picks=name)[0]
                )
        self.extra_info["node_ch_types"] = node_ch_types

    def _generate_node_ch_reref_types(self) -> None:
        """Gets the rereferencing types of channels in the connectivity
        results."""
        node_reref_types = [[], []]
        groups = ["seeds", "targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_reref_types[group_i].append(
                    self.extra_info["ch_reref_types"][name]
                )
        self.extra_info["node_ch_reref_types"] = node_reref_types

    def _generate_node_ch_coords(self) -> None:
        """Gets the coordinates of channels in the connectivity results,
        averaged across for each channel in the seeds and targets."""
        node_ch_coords = [[], []]
        groups = ["seeds", "targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_coords[group_i].append(self.signal.get_coordinates(name)[0])
        self.extra_info["node_ch_coords"] = node_ch_coords

    def _generate_node_ch_regions(self) -> None:
        """Gets the regions of channels in the connectivity results."""
        node_ch_regions = [[], []]
        groups = ["seeds", "targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_regions[group_i].append(self.extra_info["ch_regions"][name])
        self.extra_info["node_ch_regions"] = node_ch_regions

    def _generate_node_ch_subregions(self) -> None:
        """Gets the subregions of channels in the connectivity results."""
        node_ch_subregions = [[], []]
        groups = ["seeds", "targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_subregions[group_i].append(
                    self.extra_info["ch_subregions"][name]
                )
        self.extra_info["node_ch_subregions"] = node_ch_subregions

    def _generate_node_ch_hemispheres(self) -> None:
        """Gets the hemispheres of channels in the connectivity results."""
        node_ch_hemispheres = [[], []]
        groups = ["seeds", "targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_hemispheres[group_i].append(
                    self.extra_info["ch_hemispheres"][name]
                )
        self.extra_info["node_ch_hemispheres"] = node_ch_hemispheres

    def _generate_node_lateralisation(self) -> None:
        """Gets the lateralisation of the channels in the connectivity node.

        Can either be "contralateral" if the seed and target are from different
        hemispheres, or "ipsilateral" if the seed and target are from the same
        hemisphere.
        """
        node_lateralisation = []
        node_ch_hemispheres = self.extra_info["node_ch_hemispheres"]
        for node_i in range(len(node_ch_hemispheres[0])):
            if node_ch_hemispheres[0][node_i] != node_ch_hemispheres[1][node_i]:
                lateralisation = "contralateral"
            else:
                lateralisation = "ipsilateral"
            node_lateralisation.append(lateralisation)
        self.extra_info["node_lateralisation"] = node_lateralisation

    def _generate_node_ch_epoch_orders(self) -> None:
        """Gets the epoch orders of channels in the connectivity results.

        If either the seed or target has a "shuffled" epoch order, the epoch
        order of the node is "shuffled", otherwise it is "original".
        """
        node_epoch_orders = []
        for seed_name, target_name in zip(self.seeds, self.targets):
            if (
                self.extra_info["ch_epoch_orders"][seed_name] == "original"
                and self.extra_info["ch_epoch_orders"][target_name] == "original"
            ):
                order = "original"
            else:
                order = "shuffled"
            node_epoch_orders.append(order)
        self.extra_info["node_ch_epoch_orders"] = node_epoch_orders


class ProcMultivariateConnectivity(ProcConnectivity):
    """Class for processing multivariate connectivity results. A subclass of
    'ProcConnectivity'.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process (abstract)
    -   Processes the data.

    save_object (abstract)
    -   Saves the object as a .pkl file.

    save_results (abstract)
    -   Converts the results and additional information to a dictionary and
        saves them as a file.

    results_as_dict (abstract)
    -   Organises the results and additional information into a dictionary.
    """

    def __init__(self, signal: Signal, verbose: bool) -> None:
        super().__init__(signal, verbose)

        self.absolute_connectivity = False

        self.n_components = None
        self.n_lags = None

        self._seeds_list = None
        self._targets_list = None
        self._comb_names_str = None
        self._comb_names_list = None

    @abstractmethod
    def process(self) -> None:
        """Processes the data."""

    def _sort_processing_inputs(self) -> None:
        """Checks that the processing inputs are appropriate and implements them
        appropriately."""
        self._sort_seeds_targets()
        self._sort_n_components()
        super()._sort_processing_inputs()

    def _sort_seeds_targets(self) -> None:
        """Sorts the names of the seeds and targets for the connectivity
        analysis, and generates the corresponding channel indices.

        If the seeds and/or targets are dictionaries, the names of the seeds and
        targets will be automatically generated based on the information in the
        dictionaries, and then expanded, such that connectivity is calculated
        between every seed and every target.

        If the seeds and targets are both lists, the channel names in these
        lists are taken as the seeds and targets and no expansion is performed.
        """
        groups = ["seeds", "targets"]
        features = self._features_to_df()
        groups_vals = [getattr(self, group) for group in groups]
        expand_seeds_targets = False
        for group_i, group in enumerate(groups):
            group_vals = groups_vals[group_i]
            if isinstance(group_vals, dict):
                expand_seeds_targets = True
                eligible_idcs = get_eligible_idcs_lists(
                    features, group_vals["eligible_entries"]
                )
                group_idcs = get_group_names_idcs(
                    features,
                    group_vals["grouping"],
                    eligible_idcs=eligible_idcs,
                    replacement_idcs=eligible_idcs,
                )
                names_list = []
                names_str = []
                for idcs in group_idcs.values():
                    names_list.append(
                        [self.signal.data[0].ch_names[idx] for idx in idcs]
                    )
                    names_str.append(
                        combine_vals_list(
                            [self.signal.data[0].ch_names[idx] for idx in idcs]
                        )
                    )
                setattr(self, f"_{group}_list", names_list)
                setattr(self, group, names_str)
            elif isinstance(group_vals, list):
                names_str = [combine_vals_list(val for val in group_vals)]
                setattr(self, group, names_str)
                setattr(self, f"_{group}_list", group_vals)
            else:
                raise TypeError(
                    "Seeds and targets must given as lists, or as dictionaries "
                    "with instructions for generating these lists, however the "
                    f"{group} are of type {type(group_vals)}."
                )

        if expand_seeds_targets:
            self._expand_seeds_targets()

        if len(self._seeds_list) != len(self._targets_list):
            raise ValueError(
                "Seeds and targets must contain the same number of entries, "
                f"but do not ({len(self._seeds_list)} and "
                f"{len(self._targets_list)}, respectively)."
            )

        self._get_names_indices_mne()

    def _expand_seeds_targets(self) -> None:
        """Expands the channels in the seed and target groups such that
        connectivity is computed bwteen each seed and each target group.

        Should be used when seeds and/or targets have been automatically
        generated based on channel types.
        """
        seeds_list = []
        targets_list = []
        seeds_str = []
        targets_str = []
        for seed in self._seeds_list:
            for target in self._targets_list:
                seeds_list.append(seed)
                targets_list.append(target)
                seeds_str.append(combine_vals_list(seed))
                targets_str.append(combine_vals_list(target))

        self._seeds_list = seeds_list
        self._targets_list = targets_list
        self.seeds = seeds_str
        self.targets = targets_str

    def _get_names_indices_mne(self) -> None:
        """Gets the names and indices of seed and targets in the connectivity
        analysis for use in an MNE connectivity object.

        As MNE connectivity objects only support seed-target pair names and
        indices between two channels, the names of channels in each group of
        seeds and targets are combined together, and the indices then derived
        from these combined names.
        """
        seed_names_str = []
        target_names_str = []
        for seeds, targets in zip(self._seeds_list, self._targets_list):
            seed_names_str.append(combine_vals_list(seeds))
            target_names_str.append(combine_vals_list(targets))
        unique_names_str = [*unique(seed_names_str), *unique(target_names_str)]
        unique_names_list = [
            *unique(self._seeds_list),
            *unique(self._targets_list),
        ]

        self._comb_names_str = unique_names_str
        self._comb_names_list = unique_names_list
        self._generate_indices()

    def _generate_indices(self) -> None:
        """Generates the indices for computing multivariate connectivity."""
        indices = [[], []]
        ch_names = self.signal.data[0].ch_names
        for seeds, targets in zip(self._seeds_list, self._targets_list):
            indices[0].append([ch_names.index(seed) for seed in seeds])
            indices[1].append([ch_names.index(target) for target in targets])
        self.indices = tuple(indices)

    def _sort_n_components(self) -> None:
        """Sort n_components input argument."""
        if isinstance(self.n_components, (list, tuple)):
            if len(self.n_components) != 2:
                raise ValueError("`n_components` must have a length of 2.")
            if all(isinstance(n_comps, int) for n_comps in self.n_components):
                n_components = [[], []]
                for i in range(2):
                    n_components[i] = [
                        self.n_components[i] for _ in range(len(self.indices[i]))
                    ]
                self.n_components = tuple(n_components)
            elif not all(isinstance(n_comps, list) for n_comps in self.n_components):
                raise TypeError(
                    "`n_components` must be a list of lists of ints or a list "
                    "of ints."
                )
        elif self.n_components is not None and not isinstance(self.n_components, str):
            raise TypeError("`n_components` must be None, a tuple, or a str.")

    def _get_results(self) -> None:
        """Performs the connectivity analysis."""
        if self.verbose:
            self._progress_bar = ProgressBar(
                n_steps=len(self.signal.data),
                title="Computing connectivity",
            )

        self.results = [[] for _ in range(len(self.con_methods))]
        for window_idx, window_data in enumerate(self.signal.data):
            if self.verbose:
                print(
                    f"Computing connectivity for window {window_idx+1} of "
                    f"{len(self.signal.data)}.\n"
                )

            window_connectivity = multivariate_spectral_connectivity_epochs(
                data=window_data,
                indices=self.indices,
                method=self.con_methods,
                mode=self.power_method,
                tmin=self.tmin,
                tmax=self.tmax,
                fmin=self.fmin,
                fmax=self.fmax,
                fskip=self.fskip,
                faverage=self.faverage,
                cwt_freqs=self.cwt_freqs,
                mt_bandwidth=self.mt_bandwidth,
                mt_adaptive=self.mt_adaptive,
                mt_low_bias=self.mt_low_bias,
                cwt_n_cycles=self.cwt_n_cycles,
                n_components=self.n_components,
                gc_n_lags=self.n_lags,
                block_size=self.block_size,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )

            if not isinstance(window_connectivity, list):
                window_connectivity = [window_connectivity]

            self._check_connectivity_classes(window_connectivity)

            self._n_freqs = len(window_connectivity[0].freqs)
            if hasattr(window_connectivity[0], "times"):
                self._n_times = len(window_connectivity[0].times)

            for method_i, con_method in enumerate(window_connectivity):
                self.results[method_i].append(con_method)

            if self._progress_bar is not None:
                self._progress_bar.update_progress()

        if self._progress_bar is not None:
            self._progress_bar.close()

        self._sort_dimensions()
        self._generate_extra_info()
        self._processed = True

    def _check_connectivity_classes(
        self,
        connectivity: list[
            Union[
                MultivariateSpectralConnectivity,
                MultivariateSpectroTemporalConnectivity,
            ]
        ],
    ) -> None:
        """Checks that the connectivity object returned from MNE is of the
        expected type.

        PARAMETERS
        ----------
        connectivity : list of MNE MultivariateSpectralConnectivity or MNE
            MultivariateSpectroTemporalConnectivity
        -   The MNE connectivity objects.
        """
        for con_class in connectivity:
            correct_class = True
            if not isinstance(
                con_class, MultivariateSpectralConnectivity
            ) and not isinstance(con_class, MultivariateSpectroTemporalConnectivity):
                correct_class = False
            assert correct_class, (
                "The MNE connectivity object is expected to be of type "
                "MultivariateSpectralConnectivity or "
                "MultivariateSpectroTemporalConnectivity."
            )

    def _generate_extra_info(self) -> None:
        """Generates additional information related to the connectivity
        analysis."""
        self._generate_node_ch_types()
        self._generate_node_ch_coords()

        key_method_mapping = {
            "ch_reref_types": self._generate_node_ch_reref_types,
            "ch_regions": self._generate_node_ch_regions,
            "ch_subregions": self._generate_node_ch_subregions,
            "ch_hemispheres": self._generate_node_ch_hemispheres,
            "ch_epoch_orders": self._generate_node_ch_epoch_orders,
        }
        for key, method in key_method_mapping.items():
            if self.signal.extra_info[key] is not None:
                if key == "ch_hemispheres":
                    node_single_hemispheres = method()
                    self._generate_node_lateralisation(node_single_hemispheres)
                else:
                    method()

    def _generate_node_ch_types(self) -> None:
        """Gets the types of channels in the connectivity results.

        If the types of each channel in a seed/target for a given node are
        identical, this type is given as a string, otherwise the unique types
        are taken and joined into a single string by the " & " characters.
        """
        ch_types = {}
        for ch_i, combined_name in enumerate(self._comb_names_str):
            types = []
            for single_name in self._comb_names_list[ch_i]:
                types.append(
                    self.signal.data[0].get_channel_types(picks=single_name)[0]
                )
            ch_types[combined_name] = combine_vals_list(unique(types))

        node_ch_types = [[], []]
        groups = ["seeds", "targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_types[group_i].append(ch_types[name])
        self.extra_info["node_ch_types"] = node_ch_types

    def _generate_node_ch_reref_types(self) -> None:
        """Gets the rereferencing types of channels in the connectivity results.

        If the rereferencing types of each channel in a seed/target for a given
        node are identical, this type is given as a string, otherwise the unique
        types are taken and joined into a single string by the " & " characters.
        """
        ch_reref_types = {}
        for ch_i, combined_name in enumerate(self._comb_names_str):
            reref_types = ordered_list_from_dict(
                list_order=self._comb_names_list[ch_i],
                dict_to_order=self.extra_info["ch_reref_types"],
            )
            unique_types = unique(reref_types)
            ch_reref_types[combined_name] = combine_vals_list(unique_types)

        node_reref_types = [[], []]
        groups = ["seeds", "targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_reref_types[group_i].append(ch_reref_types[name])
        self.extra_info["node_ch_reref_types"] = node_reref_types

    def _generate_node_ch_coords(self) -> None:
        """Gets the coordinates of channels in the connectivity results,
        averaged across for each channel in the seeds and targets."""
        ch_coords = {}
        for ch_i, combined_name in enumerate(self._comb_names_str):
            ch_coords[combined_name] = np.mean(
                [
                    self.signal.get_coordinates(single_name)[0]
                    for single_name in self._comb_names_list[ch_i]
                ],
                axis=0,
            ).tolist()

        node_ch_coords = [[], []]
        groups = ["seeds", "targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_coords[group_i].append(ch_coords[name])
        self.extra_info["node_ch_coords"] = node_ch_coords

    def _generate_node_ch_regions(self) -> None:
        """Gets the regions of channels in the connectivity results.

        If the regions of each channel in a seed/target for a given node are
        identical, this regions is given as a string, otherwise the unique
        regions are taken and joined into a single string by the " & "
        characters.
        """
        ch_regions = {}
        for node_i, combined_name in enumerate(self._comb_names_str):
            regions = ordered_list_from_dict(
                list_order=self._comb_names_list[node_i],
                dict_to_order=self.extra_info["ch_regions"],
            )
            ch_regions[combined_name] = combine_vals_list(unique(regions))

        node_ch_regions = [[], []]
        groups = ["seeds", "targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_regions[group_i].append(ch_regions[name])
        self.extra_info["node_ch_regions"] = node_ch_regions

    def _generate_node_ch_subregions(self) -> None:
        """Gets the subregions of channels in the connectivity results.

        If the subregions of each channel in a seed/target for a given node are
        identical, these subregions are given as a string, otherwise the unique
        subregions are taken and joined into a single string by the " & "
        characters.
        """
        ch_subregions = {}
        for node_i, combined_name in enumerate(self._comb_names_str):
            subregions = ordered_list_from_dict(
                list_order=self._comb_names_list[node_i],
                dict_to_order=self.extra_info["ch_subregions"],
            )
            ch_subregions[combined_name] = combine_vals_list(unique(subregions))

        node_ch_subregions = [[], []]
        groups = ["seeds", "targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_subregions[group_i].append(ch_subregions[name])
        self.extra_info["node_ch_subregions"] = node_ch_subregions

    def _generate_node_ch_hemispheres(self) -> list[list[bool]]:
        """Gets the hemispheres of channels in the connectivity results.

        If the hemispheres of each channel in a seed/target for a given node are
        identical, this hemispheres is given as a string, otherwise the unique
        hemispheres are taken and joined into a single string by the " & "
        characters.

        RETURNS
        -------
        node_single_hemispheres : list[list[bool]]
        -   list containing two sublists of bools stating whether the channels
            in the seeds/targets of each node were derived from the same
            hemisphere.
        """
        ch_hemispheres = {}
        single_hemispheres = {name: True for name in self._comb_names_str}
        for ch_i, combined_name in enumerate(self._comb_names_str):
            hemispheres = ordered_list_from_dict(
                list_order=self._comb_names_list[ch_i],
                dict_to_order=self.extra_info["ch_hemispheres"],
            )
            unique_types = unique(hemispheres)
            if len(unique_types) > 1:
                single_hemispheres[combined_name] = False
            ch_hemispheres[combined_name] = combine_vals_list(unique_types)

        node_ch_hemispheres = [[], []]
        node_single_hemispheres = [[], []]
        groups = ["seeds", "targets"]
        for group_i, group in enumerate(groups):
            for name in getattr(self, group):
                node_ch_hemispheres[group_i].append(ch_hemispheres[name])
                node_single_hemispheres[group_i].append(single_hemispheres[name])
        self.extra_info["node_ch_hemispheres"] = node_ch_hemispheres

        return node_single_hemispheres

    def _generate_node_lateralisation(
        self, node_single_hemispheres: list[list[bool]]
    ) -> None:
        """Gets the lateralisation of the channels in the connectivity node.

        Can either be "contralateral" if the seed and target are from different
        hemispheres, "ipsilateral" if the seed and target are from the same
        hemisphere, or "ipsilateral & contralateral" if the seed and target are
        from a mix of same and different hemispheres.

        PARAMETERS
        ----------
        node_single_hemispheres : list[list[bool]]
        -   list containing two sublists of bools stating whether the channels
            in the seeds/targets of each node were derived from the same
            hemisphere.
        """
        node_lateralisation = []
        node_ch_hemispheres = self.extra_info["node_ch_hemispheres"]
        for node_i in range(len(node_ch_hemispheres[0])):
            if node_ch_hemispheres[0][node_i] != node_ch_hemispheres[1][node_i]:
                if (
                    not node_single_hemispheres[0][node_i]
                    or not node_single_hemispheres[1][node_i]
                ):
                    lateralisation = "ipsilateral & contralateral"
                else:
                    lateralisation = "contralateral"
            else:
                lateralisation = "ipsilateral"
            node_lateralisation.append(lateralisation)
        self.extra_info["node_lateralisation"] = node_lateralisation

    def _generate_node_ch_epoch_orders(self) -> None:
        """Gets the epoch orders of channels in the connectivity results.

        If either the seed or target has a "shuffled" epoch order, the epoch
        order of the node is "shuffled", otherwise it is "original".
        """
        ch_epoch_orders = {}
        for ch_i, combined_name in enumerate(self._comb_names_str):
            epoch_orders = ordered_list_from_dict(
                list_order=self._comb_names_list[ch_i],
                dict_to_order=self.extra_info["ch_epoch_orders"],
            )
            ch_epoch_orders[combined_name] = combine_vals_list(unique(epoch_orders))

        node_epoch_orders = []
        for seed_name, target_name in zip(self.seeds, self.targets):
            if (
                ch_epoch_orders[seed_name] == "original"
                and ch_epoch_orders[target_name] == "original"
            ):
                order = "original"
            else:
                order = "shuffled"
            node_epoch_orders.append(order)
        self.extra_info["node_ch_epoch_orders"] = node_epoch_orders
