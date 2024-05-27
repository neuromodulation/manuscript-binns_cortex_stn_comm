"""Classes for calculating connectivity between signals.

CLASSES
-------
ConnectivityCoherence : subclass of the abstract base class 'ProcMethod'
-   Calculates the coherence (standard or imaginary) between signals.

ConectivityMultivariate : subclass of the abstract base class 'ProcMethod'
-   Calculates the multivariate connectivity (multivariate interaction measure,
    MIM, or maximised imaginary coherence, MIC) between signals.
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np
from scipy.stats import t, sem

from pybispectra import TDE, compute_fft, set_precision

from coh_exceptions import ProcessingOrderError
from coh_connectivity_processing_methods import (
    ProcBivariateConnectivity,
    ProcMultivariateConnectivity,
)
from coh_handle_entries import combine_dicts
from coh_progress_bar import ProgressBar
from coh_saving import save_dict
import coh_signal


set_precision("single")  # set pybispectra precision


class ConnectivityCoherence(ProcBivariateConnectivity):
    """Calculates the coherence (standard or imaginary) between signals.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool; default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process
    -   Performs coherence analysis.

    save_object
    -   Saves the object as a .pkl file.

    save_results
    -   Saves the results and additional information as a file.

    results_as_dict
    -   Returns the results and additional information as a dictionary.

    get_results
    -   Extracts and returns results.
    """

    def __init__(self, signal: coh_signal.Signal, verbose: bool = True) -> None:
        super().__init__(signal, verbose)
        super()._sort_inputs()

        self._accepted_con_methods = ["coh", "cohy", "imcoh"]

    def process(
        self,
        con_methods: list[str],
        power_method: str,
        seeds: Union[str, list[str], dict],
        targets: Union[str, list[str], dict],
        fmin: Union[Union[float, tuple], None] = None,
        fmax: Union[float, tuple] = np.inf,
        fskip: int = 0,
        faverage: bool = False,
        tmin: Union[float, None] = None,
        tmax: Union[float, None] = None,
        mt_bandwidth: Union[float, None] = None,
        mt_adaptive: bool = False,
        mt_low_bias: bool = True,
        cwt_freqs: Union[NDArray, None] = None,
        cwt_n_cycles: Union[float, NDArray] = 7.0,
        average_windows: bool = False,
        average_timepoints: bool = False,
        absolute_connectivity: bool = False,
        block_size: int = 1000,
        n_jobs: int = 1,
    ) -> None:
        """Applies the connectivity analysis using the
        spectral_connectivity_epochs function of the mne-connectivity package.

        PARAMETERS
        ----------
        con_methods : list[str]
        -   The methods for calculating connectivity.
        -   Supported inputs are: 'coh' - standard coherence; 'cohy' -
            coherency; and 'imcoh' - imaginary part of coherence.

        power_method : str
        -   The mode for calculating connectivity using 'method'.
        -   Supported inputs are: 'multitaper'; 'fourier'; and 'cwt_morlet'.

        seeds : str | list[str]
        -   The channels to use as seeds for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'.
        -   If a list of strings, each entry of the list should be a channel
            name.

        targets : str | list[str]
        -   The channels to use as targets for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'.
        -   If a list of strings, each entry of the list should be a channel
            name.

        fmin : float | tuple | None
        -   The lower frequency of interest.
        -   If a float, this frequency is used.
        -   If a tuple, multiple bands are defined, with each entry being the
            lower frequency for that band. E.g. (8., 20.) would give two bands
            using 8 Hz and 20 Hz, respectively, as their lower frequencies.
        -   If None, no lower frequency is used.

        fmax : float | tuple; default infinite
        -   The higher frequency of interest.
        -   If a float, this frequency is used.
        -   If a tuple, multiple bands are defined, with each entry being the
            higher frequency for that band. E.g. (8., 20.) would give two bands
            using 8 Hz and 20 Hz, respectively, as their higher frequencies.
        -   If an infinite float, no higher frequency is used.

        fskip : int; default 0
        -   Omit every 'fskip'+1th frequency bin to decimate the frequency
            domain.
        -   If 0, no bins are skipped.

        faverage : bool; default False
        -   Whether or not to average the connectivity values for each frequency
            band.

        tmin : float | None; default None
        -   Time to start the connectivity estimation.
        -   If None, the data is used from the beginning.

        tmax : float | None; default None
        -   Time to end the connectivity estimation.
        -   If None, the data is used until the end.

        mt_bandwidth : float | None
        -   The bandwidth, in Hz, of the multitaper windowing function.
        -   Only used if 'mode' is 'multitaper'.

        mt_adaptive : bool; default False
        -   Whether or not to use adaptive weights to combine the tapered
            spectra into power spectra.
        -   Only used if 'mode' is 'multitaper'.

        mt_low_bias : bool: default True
        -   Whether or not to only use tapers with > 90% spectral concentration
            within bandwidth.
        -   Only used if 'mode' is 'multitaper'.

        cwt_freqs: array of float | None; default None
        -   The frequencies of interest to calculate connectivity for.
        -   Only used if 'mode' is 'cwt_morlet'. In this case, 'cwt_freqs'
            cannot be None.

        cwt_n_cycles: float | array of float; default 7.0
        -   The number of cycles to use when calculating connectivity.
        -   If an single integer or float, this number of cycles is for each
            frequency.
        -   If an array, the entries correspond to the number of cycles to use
            for each frequency being analysed.
        -   Only used if 'mode' is 'cwt_morlet'.

        average_windows : bool; default False
        -   Whether or not to average connectivity results across windows.

        average_timepoints : bool; default False
        -   Whether or not to average connectivity results across timepoints.

        absolute_connectivity : bool; default False
        -   Whether or not to take the absolute connectivity values.

        block_size : int; default 1000
        -   The number of connections to compute at once.

        n_jobs : int; default 1
        -   The number of epochs to calculate connectivity for in parallel.
        """
        if self._processed:
            ProcessingOrderError(
                "The data in this object has already been processed. "
                "Initialise a new instance of the object if you want to "
                "perform other analyses on the data."
            )

        self.con_methods = con_methods
        self.power_method = power_method
        self.seeds = seeds
        self.targets = targets
        self.fmin = fmin
        self.fmax = fmax
        self.fskip = fskip
        self.faverage = faverage
        self.tmin = tmin
        self.tmax = tmax
        self.mt_bandwidth = mt_bandwidth
        self.mt_adaptive = mt_adaptive
        self.mt_low_bias = mt_low_bias
        self.cwt_freqs = cwt_freqs
        self.cwt_n_cycles = cwt_n_cycles
        self.average_windows = average_windows
        self.average_timepoints = average_timepoints
        self.absolute_connectivity = absolute_connectivity
        self.block_size = block_size
        self.n_jobs = n_jobs

        self._sort_processing_inputs()

        self._get_results()

    def _sort_processing_inputs(self) -> None:
        """Checks that the processing inputs are appropriate and implements them
        appropriately."""
        if not set(self.con_methods).issubset(self._accepted_con_methods):
            raise NotImplementedError(
                f"Not all the connectivity methods {self.con_methods} are "
                f"supported: {self._accepted_con_methods}."
            )

        self._sort_seeds_targets()
        super()._sort_processing_inputs()

        self._sort_used_settings()


class ConnectivityTDE(ProcBivariateConnectivity):
    """Calculate estimates of time delay between signals.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool; default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process
    -   Performs time delay estimation analysis.

    save_object
    -   Saves the object as a .pkl file.

    save_results
    -   Saves the results and additional information as a file.

    results_as_dict
    -   Returns the results and additional information as a dictionary.

    get_results
    -   Extracts and returns results.
    """

    def __init__(self, signal: coh_signal.Signal, verbose: bool = True) -> None:
        super().__init__(signal, verbose)
        super()._sort_inputs()

    def process(
        self,
        seeds: Union[str, list[str], dict],
        targets: Union[str, list[str], dict],
        freq_bands: dict = {"all": (0.0, np.inf)},
        method: int | tuple[int] = 1,
        antisym: bool | tuple[bool] = False,
        window_func: str = "hamming",
        conf_interval: float = 0.99,
        average_windows: bool = False,
        n_jobs: int = 1,
    ) -> None:
        """Applies the connectivity analysis using the
        spectral_connectivity_epochs function of the mne-connectivity package.

        PARAMETERS
        ----------
        seeds : str | list[str]
        -   The channels to use as seeds for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'.
        -   If a list of strings, each entry of the list should be a channel
            name.

        targets : str | list[str]
        -   The channels to use as targets for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'.
        -   If a list of strings, each entry of the list should be a channel
            name.

        freq_bands : dict (default {"all": (0.0, np.inf)})
            The frequency bands to use for the analysis, where the keys are the
            names of the bands and the values are tuples of the lower and upper
            frequencies of the bands, respectively. Takes all frequencies by
            default.

        method : int | tuple of int (default 1)
            Method(s) to use to compute TDE, as in Nikias & Pan (1988).

        antisym : bool | tuple of bool (default False)
            Whether or not to antisymmetrise the bispectrum when computing TDE.

        window_func : str (default "hamming")
            Window function to apply when computing the FFT of the data. Can be
            "hanning" or "hamming".

        conf_interval : float (default 0.99)
            Confidence interval to use to determine whether the time delay
            estimate can be trusted, between 0 and 1. Only used if the data
            consists of multiple windows.

        average_windows : bool (default False)
            Whether or not to average TDE results across windows.

        n_jobs : int; default 1
        -   The number of epochs to calculate connectivity for in parallel.
        """
        if self._processed:
            ProcessingOrderError(
                "The data in this object has already been processed. "
                "Initialise a new instance of the object if you want to "
                "perform other analyses on the data."
            )

        self.freq_bands = freq_bands
        self.antisym = antisym
        self.method = method
        self.seeds = seeds
        self.targets = targets
        self.window_func = window_func
        self.conf_interval = conf_interval
        self.average_windows = average_windows
        self.n_jobs = n_jobs

        self._sort_processing_inputs()

        self._get_results()

    def _sort_processing_inputs(self) -> None:
        """Checks that the processing inputs are appropriate and implements
        them appropriately."""
        self._sort_seeds_targets()
        self._sort_fbands()
        super()._sort_processing_inputs()

        method_name_mapping = {
            val: name
            for val, name in zip([1, 2, 3, 4], ["i", "ii", "iii", "iv"])
        }
        sym_name_mapping = {False: "standard", True: "antisym"}

        self.tde_methods = []
        for method in self.method:
            for symmetrise in self.antisym:
                self.tde_methods.append(
                    f"{method_name_mapping[method]}_"
                    f"{sym_name_mapping[symmetrise]}"
                )

        self._sort_used_settings()

    def _sort_fbands(self) -> None:
        """Sort `fbands` input."""
        if not isinstance(self.freq_bands, dict):
            raise TypeError("`freq_bands` must be a dictionary.")

        fmin = []
        fmax = []
        for fband in self.freq_bands.values():
            if len(fband) != 2:
                raise ValueError(
                    "The values of `freq_bands` must have length 2."
                )
            fmin.append(fband[0])
            fmax.append(fband[1])

        self._fmin = tuple(fmin)
        self._fmax = tuple(fmax)

    def _sort_used_settings(self) -> None:
        """Collects the settings that are relevant for the processing being
        performed and adds only these settings to the 'processing_steps'
        dictionary."""
        used_settings = {
            "method": self.method,
            "symmetrise": self.antisym,
            "window_func": self.window_func,
        }

        self.processing_steps["time_delay_estimation"] = used_settings

    def _get_results(self) -> None:
        """Performs the time delay estimation analysis."""
        if self.verbose:
            self._progress_bar = ProgressBar(
                n_steps=len(self.signal.data),
                title="Computing time delay estimation",
            )

        self.results = [[] for _ in range(len(self.tde_methods))]
        self.taus = [[] for _ in range(len(self.tde_methods))]
        for window_idx, window_data in enumerate(self.signal.data):
            if self.verbose:
                print(
                    f"Computing time delay for window {window_idx+1} of "
                    f"{len(self.signal.data)}.\n"
                )

            data = window_data.get_data()

            fft_coeffs, freqs = compute_fft(
                data=data,
                sampling_freq=self.signal.data[0].info["sfreq"],
                n_points=2 * data.shape[2] + 1,
                window=self.window_func,
                n_jobs=self.n_jobs,
            )

            window_tde = TDE(
                data=fft_coeffs,
                freqs=freqs,
                sampling_freq=self.signal.data[0].info["sfreq"],
                verbose=self.verbose,
            )
            window_tde.compute(
                indices=(
                    tuple(self.indices[0].tolist()),
                    tuple(self.indices[1].tolist()),
                ),
                fmin=self._fmin,
                fmax=self._fmax,
                antisym=self.antisym,
                method=self.method,
                n_jobs=self.n_jobs,
            )

            window_tde_results = window_tde.results
            if not isinstance(window_tde_results, tuple):
                window_tde_results = (window_tde_results,)

            for method_i, tde_result in enumerate(window_tde_results):
                self.results[method_i].append(tde_result.get_results())
                self.taus[method_i].append(tde_result.tau)

            if self._progress_bar is not None:
                self._progress_bar.update_progress()

        if self._progress_bar is not None:
            self._progress_bar.close()

        for method_i in range(len(self.results)):
            self.results[method_i] = np.array(self.results[method_i])
            self.taus[method_i] = np.array(self.taus[method_i])

        self._timepoints = tde_result.times

        if len(self.signal.data) > 1:
            self._confidence = []
            for taus_result in self.taus:
                conf_intervals = t.interval(
                    alpha=self.conf_interval,
                    df=len(self.signal.data) - 1,
                    loc=np.array(taus_result).mean(axis=0),
                    scale=sem(taus_result, axis=0),
                )
                self._confidence.append(
                    ~((conf_intervals[0] <= 0) & (conf_intervals[1] >= 0))
                )

        self._sort_dimensions()
        self._generate_extra_info()
        self._processed = True

    def _sort_dimensions(self) -> None:
        """Sort the dimensions of the results."""
        self._results_dims = [
            "windows",
            "connections",
            "frequencies",
            "timepoints",
        ]

        if self.average_windows:
            n_windows = len(self.results[0])

            for method_i in range(len(self.results)):
                self.results[method_i] = np.mean(
                    self.results[method_i], axis=0
                )[np.newaxis]

            self._windows_averaged = True
            if self.verbose:
                print(f"Averaging the data over {n_windows} windows.\n")

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
        """Returns core information about the TDE results.

        RETURNS
        -------
        core_info : dict
        -   The core information about the TDE results.
        """
        tau_dims = ["channels", "windows"]  # tau never averaged over windows
        if self._n_windows == 1 or self._windows_averaged:
            window_idcs = 0
        if self._windows_averaged:
            tde_dims = ["channels", "timepoints"]
        else:
            tde_dims = ["channels", "windows", "timepoints"]
            window_idcs = np.arange(self._n_windows)
            windows = (window_idcs + 1).tolist()

        tde_data = []
        tau_data = []
        confidence = []
        for method_i in range(len(self.results)):
            tde_data.append([])
            tau_data.append([])
            confidence.append([])
            for con_i in range(len(self.indices[0])):
                for fband_i in range(len(self.freq_bands.keys())):
                    tde_data[method_i].append(
                        self.results[method_i][
                            window_idcs, con_i, fband_i, :
                        ].tolist()
                    )
                    tau_data[method_i].append(
                        self.taus[method_i][:, con_i, fband_i]
                    )
                    if self._n_windows > 1:
                        confidence[method_i].append(
                            self._confidence[method_i][con_i, fband_i].tolist()
                        )

        fband_names = list(self.freq_bands.keys()) * len(self.seeds)
        fband_freqs = list(self.freq_bands.values()) * len(self.seeds)
        seeds = np.repeat(self.seeds, len(self.freq_bands.keys())).tolist()
        targets = np.repeat(self.targets, len(self.freq_bands.keys())).tolist()

        core_info = {}
        for method_i, method_name in enumerate(self.tde_methods):
            core_info[f"tde-{method_name}"] = tde_data[method_i]
            core_info[f"tde-{method_name}_dimensions"] = tde_dims
            core_info[f"tde-{method_name}_tau"] = tau_data[method_i]
            core_info[f"tde-{method_name}_tau_dimensions"] = tau_dims
            if self._n_windows > 1:
                core_info[f"tde-{method_name}_confidence"] = confidence[
                    method_i
                ]

        core_info.update(
            freq_band_names=fband_names,
            freq_band_bounds=fband_freqs,
            seed_names=seeds,
            target_names=targets,
            timepoints=self._timepoints,
            sampling_frequency=self.signal.data[0].info["sfreq"],
            processing_steps=self.processing_steps,
            subject_info=self.signal.data[0].info["subject_info"],
        )
        if not self._windows_averaged:
            core_info.update(windows=windows)

        return core_info

    def _extra_info_for_results_dict(self) -> dict:
        """Get extra info and pad according to number of frequency bands."""
        extra_info = super()._extra_info_for_results_dict()
        n_fbands = len(self.freq_bands.keys())
        for key, value in extra_info.items():
            if key != "metadata":
                rep_val = []
                for val in value:
                    rep_val.extend([val for _ in range(n_fbands)])
                extra_info[key] = np.array(rep_val).tolist()

        return extra_info


class ConnectivityMultivariateCoh(ProcMultivariateConnectivity):
    """Calculates the maximised imaginary coherence (MIC) and multivariate
    interaction measure (MIM) between signals.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool; default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process
    -   Performs multivariate connectivity analysis.

    save_object
    -   Saves the object as a .pkl file.

    save_results
    -   Saves the results and additional information as a file.

    results_as_dict
    -   Returns the results and additional information as a dictionary.

    get_results
    -   Extracts and returns results.
    """

    def __init__(self, signal: coh_signal.Signal, verbose: bool = True) -> None:
        super().__init__(signal, verbose)
        super()._sort_inputs()

        self.con_methods = ["mic", "mim"]

    def process(
        self,
        power_method: str,
        seeds: Union[str, list[str], dict],
        targets: Union[str, list[str], dict],
        fmin: Union[Union[float, tuple], None] = None,
        fmax: Union[float, tuple] = np.inf,
        fskip: int = 0,
        faverage: bool = False,
        tmin: Union[float, None] = None,
        tmax: Union[float, None] = None,
        mt_bandwidth: Union[float, None] = None,
        mt_adaptive: bool = False,
        mt_low_bias: bool = True,
        cwt_freqs: Union[NDArray, None] = None,
        cwt_n_cycles: Union[float, NDArray] = 7.0,
        n_components: Union[list[int], str] = "rank",
        average_windows: bool = False,
        average_timepoints: bool = False,
        block_size: int = 1000,
        n_jobs: int = 1,
    ) -> None:
        """Applies the connectivity analysis using the
        multivar_spectral_connectivity_epochs function of the mne-connectivity
        package.

        PARAMETERS
        ----------
        power_method : str
        -   The spectral method for calculating the cross-spectral density.
        -   Supported inputs are: 'multitaper'; 'fourier'; and 'cwt_morlet'.

        seeds : str | list[str] | dict
        -   The channels to use as seeds for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'. In this case, channels
            belonging to each type with different epoch orders and rereferencing
            types will be handled separately.
        -   If a list of strings, each entry of the list should be a channel
            name.

        targets : str | list[str] | dict
        -   The channels to use as targets for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'. In this case, channels
            belonging to each type with different epoch orders and rereferencing
            types will be handled separately.
        -   If a list of strings, each entry of the list should be a channel
            name.

        fmin : float | tuple | None
        -   The lower frequency of interest.
        -   If a float, this frequency is used.
        -   If a tuple, multiple bands are defined, with each entry being the
            lower frequency for that band. E.g. (8., 20.) would give two bands
            using 8 Hz and 20 Hz, respectively, as their lower frequencies.
        -   If None, no lower frequency is used.

        fmax : float | tuple; default infinite
        -   The higher frequency of interest.
        -   If a float, this frequency is used.
        -   If a tuple, multiple bands are defined, with each entry being the
            higher frequency for that band. E.g. (8., 20.) would give two bands
            using 8 Hz and 20 Hz, respectively, as their higher frequencies.
        -   If an infinite float, no higher frequency is used.

        fskip : int; default 0
        -   Omit every 'fskip'+1th frequency bin to decimate the frequency
            domain.
        -   If 0, no bins are skipped.

        faverage : bool; default False
        -   Whether or not to average the connectivity values for each frequency
            band.

        tmin : float | None; default None
        -   Time to start the connectivity estimation.
        -   If None, the data is used from the beginning.

        tmax : float | None; default None
        -   Time to end the connectivity estimation.
        -   If None, the data is used until the end.

        mt_bandwidth : float | None
        -   The bandwidth, in Hz, of the multitaper windowing function.
        -   Only used if 'mode' is 'multitaper'.

        mt_adaptive : bool; default False
        -   Whether or not to use adaptive weights to combine the tapered
            spectra into power spectra.
        -   Only used if 'mode' is 'multitaper'.

        mt_low_bias : bool: default True
        -   Whether or not to only use tapers with > 90% spectral concentration
            within bandwidth.
        -   Only used if 'mode' is 'multitaper'.

        cwt_freqs: numpy array[int | float] | None
        -   The frequencies of interest to calculate connectivity for.
        -   Only used if 'mode' is 'cwt_morlet'. In this case, 'cwt_freqs'
            cannot be None.

        cwt_n_cycles: float | array of float; default 7.0
        -   The number of cycles to use when calculating connectivity.
        -   If an single integer or float, this number of cycles is for each
            frequency.
        -   If an array, the entries correspond to the number of cycles to use
            for each frequency being analysed.
        -   Only used if 'mode' is 'cwt_morlet'.

        n_components ######################################################

        average_windows : bool; default False
        -   Whether or not to average connectivity results across windows.

        average_timepoints : bool; default False
        -   Whether or not to average connectivity results across timepoints.

        block_size : int; default 1000
        -   The number of connections to compute at once.

        n_jobs : int; default 1
        -   The number of epochs to calculate connectivity for in parallel.
        """
        if self._processed:
            ProcessingOrderError(
                "The data in this object has already been processed. "
                "Initialise a new instance of the object if you want to "
                "perform other analyses on the data."
            )

        self.power_method = power_method
        self.seeds = seeds
        self.targets = targets
        self.fmin = fmin
        self.fmax = fmax
        self.fskip = fskip
        self.faverage = faverage
        self.tmin = tmin
        self.tmax = tmax
        self.mt_bandwidth = mt_bandwidth
        self.mt_adaptive = mt_adaptive
        self.mt_low_bias = mt_low_bias
        self.cwt_freqs = cwt_freqs
        self.cwt_n_cycles = cwt_n_cycles
        self.n_components = n_components
        self.average_windows = average_windows
        self.average_timepoints = average_timepoints
        self.block_size = block_size
        self.n_jobs = n_jobs

        self._sort_processing_inputs()

        self._get_results()

    def _sort_processing_inputs(self) -> None:
        """Checks that the processing inputs are appropriate and implements
        them appropriately."""
        super()._sort_processing_inputs()
        self._sort_used_settings()

    def _sort_used_settings(self) -> None:
        """Collects the settings that are relevant for the processing being
        performed and adds only these settings to the 'processing_steps'
        dictionary."""
        used_settings = {
            "con_methods": self.con_methods,
            "power_method": self.power_method,
            "n_components": self.n_components,
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
        super().save_results(
            fpath=fpath, ftype=ftype, ask_before_overwrite=ask_before_overwrite
        )

        if ask_before_overwrite is None:
            ask_before_overwrite = self.verbose

        save_dict(
            to_save=self.topographies_as_dict(),
            fpath=f"{fpath}_topographies",
            ftype=ftype,
            ask_before_overwrite=ask_before_overwrite,
            convert_numpy_to_python=True,
            verbose=self.verbose,
        )

    def topographies_as_dict(self) -> dict:
        """"""
        core_info = self._core_info_for_topographies_dict()
        extra_info = self._extra_info_for_topographies_dict()

        return combine_dicts([core_info, extra_info])

    def _core_info_for_topographies_dict(self) -> dict:
        """"""
        topos, topo_dimensions = self._rearrange_topographies()

        concatenated_names = [
            name
            for names in [*self._seeds_list, *self._targets_list]
            for name in names
        ]
        core_info = {
            "connectivity-mic_topographies": topos,
            "connectivity-mic_topographies_dimensions": topo_dimensions,
            "ch_names": concatenated_names,
            "sampling_frequency": self.signal.data[0].info["sfreq"],
            "processing_steps": self.processing_steps,
            "subject_info": self.signal.data[0].info["subject_info"],
        }
        core_info.update(self._dimensions_info_for_results_dict())

        seed_names = []
        target_names = []
        con_boundaries = []
        increment_val = 0
        for con_i in range(self._n_cons):
            n_chs = len(self.indices[0][con_i]) + len(self.indices[1][con_i])
            con_boundaries.append(n_chs + increment_val)
            increment_val += n_chs
        con_i = 0
        for entry_i in range(len(concatenated_names)):
            if entry_i > con_boundaries[con_i]:
                con_i += 1
            seed_names.append(self.seeds[con_i])
            target_names.append(self.targets[con_i])
            entry_i += 1
        core_info.update(seed_names=seed_names, target_names=target_names)

        return core_info

    def _rearrange_topographies(self) -> tuple[list, list[str]]:
        """Rearrange topography results into a list containing the concatenated
        topographies for the connections of each seed, and then the connections
        of each target, with dimensions [channels x (windows) x freqs x
        (times)].

        RETURNS
        -------
        topgraphies : list
        -   The rearranged topography results.

        topography_dimensions : list of str
        -   Names of the dimensions of the topography results.
        """
        mic_idx = self.con_methods.index("mic")

        topos = []

        entry_i = 0
        for group_i in range(2):
            for con_i in range(self._n_cons):
                for ch_i in range(len(self.indices[group_i][con_i])):
                    if self._n_windows != 0:
                        topos.append([[] for _ in range(self._n_windows)])
                        for window_i in range(self._n_windows):
                            topos[entry_i][window_i] = (
                                self.results[mic_idx][window_i]
                                .topographies[group_i][con_i][ch_i]
                                .tolist()
                            )
                    else:
                        topos[entry_i] = (
                            self.results[mic_idx][0]
                            .topographies[group_i][con_i][ch_i]
                            .tolist()
                        )
                    entry_i += 1

        topo_dimensions = ["channels", "windows", "frequencies", "timepoints"]
        if "windows" not in self.results_dims:
            topo_dimensions = [
                dim for dim in topo_dimensions if dim != "windows"
            ]
        if "timepoints" not in self.results_dims:
            topo_dimensions = [
                dim for dim in topo_dimensions if dim != "timepoints"
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
        group_names = ["seed", "target"]
        extra_info_key_mappings = {
            "ch_types": "ch_types",
            "ch_coords": "ch_coords",
            "ch_regions": "ch_regions",
            "ch_subregions": "ch_subregions",
            "ch_hemispheres": "ch_hemispheres",
            "ch_reref_types": "ch_reref_types",
            "ch_epoch_orders": "ch_epoch_orders",
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
        ]

        concatenated_names = [
            name
            for names in [*self._seeds_list, *self._targets_list]
            for name in names
        ]

        con_boundaries = []
        increment_val = 0
        for con_i in range(self._n_cons):
            n_chs = len(self.indices[0][con_i]) + len(self.indices[1][con_i])
            con_boundaries.append(n_chs + increment_val)
            increment_val += n_chs

        for key, value in extra_info_key_mappings.items():
            if key == "ch_types":
                extra_info[key] = self.signal.data[0].get_channel_types(
                    picks=concatenated_names
                )
            elif key == "ch_coords":
                extra_info[key] = self.signal.get_coordinates(
                    picks=concatenated_names
                )
            elif self.extra_info[key] is not None:
                if key == "metadata":
                    extra_info[value] = self.extra_info[key]
                else:
                    entry_i = 0
                    con_i = 0
                    for ch_name in concatenated_names:
                        if entry_i > con_boundaries[con_i]:
                            con_i += 1
                        if key in same_info_seeds_targets_keys:
                            if entry_i == 0:
                                extra_info[value] = []
                            extra_info[value].append(
                                self.extra_info[key][con_i]
                            )
                        else:
                            if value[0] != "_":
                                if entry_i == 0:
                                    extra_info[key] = []
                                extra_info[key].append(
                                    self.extra_info[key][ch_name]
                                )
                            else:
                                for group_idx, group_name in enumerate(
                                    group_names
                                ):
                                    new_key = f"{group_name}{value}"
                                    if entry_i == 0:
                                        extra_info[new_key] = []
                                    extra_info[new_key].append(
                                        self.extra_info[key][group_idx][con_i]
                                    )
                        entry_i += 1

        return extra_info


class ConnectivityGranger(ProcMultivariateConnectivity):
    """Calculates multivariate spectral Granger causality between signals.

    PARAMETERS
    ----------
    signal : coh_signal.Signal
    -   The preprocessed data to analyse.

    verbose : bool; default True
    -   Whether or not to print information about the information processing.

    METHODS
    -------
    process
    -   Performs granger causality analysis.

    save_object
    -   Saves the object as a .pkl file.

    save_results
    -   Saves the results and additional information as a file.

    results_as_dict
    -   Returns the results and additional information as a dictionary.

    get_results
    -   Extracts and returns results.
    """

    def __init__(self, signal: coh_signal.Signal, verbose: bool = True) -> None:
        super().__init__(signal, verbose)
        super()._sort_inputs()

        self.con_methods = [
            "gc",
            "gc_ts",
            "net_gc",
            "gc_tr",
            "gc_tr_ts",
            "trgc",
        ]

    def process(
        self,
        power_method: str,
        seeds: Union[str, list[str], dict],
        targets: Union[str, list[str], dict],
        fmin: Union[Union[float, tuple], None] = None,
        fmax: Union[float, tuple] = np.inf,
        fskip: int = 0,
        faverage: bool = False,
        tmin: Union[float, None] = None,
        tmax: Union[float, None] = None,
        mt_bandwidth: Union[float, None] = None,
        mt_adaptive: bool = False,
        mt_low_bias: bool = True,
        cwt_freqs: Union[NDArray, None] = None,
        cwt_n_cycles: Union[float, NDArray] = 7.0,
        n_components: Union[list[int], str] = "rank",
        n_lags: int = 20,
        average_windows: bool = False,
        average_timepoints: bool = False,
        block_size: int = 1000,
        n_jobs: int = 1,
    ):
        """Performs the Granger casuality (GC) analysis on the data, computing
        GC, net GC, time-reversed GC, and net time-reversed GC.

        PARAMETERS
        ----------
        power_method : str
        -   The spectral method for computing the cross-spectral density.
        -   Supported inputs are: "multitaper"; "fourier"; and "cwt_morlet".

        seeds : str | list[str] | dict
        -   The channels to use as seeds for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'. In this case, channels
            belonging to each type with different epoch orders and rereferencing
            types will be handled separately.
        -   If a list of strings, each entry of the list should be a channel
            name.

        targets : str | list[str] | dict
        -   The channels to use as targets for the connectivity analysis.
            Connectivity is calculated from each seed to each target.
        -   If a string, can either be a single channel name, or a single
            channel type. In the latter case, the channel type should be
            preceded by 'type_', e.g. 'type_ecog'. In this case, channels
            belonging to each type with different epoch orders and rereferencing
            types will be handled separately.
        -   If a list of strings, each entry of the list should be a channel
            name.

        n_lags : int; default 20
        -   The number of lags to use when computing autocovariance. Currently,
            only positive-valued integers are supported.

        tmin : float | None; default None
        -   Time to start the connectivity estimation.
        -   If None, the data is used from the beginning.

        tmax : float | None; default None
        -   Time to end the connectivity estimation.
        -   If None, the data is used until the end.

        average_windows : bool; default True
        -   Whether or not to average connectivity results across windows.

        ensure_full_rank_data : bool; default True
        -   Whether or not to make sure that the data being processed has full
            rank by performing a singular value decomposition on the data of the
            seeds and targets and taking only the first n components, where n is
            equal to number of non-zero singular values in the decomposition
            (i.e. the rank of the data).
        -   If this is not performed, errors can arise when computing Granger
            causality as assumptions of the method are violated.

        n_jobs : int; default 1
        -   The number of epochs to calculate connectivity for in parallel.

        cwt_freqs : list[int | float] | None; default None
        -   The frequencies of interest, in Hz.
        -   Only used if 'cs_method' is "cwt_morlet", in which case 'freqs' cannot
            be 'None'.

        cwt_n_cycles: int | float | array[int | float]; default 7
        -   The number of cycles to use when calculating connectivity.
        -   If an single integer or float, this number of cycles is for each
            frequency.
        -   If an array, the entries correspond to the number of cycles to use
            for each frequency being analysed.
        -   Only used if 'cs_method' is "cwt_morlet".

        cwt_use_fft : bool; default True
        -   Whether or not FFT-based convolution is used to compute the wavelet
            transform.
        -   Only used if 'cs_method' is "cwt_morlet".

        cwt_decim : int | slice; default 1
        -   Decimation factor to use during time-frequency decomposition to
            reduce memory usage. If 1, no decimation is performed.

        mt_bandwidth : float | None; default None
        -   The bandwidth, in Hz, of the multitaper windowing function.
        -   Only used if 'cs_method' is "multitaper".

        mt_adaptive : bool; default False
        -   Whether or not to use adaptive weights to combine the tapered
            spectra into power spectra.
        -   Only used if 'cs_method' is "multitaper".

        mt_low_bias : bool: default True
        -   Whether or not to only use tapers with > 90% spectral concentration
            within bandwidth.
        -   Only used if 'cs_method' is "multitaper".

        fmt_fmin : int | float; default 0
        -   The lower frequency of interest.
        -   If a float, this frequency is used.
        -   If a tuple, multiple bands are defined, with each entry being the
            lower frequency for that band. E.g. (8., 20.) would give two bands
            using 8 Hz and 20 Hz, respectively, as their lower frequencies.
        -   Only used if 'cs_method' is "fourier" or "multitaper".

        fmt_fmax : int | float; default infinity
        -   The higher frequency of interest.
        -   If a float, this frequency is used.
        -   If a tuple, multiple bands are defined, with each entry being the
            higher frequency for that band. E.g. (8., 20.) would give two bands
            using 8 Hz and 20 Hz, respectively, as their higher frequencies.
        -   If infinity, no higher frequency is used.
        -   Only used if 'cs_method' is "fourier" or "multitaper".

        fmt_n_fft : int | None; default None
        -   Length of the FFT.
        -   If 'None', the number of samples between 'tmin' and 'tmax' is used.
        -   Only used if 'cs_method' is "fourier" or "multitaper".
        """
        if self._processed:
            ProcessingOrderError(
                "The data in this object has already been processed. "
                "Initialise a new instance of the object if you want to "
                "perform other analyses on the data."
            )

        self.power_method = power_method
        self.seeds = seeds
        self.targets = targets
        self.fmin = fmin
        self.fmax = fmax
        self.fskip = fskip
        self.faverage = faverage
        self.tmin = tmin
        self.tmax = tmax
        self.mt_bandwidth = mt_bandwidth
        self.mt_adaptive = mt_adaptive
        self.mt_low_bias = mt_low_bias
        self.cwt_freqs = cwt_freqs
        self.cwt_n_cycles = cwt_n_cycles
        self.n_components = n_components
        self.n_lags = n_lags
        self.average_windows = average_windows
        self.average_timepoints = average_timepoints
        self.block_size = block_size
        self.n_jobs = n_jobs

        self._sort_processing_inputs()

        self._get_results()

    def _sort_processing_inputs(self) -> None:
        """Checks that inputs for processing the data are appropriate."""
        super()._sort_processing_inputs()
        self._sort_used_settings()

    def _sort_used_settings(self) -> None:
        """Collects the settings that are relevant for the processing being
        performed and adds only these settings to the 'processing_steps'
        dictionary."""
        used_settings = {
            "con_methods": self.con_methods,
            "power_method": self.power_method,
            "n_lags": self.n_lags,
            "n_components": self.n_components,
            "average_windows": self.average_windows,
            "average_timepoints": self.average_timepoints,
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
