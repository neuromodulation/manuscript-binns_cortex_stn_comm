"""Methods for dealing with plotting data."""

from platform import system
from matplotlib import pyplot as plt


def get_plot_colours() -> list[str]:
    """Returns the matplotlib default set of ten colours.

    RETURNS
    -------
    colours : list[str]
    -   The ten default matplotlib colours.
    """

    colours = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    return colours


def maximise_figure_windows() -> None:
    """Maximises the windows of figures being managed by the matplotlib pyplot
    figure manager.

    RAISES
    ------
    NotImplementedError
    -   Raised if maximising figure windows is not supported for the combination
        of operating system and plotting backend.
    """

    backend = plt.get_backend()
    cfm = plt.get_current_fig_manager()
    if backend == "wxAgg":
        cfm.frame.Maximize(True)
    elif backend == "TkAgg":
        if system() == "Windows":
            cfm.window.state("zoomed")  # This is windows only
        else:
            cfm.resize(*cfm.window.maxsize())
    elif backend == "QT4Agg" or backend == "QtAgg":
        cfm.window.showMaximized()
    elif callable(getattr(cfm, "full_screen_toggle", None)):
        if not getattr(cfm, "flag_is_max", None):
            cfm.full_screen_toggle()
            cfm.flag_is_max = True
    else:
        raise NotImplementedError(
            "Error when trying to make a figure fullscreen:\nMaking a figure "
            "fullscreen is not supported for the current operating system and "
            "plotting backend combination."
        )
