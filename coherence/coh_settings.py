"""Methods for handling information in the data and analysis settings files.

METHODS
-------
-   Extracts metadata information from a settings dictionary into a dictionary
    of key:value pairs corresponding to the various metadata aspects of the
    data.
"""


from coh_exceptions import MissingAttributeError


def extract_metadata(
    settings: dict,
    info_keys: list[str] = [
        "cohort",
        "sub",
        "med",
        "stim",
        "ses",
        "task",
        "run",
    ],
    missing_key_error: bool = True,
) -> dict:
    """Extracts metadata information from a settings dictionary into a
    dictionary of key:value pairs corresponding to the various metadata aspects
    of the data.

    PARAMETERS
    ----------
    settings : dict
    -   Dictionary of key:value pairs containing information about the data
        being analysed.

    info_keys : list[str], optional
    -   List of strings containing the keys in the settings dictionary that
        should be extracted into the metadata information dictionary.

    missing_key_error : bool, optional
    -   Whether or not an error should be raised if a key in info_keys is
        missing from the settings dictionary. Default True. If False, None is
        given as the value of that key in the metadata information dictionary.

    RETURNS
    -------
    metadata : dict
    -   Extracted metadata.

    RAISES
    ------
    MissingAttributeError
    -   Raised if a metadata information key is missing from the settings
        dictionary and 'missing_key_error' is 'True'.
    """

    metadata = {}
    for key in info_keys:
        if key in settings.keys():
            metadata[key] = settings[key]
        else:
            if missing_key_error:
                raise MissingAttributeError(
                    f"The metadata information '{key}' is not present in "
                    "the settings file, and thus cannot be extracted."
                )
            metadata[key] = None

    return metadata
