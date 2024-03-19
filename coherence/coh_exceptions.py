"""Custom exception classes.

CLASSES
-------
ProcessingOrderError : Exception
-   Class for raising exceptions/errors associated with performing processing
    steps in an incorrect order.

UnavailableProcessingError : Exception
-   Class for raising exceptions/error associated with trying to perform
    processing steps that cannot be done.

MissingAttributeError : Exception
-   Class for raising exceptions/errors associated with attributes of an
    object that have not been instantiated.

EntryLengthError : Exception
-   Class for raising exceptions/errors associated with entries within a list
    having a nonidentical length.

DuplicateEntryError : Exception
-   Class for raising exceptions/errors associated with duplicate entries
    occuring within an object.

MissingEntryError : Exception
-   Class for raising exceptions/errors associated with missing entries between
    objects.

ChannelOrderError : Exception
-   Class for raising exceptions/errors associated with lists of channel names
    being in different orders.

ChannelAttributeError : Exception
-   Class for raising exceptions/errors associated with trying to handle
    channels with different attributes.

UnidenticalEntryError : Exception
-   Class for raising exceptions/errors associated with two objects being
    unidentical.

PreexistingAttributeError
-   Class for raising exceptions/errors associated with an attribute already
    existing within an object.

MissingFileExtensionError : Exception
-   Class for raising exceptions/errors associated with a filename string
    missing a filetype extension.

UnsupportedFileExtensionError : Exception
-   Class for raising exceptions/errors associated with a filename string
    containing a filetype extension for which saving is not supported.
"""


class ProcessingOrderError(Exception):
    """Class for raising exceptions/errors associated with performing processing
    steps in an incorrect order."""


class UnavailableProcessingError(Exception):
    """Class for raising exceptions/error associated with trying to perform
    processing steps that cannot be done."""


class MissingAttributeError(Exception):
    """Class for raising exceptions/errors associated with attributes of an
    object that have not been instantiated."""


class EntryLengthError(Exception):
    """Class for raising exceptions/error associated with entries within a list
    having a nonidentical length."""


class DuplicateEntryError(Exception):
    """Class for raising exceptions/errors associated with duplicate entries
    occuring within an object.
    """


class MissingEntryError(Exception):
    """Class for raising exceptions/errors associated with missing entries
    between objects."""


class ChannelOrderError(Exception):
    """Class for raising exceptions/errors associated with lists of channel
    names being in different orders.
    """


class ChannelAttributeError(Exception):
    """Class for raising exceptions/errors associated with trying to handle
    channels with different attributes."""


class UnidenticalEntryError(Exception):
    """Class for raising exceptions/errors associated with two objects being
    unidentical."""


class PreexistingAttributeError(Exception):
    """Class for raising exceptions/errors associated with an attribute already
    existing within an object."""


class MissingFileExtensionError(Exception):
    """Class for raising exceptions/errors associated with a filename string
    missing a filetype extension."""


class UnsupportedFileExtensionError(Exception):
    """Class for raising exceptions/errors associated with a filename string
    containing a filetype extension for which saving is not supported."""
