"""Custom exceptions for the niji library."""


class NijiError(Exception):
    """Base exception for all niji-related errors."""


class ModelNotFoundError(NijiError):
    """Raised when a model checkpoint cannot be found."""


class DataNotFoundError(NijiError):
    """Raised when required data files cannot be found."""
