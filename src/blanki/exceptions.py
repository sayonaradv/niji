"""Custom exceptions for the blanki library."""


class BlankiError(Exception):
    """Base exception for all blanki-related errors."""


class ModelNotFoundError(BlankiError):
    """Raised when a model checkpoint cannot be found."""


class DataNotFoundError(BlankiError):
    """Raised when required data files cannot be found."""
