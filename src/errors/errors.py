class CallMeMaybeError(Exception):
    """Base class for all errors of this project."""


class GenerationError(CallMeMaybeError):
    """Raised when the generation does not finish in time."""


class FileReadError(CallMeMaybeError):
    """Raised when an input file cannot be read or is not valid."""


class FileWriteError(CallMeMaybeError):
    """Raised when the output file cannot be written."""


class TrieError(CallMeMaybeError):
    """Raised when a trie is missing or a token leaves it."""


class SchemaError(CallMeMaybeError):
    """Raised when a value does not match the function schema."""


class BoolError(CallMeMaybeError):
    """Raised when a boolean value being written is not valid."""


class FuncSelectError(CallMeMaybeError):
    """Raised when the chosen function does not exist."""
