class CallMeMaybeError(Exception):
    """Base class for all errors of this project."""
    pass


class FileReadError(CallMeMaybeError):
    pass


class FileWriteError(CallMeMaybeError):
    pass


class TrieError(CallMeMaybeError):
    pass


class SchemaError(CallMeMaybeError):
    pass


class BoolError(CallMeMaybeError):
    pass


class FuncSelectError(CallMeMaybeError):
    pass
