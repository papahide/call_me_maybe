from enum import Enum
from ..models.schema import ParameterType
import string

class JsonState(str, Enum):
    START = "start"
    START_OBJK = "start_to_object_key"
    OBJECT_KEY = "key"
    AFTER_KEY = "colon"
    BEFORE_VALUE = "before_value"
    IN_BOOL_VALUE = "bool_value"
    IN_STRING_VALUE = "abc"
    OBJECT_VALUE = "number"
    AFTER_VALUE = "after_value"
    CLOSE = "close"

class JsonValidation():
    VALID_CHARS: dict[JsonState, set[str]] = {
    JsonState.START: {"{"},
    JsonState.START_OBJK: {"\""},
    JsonState.OBJECT_KEY: set(string.ascii_letters) | {"\""},
    JsonState.AFTER_KEY: {":"},
    JsonState.BEFORE_VALUE: set(string.digits) | {"\"", "-"},
    JsonState.IN_BOOL_VALUE: set(),
    JsonState.IN_STRING_VALUE: set(string.printable) | {"\""},
    JsonState.OBJECT_VALUE: set(string.digits) | {",", "}"},
    JsonState.AFTER_VALUE: {",", "}"},
    JsonState.CLOSE: {"}"}
    }

    VALUE_START_CHARS: dict[ParameterType, set[str]] = {
    ParameterType.STRING: {'"'},
    ParameterType.NUMBER: set(string.digits) | {"-"},
    ParameterType.BOOLEAN: {"t", "f"}
    }

class Transitions():
    TRANSITIONS: dict[tuple[JsonState, str], JsonState] = {
        (JsonState.START, "{"): JsonState.START_OBJK,
        (JsonState.START_OBJK, "\""): JsonState.OBJECT_KEY,
        (JsonState.OBJECT_KEY, "\""): JsonState.AFTER_KEY,
        (JsonState.AFTER_KEY, ":"): JsonState.BEFORE_VALUE,
        (JsonState.BEFORE_VALUE, "\""): JsonState.IN_STRING_VALUE,
        (JsonState.BEFORE_VALUE, "-"): JsonState.OBJECT_VALUE,
        **{(JsonState.BEFORE_VALUE, d): JsonState.OBJECT_VALUE for d in string.digits},
        (JsonState.BEFORE_VALUE, "t"): JsonState.IN_BOOL_VALUE,
        (JsonState.BEFORE_VALUE, "f"): JsonState.IN_BOOL_VALUE,
        (JsonState.IN_STRING_VALUE, "\""): JsonState.AFTER_VALUE,
        (JsonState.OBJECT_VALUE, ","): JsonState.START_OBJK,
        (JsonState.OBJECT_VALUE, "}"): JsonState.CLOSE,
        (JsonState.AFTER_VALUE, ","): JsonState.START_OBJK,
        (JsonState.AFTER_VALUE, "}"): JsonState.CLOSE,
    }

    @staticmethod
    def next_state(curr_state: JsonState, value: str) -> JsonState:
        return Transitions.TRANSITIONS.get((curr_state, value), curr_state)
