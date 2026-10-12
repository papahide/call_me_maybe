from enum import Enum
from pydantic import BaseModel
from typing import ClassVar
from ..models.schema import ParameterType
import string


class JsonState(str, Enum):
    """Position of the generator inside the JSON object being built.

    Each state decides which characters can come next.

    Attributes:
        START: Nothing generated yet, only ``{`` is valid.
        START_OBJK: After ``{`` or ``,``, only the opening quote of a key.
        OBJECT_KEY: Inside a key name, driven by the trie of parameters.
        AFTER_KEY: Key closed, only ``:`` is valid.
        BEFORE_VALUE: After ``:``, waiting for the first char of a value.
        IN_BOOL_VALUE: Inside ``true`` or ``false``.
        IN_STRING_VALUE: Inside a string value.
        OBJECT_VALUE: Inside a numeric value.
        AFTER_VALUE: String closed, only ``,`` or ``}`` can follow.
        CLOSE: The object is complete.
    """
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


class JsonValidation(BaseModel):
    """Character rules used to constrain each state of the JSON.

    It only holds constants, so it is never instantiated.

    Attributes:
        VALID_CHARS: Characters allowed at each state.
        VALUE_START_CHARS: First characters allowed for each value type.
        STRING_CHARS: Characters allowed inside a string value.
    """
    VALID_CHARS: ClassVar[dict[JsonState, set[str]]] = {
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

    VALUE_START_CHARS: ClassVar[dict[ParameterType, set[str]]] = {
        ParameterType.STRING: {'"'},
        ParameterType.NUMBER: set(string.digits) | {"-"},
        ParameterType.BOOLEAN: {"t", "f"}
    }

    STRING_CHARS: ClassVar[set[str]] = set(string.printable) - {'"', "\\"}

    @staticmethod
    def is_valid_string_token(text: str) -> bool:
        """Check whether a token can be part of a string value.

        A token is valid when all its characters are allowed inside a
        JSON string. It may end with the closing quote, but a quote
        anywhere else is rejected.

        Args:
            text: Text of the token to check.

        Returns:
            True if the token can be written inside a string value.
        """
        if not text:
            return False
        body = text[:-1] if text.endswith('"') else text
        return all(c in JsonValidation.STRING_CHARS for c in body)


class Transitions(BaseModel):
    """State machine that tracks where the JSON generation is.

    Attributes:
        TRANSITIONS: Maps (state, character) to the next state. Pairs
            that are not listed keep the current state.
    """
    TRANSITIONS: ClassVar[dict[tuple[JsonState, str], JsonState]] = {
        (JsonState.START, "{"): JsonState.START_OBJK,
        (JsonState.START_OBJK, "\""): JsonState.OBJECT_KEY,
        (JsonState.OBJECT_KEY, "\""): JsonState.AFTER_KEY,
        (JsonState.AFTER_KEY, ":"): JsonState.BEFORE_VALUE,
        (JsonState.BEFORE_VALUE, "\""): JsonState.IN_STRING_VALUE,
        (JsonState.BEFORE_VALUE, "-"): JsonState.OBJECT_VALUE,
        **{(JsonState.BEFORE_VALUE, d):
           JsonState.OBJECT_VALUE for d in string.digits},
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
        """Return the state reached after generating a token.

        Inside a string value only the last character matters, because
        a token ending in a quote closes the string.

        Args:
            curr_state: State before generating the token.
            value: Text of the generated token.

        Returns:
            The next state, or ``curr_state`` if there is no transition.
        """
        if curr_state == JsonState.IN_STRING_VALUE:
            value = value[-1]
        return Transitions.TRANSITIONS.get((curr_state, value), curr_state)
