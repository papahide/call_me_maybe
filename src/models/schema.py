from pydantic import BaseModel
from enum import Enum


class ParameterType(str, Enum):
    """Types that a parameter or a return value can have.

    Attributes:
        NUMBER: Any number, written with or without decimals.
        STRING: A text.
        BOOLEAN: ``true`` or ``false``.
        INTEGER: A number without decimals.
    """

    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    INTEGER = "integer"


class ParameterSchema(BaseModel):
    """Description of the type of one parameter or return value.

    Attributes:
        type: Type of the value.
    """

    type: ParameterType


class FunctionDefinition(BaseModel):
    """Definition of a function that the model can call.

    Attributes:
        name: Name of the function.
        description: What the function does, shown to the model.
        parameters: Maps each parameter name to its type.
        returns: Type of the value that the function returns.
    """

    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema


class PromptInput(BaseModel):
    """A prompt written in natural language.

    Attributes:
        prompt: The text of the request.
    """

    prompt: str


class ModelOutput(BaseModel):
    """A prompt translated into a function call.

    Attributes:
        prompt: The original text of the request.
        name: Name of the chosen function.
        parameters: Maps each parameter name to its value.
    """

    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]
