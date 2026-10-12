from pydantic import BaseModel
from enum import Enum


class ParameterType(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


class ParameterSchema(BaseModel):
    type: ParameterType


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema


class PromptInput(BaseModel):
    prompt: str


class ModelOutput(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, str | int | float | bool]
