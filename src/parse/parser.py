from ..models import FunctionDefinition, PromptInput, ModelOutput
from pydantic import ValidationError
from ..errors import FileReadError, FileWriteError
from pathlib import Path
from typing import Any
import json


class FileParsing():
    @staticmethod
    def read_file(file_name: str) -> dict[str, Any] | list[Any]:
        if not file_name:
            raise FileReadError("No file provided")
        file = Path(file_name)
        if file.suffix != ".json":
            raise FileReadError("File not .json")
        try:
            with open(file_name, "r") as file_to_read:
                content: dict[str, Any] | list[Any] = json.load(file_to_read)
                if not content:
                    raise FileReadError("Provided empty file")
                return content
        except json.JSONDecodeError as err:
            raise FileReadError(f"Incorrect Json format: {err}")
        except FileNotFoundError as err:
            raise FileReadError(f"File not found: {err}")
        except PermissionError as err:
            raise FileReadError(f"File insufficient "
                                  f"permissions: {err}")
        except IsADirectoryError as err:
            raise FileReadError(f"File not found, "
                                  f"'{file_name}' is a directory : {err}")

    @staticmethod
    def parse_definitions(data: dict[str, Any] | list[Any]) -> list[FunctionDefinition]:
        try:
            func: list[FunctionDefinition] = []
            for content in data:
                func.append(FunctionDefinition.model_validate(content))
            return func
        except ValidationError as err:
            raise FileReadError(f"Invalid function definition: {err}")

    @staticmethod
    def parse_tests(data: dict[str, Any] | list[Any]) -> list[PromptInput]:
        try:
            test: list[PromptInput] = []
            for content in data:
                test.append(PromptInput.model_validate(content))
            return test
        except ValidationError as err:
            raise FileReadError(f"Invalid prompt: {err}")

    @staticmethod
    def write_output(results: list[ModelOutput], output_path: str) -> None:
        final_prompt: list[dict] = [i.model_dump() for i in results]
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as file:
                json.dump(final_prompt, file, indent=2)
        except FileNotFoundError as err:
            raise FileWriteError(f"File not found: {err}")
        except PermissionError as err:
            raise FileWriteError(f"File insufficient "
                                  f"permissions: {err}")
        except IsADirectoryError as err:
            raise FileWriteError(f"File not found, "
                                  f"'{output_path}' is a directory : {err}")
