from ..models import FunctionDefinition, PromptInput, ModelOutput
from pydantic import ValidationError, BaseModel
from ..errors import FileReadError, FileWriteError
from pathlib import Path
from typing import Any
import json


class FileParsing(BaseModel):
    """Read the input files and write the output file.

    It only groups static methods, so it is never instantiated.
    """
    @staticmethod
    def read_file(file_name: str) -> dict[str, Any] | list[Any]:
        """Read a JSON file and return its content.

        Args:
            file_name: Path of the JSON file.

        Returns:
            The content of the file, already parsed.

        Raises:
            FileReadError: If the path is empty, the file is not a
                ``.json`` file, it cannot be opened, it is empty or it
                does not contain valid JSON.
        """
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
        except UnicodeDecodeError as err:
            raise FileReadError(f"File is not valid text: {err}")

    @staticmethod
    def parse_definitions(
            data: dict[str, Any] | list[Any]) -> list[FunctionDefinition]:
        """Validate the function definitions read from a file.

        Args:
            data: Content of the functions definition file.

        Returns:
            One validated definition for each function.

        Raises:
            FileReadError: If an item is not a valid function definition.
        """
        try:
            func: list[FunctionDefinition] = []
            for content in data:
                func.append(FunctionDefinition.model_validate(content))
            return func
        except ValidationError as err:
            raise FileReadError(f"Invalid function definition: {err}")

    @staticmethod
    def parse_tests(data: dict[str, Any] | list[Any]) -> list[PromptInput]:
        """Validate the prompts read from a file.

        Args:
            data: Content of the prompts file.

        Returns:
            One validated prompt for each item.

        Raises:
            FileReadError: If an item is not a valid prompt.
        """
        try:
            test: list[PromptInput] = []
            for content in data:
                test.append(PromptInput.model_validate(content))
            return test
        except ValidationError as err:
            raise FileReadError(f"Invalid prompt: {err}")

    @staticmethod
    def write_output(results: list[ModelOutput], output_path: str) -> None:
        """Write the results to a JSON file.

        The folder of the file is created if it does not exist.

        Args:
            results: Function calls to save.
            output_path: Path of the file to write.

        Raises:
            FileWriteError: If the file cannot be written.
        """
        final_prompt: list[dict[str, Any]] = [
            i.model_dump() for i in results
        ]
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
