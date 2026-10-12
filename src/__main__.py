from .models import FunctionDefinition, PromptInput
from .pipeline import process_prompts
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model
from .parse import FileParsing
from .errors import errors
from typing import Any
import argparse
import json
import sys


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json")

    args = parser.parse_args()

    func_content: dict[str, Any] | list[Any] = (
        FileParsing.read_file(args.functions_definition)
    )
    func_defs: list[FunctionDefinition] = (
        FileParsing.parse_definitions(func_content)
    )
    input_content: dict[str, Any] | list[Any] = (
        FileParsing.read_file(args.input)
    )
    input_tests: list[PromptInput] = (
        FileParsing.parse_tests(input_content)
    )

    model = Small_LLM_Model()
    vocab_path: str = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as f:
        data: dict[str, int] = json.load(f)
    id_to_token: dict[int, str] = {
        v: k.replace("Ġ", " ") for k, v in data.items()
    }

    prompts_processed = (
        process_prompts(input_tests, model, func_defs, id_to_token)
    )
    FileParsing.write_output(prompts_processed, args.output)


def main() -> None:
    try:
        run()
    except errors.CallMeMaybeError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except (OSError, json.JSONDecodeError, ValidationError) as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
