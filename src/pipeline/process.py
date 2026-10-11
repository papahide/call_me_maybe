from llm_sdk import Small_LLM_Model
from ..decoder import select_function, find_function_by_name, generate_json
from ..models import PromptInput, FunctionDefinition, ModelOutput
from typing import cast
import json


def process_prompt(
    model: Small_LLM_Model,
    prompt: PromptInput,
    func_defs: list[FunctionDefinition],
    id_to_token: dict[int, str],
) -> ModelOutput:
    names: list[str] = [func.name for func in func_defs]
    functions_context = "\n".join(
        f"- {func.name}: {func.description}"
        for func in func_defs
    )
    select_text = (
        "You are a function-calling assistant. "
        "Choose the best function for the request.\n\n"
        "Available functions:\n"
        f"{functions_context}\n\n"
        f"Request: {prompt.prompt}\n"
        "Function: "
    )
    print(select_text, "\n")

    select_ids: list[int] = cast(
        list[int], model.encode(select_text).tolist()[0]
    )
    name = select_function(model, select_ids, id_to_token, names)
    function = find_function_by_name(func_defs, name)
    args_text = (
        "You are a function-calling assistant.\n"
        f"Function: {function.name} - {function.description}\n"
        f"Request: {prompt.prompt}\n"
        "Arguments: "
    )
    args_ids: list[int] = cast(
        list[int], model.encode(args_text).tolist()[0]
    )
    json_func = generate_json(model, args_ids, id_to_token, function)
    json_dict = json.loads(json_func)
    final = ModelOutput(
        prompt=prompt.prompt,
        name=function.name,
        parameters=json_dict,
    )
    return final


def process_prompts(prompts: list[PromptInput],
                    model: Small_LLM_Model,
                    func_defs: list[FunctionDefinition],
                    id_to_token: dict[int, str],) -> list[ModelOutput]:
    processed: list[ModelOutput] = [process_prompt(model, 
                                                   prompt, 
                                                   func_defs, 
                                                   id_to_token) for prompt in prompts]
    return processed