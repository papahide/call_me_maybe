from ..decoder import select_function, find_function_by_name, generate_json
from ..models import PromptInput, FunctionDefinition, ModelOutput
from ..models.schema import ParameterType
from llm_sdk import Small_LLM_Model
from typing import cast
import json


def process_prompt(
    model: Small_LLM_Model,
    prompt: PromptInput,
    func_defs: list[FunctionDefinition],
    id_to_token: dict[int, str],
) -> ModelOutput:
    names: list[str] = [func.name for func in func_defs]
    tools = "\n".join(
        json.dumps(func.model_dump(mode="json")) for func in func_defs
    )
    header = (
        "<|im_start|>system\n"
        "You are a function-calling assistant. Available functions:\n"
        f"<tools>\n{tools}\n</tools>\n"
        "For each call, return a json object within "
        "<tool_call></tool_call> tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call><|im_end|>\n"
        f"<|im_start|>user\n{prompt.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        '<tool_call>\n{"name": "'
    )
    select_ids: list[int] = cast(
        list[int], model.encode(header).tolist()[0]
    )
    name = select_function(model, select_ids, id_to_token, names)
    function = find_function_by_name(func_defs, name)
    args_text = header + f'{function.name}", "arguments": '
    args_ids: list[int] = cast(
        list[int], model.encode(args_text).tolist()[0]
    )
    json_func = generate_json(model, args_ids, id_to_token, function)
    json_dict = json.loads(json_func)
    for key, value in json_dict.items():
        param_type = function.parameters[key].type
        if param_type == ParameterType.NUMBER and isinstance(value, int):
            json_dict[key] = float(value)
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
    processed: list[ModelOutput] = [
        process_prompt(model, prompt, func_defs,
                       id_to_token) for prompt in prompts]
    return processed
