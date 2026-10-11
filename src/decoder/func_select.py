from llm_sdk import Small_LLM_Model
from .trie import build_trie, get_valid_chars_from_trie, TrieNode
from ..models import FunctionDefinition
from .selector import get_token
from ..errors import errors
import math

def mask_logits_for_trie(
    logits: list[float],
    id_to_token: dict[int, str],
    trie_node: TrieNode,
) -> list[float]:
    permitted: set[str] = get_valid_chars_from_trie(trie_node)
    final: list[float] = logits.copy()
    for i, _ in enumerate(logits):
        if id_to_token.get(i) not in permitted:
            final[i] = -math.inf
    return final


def select_function(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    id_to_token: dict[int, str],
    func_names: list[str]
) -> str:
    result: str = ""
    generated_ids = prompt_ids.copy()
    func_trie = build_trie(func_names)
    curr_node = func_trie

    while "" not in curr_node:
        logits = model.get_logits_from_input_ids(generated_ids)
        logits_masked = mask_logits_for_trie(logits, id_to_token, curr_node)
        token: int = get_token(logits_masked)
        curr_node = curr_node[id_to_token[token]]
        if id_to_token[token][0] == "Ġ":
            raise errors.TrieError("There cannot me a space in func name")
        generated_ids.append(token)
        result = result + id_to_token[token]
    return result

def find_function_by_name(
    func_defs: list[FunctionDefinition],
    selected_name: str,
) -> FunctionDefinition:
    selected = next(
        (func for func in func_defs if func.name == selected_name),
        None
    )
    if selected is None:
        raise errors.FuncSelectError("Function cannot be None")
    return selected