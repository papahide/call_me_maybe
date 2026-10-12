from llm_sdk import Small_LLM_Model
from .trie import build_trie, walk_trie, TrieNode
from ..models import FunctionDefinition
from .selector import get_token
from ..errors import errors
import math


def mask_logits_for_trie(
    logits: list[float],
    id_to_token: dict[int, str],
    trie_node: TrieNode,
) -> list[float]:
    final: list[float] = logits.copy()
    for i, _ in enumerate(logits):
        text = id_to_token.get(i)
        if text is None or walk_trie(trie_node, text) is None:
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
        text = id_to_token[token]
        next_node = walk_trie(curr_node, text)
        if next_node is None:
            raise errors.TrieError("Token outside the function trie")
        curr_node = next_node
        generated_ids.append(token)
        result = result + text
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
