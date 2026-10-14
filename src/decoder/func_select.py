from .trie import build_trie, walk_trie, TrieNode
from .selector import get_token, boost_token
from ..models import FunctionDefinition
from llm_sdk import Small_LLM_Model
from ..errors import errors
import math


def mask_logits_for_trie(
    logits: list[float],
    id_to_token: dict[int, str],
    trie_node: TrieNode,
) -> list[float]:
    """Forbid every token that does not continue a word of the trie.

    A token is allowed when all its characters can be followed from
    the current node, so tokens with several letters are accepted.

    Args:
        logits: Scores given by the model for every token id.
        id_to_token: Maps each token id to its text.
        trie_node: Current node of the trie.

    Returns:
        A new list of logits where forbidden tokens are ``-inf``.
    """
    final: list[float] = logits.copy()
    for i, _ in enumerate(logits):
        text = id_to_token.get(i)
        ends_here = text == '"' and "" in trie_node
        if text is None or (
            not ends_here and walk_trie(trie_node, text) is None
        ):
            final[i] = -math.inf
    if "" in trie_node:
        boost_token(final, logits, id_to_token, '"')
    return final


def select_function(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    id_to_token: dict[int, str],
    func_names: list[str]
) -> str:
    """Let the model choose a function name, token by token.

    The tokens are restricted with a trie, so the model can only write
    the name of one of the available functions.

    Args:
        model: Language model used to get the logits.
        prompt_ids: Token ids of the prompt that precedes the name.
        id_to_token: Maps each token id to its text.
        func_names: Names of the available functions.

    Returns:
        The name of the chosen function.

    Raises:
        TrieError: If the model picks a token outside the trie.
    """
    result: str = ""
    generated_ids = prompt_ids.copy()
    func_trie = build_trie(func_names)
    curr_node = func_trie

    while True:
        logits = model.get_logits_from_input_ids(generated_ids)
        logits_masked = mask_logits_for_trie(logits, id_to_token, curr_node)
        token: int = get_token(logits_masked)
        text = id_to_token[token]
        if text == '"':
            return result
        next_node = walk_trie(curr_node, text)
        if next_node is None:
            raise errors.TrieError("Token outside the function trie")
        curr_node = next_node
        generated_ids.append(token)
        result = result + text


def find_function_by_name(
    func_defs: list[FunctionDefinition],
    selected_name: str,
) -> FunctionDefinition:
    """Look for a function definition by its name.

    Args:
        func_defs: All the available function definitions.
        selected_name: Name of the function to look for.

    Returns:
        The definition whose name matches.

    Raises:
        FuncSelectError: If no function has that name.
    """
    selected = next(
        (func for func in func_defs if func.name == selected_name),
        None
    )
    if selected is None:
        raise errors.FuncSelectError("Function cannot be None")
    return selected
