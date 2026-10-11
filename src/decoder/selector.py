from .trie import get_valid_chars_from_trie, TrieNode
from .bool_node import get_valid_bool_char
from .generic import JsonState, JsonValidation
from ..models.schema import ParameterType
from ..errors import errors
import math


def mask_logits(
    logits: list[float],
    curr_state: JsonState,
    id_to_token: dict[int, str],
    trie_node: TrieNode | None = None,
    current_bool: str | None = None,
    param_type: ParameterType | None = None
) -> list[float]:
    permitted: set[str] = set()
    if curr_state == JsonState.OBJECT_KEY:
        if trie_node is None:
            raise errors.TrieError("Key cannot be none")
        permitted = get_valid_chars_from_trie(trie_node)
    elif curr_state == JsonState.IN_BOOL_VALUE:
        permitted = get_valid_bool_char(current_bool)
    elif curr_state == JsonState.BEFORE_VALUE and param_type is not None:
        permitted = JsonValidation.VALUE_START_CHARS[param_type]
    else:
        permitted = JsonValidation.VALID_CHARS[curr_state]
    final: list[float] = logits.copy()
    for i, _ in enumerate(logits):
        if id_to_token.get(i) not in permitted:
            final[i] = -math.inf
    return final

def get_token(logits: list[float]) -> int:
    return max(enumerate(logits), key=lambda pair: pair[1])[0]
