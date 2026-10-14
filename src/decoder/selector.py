from .trie import get_valid_chars_from_trie, TrieNode
from .bool_node import get_valid_bool_char
from .generic import JsonState, JsonValidation
from ..models.schema import ParameterType
from ..errors import errors
import math


def boost_token(
    final: list[float],
    logits: list[float],
    id_to_token: dict[int, str],
    char: str,
) -> None:
    """Give a character the best score of the tokens that start with it.

    The model often wants a token that glues the character to other
    ones, such as ``}}``. The score of ``char`` alone becomes the best
    score of all those tokens.

    Args:
        final: Masked logits, modified in place.
        logits: Original scores given by the model.
        id_to_token: Maps each token id to its text.
        char: The character to boost.
    """
    family = [i for i, t in id_to_token.items() if t.startswith(char)]
    alone = next(i for i in family if id_to_token[i] == char)
    final[alone] = max(logits[i] for i in family)


def mask_logits(
    logits: list[float],
    curr_state: JsonState,
    id_to_token: dict[int, str],
    trie_node: TrieNode | None = None,
    current_bool: str | None = None,
    param_type: ParameterType | None = None,
    remaining_param: bool = True
) -> list[float]:
    """Forbid every token that would break the JSON structure.

    The allowed tokens depend on the current state. The logit of every
    other token is set to minus infinity. Inside a string value any
    token made of valid string characters is allowed, and the closing
    quote gets the best score among all tokens that start with a quote.

    Args:
        logits: Scores given by the model for every token id.
        curr_state: Current state of the JSON generation.
        id_to_token: Maps each token id to its text.
        trie_node: Current trie node, required in ``OBJECT_KEY``.
        current_bool: Letters of the boolean written so far.
        param_type: Type of the value that is being written.
        remaining_param: True if there are keys left to write, so ``,``
            is allowed and ``}`` is not. False means the opposite.

    Returns:
        A new list of logits where forbidden tokens are ``-inf``.

    Raises:
        TrieError: If the state is ``OBJECT_KEY`` and there is no
            trie node.
    """
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
    if curr_state in (JsonState.AFTER_VALUE,
                      JsonState.OBJECT_VALUE,
                      JsonState.DECIMAL_VALUE,
                      JsonState.IN_BOOL_VALUE):
        if remaining_param:
            permitted = permitted - {"}"}
        else:
            permitted = permitted - {","}
    final: list[float] = logits.copy()
    for i, _ in enumerate(logits):
        text = id_to_token.get(i)
        if curr_state == JsonState.IN_STRING_VALUE:
            valid = (text is not None
                     and JsonValidation.is_valid_string_token(text))
        elif (curr_state == JsonState.BEFORE_VALUE
              and param_type in (ParameterType.NUMBER,
                                 ParameterType.INTEGER)):
            valid = (text is not None
                     and text.lstrip(" ") in permitted)
        else:
            valid = text in permitted
        if not valid:
            final[i] = -math.inf
    if curr_state == JsonState.IN_STRING_VALUE:
        boost_token(final, logits, id_to_token, '"')
    elif curr_state in (JsonState.OBJECT_VALUE, JsonState.DECIMAL_VALUE):
        for char in permitted & {",", "}"}:
            boost_token(final, logits, id_to_token, char)
    return final


def get_token(logits: list[float]) -> int:
    """Pick the token with the highest score.

    Args:
        logits: Scores for every token id, already masked.

    Returns:
        The id of the token with the highest logit.
    """
    return max(enumerate(logits), key=lambda pair: pair[1])[0]
