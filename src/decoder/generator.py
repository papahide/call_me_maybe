from .selector import mask_logits, get_token
from ..models import FunctionDefinition
from .trie import build_trie, TrieNode
from .generic import JsonState, Transitions
from llm_sdk import Small_LLM_Model
from ..errors import errors


def generate_json(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    id_to_token: dict[int, str],
    func_def: FunctionDefinition,
    max_tokens: int = 500,
) -> str:
    """Generate the arguments of a function as a valid JSON object.

    On every step it asks the model for the logits, forbids what the
    schema does not allow, picks the best token and moves to the next
    state. Keys already written leave the trie, so none is repeated
    and the object closes when all the parameters are done.

    A function without parameters returns ``{}`` without asking the
    model anything.

    Args:
        model: Language model used to get the logits.
        prompt_ids: Token ids of the prompt that precedes the JSON.
        id_to_token: Maps each token id to its text.
        func_def: Definition of the function, with its parameters.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        The generated JSON object as text.

    Raises:
        GenerationError: If the JSON is not complete after max_tokens.
    """
    if not func_def.parameters:
        return "{}"
    curr_state = JsonState.START
    generated_ids = prompt_ids.copy()
    result: str = ""
    current_key: str = ""
    current_bool: str = ""
    param_trie: TrieNode = build_trie(list(func_def.parameters.keys()))
    curr_trie_node: TrieNode = param_trie
    pending_keys: list[str] = list(func_def.parameters)

    steps = 0
    while curr_state != JsonState.CLOSE:
        steps += 1
        if steps > max_tokens:
            raise errors.GenerationError(
                f"JSON generation exceeded {max_tokens} tokens"
            )
        logits = model.get_logits_from_input_ids(generated_ids)
        if curr_state == JsonState.BEFORE_VALUE:
            param_type = func_def.parameters[current_key].type
            logits_masked = mask_logits(logits, curr_state,
                                        id_to_token,
                                        trie_node=curr_trie_node,
                                        current_bool=None,
                                        param_type=param_type,
                                        remaining_param=bool(pending_keys))
        elif curr_state == JsonState.IN_BOOL_VALUE:
            logits_masked = mask_logits(logits, curr_state,
                                        id_to_token, trie_node=None,
                                        current_bool=current_bool,
                                        param_type=None,
                                        remaining_param=bool(pending_keys))
        else:
            logits_masked = (
                mask_logits(logits, curr_state, id_to_token,
                            curr_trie_node,
                            remaining_param=bool(pending_keys))
            )
        token: int = get_token(logits_masked)
        generated_ids.append(token)
        if curr_state == JsonState.OBJECT_KEY and id_to_token[token] != "\"":
            curr_trie_node = curr_trie_node[id_to_token[token]]
            current_key = current_key + id_to_token[token]
        elif curr_state == JsonState.IN_BOOL_VALUE or (
            curr_state == JsonState.BEFORE_VALUE
            and id_to_token[token] in ("t", "f")
        ):
            current_bool = current_bool + id_to_token[token]
        result = result + id_to_token[token]
        curr_state = Transitions.next_state(curr_state, id_to_token[token])
        if curr_state == JsonState.AFTER_KEY:
            pending_keys.remove(current_key)
        if curr_state == JsonState.START_OBJK:
            curr_trie_node = build_trie(pending_keys)
            current_key = ""
            current_bool = ""
    return result
