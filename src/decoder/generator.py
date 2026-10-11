from .selector import mask_logits, get_token
from ..models import FunctionDefinition
from .trie import build_trie, TrieNode
from .generic import JsonState, Transitions
from llm_sdk import Small_LLM_Model

def generate_json(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    id_to_token: dict[int, str],
    func_def: FunctionDefinition
) -> str:
    curr_state = JsonState.START
    generated_ids = prompt_ids.copy()
    result: str = ""
    current_key: str = ""
    current_bool: str = ""
    param_trie: TrieNode = build_trie(list(func_def.parameters.keys()))
    curr_trie_node: TrieNode = param_trie
    pending_keys: list[str] = list(func_def.parameters)

    while curr_state != JsonState.CLOSE:
        logits = model.get_logits_from_input_ids(generated_ids)
        if curr_state == JsonState.BEFORE_VALUE:
            param_type = func_def.parameters[current_key].type
            logits_masked = mask_logits(logits, curr_state, id_to_token, trie_node=curr_trie_node, current_bool=None, param_type=param_type, remaining_param=bool(pending_keys))
        elif curr_state == JsonState.IN_BOOL_VALUE:
            logits_masked = mask_logits(logits, curr_state, id_to_token, trie_node=None, current_bool=current_bool, param_type=None, remaining_param=bool(pending_keys))
        else:
            logits_masked: list[float] = mask_logits(logits, curr_state, id_to_token, curr_trie_node, remaining_param=bool(pending_keys))
        token: int = get_token(logits_masked)

        if curr_state == JsonState.IN_STRING_VALUE:
            wanted = get_token(logits)
            print("quería:", repr(id_to_token.get(wanted)),
            "| eligió:", repr(id_to_token[token])
        )

        print(curr_state, repr(id_to_token[token]))

        generated_ids.append(token)
        if curr_state == JsonState.OBJECT_KEY and id_to_token[token] != "\"":
            curr_trie_node = curr_trie_node[id_to_token[token]]
            current_key = current_key + id_to_token[token]
        elif curr_state == JsonState.IN_BOOL_VALUE:
            current_bool = current_bool + id_to_token[token]
        if id_to_token[token][0] == "Ġ":
            result = result + " " + id_to_token[token][1:]
        else:
            result = result + id_to_token[token]
        curr_state = Transitions.next_state(curr_state, id_to_token[token])
        if curr_state == JsonState.AFTER_KEY:
            pending_keys.remove(current_key)
        if curr_state == JsonState.START_OBJK:
            curr_trie_node = build_trie(pending_keys)
            current_key = ""
            current_bool = ""
    return result