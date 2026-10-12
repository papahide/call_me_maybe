from ..errors import errors


def get_valid_bool_char(current_bool: str | None) -> set[str]:
    if not current_bool:
        return {"t", "f"}
    elif current_bool[0] == "t":
        word = "true"
    elif current_bool[0] == "f":
        word = "false"
    else:
        raise errors.BoolError("Bool value cannot be NONE")
    if current_bool == word:
        return {",", "}"}
    else:
        return {word[len(current_bool)]}
