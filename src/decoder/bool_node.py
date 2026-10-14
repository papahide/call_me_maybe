from ..errors import errors


def get_valid_bool_char(current_bool: str | None) -> set[str]:
    """Return the characters allowed next while writing a boolean.

    It compares what has been written so far with ``true`` or
    ``false``. When the word is complete, only the characters that
    end the value are allowed.

    Args:
        current_bool: Letters of the boolean written so far, empty or
            None if nothing has been written yet.

    Returns:
        The set with the next valid characters.

    Raises:
        BoolError: If the text does not start with ``t`` or ``f``.
    """
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
