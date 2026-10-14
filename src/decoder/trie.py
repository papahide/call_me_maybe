from typing import Any

TrieNode = dict[str, Any]


def build_trie(words: list[str]) -> TrieNode:
    """Build a trie with the given words.

    Each node maps a letter to its child node. The key ``""`` marks
    that a word ends in that node.

    Args:
        words: Words to store in the trie.

    Returns:
        The root node of the trie.
    """
    trie: TrieNode = {}
    for word in words:
        node: TrieNode = trie
        for c in word:
            node = node.setdefault(c, {})
        node[""] = True
    return trie


def get_valid_chars_from_trie(node: TrieNode) -> set[str]:
    """Return the characters that can follow a trie node.

    The end-of-word mark is returned as a closing quote, because that
    is the character that ends a key in the JSON.

    Args:
        node: Node of the trie.

    Returns:
        The set of valid next characters.
    """
    return {'"' if k == "" else k for k in node}


def walk_trie(node: TrieNode, text: str) -> TrieNode | None:
    """Follow a text through the trie.

    Args:
        node: Node where the walk starts.
        text: Characters to follow, one by one.

    Returns:
        The node reached after the last character, or None if the text
        is empty or leaves the trie.
    """
    if not text:
        return None
    for c in text:
        if c not in node:
            return None
        node = node[c]
    return node
