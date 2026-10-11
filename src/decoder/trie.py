from typing import Any

TrieNode = dict[str, Any]

def build_trie(words: list[str]) -> TrieNode:
    trie: TrieNode = {}
    for word in words:
        node: TrieNode = trie
        for c in word:
            node = node.setdefault(c, {})
        node[""] = True
    return trie

def get_valid_chars_from_trie(node: TrieNode) -> set[str]:
    return {'"' if k == "" else k for k in node}

