"""Conservative normalized phrase matching for deterministic verifiers."""

from __future__ import annotations

import re
import unicodedata


_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_NEGATORS = {
    "not",
    "no",
    "without",
    "never",
    "khong",
    "không",
    "chẳng",
    "chả",
}


def normalize_text(text: str) -> str:
    return " ".join(tokens(text))


def tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_PATTERN.findall(normalized)


def contains_phrase(text: str, phrase: str) -> bool:
    return bool(_phrase_starts(tokens(text), tokens(phrase)))


def contains_affirmative_phrase(text: str, phrase: str) -> bool:
    text_tokens = tokens(text)
    phrase_tokens = tokens(phrase)
    for start in _phrase_starts(text_tokens, phrase_tokens):
        preceding = text_tokens[max(0, start - 3) : start]
        if not any(token in _NEGATORS for token in preceding):
            return True
    return False


def _phrase_starts(text_tokens: list[str], phrase_tokens: list[str]) -> list[int]:
    if not phrase_tokens or len(phrase_tokens) > len(text_tokens):
        return []
    width = len(phrase_tokens)
    return [
        index
        for index in range(len(text_tokens) - width + 1)
        if text_tokens[index : index + width] == phrase_tokens
    ]
