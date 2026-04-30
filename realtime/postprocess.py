import re
from typing import Iterable


def _normalize_spaces(text: str) -> str:
    text = text.replace("\u2581", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _capitalize_first_alpha(text: str) -> str:
    for idx, ch in enumerate(text):
        if ch.isalpha():
            return text[:idx] + ch.upper() + text[idx + 1 :]
    return text


def _needs_terminal_punctuation(text: str) -> bool:
    return bool(text) and text[-1] not in ".!?"


def apply_final_punctuation(raw_text: str, context: Iterable[str] | None = None) -> str:
    """Deterministic punctuation/casing pass for finalized utterances.

    `context` is accepted for forward compatibility but only lightly used in v1.
    """
    text = _normalize_spaces(raw_text)
    if not text:
        return text

    text = _capitalize_first_alpha(text)

    # If previous context ends with sentence punctuation, keep normal casing.
    # If not, still capitalize current utterance to prefer readable output.
    _ = context  # reserved for configurable future heuristics

    if _needs_terminal_punctuation(text):
        text += "."

    return text
