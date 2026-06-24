"""Schema-compatible parameter candidates for constrained decoding."""

from __future__ import annotations

import re
from typing import TypeVar
from src.extract import Evidence
from src.models import ParameterDefinition

REGEX_LIT = re.compile(r"\\(?:d\+|w\+|s\+|\S)|\[[^\]]+\]")
WITH_VALUE = re.compile(
    r"\bwith\s+'([^']+)'|\bwith\s+\"([^\"]+)\"|\bwith\s+(\S+)",
    re.I,
)
STOP_WORDS = set(
    "a all an and are as by for from if in is it me of on or the "
    "this to with".split()
)
SYMBOL_WORDS = {"asterisk": "*", "asterisks": "*"}
REGEX_HINTS = {
    "digit": r"\d+",
    "digits": r"\d+",
    "number": r"\d+",
    "numbers": r"\d+",
    "vowel": "[aeiouAEIOU]",
    "vowels": "[aeiouAEIOU]",
}
ValueT = TypeVar("ValueT")


def clean_token(value: str | None) -> str:
    """Normalize a short value captured from a prompt."""
    return "" if value is None else value.strip(" \t\r\n.,?!'\"")


def unique(values: list[ValueT]) -> list[ValueT]:
    """Return non-empty unique values, preserving order."""
    seen: set[str] = set()
    out: list[ValueT] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(value)
    return out


def number_options(ev: Evidence, integer: bool) -> list[int | float]:
    """Return numeric candidates compatible with a number parameter."""
    values: list[int | float] = []
    for number in ev.numbers:
        if integer and number == int(number):
            values.append(int(number))
        elif not integer:
            values.append(float(number))
    return unique(values)


def boolean_options(ev: Evidence) -> list[bool]:
    """Return boolean candidates, falling back to both valid values."""
    return unique(ev.booleans or [True, False])


def string_options(
    prompt: str,
    name: str,
    spec: ParameterDefinition,
    ev: Evidence,
    vocab: set[str],
) -> list[str]:
    """Return broad string candidates; the LLM chooses the semantic fit."""
    role = _role(name, spec)
    literals = REGEX_LIT.findall(spec.description or "")
    values: list[str] = []

    if role == "source" and ev.quotes:
        values.extend(sorted(ev.quotes, key=len, reverse=True))
    elif role == "replacement":
        values.extend(_with_values(prompt))
        values.extend(ev.quotes)
    elif role == "regex":
        hints = [
            REGEX_HINTS[word.lower()]
            for word in ev.words
            if REGEX_HINTS.get(word.lower()) in literals
        ]
        values.extend(hints or ev.quotes)

    values.extend(literals)
    for word in ev.words:
        token = clean_token(word)
        lower = token.lower()
        if lower in SYMBOL_WORDS:
            values.append(SYMBOL_WORDS[lower])
        if token and lower not in STOP_WORDS and lower not in vocab:
            values.append(token)
    return unique(values or [prompt])


def _role(name: str, spec: ParameterDefinition) -> str:
    """Infer a broad parameter role from schema names/descriptions."""
    text = f"{name} {spec.description or ''}".lower()
    if "regex" in text or REGEX_LIT.search(spec.description or ""):
        return "regex"
    if "replacement" in text:
        return "replacement"
    if "source" in text or "input text" in text or "search" in text:
        return "source"
    if "target" in text or "substring" in text or "to find" in text:
        return "target"
    return "other"


def _with_values(prompt: str) -> list[str]:
    """Return values introduced by natural-language 'with ...' phrasing."""
    values: list[str] = []
    for item in WITH_VALUE.finditer(prompt):
        value = clean_token(item.group(1) or item.group(2) or item.group(3))
        values.append(SYMBOL_WORDS.get(value.lower(), value))
    return [value for value in values if value]
