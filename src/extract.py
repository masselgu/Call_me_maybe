"""Generic prompt evidence and schema-compatible parameter candidates."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from src.models import FunctionDefinition, ParameterDefinition

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
QUOTE = re.compile(r'"([^"]*)"|\'([^\']*)\'')
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
REGEX_LIT = re.compile(r"\\(?:d\+|w\+|s\+|\S)|\[[^\]]+\]")
BOOL = re.compile(
    r"\b(true|false|yes|no|on|off|enable|enabled|disable|disabled)\b",
    re.I,
)
STOP_WORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
SYMBOL_WORDS = {
    "asterisk": "*",
    "asterisks": "*",
}


class Evidence(BaseModel):
    """Values extracted once from a user prompt."""

    quotes: list[str] = Field(default_factory=list)
    numbers: list[float] = Field(default_factory=list)
    words: list[str] = Field(default_factory=list)
    booleans: list[bool] = Field(default_factory=list)


def extract(prompt: str) -> Evidence:
    """Build prompt evidence once."""
    text_without_quotes = QUOTE.sub(" ", prompt)
    numbers = [float(m.group()) for m in NUM.finditer(text_without_quotes)]
    quotes = [
        m.group(1) if m.group(1) is not None else m.group(2) or ""
        for m in QUOTE.finditer(prompt)
    ]
    words = WORD.findall(prompt)
    lower_words = [word.lower() for word in words]
    for word in lower_words:
        if word in NUMBER_WORDS:
            numbers.append(float(NUMBER_WORDS[word]))

    booleans: list[bool] = []
    for item in BOOL.finditer(prompt):
        booleans.append(
            item.group().lower() in {"true", "yes", "on", "enable", "enabled"}
        )

    return Evidence(
        quotes=quotes,
        numbers=numbers,
        words=words,
        booleans=booleans,
    )


def schema_words(fn: FunctionDefinition) -> set[str]:
    """Return lowercase words that belong to a function schema."""
    parts = [fn.name, fn.description]
    for name, spec in fn.parameters.items():
        parts.extend([name, spec.description or ""])
    words: set[str] = set()
    for part in parts:
        words.update(
            piece.lower()
            for piece in re.split(r"[_\W]+", part)
            if piece
        )
    return words


def _regex_literals(spec: ParameterDefinition) -> list[str]:
    """Return regex literals mentioned in a parameter description."""
    if not spec.description:
        return []
    out: list[str] = []
    for chunk in spec.description.split(";"):
        out.extend(REGEX_LIT.findall(chunk))
    return out


def _uniq(raw: list[Any], used: set[str]) -> list[Any]:
    """Return non-empty unique values, preserving order."""
    seen: set[str] = set()
    out: list[Any] = []
    for value in raw:
        text = str(value)
        if text and text not in used and text not in seen:
            seen.add(text)
            out.append(value)
    return out


def clean_token(value: str | None) -> str:
    """Normalize a short value captured from a prompt."""
    if value is None:
        return ""
    return value.strip(" \t\r\n.,?!'\"")


def number_options(ev: Evidence, integer: bool) -> list[int | float]:
    """Return numeric candidates compatible with a number parameter."""
    out: list[int | float] = []
    for value in ev.numbers:
        if integer:
            if value == int(value):
                out.append(int(value))
        else:
            out.append(float(value))
    return _uniq(out, set())


def boolean_options(ev: Evidence) -> list[bool]:
    """Return boolean candidates, falling back to both valid values."""
    return _uniq(ev.booleans or [True, False], set())


def string_options(
    prompt: str,
    spec: ParameterDefinition,
    ev: Evidence,
    vocab: set[str],
) -> list[str]:
    """Return broad string candidates; the LLM chooses the semantic fit."""
    raw: list[Any] = []
    raw.extend(_regex_literals(spec))
    raw.extend(ev.quotes)
    for word in ev.words:
        cleaned = clean_token(word)
        lower = cleaned.lower()
        if lower in SYMBOL_WORDS:
            raw.append(SYMBOL_WORDS[lower])
        if cleaned and lower not in STOP_WORDS and lower not in vocab:
            raw.append(cleaned)

    if not raw:
        raw.append(prompt)
    return _uniq(raw, set())
