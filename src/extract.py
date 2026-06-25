"""Generic prompt evidence and schema-compatible parameter candidates."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.models import FunctionDefinition

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
QUOTE = re.compile(r'"([^"]*)"|\'([^\']*)\'')
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
BOOL = re.compile(
    r"\b(true|false|yes|no|on|off|enable|enabled|disable|disabled)\b",
    re.I,
)
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10,
}


class Evidence(BaseModel):
    """Values extracted once from a user prompt."""

    quotes: list[str] = Field(default_factory=list)
    numbers: list[float] = Field(default_factory=list)
    words: list[str] = Field(default_factory=list)
    booleans: list[bool] = Field(default_factory=list)


def extract(prompt: str) -> Evidence:
    """Build prompt evidence once."""
    numbers = [float(m.group()) for m in NUM.finditer(QUOTE.sub(" ", prompt))]
    quotes = [
        m.group(1) if m.group(1) is not None else m.group(2) or ""
        for m in QUOTE.finditer(prompt)
    ]
    words = WORD.findall(prompt)
    for word in words:
        if word.lower() in NUMBER_WORDS:
            numbers.append(float(NUMBER_WORDS[word.lower()]))

    booleans = [
        item.group().lower() in {"true", "yes", "on", "enable", "enabled"}
        for item in BOOL.finditer(prompt)
    ]
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

    result: set[str] = set()
    for part in parts:
        pieces = re.split(r"[_\W]+", part)
        for piece in pieces:
            if piece:
                result.add(piece.lower())
    return result
