"""Function filtering and schema-compatible option helpers."""

from __future__ import annotations

from typing import Any
from src.candidates import boolean_options, number_options, string_options
from src.extract import Evidence, schema_words
from src.models import FunctionDefinition, ParameterDefinition


def can_fill(prompt: str, fn: FunctionDefinition, ev: Evidence) -> bool:
    """Check that each parameter has at least one typed candidate."""
    numbers, _, booleans = type_counts(fn)
    if numbers and len(number_options(ev, integer=False)) < numbers:
        return False
    if booleans and not ev.booleans:
        return False
    vocab = schema_words(fn)
    return all(
        options(prompt, "", spec, ev, vocab)
        for spec in fn.parameters.values()
    )


def score_function(prompt: str, ev: Evidence, fn: FunctionDefinition) -> int:
    """Rank generic schema/prompt compatibility before LLM selection."""
    numbers, strings, booleans = type_counts(fn)
    prompt_words = {word.lower() for word in ev.words}
    score = len(prompt_words & schema_words(fn)) * 10
    score += 20 if numbers and len(ev.numbers) == numbers else 0
    score += 8 if strings and ev.quotes else 0
    score += 20 if booleans and ev.booleans else 0
    score += 5 if "?" in prompt and fn.returns.type == "boolean" else 0
    return score


def type_counts(fn: FunctionDefinition) -> tuple[int, int, int]:
    """Count numeric, string, and boolean parameters."""
    numbers = strings = booleans = 0
    for spec in fn.parameters.values():
        numbers += int(spec.type in {"number", "integer"})
        strings += int(spec.type == "string")
        booleans += int(spec.type == "boolean")
    return numbers, strings, booleans


def options(
    prompt: str,
    name: str,
    spec: ParameterDefinition,
    ev: Evidence,
    vocab: set[str],
) -> list[Any]:
    """Build schema-compatible values; semantics are chosen by the LLM."""
    if spec.type == "integer":
        return list(number_options(ev, integer=True))
    if spec.type == "number":
        return list(number_options(ev, integer=False))
    if spec.type == "boolean":
        return list(boolean_options(ev))
    return list(string_options(prompt, name, spec, ev, vocab))
