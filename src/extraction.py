"""Candidate extraction for schema-constrained argument decoding."""

from __future__ import annotations

import json
import re
from typing import Any

from src.models import FunctionDefinition, ParameterDefinition


NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
QUOTED_RE = re.compile(r'"([^"]*)"|\'([^\']*)\'')
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def candidate_json_values(
    prompt: str,
    function: FunctionDefinition,
    parameter_name: str,
    parameter: ParameterDefinition,
) -> list[str]:
    """Return valid JSON literals for one parameter."""
    values: list[Any]
    if parameter.type in {"number", "integer"}:
        values = _number_values(prompt, parameter.type == "integer")
    elif parameter.type == "boolean":
        values = _boolean_values(prompt)
    else:
        values = _string_values(prompt, function, parameter_name)

    return _dedupe_json(values)


def _number_values(prompt: str, integer: bool) -> list[Any]:
    """Extract number candidates from the prompt."""
    values: list[Any] = []
    for match in NUMBER_RE.finditer(prompt):
        raw: str = match.group(0)
        try:
            if integer:
                values.append(int(float(raw)))
            else:
                values.append(float(raw))
        except ValueError:
            continue
    if not values:
        values.append(0 if integer else 0.0)
    return values


def _boolean_values(prompt: str) -> list[Any]:
    """Extract boolean candidates from common natural language forms."""
    lower: str = prompt.lower()
    values: list[Any] = []
    if any(word in lower for word in ("true", "yes", "enable", "on")):
        values.append(True)
    if any(word in lower for word in ("false", "no", "disable", "off")):
        values.append(False)
    if not values:
        values.extend([True, False])
    return values


def _string_values(
    prompt: str,
    function: FunctionDefinition,
    parameter_name: str,
) -> list[Any]:
    """Build string candidates from quoted spans and prompt hints."""
    quoted_values: list[str] = _quoted_strings(prompt)
    lowered_name: str = parameter_name.lower()
    values: list[str] = []
    if "source" in lowered_name or lowered_name in {"s", "text", "string"}:
        values.extend(sorted(quoted_values, key=len, reverse=True))
    else:
        values.extend(quoted_values)
    values.extend(_tail_phrases(prompt))
    values.extend(_regex_like_values(prompt))
    values.extend(_replacement_values(prompt))
    values.extend(_word_values(prompt))

    if "regex" in lowered_name or "pattern" in lowered_name:
        values = _prioritize_regex_values(prompt, values)
    if "replace" in lowered_name or "replacement" in lowered_name:
        values = _prioritize_replacement_values(prompt, values)

    if not values:
        values.append("")

    description_words: list[str] = WORD_RE.findall(function.description)
    values.extend(description_words[:3])
    return values


def _quoted_strings(prompt: str) -> list[str]:
    """Return strings enclosed in single or double quotes."""
    values: list[str] = []
    for match in QUOTED_RE.finditer(prompt):
        first: str | None = match.group(1)
        second: str | None = match.group(2)
        values.append(first if first is not None else second or "")
    return values


def _tail_phrases(prompt: str) -> list[str]:
    """Extract useful unquoted tails, such as names in greeting prompts."""
    values: list[str] = []
    patterns: list[str] = [
        r"\bgreet\s+(.+)$",
        r"\breverse\s+(?:the\s+string\s+)?(.+)$",
        r"\bwith\s+([A-Za-z0-9_*+-]+)$",
    ]
    for pattern in patterns:
        match: re.Match[str] | None = re.search(pattern, prompt, re.I)
        if match is not None:
            values.append(match.group(1).strip(" .?!'\""))
    return values


def _regex_like_values(prompt: str) -> list[str]:
    """Infer regex candidates from natural-language replacement requests."""
    lower: str = prompt.lower()
    values: list[str] = []
    if "number" in lower or "digit" in lower:
        values.append(r"\d+")
    if "vowel" in lower:
        values.append(r"[aeiouAEIOU]")
    if "letter" in lower:
        values.append(r"[A-Za-z]")
    return values


def _replacement_values(prompt: str) -> list[str]:
    """Infer replacement candidates from common wording."""
    values: list[str] = []
    with_match: re.Match[str] | None
    with_match = re.search(r"\bwith\s+([A-Za-z0-9_*+-]+)", prompt, re.I)
    if with_match is not None:
        raw: str = with_match.group(1).strip(" .?!'\"")
        values.append(_normalize_replacement(raw))
    if "asterisk" in prompt.lower():
        values.append("*")
    return values


def _normalize_replacement(value: str) -> str:
    """Normalize natural replacement names into literal strings."""
    if value.lower() in {"asterisk", "asterisks"}:
        return "*"
    return value


def _word_values(prompt: str) -> list[str]:
    """Return word candidates for simple unquoted string arguments."""
    ignored: set[str] = {
        "what",
        "is",
        "the",
        "sum",
        "of",
        "and",
        "greet",
        "reverse",
        "string",
        "replace",
        "all",
        "with",
        "in",
        "calculate",
        "square",
        "root",
        "substitute",
        "word",
    }
    return [
        word
        for word in WORD_RE.findall(prompt)
        if word.lower() not in ignored
    ]


def _prioritize_regex_values(prompt: str, values: list[str]) -> list[str]:
    """Move regex-looking candidates before generic strings."""
    regex_values: list[str] = _regex_like_values(prompt)
    quoted: list[str] = _quoted_strings(prompt)
    return regex_values + quoted + values


def _prioritize_replacement_values(
    prompt: str,
    values: list[str],
) -> list[str]:
    """Move likely replacement candidates before generic strings."""
    replacements: list[str] = _replacement_values(prompt)
    return replacements + values


def _dedupe_json(values: list[Any]) -> list[str]:
    """Serialize candidates as JSON literals and remove duplicates."""
    seen: set[str] = set()
    encoded: list[str] = []
    for value in values:
        text: str = json.dumps(value, ensure_ascii=False)
        if text not in seen:
            seen.add(text)
            encoded.append(text)
    return encoded
