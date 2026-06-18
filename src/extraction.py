"""Candidate extraction for schema-constrained argument decoding."""

from __future__ import annotations

import json
import re
from typing import Any

from src.models import FunctionDefinition, ParameterDefinition


NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
QUOTED_RE = re.compile(r'"([^"]*)"|\'([^\']*)\'')
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
GREET_TAIL_RE = re.compile(r"\bgreet\s+(.+)$", re.I)
REVERSE_TAIL_RE = re.compile(
    r"\breverse\s+(?:the\s+string\s+)?(.+)$",
    re.I,
)
SUBSTITUTE_WORD_RE = re.compile(
    r"\b(?:word|words)\s+'([^']+)'",
    re.I,
)
SUBSTITUTE_WITH_QUOTED_RE = re.compile(
    r"\bwith\s+'([^']+)'",
    re.I,
)
IN_QUOTED_RE = re.compile(r"\bin\s+'([^']+)'", re.I)
WITH_TOKEN_RE = re.compile(r"\bwith\s+([A-Za-z0-9_*]+)\b", re.I)
IN_DOUBLE_QUOTED_RE = re.compile(r'\bin\s+"([^"]+)"', re.I)

COMMON_WORDS: set[str] = {
    "a",
    "an",
    "all",
    "and",
    "calculate",
    "for",
    "fun",
    "greet",
    "in",
    "is",
    "of",
    "old",
    "on",
    "or",
    "replace",
    "return",
    "reverse",
    "root",
    "square",
    "string",
    "substitute",
    "sum",
    "the",
    "what",
    "with",
    "word",
    "years",
}


def candidate_json_values(
    prompt: str,
    function: FunctionDefinition,
    parameter_name: str,
    parameter: ParameterDefinition,
    already_selected: dict[str, Any] | None = None,
) -> list[str]:
    """Return valid JSON literals for one parameter."""
    selected: dict[str, Any] = already_selected or {}
    values: list[Any]
    if parameter.type in {"number", "integer"}:
        values = _number_values(prompt, parameter.type == "integer", selected)
    elif parameter.type == "boolean":
        values = _boolean_values(prompt)
    else:
        values = _string_values(prompt, parameter_name, selected)

    return _dedupe_json(values)


def _number_values(
    prompt: str,
    integer: bool,
    already_selected: dict[str, Any],
) -> list[Any]:
    """Extract number candidates and avoid reusing selected numbers."""
    values: list[Any] = []
    for match in NUMBER_RE.finditer(prompt):
        raw: str = match.group(0)
        try:
            value: Any = int(float(raw)) if integer else float(raw)
        except ValueError:
            continue
        values.append(value)

    used_numbers: set[Any] = {
        value
        for value in already_selected.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    unused: list[Any] = [
        value for value in values if value not in used_numbers
    ]
    if unused:
        return unused
    if values:
        return values
    return [0 if integer else 0.0]


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
    parameter_name: str,
    already_selected: dict[str, Any],
) -> list[Any]:
    """Build string candidates from prompt structure and parameter role."""
    lowered_name: str = parameter_name.lower()
    quoted_values: list[str] = _quoted_strings(prompt)
    used_strings: set[str] = {
        str(value)
        for value in already_selected.values()
        if isinstance(value, str)
    }

    if lowered_name in {"name"}:
        values = _name_values(prompt)
    elif "source" in lowered_name or lowered_name in {"s", "text", "string"}:
        values = _source_string_values(prompt, quoted_values)
    elif "regex" in lowered_name or "pattern" in lowered_name:
        values = _regex_values(prompt, quoted_values)
    elif "replace" in lowered_name or "replacement" in lowered_name:
        values = _replacement_param_values(prompt, quoted_values)
    else:
        values = quoted_values + _tail_phrases(prompt) + _word_values(prompt)

    filtered: list[str] = [
        value
        for value in values
        if value not in used_strings and _is_useful_string(value)
    ]
    if filtered:
        return filtered
    if values:
        return values
    return [""]


def _name_values(prompt: str) -> list[str]:
    """Extract likely person names from greeting-style prompts."""
    values: list[str] = []
    greet_match: re.Match[str] | None = GREET_TAIL_RE.search(prompt)
    if greet_match is not None:
        values.append(greet_match.group(1).strip(" .?!'\""))
    values.extend(_word_values(prompt))
    return values


def _source_string_values(
    prompt: str,
    quoted_values: list[str],
) -> list[str]:
    """Extract the text to transform, preferring the longest quoted span."""
    values: list[str] = sorted(quoted_values, key=len, reverse=True)
    in_match: re.Match[str] | None = IN_QUOTED_RE.search(prompt)
    if in_match is not None:
        values.insert(0, in_match.group(1))
    in_double_match: re.Match[str] | None = IN_DOUBLE_QUOTED_RE.search(prompt)
    if in_double_match is not None:
        values.insert(0, in_double_match.group(1))
    return values


def _regex_values(prompt: str, quoted_values: list[str]) -> list[str]:
    """Extract regex patterns from prompt wording or substitution targets."""
    values: list[str] = _regex_like_values(prompt)
    substitute_match: re.Match[str] | None = SUBSTITUTE_WORD_RE.search(prompt)
    if substitute_match is not None:
        values.append(substitute_match.group(1))
    if "replace" in prompt.lower() and len(quoted_values) == 1:
        values.extend(quoted_values)
    return values


def _replacement_param_values(
    prompt: str,
    quoted_values: list[str],
) -> list[str]:
    """Extract replacement literals from trailing or quoted values."""
    values: list[str] = []
    substitute_match: re.Match[str] | None = SUBSTITUTE_WITH_QUOTED_RE.search(
        prompt,
    )
    if substitute_match is not None:
        values.append(substitute_match.group(1))
    with_match: re.Match[str] | None = WITH_TOKEN_RE.search(prompt)
    if with_match is not None:
        values.append(_normalize_replacement(with_match.group(1)))
    if "asterisk" in prompt.lower():
        values.append("*")
    if len(quoted_values) >= 2:
        values.append(quoted_values[1])
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
    """Extract useful unquoted tails from common prompt patterns."""
    values: list[str] = []
    for pattern in (GREET_TAIL_RE, REVERSE_TAIL_RE):
        match: re.Match[str] | None = pattern.search(prompt)
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


def _normalize_replacement(value: str) -> str:
    """Normalize natural replacement names into literal strings."""
    if value.lower() in {"asterisk", "asterisks"}:
        return "*"
    return value


def _word_values(prompt: str) -> list[str]:
    """Return meaningful word candidates for unquoted string arguments."""
    return [
        word
        for word in WORD_RE.findall(prompt)
        if word.lower() not in COMMON_WORDS and len(word) > 1
    ]


def _is_useful_string(value: str) -> bool:
    """Drop noisy one-letter or generic fragments from string candidates."""
    if not value:
        return True
    if len(value) == 1 and value.lower() not in {"*"}:
        return False
    return value.lower() not in COMMON_WORDS


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
