"""Prompt extraction, function matching, and parameter candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.models import FunctionDefinition, ParameterDefinition

NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
QUOTE = re.compile(r'"([^"]*)"|\'([^\']*)\'')
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
REGEX_LIT = re.compile(r"\\(?:d\+|w\+|s\+|\S)|\[[^\]]+\]")
BOOL = re.compile(r"\b(true|false|yes|no|on|off)\b", re.I)
WITH = re.compile(
    r"\bwith\s+'([^']+)'|\bwith\s+\"([^\"]+)\"|\bwith\s+(\S+)",
    re.I,
)
REL = re.compile(r"\b(to|for|with)\s+(\S+)", re.I)
WORD_IN_PROMPT = re.compile(r"\bword\s+'([^']+)'", re.I)
REGEX_TOPIC = re.compile(
    r"\b(numbers|vowels|vowel|digits|digit|regex|pattern)\b",
    re.I,
)


@dataclass
class Evidence:
    """Values extracted once from a user prompt."""

    quotes: list[str]
    numbers: list[float]
    tokens: list[str]
    has_bool: bool
    has_with: bool
    has_relation: bool
    wants_regex: bool


def extract(prompt: str) -> Evidence:
    """Build prompt evidence once."""
    sub = re.search(
        r"\bsubtract\s+([-+]?\d+(?:\.\d+)?)\s+from\s+([-+]?\d+(?:\.\d+)?)",
        prompt,
        re.I,
    )
    if sub:
        numbers = [float(sub.group(2)), float(sub.group(1))]
    else:
        minus = re.search(
            r"([-+]?\d+(?:\.\d+)?)\s+minus\s+([-+]?\d+(?:\.\d+)?)",
            prompt,
            re.I,
        )
        if minus:
            numbers = [float(minus.group(1)), float(minus.group(2))]
        else:
            text = QUOTE.sub(" ", prompt)
            numbers = [float(m.group()) for m in NUM.finditer(text)]

    quotes = [
        m.group(1) if m.group(1) is not None else m.group(2) or ""
        for m in QUOTE.finditer(prompt)
    ]
    return Evidence(
        quotes=quotes,
        numbers=numbers,
        tokens=WORD.findall(prompt),
        has_bool=bool(BOOL.search(prompt)),
        has_with=bool(WITH.search(prompt)),
        has_relation=bool(REL.search(prompt)),
        wants_regex=bool(REGEX_TOPIC.search(prompt)),
    )


def _counts(fn: FunctionDefinition) -> tuple[int, int, int]:
    n = s = b = 0
    for spec in fn.parameters.values():
        if spec.type in {"number", "integer"}:
            n += 1
        elif spec.type == "string":
            s += 1
        elif spec.type == "boolean":
            b += 1
    return n, s, b


def schema_words(fn: FunctionDefinition) -> set[str]:
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


def _overlap(prompt: str, fn: FunctionDefinition) -> int:
    prompt_words = {w.lower() for w in WORD.findall(prompt)}
    return len(prompt_words & schema_words(fn))


def _desc_patterns(description: str | None, prompt: str) -> list[str]:
    if not description:
        return []
    lower = prompt.lower()
    out: list[str] = []
    for chunk in description.split(";"):
        chunk = chunk.strip()
        pats = REGEX_LIT.findall(chunk)
        if not pats:
            continue
        label = (
            chunk.split(" for ", 1)[-1]
            if re.search(r"\bfor\b", chunk, re.I)
            else chunk
        )
        words = [w.lower() for w in WORD.findall(label) if len(w) > 2]
        if any(
            w in lower or w.rstrip("s") in lower or w + "s" in lower
            for w in words
        ):
            out.extend(pats)
    return out or REGEX_LIT.findall(description)


def _param_role(name: str, spec: ParameterDefinition) -> str:
    desc = (spec.description or "").lower()
    if REGEX_LIT.search(spec.description or ""):
        return "regex"
    if "replacement" in name or "replacement" in desc:
        return "replacement"
    if "input text" in desc or "search" in desc:
        return "source"
    return "other"


def _uniq(raw: list[Any], used: set[str]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in raw:
        text = str(value)
        if text and text not in used and text not in seen:
            seen.add(text)
            out.append(value)
    return out


def string_options(
    prompt: str,
    name: str,
    spec: ParameterDefinition,
    ev: Evidence,
    used: set[str],
    vocab: set[str],
    bias: list[str],
) -> list[Any]:
    role = _param_role(name, spec)
    patterns = _desc_patterns(spec.description, prompt)

    if role == "regex" and patterns:
        return _uniq(patterns, used)

    raw: list[Any] = list(bias)
    raw.extend(patterns)

    if role == "replacement":
        for item in WITH.finditer(prompt):
            token = item.group(1) or item.group(2) or item.group(3)
            if "*" in (spec.description or "") and "asterisk" in token.lower():
                token = "*"
            raw.append(token)
        tail = re.search(r"\s(\S+)\s*$", prompt)
        if tail and tail.group(1).lower() not in vocab:
            raw.append(tail.group(1))
        return _uniq(raw, used)

    match = WORD_IN_PROMPT.search(prompt)
    if match:
        raw.insert(0, match.group(1))
    match = re.search(r"\breplace\s+'([^']+)'", prompt, re.I)
    if match:
        raw.append(match.group(1))
    if role == "source":
        raw.extend(q for q in ev.quotes if q not in used)
    elif role == "other":
        for item in WITH.finditer(prompt):
            token = item.group(1) or item.group(2) or item.group(3)
            raw.append(token)
        rel = REL.search(prompt)
        if rel:
            raw.insert(0, rel.group(2).strip(" .?!'\""))
        tail = re.search(r"\s(\S+)\s*$", prompt)
        if tail and tail.group(1).lower() not in vocab:
            raw.append(tail.group(1))
        raw.extend(q for q in ev.quotes if q not in used)
        raw.extend(
            w for w in ev.tokens
            if w.lower() not in vocab and w not in used
        )
    return _uniq(raw, used)


def _fits(fn: FunctionDefinition, ev: Evidence) -> bool:
    n, s, b = _counts(fn)
    if n and len(ev.numbers) != n:
        return False
    if s and not ev.quotes and not ev.tokens:
        return False
    if s >= 2 and len(ev.quotes) < s and not ev.has_with:
        if not ("recipient" in fn.parameters and ev.has_relation):
            if not ("regex" in fn.parameters and ev.wants_regex and ev.quotes):
                return False
    if "regex" in fn.parameters and not ev.wants_regex:
        return False
    if ev.wants_regex and "regex" not in fn.parameters:
        return False
    if b and not n and not s and not ev.has_bool:
        return False
    return True


def score(fn: FunctionDefinition, ev: Evidence, prompt: str) -> int:
    if not _fits(fn, ev):
        return -1
    n, s, b = _counts(fn)
    sc = _overlap(prompt, fn) * 10
    if n == len(ev.numbers):
        sc += 20
    if ev.has_bool and not ev.numbers and not ev.quotes:
        if b and not n and not s:
            sc += 50
    if ev.numbers and not ev.quotes and n:
        sc += 15
    if len(ev.numbers) == 1 and not ev.quotes and n == 1 and s == 0:
        sc += 20
    if "?" in prompt and fn.returns.type == "boolean":
        sc += 15
    if (
        len(ev.numbers) == 1
        and ev.numbers[0] == int(ev.numbers[0])
        and n == 1
        and any(spec.type == "integer" for spec in fn.parameters.values())
    ):
        sc += 15
    if "recipient" in fn.parameters and ev.has_relation:
        sc += 20
    if ev.has_relation and any(
        spec.description
        and "person" in spec.description.lower()
        and "name" in spec.description.lower()
        for spec in fn.parameters.values()
    ):
        sc += 15
    return sc
