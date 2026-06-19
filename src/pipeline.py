"""Function calling with LLM-guided constrained decoding."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from src.llm_client import LlmClient
from src.models import FunctionCallResult
from src.models import FunctionDefinition
from src.models import ParameterDefinition
from src.models import PromptItem

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
QUOTED_RE = re.compile(r'"([^"]*)"|\'([^\']*)\'')
WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SKIP = {
    "a", "an", "all", "and", "calculate", "for", "fun", "greet", "in", "is",
    "of", "old", "on", "or", "replace", "return", "reverse", "root", "square",
    "string", "substitute", "sum", "the", "what", "with", "word", "years",
}


class ChoiceDecoder(BaseModel):
    """Pick one value by scoring complete candidates with LLM logits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: LlmClient

    def choose(self, context: str, choices: list[str]) -> str:
        """Return the highest-scoring valid choice."""
        if not choices:
            raise ValueError("empty choice list")
        if len(choices) == 1:
            return choices[0]
        context_ids = self.llm.encode(context)
        best, best_score = choices[0], float("-inf")
        for choice in choices:
            score = self._score(context_ids, self.llm.encode(f"{choice}\n"))
            if score > best_score:
                best, best_score = choice, score
        return best

    def _score(self, context_ids: list[int], choice_ids: list[int]) -> float:
        """Sum log-probabilities for one candidate sequence."""
        ids = list(context_ids)
        total = 0.0
        for token_id in choice_ids:
            logits = np.array(self.llm.logits(ids), dtype=np.float64)
            if token_id >= len(logits):
                return float("-inf")
            shifted = logits - np.max(logits)
            log_probs = shifted - np.log(np.sum(np.exp(shifted)))
            total += float(log_probs[token_id])
            ids.append(token_id)
        return total


class FunctionCallingPipeline(BaseModel):
    """Translate prompts into schema-valid function calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    functions: list[FunctionDefinition]
    model_name: str = "Qwen/Qwen3-0.6B"
    _llm: LlmClient = PrivateAttr()
    _decoder: ChoiceDecoder = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize LLM helpers."""
        self._llm = LlmClient(model_name=self.model_name)
        self._decoder = ChoiceDecoder(llm=self._llm)

    def run(self, prompts: list[PromptItem]) -> list[FunctionCallResult]:
        """Process all prompts."""
        results: list[FunctionCallResult] = []
        for item in prompts:
            function = self._select_function(item.prompt)
            results.append(
                FunctionCallResult(
                    prompt=item.prompt,
                    name=function.name,
                    parameters=self._extract(item.prompt, function),
                ),
            )
        return results

    def _select_function(self, prompt: str) -> FunctionDefinition:
        """Let the LLM choose a function name."""
        lines = ["Pick one function name.", "Functions:"]
        lines += [f"- {f.name}: {f.description}" for f in self.functions]
        lines += [f"Request: {prompt}", "Function:"]
        name = self._decoder.choose(
            "\n".join(lines),
            [f.name for f in self.functions],
        )
        for function in self.functions:
            if function.name == name:
                return function
        raise ValueError(f"unknown function: {name}")

    def _extract(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, Any]:
        """Extract each parameter with constrained decoding."""
        params: dict[str, Any] = {}
        for name, spec in function.parameters.items():
            choices = _candidates(prompt, name, spec, params)
            context = "\n".join([
                f"Function: {function.name}",
                f"Parameter: {name} ({spec.type})",
                f"Request: {prompt}",
                f"Selected: {json.dumps(params)}",
                f"Choices: {', '.join(choices)}",
                "Answer:",
            ])
            value = json.loads(self._decoder.choose(context, choices))
            params[name] = _coerce(value, spec)
        return params


def _quoted(prompt: str) -> list[str]:
    """Return quoted substrings from the prompt."""
    return [
        m.group(1) if m.group(1) is not None else m.group(2) or ""
        for m in QUOTED_RE.finditer(prompt)
    ]


def _words(prompt: str) -> list[str]:
    """Return meaningful words from the prompt."""
    return [
        word for word in WORD_RE.findall(prompt)
        if word.lower() not in SKIP and len(word) > 1
    ]


def _numbers(
    prompt: str,
    integer: bool,
    selected: dict[str, Any],
) -> list[Any]:
    """Return unused numeric literals found in the prompt."""
    values: list[Any] = []
    for match in NUM_RE.finditer(prompt):
        raw = match.group(0)
        values.append(int(float(raw)) if integer else float(raw))
    used = {
        value for value in selected.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    unused = [value for value in values if value not in used]
    if unused:
        return unused
    if values:
        return values
    return [0 if integer else 0.0]


def _strings(
    prompt: str,
    name: str,
    selected: dict[str, Any],
) -> list[Any]:
    """Build string candidates based on parameter role."""
    key = name.lower()
    quoted = _quoted(prompt)
    lower = prompt.lower()
    values: list[Any] = []

    if key == "name":
        match = re.search(r"\bgreet\s+(.+)$", prompt, re.I)
        if match:
            values.append(match.group(1).strip(" .?!'\""))
        match = re.search(r"\bname\s+is\s+(\w+)", prompt, re.I)
        if match:
            values.append(match.group(1))
        values.extend(_words(prompt))
    elif "source" in key or key in {"s", "text", "string"}:
        values = sorted(quoted, key=len, reverse=True)
        for pattern in (r"\bin\s+'([^']+)'", r'\bin\s+"([^"]+)"'):
            match = re.search(pattern, prompt)
            if match:
                values.insert(0, match.group(1))
    elif "regex" in key or "pattern" in key:
        if "number" in lower or "digit" in lower:
            values.append(r"\d+")
        if "vowel" in lower:
            values.append(r"[aeiouAEIOU]")
        match = re.search(r"\bword\s+'([^']+)'", prompt, re.I)
        if match:
            values.append(match.group(1))
    elif "replace" in key or "replacement" in key:
        match = re.search(r"\bwith\s+'([^']+)'", prompt, re.I)
        if match:
            values.append(match.group(1))
        match = re.search(r"\bwith\s+([A-Za-z0-9_*]+)", prompt, re.I)
        if match:
            token = match.group(1)
            if token.lower() in {"asterisk", "asterisks"}:
                values.append("*")
            else:
                values.append(token)
        if "asterisk" in lower:
            values.append("*")
        if len(quoted) >= 2:
            values.append(quoted[1])
    else:
        values = quoted + _words(prompt)

    used = {
        str(value)
        for value in selected.values()
        if isinstance(value, str)
    }
    clean = [
        value for value in values
        if str(value) not in used and _ok(str(value))
    ]
    return clean or values or [""]


def _ok(value: str) -> bool:
    """Filter noisy string fragments."""
    if not value:
        return True
    if len(value) == 1 and value != "*":
        return False
    return value.lower() not in SKIP


def _candidates(
    prompt: str,
    name: str,
    spec: ParameterDefinition,
    selected: dict[str, Any],
) -> list[str]:
    """Serialize unique JSON literals for one parameter."""
    raw: list[Any]
    if spec.type in {"number", "integer"}:
        raw = _numbers(prompt, spec.type == "integer", selected)
    elif spec.type == "boolean":
        lower = prompt.lower()
        bools: list[Any] = []
        if any(item in lower for item in ("true", "yes", "on")):
            bools.append(True)
        if any(item in lower for item in ("false", "no", "off")):
            bools.append(False)
        raw = bools or [True, False]
    else:
        raw = _strings(prompt, name, selected)

    seen: set[str] = set()
    out: list[str] = []
    for value in raw:
        text = json.dumps(value, ensure_ascii=False)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _coerce(value: Any, spec: ParameterDefinition) -> Any:
    """Cast a decoded JSON value to the schema type."""
    if spec.type == "string":
        return str(value)
    if spec.type == "integer":
        return int(value)
    if spec.type == "number":
        return float(value)
    if spec.type == "boolean":
        return bool(value)
    raise ValueError(f"unsupported type: {spec.type}")
