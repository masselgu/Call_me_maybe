"""Function calling with LLM-guided constrained decoding."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from src.extract import Evidence, extract, schema_words, score, string_options
from src.llm_client import LlmClient
from src.models import FunctionCallResult, FunctionDefinition, PromptItem


class ChoiceDecoder(BaseModel):
    """Generate one valid choice token-by-token with masked logits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm: LlmClient

    def choose(self, context: str, choices: list[str]) -> str:
        if not choices:
            raise ValueError("empty choice list")
        if len(choices) == 1:
            return choices[0]
        encoded = [
            (c, self.llm.encode(f"{c}\n")) for c in dict.fromkeys(choices)
        ]
        ctx = self.llm.encode(context)
        out: list[int] = []
        while True:
            for choice, ids in encoded:
                if ids == out:
                    return choice
            valid = self._next(encoded, out)
            out.append(self._pick(ctx + out, valid))

    def _next(
        self,
        encoded: list[tuple[str, list[int]]],
        prefix: list[int],
    ) -> set[int]:
        valid: set[int] = set()
        n = len(prefix)
        for _, ids in encoded:
            if ids[:n] == prefix and len(ids) > n:
                tok = ids[n]
                if self.llm.vocabulary.has_id(tok):
                    valid.add(tok)
        if not valid:
            raise ValueError("constrained decoding reached a dead end")
        return valid

    def _pick(self, ctx: list[int], valid: set[int]) -> int:
        logits = np.array(self.llm.logits(ctx), dtype=np.float64)
        masked = np.full_like(logits, float("-inf"))
        for tok in valid:
            if tok < len(logits):
                masked[tok] = logits[tok]
        best = int(np.argmax(masked))
        if best not in valid:
            raise ValueError("no valid token was present in model logits")
        return best


class FunctionCallingPipeline(BaseModel):
    """Translate prompts into schema-valid function calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    functions: list[FunctionDefinition]
    model_name: str = "Qwen/Qwen3-0.6B"
    _llm: LlmClient = PrivateAttr()
    _decoder: ChoiceDecoder = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._llm = LlmClient(model_name=self.model_name)
        self._decoder = ChoiceDecoder(llm=self._llm)

    def run(self, prompts: list[PromptItem]) -> list[FunctionCallResult]:
        return [
            FunctionCallResult(
                prompt=item.prompt,
                name=fn.name,
                parameters=self._fill(item.prompt, fn, extract(item.prompt)),
            )
            for item in prompts
            for fn in [self._pick(item.prompt, extract(item.prompt))]
        ]

    def _pick(self, prompt: str, ev: Evidence) -> FunctionDefinition:
        ranked = sorted(
            self.functions,
            key=lambda fn: score(fn, ev, prompt),
            reverse=True,
        )
        ranked = [fn for fn in ranked if score(fn, ev, prompt) >= 0]
        if not ranked:
            raise ValueError(f"no function matches prompt: {prompt}")
        best = score(ranked[0], ev, prompt)
        if len(ranked) == 1 or (
            len(ranked) > 1 and best > score(ranked[1], ev, prompt) + 5
        ):
            return ranked[0]
        top = ranked[:3]
        lines = ["Pick one function name.", "Functions:"]
        for fn in top:
            sig = ", ".join(f"{k}:{v.type}" for k, v in fn.parameters.items())
            lines.append(f"- {fn.name}: {fn.description} [{sig}]")
        lines += [f"Request: {prompt}", "Function:"]
        name = self._decoder.choose(
            "\n".join(lines),
            [fn.name for fn in top],
        )
        return next(fn for fn in top if fn.name == name)

    def _fill(
        self,
        prompt: str,
        fn: FunctionDefinition,
        ev: Evidence,
    ) -> dict[str, Any]:
        vocab = schema_words(fn)
        params: dict[str, Any] = {}
        used: set[str] = set()
        num_i = 0
        quotes_left = list(ev.quotes)

        for name, spec in fn.parameters.items():
            if spec.type in {"number", "integer"}:
                if num_i >= len(ev.numbers):
                    raise ValueError(f"no number for {name}")
                val = ev.numbers[num_i]
                params[name] = (
                    int(val) if spec.type == "integer" else float(val)
                )
                num_i += 1
                continue

            if spec.type == "boolean":
                lower = prompt.lower()
                opts = []
                if any(x in lower for x in ("true", "yes", "on")):
                    opts.append("true")
                if any(x in lower for x in ("false", "no", "off")):
                    opts.append("false")
                opts = opts or ["true", "false"]
                pick = opts[0] if len(opts) == 1 else self._decoder.choose(
                    f"Request: {prompt}\nChoices: {', '.join(opts)}\nAnswer:",
                    opts,
                )
                params[name] = pick == "true"
                continue

            bias: list[str] = []
            if quotes_left and name not in {"regex", "replacement"}:
                bias = [quotes_left.pop(0)]
            opts = string_options(prompt, name, spec, ev, used, vocab, bias)
            if not opts:
                raise ValueError(f"no candidates for {name}")
            choices = [json.dumps(v, ensure_ascii=False) for v in opts]
            line = f"Parameter: {name} ({spec.type})"
            if spec.description:
                line += f" — {spec.description}"
            ctx = "\n".join([
                f"Function: {fn.name} — {fn.description}",
                line,
                f"Request: {prompt}",
                f"Already chosen: {json.dumps(params)}",
                f"Pick one value for {name}: {', '.join(choices)}",
                "Answer:",
            ])
            pick = self._decoder.choose(ctx, choices)
            params[name] = json.loads(pick)
            used.add(str(params[name]))

        return params
