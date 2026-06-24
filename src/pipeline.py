"""Function calling with LLM-guided constrained decoding."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from src.extract import (
    Evidence,
    boolean_options,
    extract,
    number_options,
    schema_words,
    string_options,
)
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    ParameterDefinition,
    PromptItem,
)


class Vocabulary(BaseModel):
    """Token ids loaded from the model vocabulary JSON file."""

    ids: set[int] = Field(default_factory=set)

    @classmethod
    def from_path(cls, path: str) -> "Vocabulary":
        """Load token ids from a Hugging Face ``vocab.json`` file."""
        with open(path, "r", encoding="utf-8") as file:
            raw: Any = json.load(file)

        if not isinstance(raw, dict):
            raise ValueError("vocabulary JSON must contain an object")

        return cls(ids={int(token_id) for token_id in raw.values()})

    def has_id(self, token_id: int) -> bool:
        """Check whether a token id exists in the vocabulary."""
        return token_id in self.ids


class ChoiceDecoder(BaseModel):
    """Generate one valid choice token-by-token with masked logits."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    vocabulary: Vocabulary

    def choose(self, context: str, choices: list[str]) -> str:
        """Ask the LLM to choose one item while masking invalid tokens."""
        if not choices:
            raise ValueError("empty choice list")
        if len(choices) == 1:
            return choices[0]
        encoded = [
            (choice, self._encode(f"{choice}\n"))
            for choice in dict.fromkeys(choices)
        ]
        ctx = self._encode(context)
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
                if self.vocabulary.has_id(tok):
                    valid.add(tok)
        if not valid:
            raise ValueError("constrained decoding reached a dead end")
        return valid

    def _pick(self, ctx: list[int], valid: set[int]) -> int:
        logits = np.array(
            self.model.get_logits_from_input_ids(ctx),
            dtype=np.float64,
        )
        masked = np.full_like(logits, float("-inf"))
        for tok in valid:
            if tok < len(logits):
                masked[tok] = logits[tok]
        best = int(np.argmax(masked))
        if best not in valid:
            raise ValueError("no valid token was present in model logits")
        return best

    def _encode(self, text: str) -> list[int]:
        encoded: Any = self.model.encode(text)
        raw: Any = encoded.tolist()
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            return [int(item) for item in raw[0]]
        if isinstance(raw, list):
            return [int(item) for item in raw]
        raise ValueError("LLM SDK returned unsupported token ids")


class FunctionCallingPipeline(BaseModel):
    """Translate prompts into schema-valid function calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    functions: list[FunctionDefinition]
    model_name: str = "Qwen/Qwen3-0.6B"
    _model: Any = PrivateAttr(default=None)
    _decoder: ChoiceDecoder = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        try:
            from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

            self._model = Small_LLM_Model(model_name=self.model_name)
            vocabulary = Vocabulary.from_path(
                self._model.get_path_to_vocab_file()
            )
            self._decoder = ChoiceDecoder(
                model=self._model,
                vocabulary=vocabulary,
            )
        except Exception as exc:
            raise ValueError(
                "failed to load the LLM model. Run `uv sync` first"
            ) from exc

    def run(self, prompts: list[PromptItem]) -> list[FunctionCallResult]:
        """Convert every prompt into one schema-valid function call."""
        results: list[FunctionCallResult] = []
        for item in prompts:
            ev = extract(item.prompt)
            fn = self._pick(item.prompt, ev)
            results.append(
                FunctionCallResult(
                    prompt=item.prompt,
                    name=fn.name,
                    parameters=self._fill(item.prompt, fn, ev),
                )
            )
        return results

    def _pick(self, prompt: str, ev: Evidence) -> FunctionDefinition:
        """Choose the function name using constrained decoding."""
        functions = [
            fn for fn in self.functions
            if self._can_fill(prompt, fn, ev)
        ]
        if not functions:
            raise ValueError(f"no function can be filled for prompt: {prompt}")

        lines = [
            "Choose the function that best matches the request.",
            "Return only one function name.",
            "All listed functions can be filled with valid JSON arguments.",
            "Functions:",
        ]
        for fn in functions:
            sig = ", ".join(
                f"{name}:{spec.type}"
                for name, spec in fn.parameters.items()
            )
            lines.append(f"- {fn.name}({sig}): {fn.description}")
        lines += [f"Request: {prompt}", "Function:"]

        name = self._decoder.choose(
            "\n".join(lines),
            [fn.name for fn in functions],
        )
        return next(fn for fn in functions if fn.name == name)

    def _fill(
        self,
        prompt: str,
        fn: FunctionDefinition,
        ev: Evidence,
    ) -> dict[str, Any]:
        """Choose every parameter value using constrained decoding."""
        vocab = schema_words(fn)
        params: dict[str, Any] = {}

        for name, spec in fn.parameters.items():
            opts = self._options(prompt, spec, ev, vocab)
            if not opts:
                raise ValueError(f"no candidates for {name}")

            choices = [json.dumps(value, ensure_ascii=False) for value in opts]
            line = f"Parameter: {name} ({spec.type})"
            if spec.description:
                line += f" - {spec.description}"
            ctx = "\n".join([
                f"Function: {fn.name} - {fn.description}",
                line,
                f"Request: {prompt}",
                f"Already chosen: {json.dumps(params)}",
                "Choose by meaning, not by candidate order.",
                f"Valid values for {name}: {', '.join(choices)}",
                "Answer:",
            ])
            pick = self._decoder.choose(ctx, choices)
            params[name] = json.loads(pick)

        return params

    def _can_fill(
        self,
        prompt: str,
        fn: FunctionDefinition,
        ev: Evidence,
    ) -> bool:
        """Check that each parameter has at least one typed candidate."""
        vocab = schema_words(fn)
        for spec in fn.parameters.values():
            if spec.type == "boolean" and not ev.booleans:
                return False
            if not self._options(prompt, spec, ev, vocab):
                return False
        return True

    def _options(
        self,
        prompt: str,
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
        return list(string_options(prompt, spec, ev, vocab))
