"""Function calling with LLM-guided constrained decoding."""

from __future__ import annotations

import json
from typing import Any
from pydantic import BaseModel, PrivateAttr
from src.decoder import ChoiceDecoder, Vocabulary
from src.extract import Evidence, extract, schema_words
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    ParameterDefinition,
    PromptItem,
)
from src.selection import can_fill, options, score_function

MAX_FUNCTION_CHOICES = 6


class FunctionCallingPipeline(BaseModel):
    """Translate prompts into schema-valid function calls."""

    functions: list[FunctionDefinition]
    model_name: str = "Qwen/Qwen3-0.6B"
    _model: Any = PrivateAttr(default=None)
    _decoder: ChoiceDecoder = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Load the SDK model and constrained decoder."""
        try:
            from llm_sdk import Small_LLM_Model

            self._model = Small_LLM_Model(model_name=self.model_name)
            vocab = Vocabulary.from_path(self._model.get_path_to_vocab_file())
            self._decoder = ChoiceDecoder(model=self._model, vocabulary=vocab)
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
        choices = [fn for fn in self.functions if can_fill(prompt, fn, ev)]
        if not choices:
            raise ValueError(f"no function can be filled for prompt: {prompt}")
        choices = sorted(
            choices,
            key=lambda fn: score_function(prompt, ev, fn),
            reverse=True,
        )[:MAX_FUNCTION_CHOICES]

        lines = ["Choose one function.", "Functions:"]
        for fn in choices:
            args = ", ".join(
                f"{name}:{spec.type}"
                for name, spec in fn.parameters.items()
            )
            lines.append(f"- {fn.name}({args}): {fn.description}")
        lines += [f"Request: {prompt}", "Function:"]

        name = self._decoder.choose(
            "\n".join(lines),
            [fn.name for fn in choices],
        )
        return next(fn for fn in choices if fn.name == name)

    def _fill(
        self,
        prompt: str,
        fn: FunctionDefinition,
        ev: Evidence,
    ) -> dict[str, Any]:
        """Choose every parameter value using constrained decoding."""
        params: dict[str, Any] = {}
        used: set[str] = set()
        vocab = schema_words(fn)

        for name, spec in fn.parameters.items():
            values = options(prompt, name, spec, ev, vocab)
            unused = [value for value in values if str(value) not in used]
            values = unused or values
            if not values:
                raise ValueError(f"no candidates for {name}")

            choices = [
                json.dumps(value, ensure_ascii=False)
                for value in values
            ]
            ctx = self._parameter_prompt(
                prompt, fn, name, spec, params, choices
            )
            pick = self._decoder.choose(
                ctx, choices,
            )
            params[name] = json.loads(pick)
            used.add(str(params[name]))
        return params

    def _parameter_prompt(
        self,
        prompt: str,
        fn: FunctionDefinition,
        name: str,
        spec: ParameterDefinition,
        params: dict[str, Any],
        choices: list[str],
    ) -> str:
        """Build the prompt for choosing one parameter value."""
        line = f"Parameter: {name} ({spec.type})"
        if spec.description:
            line += f" - {spec.description}"
        return "\n".join([
            f"Function: {fn.name} - {fn.description}",
            line,
            f"Request: {prompt}",
            f"Already chosen: {json.dumps(params)}",
            f"Valid values: {', '.join(choices)}",
            "Answer:",
        ])
