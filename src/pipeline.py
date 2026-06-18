"""Main function-calling pipeline."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from src.constrained import ConstrainedChoiceDecoder
from src.extraction import candidate_json_values
from src.llm_client import LlmClient
from src.models import FunctionCallResult
from src.models import FunctionDefinition
from src.models import ParameterDefinition
from src.models import PromptItem


class FunctionCallingPipeline(BaseModel):
    """Translate natural-language prompts into structured function calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    functions: list[FunctionDefinition]
    model_name: str = "Qwen/Qwen3-0.6B"
    _llm: LlmClient = PrivateAttr()
    _decoder: ConstrainedChoiceDecoder = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Create runtime helpers after Pydantic validates fields."""
        self._llm = LlmClient(model_name=self.model_name)
        self._decoder = ConstrainedChoiceDecoder(llm=self._llm)

    def run(self, prompts: list[PromptItem]) -> list[FunctionCallResult]:
        """Process all input prompts."""
        results: list[FunctionCallResult] = []
        for item in prompts:
            function: FunctionDefinition = self._select_function(item.prompt)
            parameters: dict[str, Any] = self._extract_parameters(
                item.prompt,
                function,
            )
            results.append(
                FunctionCallResult(
                    prompt=item.prompt,
                    name=function.name,
                    parameters=parameters,
                ),
            )
        return results

    def _select_function(self, prompt: str) -> FunctionDefinition:
        """Choose a function using constrained LLM decoding."""
        context: str = self._function_context(prompt)
        choices: list[str] = [function.name for function in self.functions]
        selected_name: str = self._decoder.choose(context, choices)
        for function in self.functions:
            if function.name == selected_name:
                return function
        raise ValueError(f"unknown function selected: {selected_name}")

    def _extract_parameters(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, Any]:
        """Extract each argument with schema-constrained choices."""
        parameters: dict[str, Any] = {}
        for name, definition in function.parameters.items():
            json_choices: list[str] = candidate_json_values(
                prompt,
                function,
                name,
                definition,
            )
            context: str = self._parameter_context(
                prompt,
                function,
                name,
                definition,
                parameters,
                json_choices,
            )
            chosen_json: str = self._decoder.choose(context, json_choices)
            parameters[name] = self._coerce_value(
                json.loads(chosen_json),
                definition,
            )
        return parameters

    def _function_context(self, prompt: str) -> str:
        """Build a compact selection prompt for the model."""
        lines: list[str] = [
            "Select the single best function for the user request.",
            "Return only one function name.",
            "",
            "Available functions:",
        ]
        for function in self.functions:
            lines.append(f"- {function.name}: {function.description}")
        lines.extend(["", f"User request: {prompt}", "Function name: "])
        return "\n".join(lines)

    @staticmethod
    def _parameter_context(
        prompt: str,
        function: FunctionDefinition,
        name: str,
        definition: ParameterDefinition,
        previous: dict[str, Any],
        json_choices: list[str],
    ) -> str:
        """Build a parameter extraction prompt for the model."""
        return "\n".join(
            [
                "Extract exactly one JSON literal for the parameter.",
                f"Function: {function.name}",
                f"Description: {function.description}",
                f"User request: {prompt}",
                f"Parameter: {name}",
                f"Parameter type: {definition.type}",
                f"Already selected: {json.dumps(previous)}",
                "Valid JSON choices:",
                ", ".join(json_choices),
                "Chosen JSON literal: ",
            ],
        )

    @staticmethod
    def _coerce_value(value: Any, definition: ParameterDefinition) -> Any:
        """Coerce a decoded value into the exact schema type."""
        if definition.type == "string":
            return str(value)
        if definition.type == "integer":
            return int(value)
        if definition.type == "number":
            return float(value)
        if definition.type == "boolean":
            return bool(value)
        raise ValueError(f"unsupported parameter type: {definition.type}")
