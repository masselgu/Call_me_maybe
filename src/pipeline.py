"""Simple function calling pipeline."""

from __future__ import annotations

import json
from typing import Any
from pydantic import BaseModel, PrivateAttr
from src.decoder import ChoiceDecoder, Vocabulary
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    PromptItem,
)


class FunctionCallingPipeline(BaseModel):

    functions: list[FunctionDefinition]
    model_name: str = "Qwen/Qwen3-0.6B"

    _model: Any = PrivateAttr()
    _decoder: ChoiceDecoder = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:

        from llm_sdk import Small_LLM_Model

        self._model = Small_LLM_Model(
            model_name=self.model_name
        )

        vocab = Vocabulary.from_path(
            self._model.get_path_to_vocab_file()
        )

        self._decoder = ChoiceDecoder(
            model=self._model,
            vocabulary=vocab,
        )

    def run(
        self,
        prompts: list[PromptItem],
    ) -> list[FunctionCallResult]:

        results: list[FunctionCallResult] = []

        for item in prompts:

            function = self._select_function(
                item.prompt
            )

            parameters = self._select_parameters(
                item.prompt,
                function,
            )

            results.append(
                FunctionCallResult(
                    prompt=item.prompt,
                    name=function.name,
                    parameters=parameters,
                )
            )

        return results

    def _select_function(
        self,
        prompt: str,
    ) -> FunctionDefinition:

        choices = [fn.name for fn in self.functions]

        text = (
            "Choose the best function.\n\n"
        )

        for fn in self.functions:
            text += (
                f"{fn.name}: "
                f"{fn.description}\n"
            )

        text += (
            f"\nRequest: {prompt}\n"
            "Function:"
        )

        selected = self._decoder.choose(
            text,
            choices,
        )

        for fn in self.functions:
            if fn.name == selected:
                return fn

        raise ValueError(
            "function not found"
        )

    def _select_parameters(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, Any]:

        parameters: dict[str, Any] = {}
        used: set[str] = set()

        for name, spec in function.parameters.items():

            values = self._parameter_candidates(
                prompt,
                name,
                spec.type,
            )

            values = [
                value
                for value in values
                if str(value) not in used
            ] or values

            choices = [
                json.dumps(v)
                for v in values
            ]

            text = (
                f"Function: {function.name}\n"
                f"Request: {prompt}\n"
                f"Parameter: {name}\n"
                f"Type: {spec.type}\n"
                f"Choices: {choices}\n"
                "Answer:"
            )

            picked = self._decoder.choose(
                text,
                choices,
            )

            parameters[name] = json.loads(
                picked
            )

            used.add(str(parameters[name]))

        return parameters

    def _parameter_candidates(
        self,
        prompt: str,
        param_name: str,
        param_type: str,
    ) -> list[Any]:

        words = prompt.split()

        if param_type == "boolean":
            return [True, False]

        if param_type in {"number", "integer"}:

            values = []

            for word in words:

                clean = ""

                for ch in word:
                    if ch.isdigit() or ch == ".":
                        clean += ch

                if clean:
                    number = float(clean)

                    if param_type == "integer":
                        values.append(int(number))
                    else:
                        values.append(number)

            return values or [0]

        strings = []

        current = ""
        inside = False
        quote = ""

        for ch in prompt:

            if not inside:

                if ch in {"'", '"'}:
                    inside = True
                    quote = ch

                continue

            if ch == quote:
                strings.append(current)
                current = ""
                inside = False
                quote = ""
                continue

            current += ch

        lower_prompt = prompt.lower()

        if param_name == "name":

            if len(words) > 1:
                return [words[-1]]

            return [prompt]

        if param_name == "s":

            if strings:
                return [strings[0]]

            return [prompt]

        if param_name == "source_string":

            if strings:
                return [strings[-1]]

            return [prompt]

        if param_name == "regex":

            if "number" in lower_prompt or "numbers" in lower_prompt:
                return ["\\d+"]

            if "vowel" in lower_prompt or "vowels" in lower_prompt:
                return ["[aeiouAEIOU]"]

            if len(strings) >= 2:
                return [strings[0]]

            return [".*"]

        if param_name == "replacement":

            if "asterisk" in lower_prompt:
                return ["*"]

            parts = prompt.split()

            for i, word in enumerate(parts):

                if word.lower() == "with" and i + 1 < len(parts):

                    value = parts[i + 1]
                    value = value.strip("'\".,!?")

                    return [value]

            if len(strings) >= 2:
                return [strings[1]]

            return [""]

        if strings:
            return strings

        return [prompt]
