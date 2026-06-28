"""Simple function calling pipeline."""

import json
from typing import Any
import numpy as np
import time
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
        text = "Choose the best function.\n\n"

        for fn in self.functions:
            text += f"{fn.name}: {fn.description}\n"

        text += f"\nRequest: {prompt}\nFunction:"
        selected = self._decoder.choose(
            text,
            choices,
        )

        for fn in self.functions:
            if fn.name == selected:
                return fn

        raise ValueError("function not found")

    def _select_parameters(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, Any]:
        llm_candidates = self._llm_propose_values(
            prompt,
            function,
        )
        parameters: dict[str, Any] = {}
        used: set[str] = set()
        for name, spec in function.parameters.items():
            values = self._parameter_candidates(
                llm_candidates,
                name,
                spec.type,
            )

            values = [
                value
                for value in values
                if str(value) not in used
            ] or values

            choices = [json.dumps(value) for value in values]
            text = (
                f"Function: {function.name}\n"
                f"Request: {prompt}\n"
                f"Parameter: {name}\n"
                f"Type: {spec.type}\n"
                f"Choices: {choices}\n"
                "Answer:"
            )
            picked = self._decoder.choose(text, choices)
            parameters[name] = json.loads(picked)
            used.add(str(parameters[name]))

        return parameters

    def _parameter_candidates(
        self,
        llm_candidates: dict[str, list[Any]],
        param_name: str,
        param_type: str,
    ) -> list[Any]:

        values = []

        for value in llm_candidates.get(param_name, []):

            try:

                if param_type == "integer":
                    values.append(int(value))

                elif param_type == "number":
                    values.append(float(value))

                elif param_type == "boolean":

                    if isinstance(value, bool):
                        values.append(value)

                else:
                    values.append(str(value))

            except (ValueError, TypeError):
                continue

        if values:
            return values

        if param_type == "boolean":
            return [True, False]

        if param_type == "integer":
            return [0]

        if param_type == "number":
            return [0.0]

        return [""]

    def _free_generate(
        self,
        prompt: str,
        max_tokens: int = 15,
    ) -> str:

        ids = self._decoder._encode(prompt)

        generated: list[int] = []
        start = time.time()
        for i in range(max_tokens):

            logits = self._model.get_logits_from_input_ids(
                ids + generated
            )
            elapsed = time.time() - start

            print(
                f"token={i} "
                f"context_len={len(ids + generated)} "
                f"time={elapsed:.4f}s"
            )
            next_id = int(np.argmax(logits))
            generated.append(next_id)

            text = self._model.decode(generated)

            if "}" in text:
                print("FOUND CLOSING BRACE")
                break

        return self._model.decode(generated)

    def _llm_propose_values(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, list[Any]]:

        param_spec = ", ".join(
            f"{name}: {spec.type}"
            for name, spec in function.parameters.items()
        )

        llm_prompt = (
            "You are a function-calling assistant.\n\n"

            "IMPORTANT RULES:\n"
            "- Do NOT execute the function.\n"
            "- Do NOT compute the result.\n"
            "- Do NOT modify the input.\n"
            "- Do NOT apply the transformation.\n"
            "- ONLY extract parameter values from the request.\n"
            "- Return ONLY valid JSON.\n\n"

            "Example:\n"
            "Function: fn_substitute_string_with_regex("
            "source_string: string, regex: string, replacement: string)\n"
            "Request: Replace all vowels in 'Programming is fun' with asterisks\n"
            'JSON: {"source_string":["Programming is fun"],'
            '"regex":["[aeiouAEIOU]"],'
            '"replacement":["*"]}\n\n'

            "Example:\n"
            "Function: fn_substitute_string_with_regex("
            "source_string: string, regex: string, replacement: string)\n"
            "Request: Replace all numbers in "
            "\"Hello 34 I'm 233 years old\" with NUMBERS\n"
            'JSON: {"source_string":["Hello 34 I\'m 233 years old"],'
            '"regex":["\\\\d+"],'
            '"replacement":["NUMBERS"]}\n\n'

            f"Function: {function.name}({param_spec})\n"
            f"Request: {prompt}\n"
            "JSON:"
        )

        raw = self._free_generate(
            llm_prompt,
            max_tokens=30,
        )

        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1

            parsed = json.loads(raw[start:end])

            result = {}

            for name in function.parameters:
                value = parsed.get(name, [])

                if not isinstance(value, list):
                    value = [value]

                result[name] = value

            return result

        except Exception:
            return {
                name: []
                for name in function.parameters
            }
