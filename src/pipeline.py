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
        print(f"Loading model {self.model_name}...")
        self._model = Small_LLM_Model(
            model_name=self.model_name
        )
        print(f"Model {self.model_name} loaded successfully.\n")
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
        print(f"Processing {len(prompts)} prompts...\n")
        start_time = time.time()
        for i, item in enumerate(prompts, 1):
            print(f"Processing prompt {i}/{len(prompts)}: {item.prompt}")
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
            print(f"successfully processed prompt: {i}/{len(prompts)} "
                  f"{item.prompt}\n")

        print(f"\nTotal time elapsed: {(time.time() - start_time) / 60:.2f} "
              "minutes")
        return results

    def _select_function(
        self,
        prompt: str,
    ) -> FunctionDefinition:

        choices = [fn.name for fn in self.functions]

        function_list_text = (
            "You are an AI function calling assistant.\n\n"
            "Your task is to choose the BEST function for the user request.\n"
            "Carefully count how many distinct values the request provides, "
            "and match it to the function with the same number of parameters.\n"
            "Only answer with the exact function name. Do not explain "
            "anything.\n\n"
            "Available functions:\n"
        )
        for fn in self.functions:
            params = ", ".join(
                f"{k}: {v.type}" for k, v in fn.parameters.items()
            )
            param_count = len(fn.parameters)
            function_list_text += (
                f"- {fn.name}({params})"
                f" [requires exactly {param_count} parameter(s)]: "
                f"{fn.description}\n"
            )

        context_text = (
            function_list_text
            + f"\nRequest: {prompt}\nFunction:"
        )
        selected = self._decoder.choose(context_text, choices)

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
        stop_chars: tuple[str, ...] = ("}",),
    ) -> str:
        """Greedy decode, stopping as soon as any char in stop_chars
        appears in the newly generated piece.

        Decodes only the new token each step instead of re-decoding
        the whole growing sequence every iteration -- O(1) per step
        instead of O(n). stop_chars is configurable so different
        callers can stop on what THEY consider "done" (a closing
        brace for JSON extraction, a newline/period for free-form
        reasoning) instead of every call sharing one hardcoded
        condition that only fits one of the two use cases.
        """

        ids = self._decoder._encode(prompt)

        generated: list[int] = []
        pieces: list[str] = []
        for _ in range(max_tokens):

            logits = self._model.get_logits_from_input_ids(
                ids + generated
            )
            next_id = int(np.argmax(logits))
            generated.append(next_id)

            piece = self._model.decode([next_id])
            pieces.append(piece)

            if any(ch in piece for ch in stop_chars):
                break

        return "".join(pieces)

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
            "Extract parameter values from a request. "
            "Copy text exactly as it appears -- do not compute, solve, "
            "or transform anything. "
            "The parameter name may differ from the wording in the request -- "
            "extract by meaning, not by name.\n\n"

            # teaches: numbers, multiple params
            "Function: fn_calculate_discount(price: number, percent: number)\n"
            "Request: Apply a discount of 15 percent to a price of 80\n"
            'JSON: {"price":[80],"percent":[15]}\n\n'

            # teaches: multiple string params
            "Function: fn_lookup_record(record_id: string, table: string)\n"
            "Request: Find the entry named 'Alpha-12' in table 'inventory'\n"
            'JSON: {"record_id":["Alpha-12"],"table":["inventory"]}\n\n'

            f"Function: {function.name}({param_spec}) "
            f"Description: {function.description}\n"
            f"Request: {prompt}\n"
            "JSON:"
        )

        raw = self._free_generate(
            llm_prompt,
            max_tokens=15 * len(function.parameters),
            stop_chars=("}",),
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
