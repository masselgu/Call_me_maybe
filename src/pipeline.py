"""Simple function calling pipeline."""

import json
import sys
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
        try:
            from llm_sdk import Small_LLM_Model
            print(f"Loading model {self.model_name}...")
            self._model = Small_LLM_Model(
                model_name=self.model_name
            )
            print(f"Model {self.model_name} loaded successfully.\n")
        except ImportError:
            print("error: llm_sdk not found. Please install it.")
            sys.exit(1)
        except Exception as e:
            print(f"error: failed to load model '{self.model_name}': {e}")
            sys.exit(1)

        try:
            vocab = Vocabulary.from_path(
                self._model.get_path_to_vocab_file()
            )
            self._decoder = ChoiceDecoder(
                model=self._model,
                vocabulary=vocab,
            )
        except Exception as e:
            print(f"error: failed to load vocabulary: {e}")
            sys.exit(1)

    def run(
        self,
        prompts: list[PromptItem],
    ) -> list[FunctionCallResult]:
        """Run the pipeline on a list of prompts.

        Args:
            prompts: list of PromptItem to process.

        Returns:
            list of FunctionCallResult with name and parameters.
        """
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
            print(
                f"successfully processed prompt: {i}/{len(prompts)} "
                f"{item.prompt}\n"
            )

        print(
            f"\nTotal time elapsed: {(time.time() - start_time) / 60:.2f} "
            "minutes"
        )
        return results

    def _select_function(
        self,
        prompt: str,
    ) -> FunctionDefinition:
        """Select the best matching function for the prompt.

        Uses constrained decoding via ChoiceDecoder to guarantee
        the selected name is always one of the known function names.

        Args:
            prompt: natural language user request.

        Returns:
            The matching FunctionDefinition.
        """
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
        """Extract and validate all parameters for the selected function.

        Args:
            prompt: natural language user request.
            function: the selected FunctionDefinition.

        Returns:
            dict mapping parameter names to typed values.
        """
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
        """Convert raw candidate values to the correct Python type.

        Args:
            llm_candidates: raw values proposed by _llm_propose_values.
            param_name: name of the parameter being resolved.
            param_type: expected type (number, integer, boolean, string).

        Returns:
            list of typed candidate values, with safe fallbacks.
        """
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

    def _llm_propose_values(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict[str, list[Any]]:
        """Extract parameter values using constrained token-by-token decoding.

        At each step logits are masked so the model can only emit tokens
        that keep the output both syntactically valid JSON and
        schema-compliant:
        - number/integer fields: only digit tokens and terminators
        - string fields (first value): only tokens verbatim in the prompt
        - string fields (subsequent): any non-newline token
        - boolean fields: handled via string path (true/false in prompt)

        Args:
            prompt: natural language user request.
            function: the selected FunctionDefinition.

        Returns:
            dict mapping parameter names to lists of candidate values.
        """
        param_spec = ", ".join(
            f"{name}: {spec.type}"
            for name, spec in function.parameters.items()
        )

        llm_prompt = (
            "Extract parameter values from a request. "
            "Copy text exactly as it appears -- do not compute, solve, "
            "or transform anything. "
            "The parameter name may differ from the wording in the request"
            " -- extract by meaning, not by name.\n\n"

            "Function: fn_calculate_discount(price: number, percent: number)\n"
            "Request: Apply a discount of 15 percent to a price of 80\n"
            'JSON: {"price":[80],"percent":[15]}\n\n'

            "Function: fn_lookup_record(record_id: string, table: string)\n"
            "Request: Find the entry named 'Alpha-12' in table 'inventory'\n"
            'JSON: {"record_id":["Alpha-12"],"table":["inventory"]}\n\n'

            f"Function: {function.name}({param_spec}) "
            f"Description: {function.description}\n"
            f"Request: {prompt}\n"
            "JSON:"
        )

        arg_types = [
            spec.type for spec in function.parameters.values()
        ]

        ids = self._decoder._encode(llm_prompt)
        generated: list[int] = []
        pieces: list[str] = []
        chosen_text = ""
        max_tokens = 15 * len(function.parameters)
        terminators = {'"', ","}
        valids: list[int] = []
        count = 0

        for _ in range(max_tokens):
            logits = np.array(
                self._model.get_logits_from_input_ids(ids + generated),
                dtype=np.float64,
            )

            if ":" in chosen_text.strip():
                count += 1
                valids = self._allowed_tokens(arg_types, prompt, count)

            if valids:
                masked = np.full(
                    len(logits), float("-inf"), dtype=np.float64
                )
                for tid in valids:
                    if tid < len(logits):
                        masked[tid] = logits[tid]
            else:
                masked = logits

            next_id = int(np.argmax(masked))
            generated.append(next_id)
            chosen_text = self._model.decode([next_id])
            pieces.append(chosen_text)

            if any(ch in chosen_text.strip() for ch in terminators):
                valids = []

            if "}" in chosen_text:
                break

        raw = "".join(pieces)

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

    def _allowed_tokens(
        self,
        arg_types: list[str],
        prompt: str,
        count: int,
    ) -> list[int]:
        """Return valid token IDs for the current value position.

        Enforces schema compliance at the token level:
        - number/integer: only digit tokens and numeric terminators
        - string (count==1, strict): only tokens verbatim in the prompt
        - string (count>1, loose): any non-newline token

        Args:
            arg_types: ordered list of parameter types.
            prompt: original user prompt for verbatim string extraction.
            count: number of values generated so far (1-indexed).

        Returns:
            list of allowed token IDs for the current generation step.
        """
        valid: list[int] = []
        arg_type = arg_types[0] if arg_types else "string"
        strict_copy = (count == 1)
        terminate_number = {",", '"', "'", "}", "."}
        terminate_string = {",", '"', "'", "}"}

        for tid in list(self._decoder.vocabulary.ids):
            try:
                raw_token = self._model.decode([tid])
            except Exception:
                continue

            if raw_token.replace(
                "\u0120", " "
            ).replace("\u010a", "\n") == " ":
                if arg_type not in {"number", "integer"}:
                    valid.append(tid)
                continue

            text = (
                raw_token
                .replace("\u0120", " ")
                .replace("\u010a", "\n")
                .strip()
            )

            if not text:
                continue

            if arg_type in {"number", "integer"}:
                if text.isdigit():
                    valid.append(tid)
                if text.lstrip("-").isdigit():
                    valid.append(tid)
                if any(s in text for s in terminate_number):
                    valid.append(tid)
            else:
                if strict_copy:
                    if text and text in prompt:
                        if "\n" not in text and "\r" not in text:
                            valid.append(tid)
                    if any(s in text for s in terminate_string):
                        valid.append(tid)
                else:
                    if text and "\n" not in text and "\r" not in text:
                        valid.append(tid)

        return valid