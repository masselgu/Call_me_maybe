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

    def run(self, prompts: list[PromptItem]) -> list[FunctionCallResult]:
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

    def _select_function(self, prompt: str) -> FunctionDefinition:
        choices = [fn.name for fn in self.functions]

        function_list_text = (
            "You are an AI function calling assistant.\n\n"
            "Your task is to choose the BEST function for the user request.\n"
            "Carefully count how many distinct values the request provides, "
            "and match it to the function with the same number of parameters."
            "\nOnly answer with the exact function name. Do not explain "
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

        context_text = (function_list_text + f"\nRequest: {prompt}\nFunction:")
        selected = self._decoder.choose(context_text, choices)

        for fn in self.functions:
            if fn.name == selected:
                return fn

        raise ValueError("function not found")

    def _select_parameters(self, prompt: str,
                           function: FunctionDefinition) -> dict[str, Any]:
        llm_candidates = self._llm_propose_values(prompt, function)
        parameters: dict[str, Any] = {}

        for name, spec in function.parameters.items():
            value = self._parameter_candidates(llm_candidates, name, spec.type)
            parameters[name] = value

        return parameters

    def _parameter_candidates(self, llm_candidates: dict[str, Any],
                              param_name: str, param_type: str) -> Any:
        value = llm_candidates.get(param_name)

        if value is not None:
            try:
                if param_type == "integer":
                    return int(value)
                elif param_type == "number":
                    return float(value)
                elif param_type == "boolean":
                    if isinstance(value, bool):
                        return value
                    if str(value).lower() == "true":
                        return True
                    if str(value).lower() == "false":
                        return False
                else:
                    return str(value)
            except (ValueError, TypeError):
                pass

        if param_type == "integer":
            return 0
        if param_type == "number":
            return 0.0
        return ""

    def _llm_propose_values(self, prompt: str,
                            function: FunctionDefinition) -> dict[str, Any]:
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

            "Function: fn_regex_replace(text: string, regex: string, sub: string)\n"
            'Request: Substitute the word "cat" with "dog" in "The cat sat on the mat with another cat"\n'
            'JSON: {"text":"The cat sat on the mat with another cat","regex":"\\\\bcat\\\\b","sub":"dog"}\n\n'

            "Function: fn_lookup_record(record_id: string, table: string)\n"
            "Request: Find the entry named 'Alpha-12' in table 'inventory'\n"
            'JSON: {"record_id":"Alpha-12","table":"inventory"}\n\n'

            f"Function: {function.name}({param_spec}) "
            f"Description: {function.description}\n"
            f"Request: {prompt}\n"
            "JSON:"
        )

        arg_types = [spec.type for spec in function.parameters.values()]

        token_cache: dict[str, list[int]] = {}
        for arg_type in arg_types:
            if arg_type not in token_cache:
                token_cache[arg_type] = self._get_valid_tokens(arg_type)

        ids = self._decoder._encode(llm_prompt)
        generated: list[int] = []
        pieces: list[str] = []
        chosen_text = ""
        max_tokens = 15 * len(function.parameters)
        valids: list[int] = []
        param_index = 0

        for _ in range(max_tokens):
            logits = np.array(
                self._model.get_logits_from_input_ids(ids + generated),
                dtype=np.float64,
            )

            if ":" in chosen_text.strip():
                arg_type = (arg_types[param_index]
                            if param_index < len(arg_types) else "string")
                valids = token_cache.get(arg_type, [])

            if valids:
                masked = np.full(len(logits), float("-inf"), dtype=np.float64)
                for tid in valids:
                    if tid < len(logits):
                        masked[tid] = logits[tid]
            else:
                masked = logits

            next_id = int(np.argmax(masked))
            generated.append(next_id)
            chosen_text = self._model.decode([next_id])
            pieces.append(chosen_text)

            if "," in chosen_text.strip():
                valids = []
                param_index += 1

            if "}" in chosen_text:
                break

        raw = "".join(pieces)

        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
            return {
                name: parsed.get(name)
                for name in function.parameters
            }
        except Exception:
            return {
                name: None
                for name in function.parameters
            }

    def _get_valid_tokens(self, arg_type: str) -> list[int]:
        valid: list[int] = []
        terminate_string: set[str] = {",", '"', "}"}
        numeric_chars: set[str] = {",", '"', "}", ".", "-"}

        for tid in list(self._decoder.vocabulary.ids):
            try:
                raw_token = self._model.decode([tid])
            except Exception:
                continue

            if raw_token.replace("\u0120", " ").replace("\u010a", "\n") == " ":
                if arg_type not in {"number", "integer"}:
                    valid.append(tid)
                continue

            text = (raw_token.replace("\u0120", " ").replace("\u010a", "\n")
                    .strip())

            if not text:
                continue

            if arg_type in {"number", "integer"}:
                if text.isdigit():
                    valid.append(tid)
                if text.lstrip("-").replace(".", "", 1).isdigit():
                    valid.append(tid)
                if text == "-":
                    valid.append(tid)
                if any(s in text for s in numeric_chars):
                    valid.append(tid)
            else:
                if any(s in text for s in terminate_string):
                    valid.append(tid)
                elif text and "\n" not in text and "\r" not in text:
                    valid.append(tid)

        return valid
