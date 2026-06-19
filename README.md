*This project has been created as part of the 42 curriculum by masselgu.*

## Description

`call_me_maybe` translates natural-language requests into structured function calls.
Given a prompt such as `What is the sum of 2 and 3?`, it writes a JSON object with
the original prompt, the selected function name, and schema-valid parameters.

The project uses the provided `llm_sdk.Small_LLM_Model` wrapper with
`Qwen/Qwen3-0.6B` by default. The LLM is used to choose the function and the
argument values. JSON validity is guaranteed by constraining decoding to known
valid completions instead of asking the model to freely write JSON.

## Instructions

Install the project dependencies:

```sh
uv sync
```

Run with the default input files:

```sh
uv run python -m src
```

Run with explicit paths:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Useful Makefile commands:

```sh
make install
make run
make lint
make clean
```

The generated output is written to `data/output/function_calling_results.json` by
default. The `data/output/` directory is ignored by git because it is generated
during execution.

## Algorithm Explanation

The solution uses finite-choice constrained decoding. For every decision, the
program builds the full set of valid completions before calling the model.

For function selection, valid completions are the function names from
`functions_definition.json`. Each name is tokenized with `encode()`. The
vocabulary JSON loaded from the SDK maps token ids to token strings. At each
generation step the decoder asks the model for logits, keeps only tokens that
can continue a valid function name, sets all other logits to negative infinity,
and selects the highest remaining score.

For parameters, schema-compatible candidates are built from the prompt and
tokenized with `encode()`. The same vocabulary JSON is used to validate token
ids during token-by-token generation. Invalid tokens are masked before selection,
so the model can only produce one valid JSON literal for the parameter.

This means the model never freely emits the final JSON file. It can only choose
among values that are already valid for the expected schema, so the output remains
parseable and contains exactly the required keys.

## Design Decisions

All input and output structures are Pydantic models. This keeps invalid files,
missing fields, unsupported parameter types, and extra output keys from silently
passing through the pipeline.

The LLM interaction is wrapped in a small `LlmClient` class. It loads the model
vocabulary JSON through the SDK, then uses `encode`, `decode`, and
`get_logits_from_input_ids` without accessing private SDK attributes.

The implementation favors a compact candidate set instead of unconstrained JSON
generation. This is faster and more reliable for a small model, while still letting
the model make the function and argument choices.

## Performance Analysis

The decoder performs one model forward pass per generated token. Function names
and JSON literals are short, so each prompt usually needs only a small number of
LLM calls. The model is loaded once and reused for all prompts.

Reliability comes from the constrained output space: invalid JSON, unknown
function names, missing parameters, and wrong primitive types cannot be emitted by
the writer. Accuracy depends on the model logits and the quality of the candidate
values extracted from the prompt. The included candidate builder covers common
cases such as arithmetic numbers, greetings, string reversal, regex replacement,
and quoted text.

## Challenges Faced

Small LLMs are not reliable when asked to produce raw JSON. The main challenge was
separating semantic choice from structural correctness. The solution lets the model
rank only valid options, while Python handles schema validation and final JSON
serialization.

Another challenge is argument extraction for arbitrary function definitions. The
candidate builder remains generic by using parameter types and prompt evidence
rather than hardcoding exact test rows. It also includes common natural-language
hints, such as `numbers` becoming `\d+` and `asterisks` becoming `*`.

## Testing Strategy

Recommended checks:

```sh
make lint
uv run python -m src
python -m json.tool data/output/function_calling_results.json
```

Manual tests should include missing files, malformed JSON, empty function lists,
prompts with multiple numbers, quoted strings, unquoted names, booleans, and
ambiguous replacement requests.

## Resources

- Python documentation: <https://docs.python.org/3/>
- Pydantic documentation: <https://docs.pydantic.dev/>
- uv documentation: <https://docs.astral.sh/uv/>
- JSON specification: <https://www.json.org/json-en.html>
- Hugging Face Transformers documentation:
  <https://huggingface.co/docs/transformers>

AI was used as a development assistant to structure the project, draft the
constrained-decoding implementation, and write this documentation. The code was
kept small and reviewed around the subject requirements so it can be explained and
modified during peer evaluation.
