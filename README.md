*This project has been created as part of the 42 curriculum by masselgu.*

# Call Me Maybe — Introduction to Function Calling in LLMs

## Description

Call Me Maybe is a function calling tool that translates natural language prompts into structured function calls using a small 0.6B parameter language model (Qwen/Qwen3-0.6B).

Given a natural language request like:
```
"What is the sum of 2 and 3?"
```

The system does not answer `5`. Instead it produces a structured function call:
```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The key challenge is that small language models are notoriously unreliable at generating structured output when prompted freely — succeeding only ~30% of the time. This project solves that problem using **constrained decoding**: a technique that guides the model token-by-token, masking invalid tokens to negative infinity at each step, guaranteeing 100% valid and schema-compliant output.

---

## Instructions

### Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone <your-repo-url>
cd Call_me_maybe
cp -r /path/to/llm_sdk .   # copy the provided llm_sdk directory here
uv sync
```

### Running the program

```bash
uv run python -m src
```

With custom paths:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Makefile targets

```bash
make install   # install dependencies
make run       # run the program
make debug     # run with pdb debugger
make clean     # remove __pycache__, .mypy_cache
make lint      # run flake8 and mypy
```

---

## Example Usage

### Input files

**`data/input/functions_definition.json`**
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {"type": "string"}
    },
    "returns": {"type": "string"}
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string and return the reversed result.",
    "parameters": {
      "s": {"type": "string"}
    },
    "returns": {"type": "string"}
  }
]
```

**`data/input/function_calling_tests.json`**
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet shrek"},
  {"prompt": "Reverse the string 'hello'"}
]
```

### Output

**`data/output/function_calling_results.json`**
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

---

## Algorithm Explanation

The pipeline has two main stages:

### 1. Function Selection — `_select_function()`

Uses `ChoiceDecoder.choose()` which performs constrained token-by-token decoding over the set of known function names. At each generation step, logits are masked to `-inf` for all tokens that are not valid continuations of any known function name. This guarantees the selected name is always one of the available functions — never a hallucinated string.

The prompt includes each function's parameter count and description, so the model can distinguish between functions that differ only by arity (e.g. `fn_add_numbers` taking 2 parameters vs `fn_add_three_numbers` taking 3).

### 2. Parameter Extraction — `_llm_propose_values()` + `_select_parameters()`

This is where the core constrained decoding for JSON happens.

**Step 1 — Constrained JSON generation (`_llm_propose_values`):**

The model generates a JSON string token by token. After each `:` character (signaling the start of a value), `_allowed_tokens()` computes which token IDs are valid for the current parameter type:

- `number` / `integer` → only digit tokens, minus sign, and JSON terminators (`,`, `"`, `}`, `.`)
- `string` (first value, strict mode) → only tokens whose text appears verbatim in the original prompt, plus terminators. This prevents the model from inventing or transforming string values.
- `string` (subsequent values, loose mode) → any non-newline token, plus terminators
- `boolean` → handled via the string path (`true`/`false` appear in the prompt)

All other tokens are masked to `-inf` before `argmax` selection. This means the model **cannot physically generate** a token that violates the schema — it is mathematically impossible.

**Step 2 — Final constrained pick (`_select_parameters`):**

The candidates extracted in Step 1 are passed to `_parameter_candidates()` which converts them to the correct Python type (float, int, bool, str). These typed candidates are then passed to `ChoiceDecoder.choose()` which does a final constrained selection — again using masked logits — to pick the best value for each parameter. This double-constrained approach guarantees both valid types and contextually correct values.

### Why this guarantees 100% valid JSON

- Step 1 constrains every token during generation → structurally valid JSON
- Step 2 assembles the final output from typed Python objects via `json.dumps()` / `json.dump()` → always parseable
- The `except Exception` fallback in `_llm_propose_values` returns empty typed defaults rather than crashing, so `_select_parameters` always has something to work with

---

## Design Decisions

**Why `ChoiceDecoder.choose()` for function selection?**
Function names are a closed, finite set known at runtime. Constraining to exact token sequences of known names is more reliable than open-ended generation. The decoder tracks which tokens are valid continuations of each candidate name at each position, making wrong selections impossible.

**Why few-shot examples in `_llm_propose_values`?**
The 0.6B model needs context to understand the extraction task. Two generic, domain-unrelated examples (discount calculation, record lookup) teach the model the pattern — extract values as-is, map by meaning not by name — without hardcoding any test-specific knowledge.

**Why `max_tokens = 15 * len(function.parameters)`?**
A fixed token budget of 30 was insufficient for functions with 4-5 parameters. Scaling by parameter count ensures the model has enough budget to generate a complete JSON object regardless of function complexity.

**Why keep `_select_parameters` on top of `_llm_propose_values`?**
`_llm_propose_values` generates candidates under structural constraints. `_select_parameters` adds a second layer of semantic constraint via `choose()` — picking the contextually best value when multiple candidates are available. This two-step approach improves accuracy beyond what either step achieves alone.

**Why `extra="forbid"` in `FunctionCallResult`?**
Pydantic's `extra="forbid"` ensures no stray keys can appear in the output, enforcing the schema requirement that each output object contains exactly `prompt`, `name`, and `parameters`.

---

## Performance Analysis

Tested on the subject's canonical 11-prompt test set (Qwen/Qwen3-0.6B, CPU):

| Metric | Result |
|---|---|
| Function selection accuracy | 11/11 (100%) |
| Parameter extraction accuracy | 11/11 (100%) |
| Output JSON validity | 11/11 (100%) |
| Total runtime | ~3.5 minutes for 11 prompts |
| Per-prompt average | ~19 seconds |

The main performance cost is `_allowed_tokens()`, which iterates over the full vocabulary (~150,000 tokens) and decodes each one individually to determine validity. This runs multiple times per prompt. Future optimization could precompute the number/string token sets once at startup in `model_post_init`, reducing per-prompt cost significantly.

The system stays well within the subject's 5-minute requirement and is deterministic (greedy `argmax` decoding), meaning results are identical across runs.

---

## Challenges Faced

**Challenge 1: Function selection ambiguity between same-arity functions**

Early versions failed to distinguish `fn_add_numbers` (2 params) from `fn_add_three_numbers` (3 params). The fix was enriching the function list prompt with explicit `[requires exactly N parameter(s)]` labels and a counting instruction, giving the model a direct signal to match arity.

**Challenge 2: `_allowed_tokens` was too slow**

Iterating over 150,000 tokens and decoding each one per generation step was extremely slow. The solution was to use `self._decoder.vocabulary.ids` (the pre-filtered valid token set from the vocabulary file) rather than iterating over all possible token IDs, reducing the iteration space significantly.

**Challenge 3: Space tokens stripped during string extraction**

BPE tokenizers encode spaces as unicode prefix `\u0120`. After replacing and stripping, pure space tokens became empty strings and were silently dropped, causing extracted strings like `"Hello 34 I'm 233 years old"` to lose their spaces. The fix was to explicitly allow pure space tokens before the strip operation, checking for `" "` before stripping.

**Challenge 4: Token budget too small for multi-parameter functions**

A fixed `max_tokens=30` was insufficient for functions with 5 parameters. The extraction would get cut off mid-JSON, causing `json.loads()` to fail and the fallback to return empty values. Scaling to `15 * len(function.parameters)` resolved this entirely.

**Challenge 5: Two-step count-then-select approach was unreliable**

An early version tried to first ask the model "how many values does this request mention?" then use that count to select the function. The 0.6B model always returned `1` or `3` regardless of the actual count — it was anchoring on the last few-shot example rather than reasoning. The fix was to drop the count step entirely and rely on a single, well-crafted selection prompt with explicit parameter count labels.

---

## Testing Strategy

**Step 1 — Subject's canonical test set (11 prompts)**
Used the exact examples from the subject's documentation (V.2 Input Files) as the primary correctness check. These cover: addition, greeting, string reversal, square root, and regex substitution with 1-3 parameters.

**Step 2 — Multi-parameter stress test (15 prompts)**
Created a custom test with functions ranging from 1 to 5 parameters, mixed types (string + integer + boolean in the same function), and pairs of functions sharing the same arity to stress-test selection accuracy. Achieved 15/15.

**Step 3 — Adversarial edge cases (25 prompts)**
Tested deliberately hard cases: negatives (`-50`), floats (`12.5`), accented characters (`García`), ambiguous semantics (`"Greet"` vs `"Farewell"` vs `"Welcome"`), no-value prompts, and gibberish input. Identified known limitations (out-of-domain prompts with no matching function) which are outside the subject's stated scope.

**Step 4 — Error handling**
Manually tested: missing input files, malformed JSON in input, empty function definitions, empty prompt lists. All handled gracefully with clear error messages and no crashes.

---

## Resources

### Technical References

- [Hugging Face — Constrained Beam Search](https://huggingface.co/blog/constrained-beam-search) — Introduction to constrained decoding concepts
- [Outlines library](https://github.com/outlines-dev/outlines) — Reference implementation of structured generation (forbidden to use, but useful to understand the concept)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B) — Model documentation
- [Pydantic documentation](https://docs.pydantic.dev/) — Used for all data validation
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259) — Reference for JSON validity rules
- [BPE Tokenization explained](https://huggingface.co/learn/nlp-course/chapter6/5) — Understanding how tokens map to text, including space prefixes

### AI Usage

AI (Claude, Anthropic) was used in the following parts of this project:

- **Debugging** — identifying why the count-step approach always returned `1`, tracing the space-token stripping bug, diagnosing token budget issues for multi-parameter functions
- **Prompt engineering** — iterating on few-shot examples for `_llm_propose_values` to teach verbatim extraction and parameter-name-to-meaning mapping
- **Code review** — reviewing `_allowed_tokens` logic and suggesting the explicit space token handling fix
- **Test design** — generating diverse test cases covering edge cases beyond the subject's examples
- **README** — this document was drafted with AI assistance and reviewed for accuracy

All AI-generated content was reviewed, tested, and validated before use. The core algorithm design, architecture decisions, and debugging reasoning were developed through understanding the subject requirements and iterative testing.