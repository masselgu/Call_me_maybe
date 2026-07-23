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
- UV package manager

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

> [!NOTE]
> **Function names and prompts should be clear and unambiguous.**
>
> - Prefer unique, descriptive function names that are easily distinguishable from one another.
> - Avoid naming patterns where one function name is simply the beginning (prefix) of another, as this can make constrained decoding less reliable.
> - Write prompts that clearly describe the intended operation to minimize ambiguity during function selection and parameter extraction.

---

## Algorithm Explanation

### 1. Function Selection — `_select_function()`

Uses `ChoiceDecoder.choose_function()` which performs constrained token-by-token
decoding over known function names. At each step, logits are masked to
`-inf` for all tokens that don't continue a valid function name —
making wrong selections impossible.

### 2. Parameter Extraction — `_llm_propose_values()`

The model generates a JSON string token by token. After each `:`,
`_get_valid_tokens()` restricts which tokens are allowed based on the
parameter type:

- `number` / `integer` → only digits, minus, dot, and terminators
- `string` → any printable character (`ord >= 32`)
- `boolean` → only `true`, `false`, and terminators

All other tokens are masked to `-inf` — the model cannot physically
generate a schema-violating token.

### Why this guarantees 100% valid JSON

- Every token during generation is constrained → structurally valid
- Final output is assembled from typed Python objects → always parseable
- Fallback returns typed defaults on parse failure → never crashes

---

## Design Decisions

- **`ChoiceDecoder` for function names** — function names are a closed
  finite set, constrained decoding over exact token sequences is more
  reliable than open-ended generation
- **Few-shot examples** — two generic domain-unrelated examples teach
  the model to extract values verbatim without hardcoding test-specific
  knowledge
- **`max_tokens = 20 * len(parameters)`** — scales token budget with
  function complexity, prevents cutoff for multi-parameter functions
- **`extra="forbid"`** — ensures output contains exactly `prompt`,
  `name`, `parameters` — no extra keys
---

## Performance Analysis

Tested on the subject's canonical 11-prompt test set (Qwen/Qwen3-0.6B, CPU):

| Metric | Result |
|---|---|
| Function selection accuracy | 11/11 (100%) |
| Parameter extraction accuracy | 11/11 (100%) |
| Output JSON validity | 11/11 (100%) |
| Total runtime | ~3.0 minutes for 11 prompts |
| Per-prompt average | ~17 seconds |

Results are deterministic and identical across runs. The pipeline
never crashes — fallbacks return typed defaults on parse failure.

---

## Challenges Faced

**Function selection ambiguity between same-arity functions**
Early versions failed to distinguish `fn_add_numbers` (2 params) from
`fn_add_three_numbers` (3 params). Fixed by adding explicit
`[requires exactly N parameter(s)]` labels to the prompt.

**Debugging token-level failures**
When output was wrong, the cause was invisible from the final JSON.
Adding raw generation prints to trace token-by-token output was
essential to understand what the model was actually generating.

---

## Testing Strategy

**Subject canonical test set (11 prompts)**
Used the exact examples from the subject's V.2 documentation as the
primary correctness check. Validated with the provided moulinette —
achieved 11/11 (100%) on the public set.

**Multi-parameter stress test (15 prompts)**
Custom test with functions from 1 to 5 parameters, mixed types, and
same-arity function pairs. Achieved 15/15.

**Error handling**
Manually tested: missing files, malformed JSON, empty function
definitions, empty prompts. All handled gracefully with clear error
messages and no crashes.
---

## Resources

### Technical References

- [Aidan Cooper — Constrained Decoding](https://www.aidancooper.co.uk/constrained-decoding/) — Practical guide explaining how constrained decoding ensures LLM outputs follow predefined formats such as JSON schemas and grammars.
- [42 School Project Subject](https://cdn.intra.42.fr/pdf/pdf/208528/en.subject.pdf) — Official assignment specification detailing project objectives, requirements, and expected deliverables.
- [Hugging Face — Function Calling Guide](https://huggingface.co/docs/hugs/en/guides/function-calling) — Documentation on enabling LLMs to call tools and APIs through structured function definitions and JSON arguments.
- [GeeksforGeeks — Getting Started with Transformers](https://www.geeksforgeeks.org/machine-learning/getting-started-with-transformers/) — Beginner-friendly introduction to Transformer architecture, self-attention, encoders, decoders, and modern NLP models.

### AI Usage

AI  was used in the following parts of this project:

- **Test design** — generating diverse test cases covering edge cases beyond the subject's examples
- **README** — AI was used to suggest improvements to the README's structure, wording, and readability.
- **Learning support** — clarifying technical documentation and summarizing external resources to accelerate understanding of the subject.
- **Project planning** — helping organize tasks, milestones, and implementation priorities before development.