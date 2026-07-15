import json
import sys
from pathlib import Path
from typing import Any
from pydantic import BaseModel, ValidationError
from src.models import FunctionCallResult, FunctionDefinition, PromptItem


DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class CliArguments(BaseModel):
    """Validated command-line arguments."""
    functions_definition: Path
    input: Path
    output: Path
    model: str


def parse_args() -> CliArguments:
    """Parse CLI args from sys.argv."""
    args = sys.argv[1:]

    result: dict[str, Any] = {
        "functions_definition": DEFAULT_FUNCTIONS,
        "input": DEFAULT_INPUT,
        "output": DEFAULT_OUTPUT,
        "model": DEFAULT_MODEL,
    }

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "--functions_definition":
            i += 1
            result["functions_definition"] = Path(args[i])
        elif arg == "--input":
            i += 1
            result["input"] = Path(args[i])
        elif arg == "--output":
            i += 1
            result["output"] = Path(args[i])
        elif arg == "--model":
            i += 1
            result["model"] = args[i]
        else:
            raise ValueError(f"unknown argument: {arg}")

        i += 1

    return CliArguments(**result)


def read_json(path: Path) -> Any:
    """Read JSON from disk with clear errors."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise ValueError(f"missing file: {path}")
    except PermissionError:
        raise ValueError(f"permission denied: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in {path}: line {e.lineno}")


def load_list(path: Path, model: Any, label: str) -> list[Any]:
    """Load and validate a JSON array."""
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{label} must be a JSON array")
    try:
        return [model(**item) for item in data]
    except ValidationError as e:
        error = e.errors()[0]
        field = ".".join(str(x) for x in error["loc"])
        msg = error["msg"]
        raise ValueError(f"in {label}: {field} -> {msg}")


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load available function definitions."""
    functions = load_list(path, FunctionDefinition, "functions")
    if not functions:
        raise ValueError("function definitions cannot be empty")
    return functions


def load_prompts(path: Path) -> list[PromptItem]:
    """Load input prompts."""
    prompts = load_list(path, PromptItem, "prompts")
    if not prompts:
        raise ValueError("prompts cannot be empty")
    return prompts


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write output JSON results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [item.model_dump(mode="json") for item in results]
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
