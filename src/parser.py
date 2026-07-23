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
    """Validated command-line arguments.

    Attributes:
        functions_definition: Path to functions definition JSON.
        input: Path to input prompts JSON.
        output: Path to write results JSON.
        model: HuggingFace model name to use.
    """
    functions_definition: Path
    input: Path
    output: Path
    model: str


def parse_args() -> CliArguments:
    """Parse CLI arguments from sys.argv.

    Returns:
        Validated CliArguments with defaults applied.

    Raises:
        ValueError: If an unknown argument is provided.
    """
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


def _check_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject JSON objects with duplicate keys.

    Args:
        pairs: List of key-value pairs from JSON parser.

    Returns:
        Dict of key-value pairs.

    Raises:
        ValueError: If any key appears more than once.
    """
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"duplicate key '{key}' in JSON object")
        seen.add(key)
    return dict(pairs)


def read_json(path: Path) -> Any:
    """Read and parse a JSON file from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        ValueError: If file is missing, unreadable, invalid JSON,
            or contains duplicate keys.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(
                file,
                object_pairs_hook=_check_duplicate_keys
            )
    except FileNotFoundError:
        raise ValueError(f"missing file: {path}")
    except PermissionError:
        raise ValueError(f"permission denied: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in {path}: line {e.lineno}")


def load_list(path: Path, model: Any, label: str) -> list[Any]:
    """Load and validate a JSON array into pydantic models.

    Args:
        path: Path to the JSON file.
        model: Pydantic model class to validate each item.
        label: Human-readable label for error messages.

    Returns:
        List of validated model instances.

    Raises:
        ValueError: If file is not an array or validation fails.
    """
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

def check_duplicate_function_names(
    functions: list[FunctionDefinition],
) -> None:
    """Reject duplicate function names.

    Args:
        functions: Validated function definitions.

    Raises:
        ValueError: If two functions share the same name.
    """
    seen: set[str] = set()

    for function in functions:
        if function.name in seen:
            raise ValueError(
                f"duplicate function name '{function.name}'"
            )
        seen.add(function.name)

def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to functions_definition.json.

    Returns:
        Non-empty list of FunctionDefinition instances.

    Raises:
        ValueError: If file is invalid or empty.
    """
    functions = load_list(path, FunctionDefinition, "functions")
    if not functions:
        raise ValueError("function definitions cannot be empty")
    
    check_duplicate_function_names(functions)
    
    return functions


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate input prompts from a JSON file.

    Args:
        path: Path to function_calling_tests.json.

    Returns:
        Non-empty list of PromptItem instances.

    Raises:
        ValueError: If file is invalid or empty.
    """
    prompts = load_list(path, PromptItem, "prompts")
    if not prompts:
        raise ValueError("prompts cannot be empty")
    return prompts


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write function call results to a JSON file.

    Args:
        path: Output file path.
        results: List of FunctionCallResult to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [item.model_dump(mode="json") for item in results]
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
