"""CLI and JSON parsing helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.models import FunctionCallResult
from src.models import FunctionDefinition
from src.models import PromptItem

ModelT = TypeVar("ModelT", bound=BaseModel)

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


class CliArguments(BaseModel):
    """Validated command-line arguments."""

    functions_definition: Path
    input: Path
    output: Path
    model: str = "Qwen/Qwen3-0.6B"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Translate prompts into schema-valid function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    return parser


def parse_args(argv: list[str] | None = None) -> CliArguments:
    """Parse and validate CLI arguments."""
    namespace = build_parser().parse_args(argv)
    return CliArguments.model_validate(vars(namespace))


def read_json(path: Path) -> Any:
    """Read JSON from disk with clear errors."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: line {exc.lineno}") from exc


def load_list(path: Path, model: type[ModelT], label: str) -> list[ModelT]:
    """Load and validate a JSON array."""
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{label} must be a JSON array")
    try:
        return [model.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load available function definitions."""
    functions = load_list(path, FunctionDefinition, "functions")
    if not functions:
        raise ValueError("function definitions cannot be empty")
    return functions


def load_prompts(path: Path) -> list[PromptItem]:
    """Load input prompts."""
    return load_list(path, PromptItem, "prompts")


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write output JSON results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.model_dump(mode="json") for item in results]
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
