"""Input and output helpers for JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.models import FunctionCallResult, FunctionDefinition, PromptItem


ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_json(path: Path) -> Any:
    """Read JSON from a file with graceful error messages."""
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise ValueError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        message: str = f"invalid JSON in {path}: line {exc.lineno}"
        raise ValueError(message) from exc
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc


def _parse_model_list(
    data: Any,
    model: type[ModelT],
    label: str,
) -> list[ModelT]:
    """Validate a JSON array against a Pydantic model."""
    if not isinstance(data, list):
        raise ValueError(f"{label} must contain a JSON array")
    try:
        return [model.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def load_functions(path: Path) -> list[FunctionDefinition]:
    """Load and validate available function definitions."""
    functions: list[FunctionDefinition]
    functions = _parse_model_list(
        _read_json(path),
        FunctionDefinition,
        "function definitions",
    )
    if not functions:
        raise ValueError("function definitions cannot be empty")
    return functions


def load_prompts(path: Path) -> list[PromptItem]:
    """Load and validate input prompts."""
    return _parse_model_list(_read_json(path), PromptItem, "input prompts")


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write result objects as strict JSON."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: list[dict[str, Any]]
        payload = [item.model_dump(mode="json") for item in results]
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except OSError as exc:
        raise ValueError(f"cannot write {path}: {exc}") from exc
