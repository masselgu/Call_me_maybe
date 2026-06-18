"""Command-line entry point for the project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.io_utils import load_functions, load_prompts, write_results
from src.pipeline import FunctionCallingPipeline


DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Translate prompts into schema-valid function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path where generated function calls will be written.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model name passed to the provided SDK.",
    )
    return parser


def main() -> int:
    """Run the CLI and return a process exit code."""
    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args()
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        pipeline = FunctionCallingPipeline(
            functions=functions,
            model_name=args.model,
        )
        results = pipeline.run(prompts)
        write_results(args.output, results)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
