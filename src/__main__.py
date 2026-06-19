"""CLI entry point."""

from __future__ import annotations

import sys

from src.parser import load_functions, load_prompts, parse_args, write_results
from src.pipeline import FunctionCallingPipeline


def main() -> int:
    """Run the function-calling CLI."""
    try:
        args = parse_args()
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        results = FunctionCallingPipeline(
            functions=functions,
            model_name=args.model,
        ).run(prompts)
        write_results(args.output, results)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
