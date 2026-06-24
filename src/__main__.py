"""CLI entry point."""

import sys
from src.parser import load_functions, load_prompts, parse_args, write_results
from src.pipeline import FunctionCallingPipeline


def main() -> None:
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
    except Exception as e:
        print(f"error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
