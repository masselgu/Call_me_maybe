UV_CACHE_DIR = $(HOME)/goinfre/.uv_cache
HF_HOME = $(HOME)/goinfre/.hf_cache

export UV_CACHE_DIR HF_HOME

install:
	@mkdir -p $(UV_CACHE_DIR) $(HF_HOME)
	@uv sync

run:
	@uv run python -m src

debug:
	@uv run python -m pdb -m src

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .mypy_cache

lint:
	@uv run flake8 src
	@uv run mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
