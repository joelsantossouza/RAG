MODEL_LINK := https://huggingface.co/enacimie/Qwen3-Embedding-0.6B-Q4_K_M-GGUF/resolve/main/qwen3-embedding-0.6b-q4_k_m.gguf

install:
	pip install uv
	uv sync
	wget $(MODEL_LINK)

run:

debug:

clean:
	find . -name "*__pycache__*" -exec rm -rf {} \;
	find . -name "*.mypy_cache*" -exec rm -rf {} \;
	find . -name "*.pyc*" -exec rm -rf {} \;

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict
