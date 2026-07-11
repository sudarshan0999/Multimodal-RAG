# Contributing

Thanks for helping improve this project.

## Workflow

1. Fork the repository and create a feature branch.
2. Make focused changes; match existing style (types, imports, minimal comments).
3. Run tests: `pytest tests/ -q`.
4. Open a pull request with a clear description of behavior changes.

## Code style

- Python 3.10+ type hints where they clarify public APIs.
- Prefer small modules aligned with `providers/`, `chunking/`, `ingestion/`, etc.
- Do **not** change the core algorithm in [`ingestion/pdf_images.py`](ingestion/pdf_images.py) unless fixing a documented bug; keep extraction logic stable.

## Environment

- Copy `.env.example` to `.env` for local keys; never commit secrets.
- Use a virtual environment and `pip install -r requirements.txt`.

## Issues

When reporting bugs, include OS, Python version, provider used, and minimal steps to reproduce (sample PDF if possible).
