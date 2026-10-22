"""Run after `uv sync --locked --no-editable` to use the installed package."""

from compactfacts.data.public_benchmarks import main

if __name__ == "__main__":
    raise SystemExit(main())
