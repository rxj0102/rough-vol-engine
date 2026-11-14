# make validate   run the numerical validation checks and re-render the README Validation table
# make check      verify the README Validation table matches results/ (fast, used in CI)
# make test       run the unit tests
.PHONY: setup validate check test

setup:
	uv sync

validate:
	uv run python scripts/validate.py

check:
	uv run python scripts/validate.py --check

test:
	MPLBACKEND=Agg uv run pytest -q
