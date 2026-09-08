SHELL := /bin/bash

.PHONY: setup play arena zip gate

setup:
	uv sync

play:
	uv run python -m harness.play --white current --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --agent current --opponent baselines/greedy

zip:
	uv run python -m harness.package --root current

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --agent current --opponent baselines/random --games 2 --base-ms 5000
	uv run python -m unittest \
	tests.test_package \
	tests.test_backtest \
	tests.test_referee \
	tests.test_agent \
	tests.test_pentanomial_sprt \
	tests.test_numba_search_scaling \
	tests.test_qs_eval_cache \
	tests.test_qs_lazy_profile \
	tests.test_qs_lazy_accumulator
