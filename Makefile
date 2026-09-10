SHELL := /bin/bash
# the measured deliverable, not the stale root copy and not a teammate's current/
AGENT ?= challengers/v9_kingnet
OUT ?= submission.zip

.PHONY: setup play arena zip gate

setup:
	uv sync

play:
	uv run python -m harness.play --white $(AGENT) --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --agent $(AGENT) --opponent baselines/greedy

zip:
	uv run python -m harness.package --root $(AGENT) --out $(OUT)

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --agent $(AGENT) --opponent baselines/random --games 2 --base-ms 5000
