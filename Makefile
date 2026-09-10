SHELL := /bin/bash
OUT ?= submission.zip

.PHONY: setup play arena zip zip-challenger gate

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

zip:
	uv run python -m harness.package

# package zips its working directory, so this must run inside challenger to include weights/
zip-challenger:
	cd challengers/$(CHALLENGER) && PYTHONPATH=$(CURDIR) uv run python -m harness.package --out $(CURDIR)/$(OUT)

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
