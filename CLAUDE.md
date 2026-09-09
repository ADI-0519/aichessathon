# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An entry for AI Chessathon, a chess-engine competition. The deliverable is a zip whose root
`agent.py` exposes `get_move(fen: str, time_left_ms: int) -> str` (UCI). The platform imports it
and calls that function once per move, in a process that stays alive for one game. The **canonical**
rules and agent contract live at https://aichessathon.com/docs and change; the local copy at
[AGENTS.md](AGENTS.md) is a convenience summary — fetch the live docs before relying on a number
(time controls, upload limits, memory caps). Read `AGENTS.md` in full before touching anything
related to the platform contract; it also covers what is prohibited to ship (borrowed engines/nets,
move-lookup tables) versus what's allowed (opening books, tablebases, engine-labelled training data).

## Commands

```bash
uv sync                                              # install/sync the environment

uv run python -m unittest discover -s tests          # full test suite
uv run python -m unittest tests.test_engine_numba     # one test module
uv run python -m unittest tests.test_engine_numba.NumbaBoardTests.test_en_passant_reference_perft  # one test

uv run ruff check .                                  # lint (line-length 100, py312)
uv run mypy                                           # strict type check (scoped to files in pyproject.toml)

make play FEN="<fen>"                                 # one real-clock game vs baselines/greedy
uv run python -m harness.play --white . --black baselines/minimax --pgn game.pgn
make arena                                            # 20 fast games vs baselines/greedy, scored
make zip                                              # build submission.zip (agent.py at the root)
make gate                                             # ruff + mypy + 2 quick games; CI runs this
```

`ruff`/`mypy` cover `agent.py`, `engine.py`, `search.py`, and `harness/` (see `[tool.mypy] files` in
`pyproject.toml`) — `champions/`, `challengers/`, `tools/`, and `docs/` scripts are not gated by
`make gate` and should be checked manually (`uv run ruff check champions/ challengers/ tools/`).

### Judging a change honestly

`make arena` (20 games from the start position) cannot resolve a change worth less than ~150 Elo.
Use `tools/paired_arena.py` instead — it plays color-swapped games from varied openings/positions
and reports a score with a 95% confidence interval, an Elo estimate, and an explicit verdict
(`stronger` / `weaker` / `NOT RESOLVED at this sample size`):

```bash
uv run python -m tools.paired_arena --candidate . --opponent champions/python_v2 \
    --base-ms 4000 --increment-ms 100 --extra-positions 20 --pgn-dir games/
```

`--extra-positions N` controls sample size (adds N random positions to 9 fixed ones, each played
twice). A result of "NOT RESOLVED" is a real answer — it means run more games, not that the change
is a weak yes. See [docs/PROMOTION_TESTING.md](docs/PROMOTION_TESTING.md) for the full promotion
gate a change must clear before it replaces the current champion.

`tools/blunder_audit.py` scores every move of a played PGN against a much-deeper fixed-node search
of the same engine, reporting average centipawn loss and inaccuracy/mistake/blunder counts — use it
on games you actually lost, at a node count well above (≥10x) what the engine searches in the real
game, or the referee just agrees with whatever was played.

## Architecture

### The root boundary and the compiled core

`agent.py` at the repo root is a thin, python-chess-backed safety boundary: it tracks the persistent
game board across moves, does time management (`_move_budget_ms`), and calls into the compiled
search — wrapping everything in a `try/except` that falls back to the first legal move on *any*
internal failure. It never trusts the compiled engine's move without checking it's still legal on
the python-chess board.

`engine.py` (bitboard representation, movegen, make/unmake, Zobrist hashing) and `search.py`
(negamax/PVS with aspiration windows, quiescence with SEE and delta pruning, LMR, a fixed-array
transposition table) are the actual engine, JIT-compiled with `@njit` (numba). Every `njit` function
is `cache=False` — **do not change this to `cache=True`**: it was tested, and numba's on-disk cache
cannot round-trip this codebase's mutually recursive search (`_negamax` calls `_quiescence` and vice
versa) — a warm cached import throws `LLVM ERROR: Symbol not found` and would be an init failure
(automatic loss) on every game after the first on the platform.

Import time (which compiles everything) is the thing to watch: it has been measured between 27s and
67s for identical code on the same machine against a 90s budget — machine-state variance, not
something fixable by requesting more cores (numba compilation is single-threaded; pinning to one
core cost nothing in testing). `search.warmup()` at the bottom of `agent.py` forces every jitted
signature to compile at import, inside the budget, rather than on the first move.

**Root-level naming risk**: the platform puts the uploaded zip first on `sys.path`. `engine.py` and
`search.py` are generic names; if either ever collided with a name the numba/torch/onnxruntime stack
imports internally, the failure would look unrelated to the actual cause (see AGENTS.md's warning
about this class of bug). Nothing currently collides, but be deliberate about adding root-level
modules with common names.

### `champions/`, `challengers/`, and promotion

- `champions/` holds frozen, non-experimental agent directories used purely as **opponents** for
  `tools/paired_arena.py` and `tools/blunder_audit.py` — `python_v2` (the engine uploaded before the
  compiled core replaced it) and `python_tuned` (a search-improved fork of it, +105 Elo over
  `python_v2` in paired testing, never itself promoted to root). Promoting a new build means copying
  it into a new `champions/<name>/` directory, not editing an existing one.
- `challengers/numba_v1/` predates the current root engine and is now redundant with it — same
  engine, promoted to root in the merge that made it the submission. It's kept as a secondary
  measurement rig (its own `agent.py`/`engine.py`/`search.py` copy plus its own tests) and hasn't
  been deleted; treat root as the source of truth if the two ever diverge.
- `harness/` mirrors the platform's protocol and clock exactly — **do not edit it**; changing it
  makes every local result meaningless as a proxy for the real platform.

### Evaluation

`search.evaluate()` is a hand-tuned tapered (middlegame/endgame-interpolated) evaluation: material,
piece-square tables, pawn structure (doubled/isolated/passed), rook-on-open-file, bishop pair, and a
king pawn-shield bonus. **No term's weight has ever been fit against data** — they're hand-guessed
constants, and there's currently no mobility or king-safety-under-attack term. This is understood to
be the main remaining lever on playing strength (see `docs/BLUNDER_REDUCTION.md` for the reasoning
and the measured evidence for it).

### Diagnostics in `docs/` and `tools/`

The `docs/*.md` files (`BLUNDER_REDUCTION.md`, `SEARCH_UPGRADE_RESULTS.md`, `PROMOTION_TESTING.md`,
`TRIAL_RECOVERY_REPORT.md`, `ROUND4_RECOVERY_PLAN.md`, `HYPERCOMPETITIVE_ROADMAP.md`) are a working
narrative of what's been measured and tried, not a fixed spec — some conclusions in the older ones
have since been measured wrong and corrected in later ones (e.g. which side we played in a given
rated game, or whether a given game round is representative). Prefer the most recently dated finding
over an older one when they conflict, and prefer re-measuring over trusting either.

`tools/` beyond the promotion tools above are development-only and never ship in `submission.zip`
(`harness/package.py` only includes root-level `*.py` plus an optional `weights/` directory):
`analyze_pgn_stockfish.py` / `stockfish_arena.py` need a local Stockfish binary; `probe_pgn_positions.py`
and `search_scaling.py` (Python-agent-specific) / `numba_search_scaling.py` (compiled-core-specific)
show what an agent would play at increasing time budgets on one position; `fuzz_numba_core.py`
differentially tests `engine.py` against python-chess.
