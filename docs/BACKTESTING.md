# Backtest Suite V2

`tools.backtest` is the reproducible strength-testing layer above the unmodified competition
referee. It is development-only and is never packaged with the chess agent.

## Guarantees

- Every selected position is played once with each colour.
- Candidate and opponent package inputs are SHA-256 fingerprinted in `manifest.json`.
- Those fingerprints are checked before every new game. Editing an engine stops the run instead of
  silently mixing versions.
- Every completed game is durably appended to `games.jsonl`; repeating the same command skips it.
- PGNs are written atomically under `games/`, and `summary.json` is refreshed after every game.
- A pre-existing output directory can only resume an identical experiment configuration.
- Voids are not misreported as draws, and technical failures are attributed to the candidate or
  opponent.
- Reports include W/D/L, colour splits, terminations, Elo from score, paired pentanomial counts,
  and an approximate paired 95% score interval.

The confidence interval is descriptive, not a full Fishtest-compatible GSPRT. Small samples remain
directional evidence.

## Suites and leakage control

The suite can be `builtin` or a UTF-8 `.epd`, `.fen`, or `.pgn` file. EPD/FEN files contain one
position per non-comment line. For a PGN opening book, the position after each main line is used.
Duplicate normalized FENs are removed.

Positions receive a stable SHA-256 assignment:

- 60% development;
- 20% validation;
- 20% holdout.

The assignment depends only on normalized FEN and `--split-seed`, so reordering or extending a
suite cannot move an existing position between splits. Accessing `holdout` or `all` requires the
deliberate `--unlock-holdout` flag. Do not use the holdout to choose features or weights.

## Local champion example

```bash
./.venv/Scripts/python.exe -m tools.backtest \
  --candidate . \
  --opponent champions/python_v2 \
  --suite builtin \
  --split validation \
  --base-ms 10000 \
  --increment-ms 100 \
  --output benchmarks/runs/v3-vs-python-v2-validation
```

## Stockfish example

Stockfish is an offline opponent only and is never included in the submission.

```bash
./.venv/Scripts/python.exe -m tools.backtest \
  --candidate . \
  --stockfish "/c/Users/adirj/AppData/Local/Microsoft/WinGet/Packages/Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe/stockfish/stockfish-windows-x86-64-avx2.exe" \
  --stockfish-nodes 2000 \
  --suite builtin \
  --split validation \
  --base-ms 10000 \
  --increment-ms 100 \
  --output benchmarks/runs/v3-vs-sf2k-validation
```

Run the exact same command after an interruption to resume. If any hashed source changed, choose a
new output directory; old and new builds must never share a report.

Use `--limit N` for a short screen and `--offset N` for an explicitly selected later slice. The
position order is deterministically shuffled by `--order-seed` before those options apply.

## Reading the output

The pentanomial keys are the candidate's total points over a colour-swapped pair:

| Key | Pair outcome |
|---:|---|
| 0.0 | two losses |
| 0.5 | loss and draw |
| 1.0 | two draws, or one win and one loss |
| 1.5 | win and draw |
| 2.0 | two wins |

The candidate must have zero technical failures. A positive score on development positions is not
enough for promotion: require confirmation on validation, external opponents, and finally the
locked holdout.
