# Residual evaluation tuning

## Objective

V3 already evaluates material, piece-square placement, bishop pairs, pawn
structure, passed pawns, rook files, king shields and tempo. The V4 model must
not discard or relearn those fundamentals from a small correlated sample.
Instead it predicts:

```text
Stockfish score from side to move - V3 static score from side to move
```

The fitted correction is eventually added to V3. The current features cover
phase-tapered mobility, king-zone pressure, hanging pieces, safe space, rook
activity, blocked passers, minor-piece outposts and king-file exposure.

## Dataset rules

- Split by normalized FEN before fitting: development, validation and holdout.
- Keep all positions from the same normalized FEN in one split.
- Include openings, middlegames, endgames, quiet positions and tactical positions.
- Clear the teacher hash between positions so fixed-node labels do not depend on
  input order.
- Record hashes for the PGN sources, sampled suite, Stockfish executable, labels
  and the exact V3 source.
- Use development to fit, validation to select ridge strength, and do not inspect
  holdout until the candidate and fitting procedure are frozen.
- Never ship Stockfish, teacher evaluations, or the labelled dataset. Only the
  small weights trained by the team may enter the submission.

## Build a diverse suite

The sampler walks one or more PGNs, samples throughout each game, de-duplicates
normalized positions and balances six phase/tactical strata. For a first real
fit, target at least 20,000 positions; 50 positions is only a plumbing test.

```bash
./.venv/Scripts/python.exe -m tools.sample_evaluation_positions \
  --source data/training/games-1.pgn data/training/games-2.pgn \
  --output benchmarks/suites/evaluation_v4_20k.epd \
  --count 20000 \
  --seed v4-residual-eval-v1 \
  --min-ply 12 \
  --max-ply 160 \
  --stride 4
```

Use games whose redistribution terms permit this use and retain their source
and licence beside the local training corpus. Do not commit a huge raw corpus.

## Generate teacher labels

Set `SF` to the local executable in Git Bash. Ten thousand nodes per position is
a reasonable first full pass; use a larger independent relabel for milestone
validation rather than repeatedly tuning against the same noisy labels.

```bash
SF="/c/Users/adirj/AppData/Local/Microsoft/WinGet/Packages/Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe/stockfish/stockfish-windows-x86-64-avx2.exe"

./.venv/Scripts/python.exe -m tools.label_positions \
  --suite benchmarks/suites/evaluation_v4_20k.epd \
  --engine "$SF" \
  --nodes 10000 \
  --split all \
  --split-seed v4-residual-eval-v1 \
  --output benchmarks/datasets/evaluation_v4_20k.jsonl \
  --resume \
  --checkpoint-every 25
```

## Fit without opening the holdout

The fitter uses standardized ridge regression solved by least squares. It
selects the ridge value on validation data and reports both floating-point and
rounded-integer accuracy. Runtime experiments use the integer parameters.

```bash
./.venv/Scripts/python.exe -m tools.fit_hce \
  --labels benchmarks/datasets/evaluation_v4_20k.jsonl \
  --output benchmarks/datasets/evaluation_v4_20k.hce.json
```

Do not continue merely because training error improved. Require validation
residual RMSE to beat the zero-correction baseline, implement the identical
features in Numba, and add Python-versus-Numba feature parity tests across
random positions.

## Promotion sequence

1. Freeze the fitted artifact and generated Numba constants.
2. Verify exact offline/runtime feature parity on at least 10,000 positions.
3. Measure evaluation throughput and complete search depth against V3.
4. Run the critical-FEN regression pack.
5. Run 30 fast paired positions against root V3, then 50 development pairs.
6. Use the validation opening split for an independent confirmation.
7. Only now report holdout metrics with `--report-holdout` and run SF2000.
8. Run official-clock smoke games, build the submission zip and inspect its root.

An evaluation fit is a candidate generator, not evidence of Elo. Paired games
remain the promotion authority.
