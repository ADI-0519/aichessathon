# Residual evaluation tuning

> Historical V3/V4 pipeline. The resulting residual approaches are recorded in
> `docs/EXPERIMENT_LEDGER.md`; the full V8 residual was rejected and is not the current engine.

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

The sampler streams `.pgn`, `.epd` and `.fen` files, de-duplicates normalized
positions across every source, and balances six phase/tactical strata. EPD and
FEN inputs contain one position per non-comment line. PGN inputs are sampled
throughout each main line using the ply bounds and stride below.

For a first real fit, build a 100,000-position source suite and label a balanced
20,000-position pilot from it. Fifty positions is only a plumbing test. Require
at least 10,000 examples in every stratum so an opening-only corpus fails before
it consumes teacher time.

### Selected full-game source

Use the official Lichess January 2013 standard-rated export. It is released
under CC0, contains 121,332 complete standard games, is 17.8 MB compressed and
expands to about 93 MB. Its published archive SHA-256 is
`aa40b3671fa3cf1072eb182892cd90b0e1e003a4a5943492f64b77e7f3fd1635`.
The first 5,000 games passed the sampler gate with thousands of positions in
every phase and tactical stratum.

`zstandard` is a development-only decompressor. It is installed only in the
local virtual environment and must never be imported by a submitted agent.

```bash
mkdir -p benchmarks/suites/sources

curl -fL \
  https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst \
  -o benchmarks/suites/sources/lichess_db_standard_rated_2013-01.pgn.zst

echo "aa40b3671fa3cf1072eb182892cd90b0e1e003a4a5943492f64b77e7f3fd1635  benchmarks/suites/sources/lichess_db_standard_rated_2013-01.pgn.zst" \
  | sha256sum -c -

uv pip install --system-certs --python .venv/Scripts/python.exe zstandard

./.venv/Scripts/python.exe -c '
from pathlib import Path
import zstandard as zstd

source = Path("benchmarks/suites/sources/lichess_db_standard_rated_2013-01.pgn.zst")
target = source.with_suffix("")
with source.open("rb") as compressed, target.open("wb") as pgn:
    zstd.ZstdDecompressor().copy_stream(compressed, pgn)
print(target)
'
```

The expected decompressed file has 121,332 `[Event ...]` headers and SHA-256
`8963b6a1620a0e9c77e5515a0744ec133e86869487188af047bb0a74400dee37`.

```bash
./.venv/Scripts/python.exe -m tools.sample_evaluation_positions \
  --source benchmarks/suites/sources/lichess_db_standard_rated_2013-01.pgn \
  --source-url https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst \
  --output benchmarks/runs/evaluation-data/evaluation_v5_lichess_100k.epd \
  --count 100000 \
  --seed v5-residual-eval-v1 \
  --min-ply 12 \
  --max-ply 160 \
  --stride 8 \
  --min-per-stratum 10000
```

With the source and sampler version documented here, this reads 121,332 games,
examines 902,946 sampled plies and emits 100,000 unique positions. Each stratum
contains 16,666 or 16,667 positions. The expected suite SHA-256 is
`586739788d4bd8007f5c29b7a6a6df6cb511da30681b399cc2a3f2afa8118de3`.

Use games whose redistribution terms permit this use and retain their source
and licence beside the local training corpus. `benchmarks/suites/sources/` and
`benchmarks/runs/` are ignored; do not commit a huge raw corpus or generated
suite. Add one `--source-url URL` per source, in the same order, when the files
were downloaded. The manifest records URLs, content hashes, source counts,
deduplication, terminal-position filtering, sampling parameters and the output
hash. Existing outputs are protected unless `--force` is explicit.

The official Stockfish `popularpos_lichess_v3.epd` corpus is CC0 and useful as
an opening supplement, but it is not a standalone evaluation corpus. In the
current 200,000-position file almost every position classifies as opening and
none as endgame. Combining it with full-game PGNs is reasonable; labelling it
alone would train the wrong distribution.

```bash
mkdir -p benchmarks/suites/sources
curl -fL \
  https://raw.githubusercontent.com/official-stockfish/books/master/popularpos_lichess_v3.epd.zip \
  -o benchmarks/suites/sources/popularpos_lichess_v3.epd.zip

./.venv/Scripts/python.exe -m zipfile -e \
  benchmarks/suites/sources/popularpos_lichess_v3.epd.zip \
  benchmarks/suites/sources
```

## Generate teacher labels

Set `SF` to the local executable in Git Bash. Ten thousand nodes per position is
a reasonable first full pass; use a larger independent relabel for milestone
validation rather than repeatedly tuning against the same noisy labels.

```bash
SF="/c/Users/adirj/AppData/Local/Microsoft/WinGet/Packages/Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe/stockfish/stockfish-windows-x86-64-avx2.exe"

./.venv/Scripts/python.exe -m tools.label_positions \
  --suite benchmarks/runs/evaluation-data/evaluation_v5_lichess_100k.epd \
  --engine "$SF" \
  --nodes 10000 \
  --split all \
  --split-seed v5-residual-eval-v1 \
  --limit 20000 \
  --output benchmarks/runs/evaluation-data/evaluation_v5_pilot_20k.jsonl \
  --resume \
  --checkpoint-every 25
```

## Fit without opening the holdout

The fitter uses standardized ridge regression solved by least squares. It
selects the ridge value on validation data and reports both floating-point and
rounded-integer accuracy. Runtime experiments use the integer parameters.

```bash
./.venv/Scripts/python.exe -m tools.fit_hce \
  --labels benchmarks/runs/evaluation-data/evaluation_v5_pilot_20k.jsonl \
  --output benchmarks/runs/evaluation-data/evaluation_v5_pilot_20k.hce.json
```

Do not continue merely because training error improved. Require validation
residual RMSE to beat the zero-correction baseline, implement the identical
features in Numba, and add Python-versus-Numba feature parity tests across
random positions.

When the frozen runtime baseline is V5 rather than V3, preserve the expensive
teacher labels and replace only their baseline score before fitting:

```bash
./.venv/Scripts/python.exe -m tools.relabel_evaluation_baseline \
  --labels benchmarks/runs/evaluation-data/evaluation_v5_pilot_20k.jsonl \
  --engine-root challengers/v5_nnue \
  --output benchmarks/runs/evaluation-data/evaluation_v5_pilot_20k.v5-baseline.jsonl

./.venv/Scripts/python.exe -m tools.fit_hce \
  --labels benchmarks/runs/evaluation-data/evaluation_v5_pilot_20k.v5-baseline.jsonl \
  --output benchmarks/runs/evaluation-data/evaluation_v5_pilot_20k.v5-residual.hce.json \
  --target-name teacher_score_cp_minus_v5_50_static_score_cp
```

The relabel manifest fingerprints the exact runtime source and proves that
teacher scores and feature vectors were retained. A compiled candidate can be
checked against the Python model with `tools.verify_residual` before any game.

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
