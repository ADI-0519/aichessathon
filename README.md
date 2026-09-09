# AIY Chessathon engine

This repository contains AIY's competition engine, frozen challengers, training and diagnostic
tools, and a local harness for [AI Chessathon](https://aichessathon.com). The canonical deployable
engine is `current/`; repository-root Python files are development infrastructure and are never the
submission agent.

```bash
git clone https://github.com/advitrocks9/aichessathon-starter
cd aichessathon-starter
uv sync
./.venv/Scripts/python.exe -m harness.play --white current --black baselines/greedy
```

That plays the current champion against a baseline over a full 120 s + 0.5 s game and prints the
result.
When you like it, build and inspect `submission.zip` before uploading it. The complete Git Bash
workflow is below. `make play`, `make arena`, `make gate`, and `make zip` remain convenient
shortcuts when `make` is available, but none of the commands in this guide require it.

## Current engine

The current champion is V7 continuous-time search plus the promoted, exact qsearch evaluation
cache. It is a Numba-compiled alpha-beta engine with incremental NNUE, handcrafted evaluation,
persistent search memory, conservative selective search, and completed-iteration timeout safety.
Packaging places the readable `current/` sources and weights at the submission zip root.
`agent.py` exposes the required function:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    return "e2e4"
```

Do not edit `current/` during an experiment. Copy the champion into a new challenger, change one
hypothesis, and compare that immutable candidate against `current/`.

The teammate's king-conditioned evaluator is retained in
`challengers/v9_kingnet/`. Its README records its exact source commit, model hash, limitations, and
the staged comparison needed before it can be combined with the current champion.

```bash
PY="./.venv/Scripts/python.exe"

"$PY" -m harness.play --white current --black baselines/minimax
"$PY" -m harness.play --white current --black baselines/minimax --fen "<fen>"
"$PY" -m harness.arena --agent current --opponent baselines/minimax --games 20
```

Anything your agent prints shows up under the result, so `print` debugging works. The platform
keeps it too. Every rated game leaves a log on your dashboard next to the PGN, holding your
output plus your init time, your time on each move, and the clock you had left. Only your team
can read it.

For reproducible paired experiments, resumable reports, source fingerprints, the pinned
500-position opening suite, and protected dataset splits, use the
[Backtest Suite V2](docs/BACKTESTING.md).

For fixed-node choice traces, independently scored root moves, legal PVs, and the rated-game
critical-position suite, use the [search diagnostics](docs/SEARCH_DIAGNOSTICS.md).
The first controlled search-profile results are recorded in the
[V4 search ablation report](docs/V4_SEARCH_ABLATIONS.md).

Start with [Current engine state](CURRENT_STATE.md) for the deployable build and immediate next
step. [Experiment ledger](docs/EXPERIMENT_LEDGER.md) records what has already been promoted,
rejected, or left inconclusive. The evidence and strategic reset are expanded in
[Learned Evaluator Track](docs/LEARNED_EVALUATOR.md) and
[Engine Strategy Reset](docs/ENGINE_STRATEGY_RESET.md).

## Git Bash development runbook

Run these commands from the repository root in Git Bash. Set the Python variable once in every new
shell:

```bash
PY="./.venv/Scripts/python.exe"
"$PY" --version
git branch --show-current
git status --short
```

The engine arguments used by the tools are directories, not zip files:

- `current` is the canonical deployable champion;
- `challengers/<name>` is an experimental engine or frozen promoted predecessor;
- `baselines/<name>` is a deliberately simple reference opponent;
- an archived submission must be extracted before it can play locally.

Do not edit either engine while a match is running. The reproducible backtester fingerprints both
inputs and will reject mixed-source results.

### Test one game or a quick match

Play one game and save its PGN:

```bash
"$PY" -m harness.play \
  --white challengers/v4_dev_safe \
  --black current \
  --base-ms 10000 \
  --increment-ms 100 \
  --pgn benchmarks/runs/v4-dev-safe-one-game.pgn
```

Play every selected position once with each colour:

```bash
"$PY" -m tools.paired_arena \
  --candidate challengers/v4_dev_safe \
  --opponent current \
  --base-ms 10000 \
  --increment-ms 100 \
  --limit 15 \
  --pgn-dir benchmarks/runs/v4-dev-safe-vs-current-pgns
```

For a small official-clock smoke test, use the same command with `--base-ms 120000`,
`--increment-ms 500`, and `--limit 2`. This is a reliability check, not enough games for a
strength claim.

### Test a branch without switching branches

There is no need to disturb a dirty working tree just to test another branch. Materialize its
canonical submission directory from the exact commit into an ignored snapshot directory:

```bash
BRANCH="dev"
COMMIT="$(git rev-parse "$BRANCH")"
SNAPSHOT="benchmarks/runs/snapshots/${BRANCH}-$(git rev-parse --short "$BRANCH")"

mkdir -p "$SNAPSHOT"
git archive "$COMMIT" current | tar -x -C "$SNAPSHOT" --strip-components=1
printf 'snapshot: %s\ncommit:   %s\n' "$SNAPSHOT" "$COMMIT"
```

Use `$SNAPSHOT` anywhere a command accepts `--candidate`, `--opponent`, `--agent`, or
`--engine-root`. To compare against an older zip, extract that zip to a directory first:

```bash
OLD="benchmarks/runs/snapshots/submission-v3"
mkdir -p "$OLD"
unzip -q -o submission_v3.zip -d "$OLD"

"$PY" -m tools.paired_arena \
  --candidate "$SNAPSHOT" \
  --opponent "$OLD" \
  --base-ms 10000 \
  --increment-ms 100 \
  --limit 15
```

### Run a reproducible backtest

Use the pinned opening suite for promotion evidence. One position produces two games because the
candidate plays both colours:

```bash
CANDIDATE="$SNAPSHOT"
RUN="v4-dev-vs-v3-development-50"

"$PY" -m tools.backtest \
  --candidate "$CANDIDATE" \
  --opponent current \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split development \
  --limit 50 \
  --base-ms 10000 \
  --increment-ms 100 \
  --output "benchmarks/runs/$RUN"
```

The output directory contains `manifest.json`, resumable `games.jsonl`, PGNs, and `summary.json`.
Repeat the exact command to resume an interrupted run. Use a new output name if the candidate,
opponent, clock, suite, or selection changes. See [Backtest Suite V2](docs/BACKTESTING.md) for the
split policy and report format.

For a behaviour-changing challenger, add a five-bin paired SPRT. The runner checks the boundary
only after the colour-swapped pair is complete and persists its state in `summary.json`:

```bash
"$PY" -m tools.backtest \
  --candidate "$CANDIDATE" \
  --opponent current \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split development \
  --limit 200 \
  --base-ms 10000 \
  --increment-ms 100 \
  --sprt --sprt-elo0 0 --sprt-elo1 20 \
  --output "benchmarks/runs/${RUN}-sprt-0-20"
```

This tests whether the result is closer to 0 Elo or +20 Elo; a lower-bound stop does not establish
negative Elo. See [the backtesting guide](docs/BACKTESTING.md) before choosing bounds or running
tests concurrently.

Confirm a promising change on the validation split rather than repeatedly tuning against it:

```bash
"$PY" -m tools.backtest \
  --candidate "$CANDIDATE" \
  --opponent current \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split validation \
  --limit 30 \
  --base-ms 10000 \
  --increment-ms 100 \
  --output benchmarks/runs/v4-dev-vs-v3-validation-30
```

### Run independent shards concurrently

Different output directories and non-overlapping offsets are mandatory. `PYTHONUNBUFFERED=1`
makes progress appear immediately in the log files:

```bash
mkdir -p benchmarks/runs/logs
RUN="v4-dev-vs-v3-shard"

for OFFSET in 0 10 20; do
  PYTHONUNBUFFERED=1 "$PY" -m tools.backtest \
    --candidate "$CANDIDATE" \
    --opponent current \
    --suite benchmarks/suites/openings_8moves_v3_500.epd \
    --split development \
    --offset "$OFFSET" \
    --limit 10 \
    --base-ms 10000 \
    --increment-ms 100 \
    --output "benchmarks/runs/${RUN}-${OFFSET}" \
    > "benchmarks/runs/logs/${RUN}-${OFFSET}.log" 2>&1 &
done

wait
```

In another Git Bash window, follow one shard with:

```bash
tail -f benchmarks/runs/logs/v4-dev-vs-v3-shard-0.log
```

After all shards finish:

```bash
for RESULT in benchmarks/runs/v4-dev-vs-v3-shard-*/summary.json; do
  printf '\n===== %s =====\n' "$RESULT"
  cat "$RESULT"
done
```

Do not run more timed shards than the machine has spare physical CPU cores. CPU contention changes
the effective search time and makes results harder to compare.

### Benchmark against Stockfish locally

Stockfish is a development-only opponent. Never put its executable, source, output, or runtime
adapter in a submission. Set its path once; this is the location used by the current Windows
development machine:

```bash
SF="/c/Users/adirj/AppData/Local/Microsoft/WinGet/Packages/Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe/stockfish/stockfish-windows-x86-64-avx2.exe"
test -f "$SF" && echo "Stockfish found"
```

Run the compact fixed-opening benchmark:

```bash
"$PY" -m tools.stockfish_arena \
  --candidate "$CANDIDATE" \
  --engine "$SF" \
  --nodes 500 \
  --base-ms 10000 \
  --increment-ms 100 \
  --limit 15 \
  --pgn-dir benchmarks/runs/v4-dev-vs-sf500-pgns
```

Or use the reproducible suite and report pipeline:

```bash
"$PY" -m tools.backtest \
  --candidate "$CANDIDATE" \
  --stockfish "$SF" \
  --stockfish-nodes 2000 \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split validation \
  --limit 50 \
  --base-ms 10000 \
  --increment-ms 100 \
  --output benchmarks/runs/v4-dev-vs-sf2000-validation-50
```

Fixed-node Stockfish is a stable measuring stick, not a model of every ladder opponent. Compare a
candidate with the frozen champion at the same Stockfish node limit and opening selection.

### Diagnose a rated PGN

Use Stockfish to identify costly decisions. For one game, `--focus` is the colour played by our
agent:

```bash
PGN="/c/Users/adirj/Downloads/aichessathon-round-28-team-i-love-fortnite.pgn"

"$PY" -m tools.analyze_pgn_stockfish \
  --engine "$SF" \
  --nodes 20000 \
  --focus black \
  "$PGN"
```

For the rated-game corpus, select AIY by its PGN player header and write mate-safe JSON. Each
position is analysed once and reused as the following ply's before-position, so this requires
roughly half as many engine calls as the text-only predecessor:

```bash
"$PY" -m tools.analyze_pgn_stockfish \
  --engine "$SF" \
  --nodes 100000 \
  --player AIY \
  --missing-player skip \
  --format json \
  --output benchmarks/runs/v5-rated-stockfish100k.json \
  /c/Users/adirj/Downloads/aichessathon-round-*.pgn

"$PY" -m tools.build_critical_suite_from_analysis \
  --analysis benchmarks/runs/v5-rated-stockfish100k.json \
  --output benchmarks/suites/v5_rated_critical.json \
  --min-cp-loss 80 \
  --limit 25
```

Mate scores stay in separate `mate` fields and are excluded from ACPL. The first move starts at
120 seconds; the increment is added only before later moves by the same colour.

Probe multiple agents at the positions before selected full moves:

```bash
"$PY" -m tools.probe_pgn_positions \
  --agent current \
  --agent "$CANDIDATE" \
  --color black \
  --fullmoves 9,11,22,28,32 \
  --time-ms 90000 \
  "$PGN"
```

These probes load fresh agent state for every position. If the move differs from the rated game,
test persistent transposition-table/history effects with the checked-in faithful replay suite:

```bash
"$PY" -m tools.search_memory_replay \
  --engine-root current \
  --warm-nodes 25000 \
  --target-nodes 300000 \
  --output benchmarks/diagnostics/v3-persistent-replay.json
```

### Inspect search decisions and ablations

Report completed depth, nodes, quiescence share, TT/LMR statistics, top root moves, scores, and
principal variations:

```bash
"$PY" -m tools.search_diagnostics \
  --engine-root "$CANDIDATE" \
  --all-cases \
  --nodes 25000,100000,300000 \
  --root-depth 5 \
  --top 8
```

Inspect an arbitrary position or one critical case more deeply:

```bash
"$PY" -m tools.search_diagnostics \
  --engine-root "$CANDIDATE" \
  --fen "<fen>" \
  --reference-move "<uci-move>" \
  --nodes 100000,300000,1000000 \
  --root-depth 7 \
  --top 12
```

Run the named V4 search profiles in isolated processes:

```bash
"$PY" -m tools.search_ablations \
  --engine-root challengers/v4_lab \
  --nodes 25000,100000,300000 \
  --root-depth 5 \
  --output benchmarks/diagnostics/v4-search-ablations.json
```

Materialize the development-only V7 lab, then classify whether rated errors come from evaluator
choice, LMR, null-move pruning, or a shared blind spot:

```bash
"$PY" -m tools.materialize_search_lab \
  --source challengers/v7_continuous_time \
  --output benchmarks/runs/candidates/v7_search_lab

"$PY" -m tools.search_ablations \
  --engine-root benchmarks/runs/candidates/v7_search_lab \
  --suite benchmarks/suites/v5_rated_critical.json \
  --trials baseline,hce-only,nnue-only,no-lmr,no-null,no-lmr-no-null \
  --nodes 25000,100000,300000,1000000 \
  --root-depth 5 \
  --trial-timeout-s 3600 \
  --output benchmarks/diagnostics/v7-rated-mechanism-matrix.json
```

The lab is generated from frozen V7 and is never a submission candidate. Every profile runs in a
fresh interpreter because Numba freezes these switches during compilation.

### Run correctness and quality gates

```bash
"$PY" -m ruff check .
"$PY" -m mypy
"$PY" -m unittest discover -s tests -v
"$PY" -m tools.fuzz_numba_core --positions 10000 --attack-interval 25
```

The differential fuzz campaign compares the compiled board against `python-chess`, including
legal moves, captures, attacks, state transitions, hashing, and undo restoration. Run it after
changing move generation, make/unmake, attack detection, or hashing. Always require zero crashes,
illegal moves, flags, and void games before considering strength results.

### Train the sparse evaluator

Training uses a separate environment so the competition-compatible `.venv` remains untouched.
The Parquet corpus and packed arrays are ignored and must never enter a submission.

Run the complete resumable setup, verification, packing, and CUDA-training pipeline with one
command from Git Bash:

```bash
bash scripts/train_nnue_v1.sh
```

Progress is written to `benchmarks/runs/nnue-v1/pipeline.log`. The expanded commands below are
useful when diagnosing a failed stage or running a controlled variant.

```bash
TRAIN_PY="./.venv-training/Scripts/python.exe"

uv venv .venv-training --python 3.12
uv --system-certs pip install --python "$TRAIN_PY" \
  "chess==1.11.2" "numpy==2.5.2" pyarrow
uv --system-certs pip install --python "$TRAIN_PY" \
  torch --index https://download.pytorch.org/whl/cu128

"$TRAIN_PY" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Download the official CC0 Lichess Fishnet shard and verify its published digest:

```bash
mkdir -p benchmarks/suites/sources
curl.exe -fL \
  'https://huggingface.co/datasets/Lichess/fishnet-evals/resolve/main/standard_rated_2014_09.parquet?download=true' \
  -o benchmarks/suites/sources/standard_rated_2014_09.parquet
echo 'b2d0d3cc3ea2f2795e6fffa4d33f5c1ff6b26d8a0c1cbb5a74768f3228b9a5ef  benchmarks/suites/sources/standard_rated_2014_09.parquet' \
  | sha256sum --check
```

Pack four million quiet training positions and a disjoint 500,000-position validation set:

```bash
"$TRAIN_PY" -m tools.pack_nnue_data \
  --source benchmarks/suites/sources/standard_rated_2014_09.parquet \
  --train-output benchmarks/runs/nnue-v1/train-4m.npy \
  --validation-output benchmarks/runs/nnue-v1/validation-500k.npy \
  --manifest benchmarks/runs/nnue-v1/data-manifest.json \
  --train-target 4000000 \
  --validation-target 500000 \
  --validation-groups 1 \
  --min-ply 12
```

Train and export our own weights:

```bash
"$TRAIN_PY" -m tools.train_nnue \
  --train benchmarks/runs/nnue-v1/train-4m.npy \
  --validation benchmarks/runs/nnue-v1/validation-500k.npy \
  --output benchmarks/runs/nnue-v1/model.npz \
  --manifest benchmarks/runs/nnue-v1/model-manifest.json \
  --accumulator 128 \
  --hidden 32 \
  --epochs 8 \
  --batch-size 8192 \
  --device cuda
```

An exported model is research output, not a champion. It must next pass feature-parity tests,
fixed-position evaluator checks, a nodes-per-second budget, paired games against exact V4,
independent validation games, and official-clock/package smoke tests.

Verify the V5 runtime against NumPy and PyTorch, exercise incremental updates (including castling,
promotion, capture, and en passant), and measure compiled inference throughput:

```bash
"$PY" -m tools.verify_v5_nnue --candidate challengers/v5_nnue
```

V5 keeps the trained artifact at `weights/model.npz`; the standard packager and backtest
fingerprint both include that directory. Never move the model to the candidate root. Create
immutable local candidates for controlled blend experiments instead of editing a candidate during
a resumable run:

```bash
for blend in 0 25 50 100; do
  "$PY" -m tools.materialize_nnue_blend \
    --source challengers/v5_nnue \
    --blend "$blend" \
    --output "benchmarks/runs/candidates/v5-nnue-${blend}-t1"
done
```

The blend is the learned evaluator's percentage; `0` isolates integration overhead, `25` is the
first conservative candidate, and `100` is pure neural evaluation. The materializer refuses to
overwrite an existing directory. Give reruns a new candidate/output name so their fingerprints
stay trustworthy.

Run a quick two-position technical smoke against the frozen V4 archive before a longer backtest:

```bash
mkdir -p benchmarks/runs/v4-exact
unzip -q -o submission_v4.zip -d benchmarks/runs/v4-exact

"$PY" -m tools.paired_arena \
  --candidate benchmarks/runs/candidates/v5-nnue-25-t1 \
  --opponent benchmarks/runs/v4-exact \
  --base-ms 10000 \
  --increment-ms 100 \
  --limit 2 \
  --pgn-dir benchmarks/runs/v5-nnue-25-t1-smoke
```

Use `tools.backtest` and the pinned development/validation splits for promotion evidence; a tiny
smoke score is only a correctness signal.

Run the first blend screen sequentially so competing searches do not steal CPU from one another:

```bash
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

for blend in 25 50 100; do
  "$PY" -m tools.backtest \
    --candidate "benchmarks/runs/candidates/v5-nnue-${blend}-t1" \
    --opponent benchmarks/runs/v4-exact \
    --suite benchmarks/suites/openings_8moves_v3_500.epd \
    --split development \
    --limit 20 \
    --base-ms 10000 \
    --increment-ms 100 \
    --output "benchmarks/runs/v5-nnue-${blend}-t1-vs-v4-development-20"
done
```

Each `--limit 20` run is 20 opening pairs (40 games). Re-running the same command resumes from its
append-only journal. Stop immediately and investigate if the candidate records any technical
failure; compare paired scores and confidence intervals only after all three runs finish.

### Train and gate the V6 king-conditioned evaluator

V6 is an isolated evaluator experiment built on the exact V5 search and 50% blend. It conditions
each piece-square feature on the friendly king square, starts from an exact lift of the proven V5
network, and keeps the V5 checkpoint if training does not improve held-out loss. The existing
four-million-position V5 arrays are reused; do not repack them.

Train on a CUDA machine with the repository's training environment. This writes an approximately
25 MiB integer-only model and a provenance manifest:

```bash
TRAIN_PY="./.venv-training/Scripts/python.exe"

"$TRAIN_PY" -m tools.train_halfkp \
  --train benchmarks/runs/nnue-v1/train-4m.npy \
  --validation benchmarks/runs/nnue-v1/validation-500k.npy \
  --initial-model challengers/v5_nnue/weights/model.npz \
  --output benchmarks/runs/v6-halfkp-full/model.npz \
  --manifest benchmarks/runs/v6-halfkp-full/manifest.json \
  --accumulator 256 \
  --hidden 32 \
  --epochs 6 \
  --batch-size 8192 \
  --learning-rate 0.0001 \
  --freeze-dense-epochs 2 \
  --device cuda
```

For Colab, select a GPU runtime and keep its CUDA-enabled PyTorch installation. Install only the
missing data dependencies; installing this project or its CPU PyTorch pin would disable the GPU.
Copy the two packed arrays to the clone, run the command above with `python`, and save both the
model and manifest to Drive before ending the runtime.

Inspect the manifest first. If its `best_epoch` is greater than zero, copy the selected artifact
into the isolated candidate, then verify the trained runtime before playing any games:

```bash
PY="./.venv/Scripts/python.exe"

cp benchmarks/runs/v6-halfkp-full/model.npz \
  challengers/v6_halfkp/weights/model.npz

"$PY" -m tools.verify_halfkp \
  --candidate challengers/v6_halfkp \
  --random-plies 500 \
  --benchmark-iterations 100000
```

If the selected epoch predates dense-head unfreezing, its added channels may be disconnected. The
exact pruner checks that every removed outgoing connection is zero and refuses an approximate
transformation:

```bash
"$PY" -m tools.prune_halfkp \
  --source benchmarks/runs/v6-halfkp-full/model.npz \
  --output benchmarks/runs/v6-halfkp-pruned-128/model.npz \
  --manifest benchmarks/runs/v6-halfkp-pruned-128/manifest.json \
  --accumulator 128
```

Maximum feature and incremental errors must be exactly zero. Next run a two-pair technical smoke,
then 20 development pairs against frozen V5. Only a technically clean candidate with encouraging
development evidence advances to a larger match; offline validation loss alone never promotes it.

```bash
"$PY" -m tools.backtest \
  --candidate challengers/v6_halfkp \
  --opponent challengers/v5_nnue \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split development \
  --limit 2 \
  --base-ms 10000 \
  --increment-ms 100 \
  --output benchmarks/runs/v6-halfkp-vs-v5-smoke

"$PY" -m tools.backtest \
  --candidate challengers/v6_halfkp \
  --opponent challengers/v5_nnue \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split development \
  --limit 20 \
  --base-ms 10000 \
  --increment-ms 100 \
  --output benchmarks/runs/v6-halfkp-vs-v5-development-20
```

The generated V6 weight is ignored by Git while it is an untrained scaffold. Remove that one
ignore rule and commit the model only after the trained artifact passes the gates. Detailed design
and verification notes live in `challengers/v6_halfkp/README.md`.

### Test the stable-timeout and continuous-clock challengers

Keep search changes isolated from clock changes. The continuous-clock candidate is compared with
the stable-timeout candidate, not with V5, because both already contain the completed-iteration
repair:

```bash
"$PY" -m tools.backtest \
  --candidate challengers/v7_continuous_time \
  --opponent challengers/v6_stable_timeout \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split development \
  --limit 20 \
  --base-ms 10000 \
  --increment-ms 100 \
  --output benchmarks/runs/v7-continuous-time-vs-v6-stable-development-20
```

The relevant rated-loss regressions are
`benchmarks/suites/v5_round46_47_losses.json` and
`benchmarks/suites/v5_round48_loss.json`. Fixed-node diagnostics explain whether a changed move is
caused by search rather than by the clock policy; paired games remain the promotion authority.

### Build and verify a versioned submission

Package the canonical champion with the harness:

```bash
VERSION="v7-current"
"$PY" -m harness.package --root current --out "submission_${VERSION}.zip"
```

To package committed source from another branch without switching to it, materialize that commit's
canonical directory and pass the snapshot to the same packager:

```bash
BRANCH="dev"
COMMIT="$(git rev-parse "$BRANCH")"
VERSION="${BRANCH}-$(git rev-parse --short "$COMMIT")"
SNAPSHOT="$(mktemp -d ./benchmarks/runs/package-${VERSION}-XXXXXX)"

git archive "$COMMIT" current | tar -x -C "$SNAPSHOT" --strip-components=1
"$PY" -m harness.package --root "$SNAPSHOT" --out "submission_${VERSION}.zip"
```

Inspect and test the exact archive, rather than trusting the source directory that produced it:

```bash
unzip -Z1 "submission_${VERSION}.zip"
unzip -t "submission_${VERSION}.zip"
unzip -l "submission_${VERSION}.zip"
sha256sum "submission_${VERSION}.zip"

mkdir -p benchmarks/runs
CHECK_DIR="$(mktemp -d ./benchmarks/runs/package-check-${VERSION}-XXXXXX)"
unzip -q "submission_${VERSION}.zip" -d "$CHECK_DIR"

"$PY" -m harness.arena \
  --agent "$CHECK_DIR" \
  --opponent baselines/minimax \
  --games 2 \
  --base-ms 5000 \
  --increment-ms 100
```

The archive must put `agent.py` directly at its root and include the engine sources plus
`weights/model.npz`; it must not contain a parent `current/` directory. After every check passes,
preserve the versioned rollback and update the generic upload file byte-for-byte:

```bash
cp "submission_${VERSION}.zip" submission.zip
sha256sum "submission_${VERSION}.zip" submission.zip
```

Upload either identical file and read the dashboard validation log. Local checks cannot certify an
upload. Record the source commit and SHA-256 with every promoted version.

### Commit and push deliberately

`submission.zip` is intentionally ignored; versioned archives such as
`submission_v7-current.zip` may be
committed when the team wants them retained:

```bash
git status --short
git add README.md CURRENT_STATE.md docs/EXPERIMENT_LEDGER.md
git add "submission_${VERSION}.zip"  # only when deliberately retaining this exact rollback
git diff --cached --stat
git commit -m "Make current the canonical champion and record experiments"
git push origin "$(git branch --show-current)"
```

Do not use `git add .` in a working tree containing unrelated teammate or experiment files. A
submission can be built from `dev` while remaining on `adi_branch`; switching branches is not part
of packaging.

### Decide whether a candidate is better

Use game score for promotion, and diagnostics to explain it. A candidate should have zero technical
failures, improvement against the frozen champion on paired development games, confirmation on a
separate split, and an official-clock smoke test. For PGN analysis, prioritize catastrophic move
losses and ACPL over raw “best move” count: many trivial best moves do not compensate for one
game-losing error. Small match samples have wide confidence intervals, so treat them as directional
screens rather than proof.

## The ladder

Measured with `harness/arena.py`. Beating greedy is a search. Beating minimax is a search plus an
evaluation worth searching with.

| Matchup | Games | Time control | Score |
|---|---|---|---|
| random vs greedy | 20 | 10 s + 0.1 s | 10.0% (+1 =2 -17) |
| greedy vs minimax | 6 | 120 s + 0.5 s | 0.0% (+0 =0 -6) |
| numba vs minimax | 6 | 10 s + 0.5 s | 66.7% (+2 =4 -0) |

- `baselines/random` plays a uniformly random legal move and is only a protocol/reliability check.
- `baselines/greedy` searches one ply on material.
- `baselines/minimax` searches two plies on material and mobility, with no time management.
- `baselines/numba` is `minimax` with the evaluation jitted. It is barely stronger, which is
  the point: jitting a shallow search buys headroom, not depth. Read it for the warm-up call
  at the bottom, which is how you keep compilation off your clock.

## What's here

```
current/agent.py     canonical public get_move boundary
current/engine.py    compiled board, move generation, make/unmake, and hashing
current/search.py    compiled evaluation and search
current/nnue.py      incremental inference for the team-trained evaluator
current/time_manager.py  continuous move-aware clock allocation
current/weights/     model artifacts included by the packager
challengers/         experimental engines and frozen promoted predecessors
baselines/           random, greedy, minimax, numba; each is a directory with an agent.py
harness/runner.py    the process the platform runs your agent in
harness/referee.py   the clock, legality, draw and adjudication rules
harness/rules.py     the event constants the harness enforces
harness/sandbox.py   the one process, spoken to as the platform speaks to a container
harness/play.py      one game between two agent directories
harness/arena.py     many games, with a score
harness/package.py   builds submission.zip with agent.py at the root
tools/search_diagnostics.py  fixed-node, root-score, and PV investigation
tools/search_ablations.py    isolated V4 search-profile comparison
tools/search_memory_replay.py  rated-game TT/history reconstruction
tools/sample_evaluation_positions.py  deterministic PGN/EPD/FEN data sampler
tools/label_positions.py     resumable fixed-node teacher labelling
tools/fit_hce.py             validation-selected residual HCE fitting
tools/relabel_evaluation_baseline.py  rebase teacher labels onto an exact runtime evaluator
tools/verify_residual.py     Python/Numba parity for a compiled residual evaluator
tools/nnue_features.py       colour-symmetric sparse piece-square encoding
tools/pack_nnue_data.py      Parquet filtering and row-group-disjoint packing
tools/train_nnue.py          from-scratch sparse-network training and export
docs/EVALUATION_TUNING.md    end-to-end evaluation-data runbook and gates
docs/LEARNED_EVALUATOR.md    chronological learned-evaluator experiment log
docs/EXPERIMENT_LEDGER.md    compact promoted/rejected/inconclusive decision record
CURRENT_STATE.md             authoritative champion, evidence, and next task
docs/IDEAS.md                early engine-development notes
```

Local games start from the normal position unless you pass `--fen`. Rated games start from
curated neutral positions.

The harness is here so your games are honest, not so you can pre-validate an upload. Acceptance
happens on the platform, and the validation log on your dashboard is the authority on it.

## The rules

[aichessathon.com/docs](https://aichessathon.com/docs) is canonical and changes. Read it before
you upload.
