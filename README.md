# AI Chessathon starter

Fork this to build an agent for [AI Chessathon](https://aichessathon.com). It gives you a working
submission, baselines to beat, and a local harness that speaks the same protocol and enforces the
same clock as the platform, so you can see whether a change actually helped before you upload it.

```bash
git clone https://github.com/advitrocks9/aichessathon-starter
cd aichessathon-starter
uv sync
./.venv/Scripts/python.exe -m harness.play --white . --black baselines/greedy
```

That plays your agent against a baseline over a full 120 s + 0.5 s game and prints the result.
When you like it, build and inspect `submission.zip` before uploading it. The complete Git Bash
workflow is below. `make play`, `make arena`, `make gate`, and `make zip` remain convenient
shortcuts when `make` is available, but none of the commands in this guide require it.

## Writing an agent

The production agent is split across three readable source files, all placed at the submission
zip root. `agent.py` exposes the one required function:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    return "e2e4"
```

The fork ships a legal random-mover, so the loop works before you write anything. Replace the body.

```bash
PY="./.venv/Scripts/python.exe"

"$PY" -m harness.play --white . --black baselines/minimax
"$PY" -m harness.play --white . --black baselines/minimax --fen "<fen>"
"$PY" -m harness.arena --agent . --opponent baselines/minimax --games 20
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

The current architecture decision, research synthesis, experiment gates, and dated build schedule
are in the [Post-Day-1 competitive engine plan](docs/POST_DAY1_DEEP_RESEARCH.md).

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

- `.` is the agent in the current working tree;
- `challengers/<name>` is an experimental engine;
- `champions/<name>` is a frozen previous champion;
- `baselines/<name>` is a deliberately simple reference opponent;
- an archived submission must be extracted before it can play locally.

Do not edit either engine while a match is running. The reproducible backtester fingerprints both
inputs and will reject mixed-source results.

### Test one game or a quick match

Play one game and save its PGN:

```bash
"$PY" -m harness.play \
  --white challengers/v4_dev_safe \
  --black . \
  --base-ms 10000 \
  --increment-ms 100 \
  --pgn benchmarks/runs/v4-dev-safe-one-game.pgn
```

Play every selected position once with each colour:

```bash
"$PY" -m tools.paired_arena \
  --candidate challengers/v4_dev_safe \
  --opponent . \
  --base-ms 10000 \
  --increment-ms 100 \
  --limit 15 \
  --pgn-dir benchmarks/runs/v4-dev-safe-vs-current-pgns
```

For a small official-clock smoke test, use the same command with `--base-ms 120000`,
`--increment-ms 500`, and `--limit 2`. This is a reliability check, not enough games for a
strength claim.

### Test a branch without switching branches

There is no need to disturb a dirty working tree just to test another branch. Materialize the
three submission files from its exact commit into an ignored snapshot directory:

```bash
BRANCH="dev"
COMMIT="$(git rev-parse "$BRANCH")"
SNAPSHOT="benchmarks/runs/snapshots/${BRANCH}-$(git rev-parse --short "$BRANCH")"

mkdir -p "$SNAPSHOT"
git archive "$COMMIT" agent.py engine.py search.py | tar -x -C "$SNAPSHOT"
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
  --opponent . \
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

Confirm a promising change on the validation split rather than repeatedly tuning against it:

```bash
"$PY" -m tools.backtest \
  --candidate "$CANDIDATE" \
  --opponent . \
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
    --opponent . \
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

Use Stockfish to identify costly decisions. `--focus` is the colour played by our agent:

```bash
PGN="/c/Users/adirj/Downloads/aichessathon-round-28-team-i-love-fortnite.pgn"

"$PY" -m tools.analyze_pgn_stockfish \
  --engine "$SF" \
  --nodes 20000 \
  --focus black \
  "$PGN"
```

Probe multiple agents at the positions before selected full moves:

```bash
"$PY" -m tools.probe_pgn_positions \
  --agent . \
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
  --engine-root . \
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

### Build and verify a versioned submission

Package the current working tree with the harness:

```bash
VERSION="v4"
"$PY" -m harness.package --out "submission_${VERSION}.zip"
```

To package committed source from another branch without switching to it, archive the exact three
files directly from its commit:

```bash
VERSION="v4"
BRANCH="dev"
COMMIT="$(git rev-parse "$BRANCH")"

git archive --format=zip \
  --output="submission_${VERSION}.zip" \
  "$COMMIT" \
  agent.py engine.py search.py
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

The archive must put `agent.py` directly at its root. For the current three-file engine, the member
list should be exactly `agent.py`, `engine.py`, and `search.py`. After every check passes, preserve
the versioned rollback and update the generic upload file byte-for-byte:

```bash
cp "submission_${VERSION}.zip" submission.zip
sha256sum "submission_${VERSION}.zip" submission.zip
```

Upload either identical file and read the dashboard validation log. Local checks cannot certify an
upload. Record the source commit and SHA-256 with every promoted version.

### Commit and push deliberately

`submission.zip` is intentionally ignored; versioned archives such as `submission_v4.zip` may be
committed when the team wants them retained:

```bash
git status --short
git add README.md submission_v4.zip
git diff --cached --stat
git commit -m "Document engine workflow and retain V4 submission"
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

- `baselines/random` plays a uniformly random legal move. It is what `agent.py` starts as.
- `baselines/greedy` searches one ply on material.
- `baselines/minimax` searches two plies on material and mobility, with no time management.
- `baselines/numba` is `minimax` with the evaluation jitted. It is barely stronger, which is
  the point: jitting a shallow search buys headroom, not depth. Read it for the warm-up call
  at the bottom, which is how you keep compilation off your clock.

## What's here

```
agent.py             public get_move boundary
engine.py            compiled board, move generation, make/unmake, and hashing
search.py            compiled evaluation and search
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
docs/IDEAS.md        where the strength actually comes from
```

Local games start from the normal position unless you pass `--fen`. Rated games start from
curated neutral positions.

The harness is here so your games are honest, not so you can pre-validate an upload. Acceptance
happens on the platform, and the validation log on your dashboard is the authority on it.

## The rules

[aichessathon.com/docs](https://aichessathon.com/docs) is canonical and changes. Read it before
you upload.
