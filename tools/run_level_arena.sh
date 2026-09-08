#!/bin/bash
# run_level_arena.sh <candidate> <opponent> <fens> <base-ms> <increment-ms> <log-dir>
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# both engines must stay single-threaded or the shards contend and the clock lies
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH="$PWD"

CANDIDATE="$1"; OPPONENT="$2"; FENS="$3"; BASE="$4"; INC="$5"; LOGDIR="$6"
TOTAL=$(grep -c . "$FENS")
SHARDS=10
mkdir -p "$LOGDIR"
rm -f "$LOGDIR"/shard-*.log

echo "$SHARDS shards over $TOTAL positions"
per=$(( (TOTAL + SHARDS - 1) / SHARDS ))
offset=0
while [ "$offset" -lt "$TOTAL" ]; do
  ./.venv/bin/python -m tools.fen_arena \
    --candidate "$CANDIDATE" --opponent "$OPPONENT" --fens "$FENS" \
    --base-ms "$BASE" --increment-ms "$INC" \
    --offset "$offset" --limit "$per" \
    > "$LOGDIR/shard-$(printf '%03d' "$offset").log" 2>&1 &
  offset=$(( offset + per ))
done
wait

./.venv/bin/python -m tools.fen_arena_aggregate --log-dir "$LOGDIR" \
  --label "candidate $CANDIDATE vs $OPPONENT at ${BASE}ms+${INC}ms, level suite $FENS"
