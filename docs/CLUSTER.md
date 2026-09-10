# Running the training experiment on the shared cluster

```bash
git clone <repo> && cd aichessathon
git checkout team-combined
bash scripts/cluster_train.sh
```

That is the whole procedure. Everything else below is what it does and how to
watch it.

## It shares the machine

The box has 64 cores, eight GPUs and a terabyte of memory, and other people are
using most of it -- at the time of writing 679 GB of RAM and seven of the eight
GPUs. The script is written to be a good neighbour before it is written to be
fast:

- **One GPU, chosen because it is idle.** It picks the first card with at least
  40 GB free and under 10% utilisation, exports `CUDA_VISIBLE_DEVICES` for that
  card alone, and every other GPU is invisible to the process for its whole life.
  If nothing is free it prints the current state and exits rather than queueing
  behind someone.
- **12 of 64 cores** for packing, and 4 native threads per process. NumPy and
  torch otherwise each try to fill the machine.
- **`nice -n 15`** on both heavy stages, so interactive work wins.
- **Writes only under `/data/$USER/aichessathon`.** Nothing lands in a shared
  directory or in `$HOME`.
- **One instance at a time**, enforced with `flock`. A second invocation exits
  rather than fighting the first for the same files.
- **Cleans up on exit**, including on Ctrl-C, so no orphaned trainer keeps a GPU.

**Choosing the card yourself:**

```bash
GPU=6 bash scripts/cluster_train.sh
```

It checks the card exists, reports what is already on it, and warns if someone
else appears to be using it -- but proceeds, because you named it. Leaving `GPU`
unset makes it find an idle card instead.

Override anything else the same way: `GPU=6 WORKERS=8 MONTHS=24 bash scripts/cluster_train.sh`.

## Memory

It reads `MemAvailable` before starting and refuses if what is free will not
cover the run plus 16 GB of headroom, because swapping on a shared box hurts
everyone. A packed position is 68 bytes and each packing worker needs about a
gigabyte for the Parquet row group it decodes, so:

| positions | workers | needs |
| --- | --- | --- |
| 60M (default) | 12 | ~19 GB |
| 200M | 12 | ~29 GB |
| 800M | 12 | ~66 GB |

Against the 327 GB free when this was written, even the largest run is
comfortable. Note that 13 GB of swap was already in use, so treat the available
figure as something that moves rather than a guarantee.

## Watching it

Each run gets `/data/$USER/aichessathon/runs/<timestamp>/`:

| file | what |
| --- | --- |
| `run.log` | everything, timestamped |
| `status.txt` | one line per stage, for a quick `cat` |
| `model.npz` | the trained network |
| `model-manifest.json` | architecture, hyperparameters, and per-epoch validation history |

```bash
tail -f /data/$USER/aichessathon/runs/*/run.log     # live
cat /data/$USER/aichessathon/runs/*/status.txt      # where it is
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # what it is using
```

## Resuming

Every stage is skipped if its output exists. Downloads resume file by file,
packed data is reused, and only training itself restarts from scratch. An
interrupted run costs the stage it was in, not the whole thing.

## What the defaults do, and why

| setting | default | reason |
| --- | --- | --- |
| `MONTHS` | 12 | roughly 97M positions; the last month is held out whole |
| `TRAIN_TARGET` | 60,000,000 | 15x the four million every net so far has been trained on |
| `ACCUMULATOR` | 256 | our 256-wide net lost at 4M positions, but that was capacity against too little data; this is the first run where the data justifies the width |
| `EPOCHS` | 12 | the 8-epoch baseline was still improving when it stopped |
| `BATCH_SIZE` | 16384 | larger than the laptop's 8192, since the card has 48 GB |

Change any of them from the environment. The single most useful variation is
`MONTHS` -- more months is the axis that has never been tested.

## Known limitation

`tools/train_nnue.py` indexes a memory-mapped array with a fresh random
permutation each epoch, on the main thread, with no prefetch. On the laptop this
was already the bottleneck: a 256-wide net trained *faster* than a 128-wide one,
which only happens when the GPU is waiting for data. With 327 GB of RAM
available the array will sit in page cache, so it should hold up here, but if
the GPU sits below about 50% utilisation during training that is the reason, and
the fix is a background loader rather than a bigger card.
