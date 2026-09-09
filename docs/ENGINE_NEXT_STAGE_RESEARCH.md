# Engine next-stage research and execution plan

**Date:** 9 September 2026  
**Decision horizon:** submission lock on 11 September at 11:00 London time  
**Scope:** turn the current V7/qcache champion and V9 KingNet evidence into the strongest safe
submission, then establish the next serious engine-development programme

## Executive decision

The best immediate move is **not** another bundle of search heuristics and **not** an unbounded
neural-network training run.

1. Finish the active comparison between `exp_kingnet75_qcache` and raw `v9_kingnet`.
2. Select qcache by deterministic parity and repeatable single-process throughput, not by a small
   self-play score. It is intended to be an exact optimisation, so a noisy game result is the
   wrong primary statistic.
3. If KingNet's existing strength result survives a direct comparison with `current/`, recover its
   exact training provenance, run the release gates, and submit it. KingNet is the only current
   candidate with evidence of a material strength jump.
4. Once a safe KingNet submission exists, run two independent development lanes:
   - a **search-depth lane** concentrated on quiescence cost and conservative selectivity;
   - an **evaluator lane** concentrated on data quality, WDL-aware targets, horizontal king
     mirroring, and model selection by games.

The realistic competition objective is to close the gap to the leading submitted agents. It is
not realistic to recreate Stockfish in the remaining time. Stockfish is the product of mature
native code, large neural datasets, and massive distributed testing; its own project explicitly
says that improving it requires a massive amount of testing.[1] The transferable lesson is the
development method and the independently implementable concepts—not its source or weights.

## What our evidence actually says

### Engine status

`current/` is V7 continuous time management plus an exact qsearch evaluation cache. It is a
reliability champion, not a demonstrated large Elo gain: the V7 validation match scored 48.75%
against V6, and the direct ten-pair validation check against submitted V5 scored 50%. The qcache
was promoted because it preserved the fixed-node search exactly and improved median throughput by
roughly 4.5–7.5% in clean measurements.

Raw `challengers/v9_kingnet/` changes the learned representation to 16 king buckets, retains a
128-wide accumulator and 32-wide hidden layer, and uses a 75% neural blend. Its first 20-pair
comparison against frozen V7 scored 18 wins, 14 draws, and 8 losses: 62.5%, approximately +89 Elo.
That is the strongest directional result presently available, but its interval remains broad and
the exact bundled model still needs a recoverable training manifest before submission.

The active `exp_kingnet75_qcache` candidate combines KingNet with the champion's exact qcache.
Fixed-node parity with raw V9 has already been observed on a probe. A single NPS probe suggested
that the cache might be slower with KingNet, so it must earn its place independently. Cache value
depends on both hit rate and the cost of the evaluation it avoids.

### What the platform games show—and what they do not

`benchmarks/diagnostics/platform-recent-stockfish.json` contains 17 platform games and 944 AIY
moves scored by Stockfish 18 at 100,000 nodes. The aggregate weighted ACPL is 18.42 cp, with:

| Error threshold | Count |
|---|---:|
| at least 50 cp | 86 |
| at least 75 cp | 54 |
| at least 100 cp | 33 |
| at least 150 cp | 23 |
| at least 200 cp | 15 |
| at least 300 cp | 6 |

The distribution by material is more informative:

| Pieces on board | Scored moves | ACPL | 100+ cp | 200+ cp |
|---|---:|---:|---:|---:|
| 27–32 | 153 | 15.83 | 2 | 1 |
| 21–26 | 167 | 21.75 | 5 | 3 |
| 13–20 | 250 | 20.88 | 10 | 1 |
| 8–12 | 171 | **33.99** | **16** | **10** |
| at most 7 | 203 | 1.49 | 0 | 0 |

This is strong evidence that reduced-material play—especially 8–12-piece endings—is a major
weakness in the submitted lineage. The largest recorded misses include 766, 538, 462, 419, 328,
and 328 cp losses. It is not evidence that tiny Syzygy endings are the main problem: the at-most-
seven-piece slice was already the cleanest, while the failures cluster above ordinary small
tablebase cardinality.

Two cautions matter:

- The repository's current state record identifies these rated losses as evidence from the
  submitted V5 lineage. They are useful for learning the position distribution, but they are not
  direct measurements of V7 or KingNet.
- A 100,000-node teacher is diagnostic rather than ground truth. At least one record reports the
  same played and best move with a large apparent loss, demonstrating score instability in the
  analysis pipeline. Before a position becomes training data or a hard regression, relabel it at
  greater depth and require teacher-move stability.

### Search profile

Across the checked-in fixed-node diagnostics, approximately 70–86% of nodes are quiescence nodes;
the common figure is around 80%. The qcache hit rate varies by position from roughly 18% to 36%.
That makes qsearch economics the most leveraged exact-speed target.

The current search already has PVS, a direct-mapped array TT, aspiration windows, quiescence,
SEE-informed ordering, killers, quiet history, one-ply LMR, and guarded null move. It does not have
the interacting selectivity system seen in mature engines: adaptive reductions, reverse futility,
late-move pruning, correction histories, capture/continuation histories, ProbCut, or singular
extensions. Current Stockfish search illustrates that these mechanisms interact with static
evaluation, TT information, history, and node type rather than existing as independent switches;
for example, its current source adjusts histories from evaluation differences and guards razoring
and futility with several search-state conditions.[2]

That observation argues against copying a checklist. Our earlier global no-LMR/no-null matrix and
the rejected counter/history-malus challenger already show that plausible isolated ideas can lose.

## Lessons from strong-engine practice

### 1. Architecture helps, but model selection is empirical

NNUE is suited to this event because sparse inputs change little between positions and permit
low-precision, low-latency CPU inference. Official NNUE documentation states that the design target
is millions of evaluations per second per thread and emphasizes quantization.[3] That is directly
aligned with the single-core, no-GPU match environment.

King conditioning is a credible improvement. Stockfish's HalfKAv2_hm representation combines the
friendly king position with piece-square features and horizontally mirrors the board so the king
is always on one half.[4] Tcheran, an independently developed strong engine, reports a horizontally
mirrored bucketed evaluator, multiple output buckets, self-play training data, and an alpha-beta
search with many pruning and history mechanisms.[5] These are architectural hypotheses, not assets
we may transplant.

Our V9 representation is materially simpler: 16 coarse 2x2 king buckets, no horizontal king-file
mirroring, a 128-wide accumulator, a 256→32→1 dense head, and a 75% blend with HCE. It is therefore
a useful intermediate network, not the end of the evaluator programme.

### 2. Dataset quality matters more than merely increasing row count

Our packed dataset currently rejects check positions and positions whose teacher best move is a
capture. That is a useful first filter, but it does not establish tactical stability, deduplicate
near-identical positions, stratify material phases, or retain game outcomes. The trainer minimizes
teacher-evaluation probability MSE only.

Official Stockfish training guidance is unusually honest: dataset quality is selected
empirically; better teacher evaluations do not automatically give more learnable data. It reports
that mixtures of different data sources and staged retraining outperform treating a single source
as universally sufficient.[6] Official NNUE documentation also supports mixing a sigmoid-mapped
engine evaluation with the eventual game result under a tunable lambda.[7]

A recent paper proposes filtering positions using disagreement between static, quiescence, and
deeper search scores. Its concrete results are from Xiangqi, not Western chess, so its exact 60/70
cp margins must not be imported as facts. The useful transferable claim is narrower: positions
that are not in check and remain stable under tactical/deeper search make cleaner static-evaluator
targets, and phase/evaluation diversity must be preserved.[8]

### 3. Offline loss is a filter, never the promotion criterion

The project has already learned this twice: HalfKP-256 improved held-out probability loss but lost
its match badly, and the linear residual improved RMSE substantially but failed critical probes
and games. Stockfish's NNUE tooling automatically converts training checkpoints and plays games to
select the best network.[9] We need the same principle at our scale: keep checkpoints, screen them
in search, then promote only through paired games.

### 4. Serious engine testing uses paired outcomes and large samples

Stockfish uses the pentanomial model because paired opening outcomes are more informative than
independent W/D/L counts, and sequential testing saves resources.[10] Its development guide also
requires one focused idea per test.[11] A strong contemporary engine release provides a useful
scale comparison: Stormphrax reported about 3,000 short-time-control games and 1,000 long-time-
control games for a release regression, and explicitly warned that self-play may not transfer
perfectly to other opponents.[12]

Our 20-pair experiments are directional screens. They are appropriate for rejecting disasters,
not for distinguishing +5 Elo from noise.

## The immediate release lane: now to submission lock

### Gate A: decide whether qcache belongs with KingNet

The active V9-vs-KingNet75-qcache game match is a technical regression screen. The actual decision
requires all of the following:

1. zero technical failures;
2. identical fixed-node move, score, completed depth, nodes, qnodes, and search statistics;
3. repeated single-process NPS on the same suite with no other CPU-heavy work;
4. a positive effect at 300k and 1M nodes that survives reversed run order.

If qcache is neutral or slower, release raw V9. There is no virtue in retaining a cache merely
because it helped the older 50:50 evaluator.

After the active match, the long-running measurement should be run with the machine idle:

```bash
PY="./.venv/Scripts/python.exe"

"$PY" -m tools.numba_search_scaling \
  --engine-root challengers/v9_kingnet \
  --suite benchmarks/suites/v5_priority_losses.json \
  --nodes 100000,300000,1000000 \
  --repeats 5 \
  --output benchmarks/diagnostics/v9-kingnet-scaling-clean.json

"$PY" -m tools.numba_search_scaling \
  --engine-root challengers/exp_kingnet75_qcache \
  --suite benchmarks/suites/v5_priority_losses.json \
  --nodes 100000,300000,1000000 \
  --repeats 5 \
  --output benchmarks/diagnostics/kingnet75-qcache-scaling-clean.json
```

Reverse the order once if the margin is below roughly 3%; thermal and background-load effects can
otherwise dominate a small difference.

### Gate B: establish that the combined release beats `current/`

The earlier 62.5% result was against frozen V7, whereas `current/` now has qcache. The selected
KingNet release candidate must face the actual champion. Use the integrated pentanomial SPRT and
four workers only if four physical/logical cores can remain free. Workers shorten game gauntlets;
they must not be used for NPS measurements.

```bash
PY="./.venv/Scripts/python.exe"

export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"$PY" -m tools.backtest \
  --candidate challengers/exp_kingnet75_qcache \
  --opponent current \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split development \
  --limit 200 \
  --base-ms 10000 \
  --increment-ms 100 \
  --workers 4 \
  --sprt --sprt-elo0 0 --sprt-elo1 20 \
  --sprt-min-pairs 25 \
  --output benchmarks/runs/kingnet75-qcache-vs-current-sprt-0-20
```

Replace the candidate path with raw `v9_kingnet` if Gate A rejects qcache. Stop on a technical
failure. If time is short, a fixed 20-pair screen is enough to decide whether to do a release
smoke, but it is not proof of +20 Elo.

### Gate C: provenance and release safety

Before upload, recover and record:

- source dataset identities and hashes;
- packing filters and split assignment;
- trainer commit and command;
- seed, epochs, batch size, learning rate, and selected checkpoint;
- exported model SHA-256;
- teammate confirmation that the team trained the network.

This is a release blocker, not paperwork. The competition may ask the team to explain how the
network was trained.

Then run the KingNet verifier, focused tests, `make gate`, build the archive from an immutable
release directory, inspect its contents/size, and play one official-clock game as each colour.

## The evaluator lane after the release is safe

### Experiment E1: complete the blend sweep on the same KingNet weights

Test 50%, 75%, and 100% with otherwise byte-identical code. The old network's pure mode losing to
its blend does not establish the optimum for KingNet. Pure KingNet is strategically important:
it removes the cost and conflicting opinion of HCE at every cache miss. First measure fixed-node
NPS and critical moves, then run paired games. Do not train a new model during this experiment.

### Experiment E2: repair the data contract, keeping the V9 architecture fixed

Add the following to the training data before changing network size:

- a stable position/game identity for exact and near-duplicate control;
- game result in WDL space;
- material count/phase and evaluation band;
- source identity and teacher settings;
- a quietness/stability field based on deeper-vs-qsearch agreement;
- explicit tactical, quiet, and endgame strata;
- group-disjoint validation by game/source, not merely the last Parquet row groups.

Build a balanced mixture with deliberately increased 8–12-piece coverage, but do not train only on
the famous losses. Relabel ladder-critical positions more deeply and reserve them for diagnosis;
otherwise the regression suite ceases to be independent.

Train a small matrix of checkpoints varying only the WDL/result mixing lambda. Select the best
offline checkpoints, quantize and verify them, then let paired games choose the network. Continue
from a good checkpoint instead of restarting every dataset stage; official guidance notes that
dataset order can matter.[6]

### Experiment E3: improve representation one axis at a time

Once E2 produces a reproducible same-architecture control:

1. **Horizontal mirroring plus finer king buckets.** Compare V9's 16 coarse buckets with a
   mirrored 32-bucket HalfKA-style mapping. This adds sparse feature capacity without increasing
   the dense-head width.
2. **Material output buckets.** Add four phase/material heads selected at inference. This directly
   addresses the observed endgame concentration and is cheap at a leaf, but it should follow a
   clean base model.
3. **Width.** Compare 128 with 192 or 256 only after deployment weights are exported directly in
   quantized form and NPS is measured. A large float `.npz` is an avoidable package/import cost.
4. **Activation/head.** Test squared clipped ReLU or a simpler direct head separately. Do not
   combine activation, buckets, width, and data changes into one uninterpretable run.

The 50 MB limit and scalar Numba inference make a 512/1024-wide C++-style architecture an unsafe
assumption. Capacity is useful only if the search can afford it.

## The search-depth lane

### S1: exact KingNet qsearch economics

Profile these separately on the selected KingNet blend:

1. qcache on/off;
2. the existing post-move lazy qsearch accumulator candidate;
3. skipping arithmetic for a perspective whose king-bucket row is already marked stale;
4. direct quantized-weight loading instead of loading floats and quantizing at import.

Items 1–3 must retain exact fixed-node search results. The existing lazy-accumulator measurements
on the old evaluator show only a small clean reversed-order gain (roughly 0.3–1.7% depending on node
limit), despite a high theoretical wasted-update count. That is not enough evidence to promote it
unchanged. KingNet must be profiled independently because its accumulator behaviour and cache
economics differ.

### S2: conservative reverse futility pruning

This is the first behaviour-changing search candidate after the evaluator release. The current
engine has no reverse futility pruning, so a guarded shallow non-PV version can save substantial
work without changing root logic. Required exclusions include check, mate-score windows, low-
material zugzwang-prone positions, and unreliable/absent static evaluation. Begin at shallow depth
with deliberately wide margins.

The gate is not “more NPS.” It is:

- no new misses on the relabelled critical suite at equal nodes;
- more completed depth or fewer nodes at equal depth;
- SPRT against the selected champion;
- independent validation if development accepts.

### S3: adaptive LMR, then correction history

Current LMR reduces every eligible fifth-or-later quiet by one ply. A depth/move-count table with
PV, killer, history, check, and improving-position adjustments is a logical next candidate, but the
existing aggressive LMR experiment already changed previously solved decisions. Start from the
moderate capped-two-ply variant and test it as a separate branch.

If that lane stabilizes, add correction history before more exotic pruning. A correction table
learns systematic static-evaluation error from completed searches and can improve both pruning
decisions and ordering. It is a better match to an imperfect learned evaluator than indiscriminately
adding singular extensions or ProbCut first.

### S4: staged move selection and qsearch pruning

The search currently scores and insertion-sorts the complete legal move list. A staged move picker
(TT move, good captures, killers, ordered quiets, bad captures) may reduce ordering work, especially
when a cutoff happens early. In qsearch, investigate whether obviously losing captures can be
rejected before making the move and updating the accumulator. This needs a correct fast check-
giving exception; otherwise it changes tactics and is not an exact optimisation.

Do not globally disable qsearch SEE/delta pruning. That ablation was already tested and did not fix
the known failures.

## Lower-priority ideas

- **Small Syzygy tables:** allowed and useful for perfect cleanup, but they do not cover the 8–12
  piece region where the historical error rate peaks. Add them only after measuring archive size,
  hit rate, and probe overhead. Mature engines support up to seven pieces and carefully gate probes
  by cardinality/depth/state.[13]
- **Opening book:** 29 of 31 downloaded starts were unique in the existing audit. Measure coverage
  on future starts before investing.
- **Opponent modelling:** curated starts and Swiss/ladder pairing uncertainty make bespoke opponent
  prediction fragile. Spend that effort on robust chess strength.
- **MCTS/Lc0 imitation:** one CPU core and no GPU make batched policy/value MCTS a poor runtime fit.
- **More Stockfish opponent games:** Stockfish is useful as an external yardstick, but beating a
  fixed 500-node configuration does not imply beating diverse ladder agents. Promotion must remain
  candidate-vs-champion paired testing.
- **Feature bundles:** do not combine tablebases, new LMR, correction history, wider NNUE, new data,
  and blend changes. A win would be impossible to attribute and a loss impossible to repair.

## Operating cadence

Every candidate follows the same funnel:

1. code/invariant tests and full incremental-evaluator verification;
2. fixed-node critical suite and teacher-stability audit;
3. single-process NPS if the change claims speed;
4. two-pair technical smoke;
5. development pentanomial SPRT against the frozen champion;
6. independent validation and a different opponent family;
7. official-clock two-colour smoke, packaging inspection, then upload;
8. immutable candidate hash and ledger entry.

Run game workers in parallel across free cores. Run microbenchmarks one process at a time on an idle
machine. Never compare NPS gathered while a game, video game, trainer, or another benchmark is
competing for the same CPU.

## Priority board

| Priority | Work | Expected value | Risk | Ship before lock? |
|---:|---|---|---|---|
| 0 | Decide V9 vs KingNet75-qcache; recover model provenance | Very high | Low | Yes |
| 1 | Selected KingNet candidate vs actual `current/`; release gates | Very high | Low | Yes |
| 2 | Same-weight KingNet 50/75/100 blend sweep | High | Low | Only if compute permits |
| 3 | Exact KingNet qsearch profiling/optimisation | Medium–high | Low | Only a clearly positive exact change |
| 4 | Data-contract repair and reproducible same-architecture retrain | Very high | Medium | Probably after lock |
| 5 | Mirrored/finer king features, then material output buckets | High | Medium | After a clean training control |
| 6 | Guarded reverse futility pruning | High | Medium | After release is safe |
| 7 | Moderate adaptive LMR, then correction history | High | Medium–high | Later, separately |
| 8 | Small tablebases/book/opponent modelling | Low–medium | Low–medium | Only measured opportunities |

The central idea is simple: **secure the large evaluator gain already in hand, then buy depth and
evaluation quality through controlled experiments.** Progress has felt slow because too many
experiments improved a proxy—offline loss, one critical move, or one small match—without clearing
the whole causal chain. This plan makes the chain explicit.

## Sources

1. [Stockfish README—massive testing through Fishtest](https://github.com/official-stockfish/Stockfish/blob/master/README.md#donating-hardware)
2. [Current Stockfish search source—history, razoring, and futility interact with search state](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)
3. [Official NNUE documentation—sparsity, incremental updates, quantization, and CPU inference](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
4. [Stockfish HalfKAv2_hm feature source—king conditioning and horizontal mirroring](https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/features/half_ka_v2_hm.h)
5. [Tcheran repository—bucketed NNUE, output buckets, self-play data, and enhanced alpha-beta](https://github.com/tcheran-chess/tcheran)
6. [Official Stockfish NNUE training-dataset guidance](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets)
7. [Official NNUE documentation—WDL-space targets and result interpolation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md#using-results-along-the-evaluation)
8. [Tan and Medina, “Study of the Proper NNUE Dataset”](https://arxiv.org/html/2412.17948v1)
9. [Official nnue-pytorch tooling—selecting checkpoints by engine games](https://github.com/official-stockfish/nnue-pytorch#automatically-run-matches-to-determine-the-best-net-generated-by-a-running-training)
10. [Fishtest mathematics—pentanomial models and sequential testing](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
11. [Fishtest development guide—small atomic changes and one idea per test](https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html)
12. [Stormphrax releases—STC/LTC sample sizes and self-play caveat](https://github.com/Ciekce/Stormphrax/releases)
13. [Stockfish Syzygy probing source](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)
