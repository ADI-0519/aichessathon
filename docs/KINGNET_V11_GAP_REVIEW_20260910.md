# KingNet V11: decisions before the 80M run

Review date: 10 September 2026. This updates the decisions in
`TOBY_COAD_GAP_STRATEGY.md` using the present source, the local width and training
measurements, and Toby Coad's public tree at
[`ec6d8e9ef7557cb5ace4aa7a7d242f5819c500fb`](https://github.com/TobyCoad/aichessathon-starter/tree/ec6d8e9ef7557cb5ace4aa7a7d242f5819c500fb).

## Decision

Finish the two 40M packs, benchmark training on the full working set, and train
the existing **256 / pair128 / hidden32 / eight-head V11 from scratch** as the
first control. Keep the planned 16 epochs, selecting the best checkpoint rather
than automatically shipping epoch 16. Begin engine tests from partial exports.

The largest practical opportunities are:

1. Make the new evaluator strong enough to remove the remaining HCE blend.
2. Reduce its CPU inference cost, which currently penalizes width severely.
3. Validate the already implemented signed-history / quiet-SEE challenger.
4. Test explicit king-relative file canonicalization as a separate representation.
5. Add stronger-label/search-distribution data after obtaining the two-month control.

Do not postpone a usable trained model for a wholesale engine rewrite. An A100
is valuable if its measured end-to-end turnaround beats the available GPU,
including transfer and setup. It is not a prerequisite for this run.

## What was actually inspected

Local: repository instructions, current board/search/evaluator/time integration,
V11 training/packing/export/runtime, relevant tests, width reports, smoke manifest,
mixed80m configuration, challenger descriptions, diagnostics and experiment ledger.
Toby: recursive repository index, current agent/board/search and trainer/packer,
handover, journal, continuous notes, candidate report, V16 training scripts and
V17 evaluation/ordering/time reports. This is a targeted engineering audit across
the repository, not execution of every historical experiment or a line-by-line
review of every file. No Toby engine or network was run or transplanted.

The canonical [contract](https://aichessathon.com/docs/agent-contract.md) and
[rules](https://aichessathon.com/docs/rules.md) were fetched directly during this
review. They specify 120s + 0.5s, one EPYC core, 2GB RAM, 90s initialization,
50MB unzipped and upload close September 11 at 11:00. Leave time for validation
before that close. The fetched close string itself does not state a timezone.
Engine-labelled training is allowed; another engine's implementation or network
must not enter the submission.

## The actual gap today

| Area | Our present source | Toby's inspected source / evidence | Consequence |
|---|---|---|---|
| Search selectivity | `current/search.py` already enables dynamic NMP, RFP, LMP, quiet futility, capture SEE and contextual LMR | Similar mechanisms plus richer history, singular search, qsearch TT, paired TT slots | Older descriptions of our fixed-R=2 search are stale; do not rebuild V10 |
| Evaluator architecture | V11 already implements pairwise products, dual activation, factorization and eight material heads | These are also in his V16 lineage | Architecture feature names alone no longer describe the gap |
| File symmetry | Random reflection in training; ordinary 2x2 king buckets | Explicit file canonicalization per king perspective | Our model still has to learn agreement between reflected inputs |
| Runtime | Fixed-point accumulator and scalar integer dense arithmetic | Float32 inference, with rejected integer experiments documented | Benchmark our arithmetic before increasing width |
| History | Current rewards quiet cutoffs by depth squared, monotonically capped | Active signed bounded history and quiet SEE | Our S1-lean challenger is the cheapest prepared search experiment |
| Time | Current uses a smooth budget capped at 4.5s without root-instability feedback | Stability/score-sensitive allocation | Revisit our existing adaptive candidate at meaningful clocks |
| Data | Planned November 2024 + February 2025 Fishnet positions | Multiple human months plus stronger-label engine-position shards | Two months increase diversity, but do not reproduce his source mixture |

`current/README.md` and portions of the ledger lag behind current search source.
The source establishes local behavior; it does not establish what is currently
uploaded. Toby's older candidate report likewise says V15 while later source and
reports describe V16f/S1. Do not equate a stale README with the deployed engine.

His [V16 script](https://github.com/TobyCoad/aichessathon-starter/blob/ec6d8e9ef7557cb5ace4aa7a7d242f5819c500fb/overnight/v16_long.sh)
describes roughly 6B samples and a 1024-wide model. Our 1.28B samples are about
21% of that stated exposure budget. These are script estimates, not independently
audited completed training counts, and different samplers make epoch counts
incomparable. There is no measured Elo gap between our current build and his in
this review.

## Packing and validation: preserve the work, audit the boundaries

The 2024 training file was still a `.staging.npy` at inspection; its 500k
validation file existed. This is consistent with an unfinished pack, not evidence
that 40M usable records are ready. The selected sources are individual months,
not all positions from the full calendar years.

The packer already reserves validation first, deduplicates validation, randomizes
training row-group order and excludes fingerprints of evaluator inputs. Keep it.
Before training, verify actual accepted counts and manifests, not just target
arguments or preallocated staging-file sizes. Packing can finish below its target
when source groups run out.

Both training packs must exclude **every validation set used for selection**:

- November 2024's 500k holdout;
- the older `nnue-v1/validation-500k.npy`, which is included in mixed80m selection.

The 2025 pack should receive both exclusions. The 2024 pack must also have excluded
the older set. An extra 2025 validation output is not automatically safe for model
selection: 2024 training has not necessarily excluded it. Keep it diagnostic unless
cross-shard exclusion is established.

The trainer's content hashes reject identical files, not partial row overlap.
Packer manifests therefore remain essential. Row-group separation plus exact-input
exclusion also does not prove game-disjoint validation: the packed schema does not
retain game identity. Near-neighbor positions from a game can remain correlated.

One additional audit is needed for stronger independence: the fingerprint hashes
the original feature sequence, while training can reflect its squares. A reflected
training input can therefore coincide with a held-out input after augmentation.
Count overlap under reflected, sorted active feature sets as well as ordinary
inputs. Any canonicalized representation should reserve whole equivalence classes.
This is a potential leakage channel, not a measured overlap count. A filtered row
index can exclude affected training rows without repeating expensive FEN packing.

Keep a release evaluation split separate from checkpoint selection. The older
holdout currently has selection weight 0.25, so despite its `diagnostic` name it
actively influences which checkpoint wins and is not an untouched final test.

## What 80M per epoch really means

`_sample_batch` selects shard/material components, then draws row indices with
replacement. Equal shard weights target equal exposure only when their available
material components match. The 16-epoch plan is **1.28B training draws**, with
156,256 optimizer steps at batch size 8192, not sixteen complete shuffled passes.

Even uniform replacement sampling of N rows N times visits only about 63.2% of
rows in expectation. Material weighting changes that substantially. Using the
2M smoke distribution as an illustration:

| Pieces | Smoke data share | Training share | Expected draws/row/epoch |
|---|---:|---:|---:|
| 2-8 | 5.86% | 10% | 1.71 |
| 9-12 | 7.47% | 22.5% | 3.01 |
| 13-16 | 10.94% | 20% | 1.83 |
| 17-24 | 33.35% | 25% | 0.75 |
| 25-32 | 42.38% | 22.5% | 0.53 |

This deliberately emphasizes endgames, but must be recalculated from the full
packs. More epochs will not create more distinct rare endgames. Retain the sampler
for the control; test a shuffled pass plus loss weights separately if guaranteed
row coverage becomes the objective. That changes optimization and should not be
silently substituted into the current recipe.

## Benchmark the A100 against the real job

The local smoke manifest records 2M training draws in 7.685s, about **260,233/s**.
At the same sustained rate, 1.28B draws take **82 minutes**, excluding initial
hashing/indexing, validation, checkpoint/export work and transfer. The GPU currently
visible locally is an RTX 3060 Laptop with 6GB; the manifest records CUDA but does
not identify the GPU used for that historical run. This is not an A100 measurement.

The smoke is only 245 steps and a small memory-mapped dataset. Do not use it as a
promise for random reads over 80M records. The packed records use 68 bytes/row:
80M occupy about 5.44GB before headers, with approximately 320MB of int32 material
row indices plus transient arrays. Use local SSD storage, enough host RAM and the
same full data working set for the A100 comparison.

Measure both cold-start and warmed throughput, at least several sustained blocks
after warmup, with GPU synchronization around timings. Include:

- CPU sampling and augmentation;
- host-to-device transfer;
- forward/backward/optimizer step;
- validation, export and checkpoint time;
- peak GPU memory and total elapsed job time.

Try batch sizes 8192, 16384 and 32768 as throughput probes; changing the final batch
also changes optimizer steps and needs a learning-quality pilot. Do not assume
tensor-core mixed precision will benefit sparse embedding work automatically.

There are concrete optimization candidates in our trainer: four synchronous `.to`
transfers, a CPU-resident king-bucket tensor moved during forward, per-step scalar
loss synchronization, and `hidden_weight[head]`. At width256/batch8192 the latter
materializes **256MiB** of selected head matrices alone. Grouped-by-head matrix
multiplication could avoid that expansion. Profile first; these are suspected
costs, not measured bottleneck shares. Preserve a float32 numerical baseline.

No remote A100 connection was established and no paid GPU job was started in this
review. If setup/transfer costs more time than it saves, run the control locally.

## CPU width economics and the next experiments

The second width benchmark used a single position, neutral synthetic networks and
300k fixed nodes. All widths searched the same fixed-node tree:

| Width | Search NPS | Relative | Eval/s | Candidate bytes |
|---|---:|---:|---:|---:|
| 128 | 134,497 | 1.000 | 349,032 | 3,560,178 |
| 256 | 102,114 | 0.759 | 179,422 | 6,837,490 |
| 512 | 67,706 | 0.503 | 91,262 | 13,392,114 |
| 1024 | 37,937 | 0.282 | 45,620 | 26,501,367 |

All fit the package ceiling locally. CPU cost is the issue. The 1024 timed run
also completed one less depth than the smaller widths. Synthetic one-position
results cannot predict trained-model Elo or official-hardware initialization.

Prioritize these bounded experiments:

1. **V11-256 at 75% versus current**, then **100% versus the winning V11 build**.
   Pure learned evaluation skips the HCE call in current source. Measure the
   actual speed gain and endgame decisions; do not inherit old pure-NNUE losses
   as a verdict on a new network, or assume pure V11 must win.
2. **Inference arithmetic on identical weights.** Profile dense head, accumulator
   update, refresh and HCE separately. Test a float32 head or mixed arithmetic
   against the integer implementation, including trained-model CP error tails,
   incremental/full-rebuild parity and overflow bounds. Floating-point changes
   need game testing if rounded scores/search decisions change. A same-arithmetic
   loop optimization can instead require exact fixed-node equivalence.
3. **Prepared S1-lean search.** Validate its V10 control against present current
   before reusing it. Test signed history separately, then quiet SEE, then their
   combination if justified. History should supply useful magnitudes to ordering
   and LMR; measure its distribution and pruning/re-search counts. Once a new net
   wins, reconfirm the combined search/net candidate.
4. **Canonical king-file representation.** Reflect each perspective's features
   when its own king is on the other half of the board, with an explicit bucket
   map and model-format metadata. Training, export, full rebuild and incremental
   updates must agree, especially crossing d/e and castling. Existing packed rows
   can support this; no FEN re-pack is inherently needed. Changing inference
   alone invalidates existing weights. Random reflection is augmentation, not
   this architectural weight sharing. Missing castling/EP input also means this
   is a useful modeling assumption, not proof of symmetry of full chess value.
5. **Data quality pilot.** Two human months still share one general collection
   mechanism. Add a measured minority of stronger-label engine/search positions,
   or relabel a targeted endgame sample; retain human validation and verify score
   perspective/units. The current `move` filter examines the human continuation,
   so it does not guarantee teacher-best-move quietness or stable targets. Preserve
   source/game/depth/result metadata in future packs where available. Do not
   invent game-result targets from the present cp-only records.

For mirroring or a new data mixture, run a short equal-compute pilot against the
256 control before committing another full training run. A 128-wide control can
also be competitive after training; 256 is a pragmatic starting point, not a
proven optimum.

## What Toby's evidence does and does not justify

His [evaluation report](https://github.com/TobyCoad/aichessathon-starter/blob/ec6d8e9ef7557cb5ace4aa7a7d242f5819c500fb/overnight/eval/v17/eval.md)
first proposes an accumulator cache, then records that its speed gain did not
repeat and leaves it disabled. It also reports integer inference slower in his
implementation. Neither statement proves our outcome, but both argue for direct
measurement rather than assuming standard optimizations transfer.

His [ordering report](https://github.com/TobyCoad/aichessathon-starter/blob/ec6d8e9ef7557cb5ace4aa7a7d242f5819c500fb/overnight/eval/v17/ordering.md)
reports a substantial S1-lean match gain with a wide error bar. That supports
testing our existing independent implementation; it does not transfer his Elo.
His [time report](https://github.com/TobyCoad/aichessathon-starter/blob/ec6d8e9ef7557cb5ace4aa7a7d242f5819c500fb/overnight/eval/v17/timett.md)
highlights instability-sensitive time allocation. Our adaptive-time experiment
already exists but has inconclusive development evidence; independent validation
and actual-clock tests are the next work, not another time-manager rewrite.

His notes also retract an apparent endgame regression after discovering a teacher
hash/depth mismatch in the scoring instrument. Treat historical intermediate
claims cautiously. Static loss, top-move ranking and game results answer different
questions. Do not ship a network solely because its aggregate MSE falls.

## Execution order and promotion

1. Complete packs; verify counts, exclusion provenance and material populations.
2. Benchmark sustained full-working-set throughput; choose the fastest practical
   training host. Freeze the control configuration and hashes.
3. Run 80M draws x 16 epochs. The 6.25% warmup is about one full epoch. Preserve
   the configured cosine schedule; do not accidentally restart it for each pilot.
4. Snapshot improving `model.best.partial.npz` files with epoch/model/config hashes
   while training continues. The `.pt` file is recovery state despite its `best.pt`
   name; the trainer preserves both current and best weights within it.
5. Compare exported/runtime predictions on broad held-out and special-move cases,
   inspect calibration and error tails by material, then run critical-position
   vetoes and multi-position CPU throughput measurements on an idle machine.
6. Paired development screen, independent opening validation, official-clock safety
   games, clean package import/size inspection, then promotion. Avoid saturating the
   same CPU with packing/training while measuring wall-clock chess performance.

Priority after the first network: pure blend and inference speed, then existing
S1-lean. Explicit mirroring and stronger-label data are the next training axes.
Singular extensions, larger/clustered TT and qsearch bounds are later isolated
search experiments; books, tablebases, ProbCut and correction history do not get
the first remaining slot without a measured failure they address.

The review added this decision record only. It did not modify `current/`, the
packer, training settings, weights or harness, and it did not start another pack,
training run, match or upload. Existing benchmark results were inspected, not
rerun under the active packing workload.
