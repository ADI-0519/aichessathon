# Engine strategy reset

**Audience:** AIY development team  
**Date:** 7 September 2026  
**Decision:** what to build after V7, given the rated games and the failed search/evaluation
experiments  
**Deadline assumption:** the final valid upload must be accepted before 11 September at 11:00
London time.

## Executive decision

Freeze V7 as the deployable baseline. Do not begin another generic NNUE run and do not bundle a
list of fashionable search heuristics into V8.

The next implementation should be a development-only V7 search laboratory that classifies the
largest real-game errors at equal node counts under six isolated configurations:

1. unchanged V7;
2. handcrafted evaluation only;
3. NNUE only;
4. LMR disabled;
5. null-move pruning disabled;
6. both LMR and null-move pruning disabled.

That experiment decides the next strength branch. In parallel, profile an exact-output pawn/HCE
cache. It is worth implementing only if it produces a material full-search throughput gain while
preserving every evaluation and move at fixed nodes.

This is the shortest credible route to a real improvement because it creates a causal bridge from
played error to mechanism. Our recent process often stopped at an offline metric or a small game
score, which allowed attractive changes to survive without explaining the moves that cost games.

## What the evidence says now

### V7 is a reliability baseline, not a proven Elo breakthrough

V6 repaired the interrupted-iteration mechanism reproduced from round 47. V7 retained that fix
and replaced the discontinuous clock schedule exposed by round 48. V7 scored 65.0% against V6 on
20 development pairs, but only 48.75% on 20 independent validation pairs. Directly against the
submitted V5 it scored exactly 50.0% over ten validation pairs. All of these matches completed
without technical failures.

The correct interpretation is narrow: V7 contains two real fixes and has no measured regression,
but the available games do not prove a general strength gain. The official-clock colour-swapped
smoke also completed without a crash, flag, illegal move or initialization failure. It was a
release test, not an Elo estimate.

### The new V5 games are not evidence that the quality gap disappeared

Rounds 52--55 produced three wins and one draw. Fixed-node Stockfish 18 analysis at 100,000 nodes
per position shows that the wins were substantially opponent-assisted:

- In round 52, `22.g4` lost about 117 cp and `27.c3` about 107 cp. The opponent's later `...Qc2`
  error made the repetition possible.
- In round 53, AIY's `...Be5`, `...Ra8`, `...Rab8`, `...h5`, and `...Qxd3+` each lost roughly
  90--121 cp at this analysis limit. The opponent supplied larger tactical errors, notably `Ba7`,
  `Qa4`, and `Ke3`.
- In round 54, AIY's `Qd2` lost about 97 cp and `Rc3` about 66 cp. The opponent then made several
  100+ cp errors and AIY converted. Mate-score changes late in the game are excluded from this
  qualitative assessment because they are not ordinary centipawn losses.
- In round 55, the attack was directionally successful, but `...Bh3`, `...exf3`, and `...Rae8`
  gave back roughly 92--164 cp before the opponent collapsed. Later mate-distance score changes
  likewise should not be read as normal ACPL.

These figures are diagnostic estimates, not ground truth: a fixed-node teacher can change its
preferred move at greater depth. Their value is the repeated pattern of medium and large errors,
not the last centipawn.

### The current evaluator is small and fast, but not richly contextual

V7's learned input is a colour-symmetric 12-by-64 piece-square representation. It has two
128-value incremental accumulators and a 256-to-32-to-1 dense head. It does not condition every
piece feature on the friendly king square. The handcrafted and learned scores are mixed 50:50 at
every evaluation.

The network therefore has less positional context than a king-conditioned NNUE, while the blend
still pays for both evaluators. On the pinned historical benchmark the 50% learned blend reduced
search throughput by about 23% relative to the classical predecessor. That cost was justified by
V5's large paired gain over V4, but it makes further evaluator complexity expensive.

### We have already tested the obvious residual correction

The pasted proposal correctly noticed that training a network toward the teacher and blending it
with an independently wrong handcrafted score can create an objective mismatch. However, the
first empirical test of this idea is complete.

The exact V5-relative 24-channel residual fit reduced validation RMSE from 267.6 to 204.0 cp and
MAE from 182.5 to 149.2 cp, with exact Python/Numba parity over all 20,000 labelled positions. It
then failed all three round-50 reference probes at one million nodes and lost the first three
completed smoke games. That candidate is rejected.

This does not disprove every neural residual model. It does prove that a better offline residual
metric is not enough, and that “correct the loss and train a larger architecture” cannot be our
next ungated leap.

### The king-conditioned run did not validate the architecture

The 256-wide HalfKP experiment improved held-out probability MSE by 3.38%, yet scored 35.0%
against V5 over 20 development pairs. Pruning inactive columns restored throughput and reached
50.0% in a smaller ten-pair screen, but did not establish a gain. Its selected epoch occurred
while the new half of the dense input remained inactive, so it did not provide a clean test of
the intended new king-conditioned capacity.

This means HalfKAv2_hm remains technically plausible, not empirically selected. It should be
revisited only with a better data and model-selection loop.

## Assessment of the proposed ideas

| Proposal | Current verdict | Reason |
|---|---|---|
| Runtime-aligned residual objective | Keep as a future neural control | Sound objective; the linear residual pilot failed in search, so a neural version needs stronger data and strict gates. |
| V7 search-leaf and disagreement data | High priority if evaluation is implicated | It matches training positions to the distribution actually evaluated by our search. |
| HalfKAv2_hm | Conditional | Official NNUE documentation supports king-conditioned, horizontally mirrored features, but architecture alone did not rescue our HalfKP run. |
| Four output buckets | Conditional after a working base network | Cheap at inference, but adds another dimension before the data/selection problem is solved. |
| Pairwise root-move ranking loss | Research track, not first implementation | It directly encodes move preferences but does not automatically produce a calibrated leaf evaluator for alpha-beta. Value/WDL training is the established first control. |
| Full threat features | Defer | High implementation and update cost; our one-core Numba engine cannot assume Stockfish's optimized SIMD economics. |
| Countermoves/history maluses | Rejected in current form | The isolated challenger scored 40.0% over 20 development pairs. |
| Aspiration windows | Already present | V7 uses a 45 cp window with full-window retry. |
| Fixed-array TT | Already present | V7 has a bounded direct-mapped NumPy/Numba TT. |
| Null-move pruning | Already present | V7 uses guarded R=2 null-move pruning. |
| Opening book from revealed starts | Low immediate priority | Among 31 downloaded rated PGNs there were 29 unique starts; after excluding a duplicate file, only one genuine exact start repeated. A broad book is worth work only after measured coverage. |
| Small tablebases | Useful polish, not the main gap | They prevent errors in tiny endings, but the observed costly errors occur with substantially more material. |
| AlphaZero/Lc0-style MCTS | Reject for this event | The runtime has one CPU core and no GPU. A batched policy/value network plus MCTS is mismatched to that environment and to the remaining time. |
| Maia-style human imitation | Reject for strength | Maia is designed to predict human moves, not maximize best-play strength. |
| Sunfish-derived rewrite | Reject | Sunfish is an excellent teaching engine, but its own documentation emphasizes minimalism and tactical speed limitations; shipping a port would also violate the event's third-party-engine rule. |

## The next experiment: mechanism classification

### Input corpus

Build one versioned JSON suite from the first major error in every rated game supplied by the team,
not only losses. Include the existing round-46, round-47, round-48 and round-50 positions and the
new round-52--55 candidates. For every case record:

- FEN and side to move;
- played move;
- teacher move and node limit;
- teacher score loss;
- clock before and time spent;
- whether the final game was won, drawn or lost;
- a status field distinguishing diagnosis from untouched validation.

The analyzer must cap or separately encode mate scores so forced-mate distance changes do not
pollute ACPL. It should emit machine-readable JSON rather than requiring tables to be copied from
terminal output.

### Equal-node ablations

Run each critical position at 25k, 100k, 300k, one million and, only when necessary, five million
nodes. Use fresh memory first and faithful game-memory replay for cases where the fresh result
differs from the played move.

Classify every case:

- **Depth-limited:** unchanged V7 finds the reference at higher nodes. Prefer exact speedups or
  time allocation.
- **NNUE-induced:** HCE-only finds the reference materially earlier while the blend/NNUE does not.
  Prioritize learned evaluation and data.
- **HCE-induced:** NNUE-only finds it materially earlier. Reconsider the 50% blend or expensive
  handcrafted terms.
- **Selectivity-induced:** no-LMR or no-null finds it at comparable nodes while unchanged V7 does
  not. Build a narrow guard or verification search; do not globally disable pruning.
- **Shared evaluation blind spot:** both evaluator modes remain wrong under nonselective search.
  Add training coverage or an explainable feature only after confirming the pattern across several
  cases.
- **Teacher-unstable:** the reference changes materially with teacher depth. Do not train or tune
  against it until relabelled.

The promotion-relevant output is the distribution across these categories. One famous position
must not decide the architecture.

## Two development branches, chosen by evidence

### Branch A: exact-score throughput

Profile the handcrafted evaluator by component inside full search. Pawn structure, rook-file and
king-shelter work is repeated across many nodes with unchanged pawn sets. Test a fixed-size Numba
cache whose key includes every state needed by the cached terms; king shelter cannot be keyed by
pawns alone.

Required gates:

1. exact score equality over the labelled corpus and randomized legal sequences;
2. identical fixed-node nodes, scores and root moves on the critical suite;
3. at least a clearly repeatable full-search NPS improvement (target 8% or more);
4. 20 development pairs against frozen V7;
5. independent validation only if development is non-negative.

An exact speedup is attractive because it can deepen every tactical and positional decision
without changing what a leaf means. If profiling shows a negligible gain, stop immediately.

### Branch B: search-distribution-aware NNUE

Take this branch only if the ablation corpus shows that learned evaluation is a common cause.

Generate data from positions V7 actually evaluates: quiet quiescence leaves, principal-variation
leaves, and root alternatives around large teacher disagreements. Keep generic balanced positions
for coverage, but do not let them dominate. Split by source game before sampling so near-duplicate
positions cannot cross train and validation.

Use a compact control before a new architecture:

1. current 128-wide piece-square network trained on the new data;
2. the same network with the loss applied to the exact runtime blend;
3. only if one of those improves root classifications, a team-written 128-wide HalfKAv2_hm
   network;
4. only after that base works, four material-count output buckets.

Train toward WDL-space teacher values and, where trustworthy game outcomes exist, test an explicit
teacher/result mixture. The official NNUE documentation describes this interpolation and warns
that network size is an accuracy/performance tradeoff. Select checkpoints by three metrics
together: held-out WDL loss, full-search NPS, and critical-root classification. Paired games remain
the promotion authority.

Do not add the pairwise move-ranking loss until a value-trained control works. When tested, keep it
as a small auxiliary loss on root siblings and verify that leaf calibration and game strength do
not regress.

## Testing discipline for the remaining days

Stockfish's Fishtest process uses short-time-control screening followed by long-time-control SPRT,
often requiring far more games than we can run. The transferable lesson is not its exact sample
size; it is isolated changes, paired openings, predefined hypotheses and automatic stopping rather
than interpreting every noisy score as Elo.

For this competition:

1. **Mechanism gate:** critical suite and equal-node diagnostics.
2. **Integrity gate:** randomized make/unmake, NNUE rebuild parity, legal-move and timeout tests.
3. **Speed gate:** repeated full-search NPS on a pinned suite with no concurrent games.
4. **Development gate:** 20--30 fast pairs against frozen V7.
5. **Independent gate:** 20 fast pairs on a split never used for design decisions.
6. **External anchor:** fixed-node Stockfish only for milestone builds.
7. **Release gate:** official 120+0.5 colour-swapped smoke and extracted ZIP test.

Never combine development and validation scores to rescue a failed validation result. Never reuse
an inspected split as “unseen.” Do not promote on ACPL alone: ACPL can reveal where to investigate,
but only games measure search/evaluation interaction.

## Proposed schedule

### 7--8 September

- Keep V7 frozen and upload it if the dashboard validation passes.
- Sync the official harness update from `origin/main` before new release tests.
- Implement the machine-readable rated-game analyzer and V7 ablation profiles.
- Classify the top 15--25 costly decisions from all supplied games.
- Profile the exact pawn/HCE-cache opportunity.

### 8--9 September

- Build exactly one strength candidate selected by the classification results.
- Run integrity, critical-position and speed gates.
- Start development pairs only if those gates pass.

### 9--10 September

- Run independent validation for the surviving candidate.
- If evaluation is clearly implicated and compute remains, train the two compact neural controls;
  do not begin a broad architecture sweep.
- Add a small tablebase or book only if measured coverage justifies it and it cannot disturb search.

### 10 September to submission freeze

- Stop feature development.
- Run official-clock and extracted-package gates.
- Preserve V7 as rollback, record hashes and upload early enough for platform validation.

## Sources and evidence ledger

- [AI Chessathon agent contract](https://aichessathon.com/docs/agent-contract.md), AI Chessathon,
  current contract accessed 7 September 2026. Supports the one-core/no-GPU environment, process
  lifecycle, package limits and failure semantics.
- [AI Chessathon rules](https://aichessathon.com/docs/rules.md), AI Chessathon, accessed
  7 September 2026. Supports the upload deadline and engine/model restrictions.
- [NNUE technical documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md),
  Stockfish project, accessed 7 September 2026. Supports HalfKP/HalfKAv2 design, incremental
  accumulators, architecture-cost tradeoffs, quantization and WDL teacher/result losses.
- [Stockfish NNUE architecture source](https://github.com/official-stockfish/Stockfish/blob/master/src/nnue/nnue_architecture.h),
  Stockfish project, accessed 7 September 2026. Confirms that the current engine uses
  HalfKAv2_hm, threat/pair feature sets and multiple layer stacks; it is an architecture reference,
  not evidence that copying every feature helps this engine.
- [Creating a test on Fishtest](https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html)
  and [Fishtest FAQ](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-FAQ.html),
  Stockfish project, accessed 7 September 2026. Supports isolated STC/LTC testing, SPRT and limits
  on repeated tuning attempts.
- [Stockfish search source](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp),
  Stockfish project, accessed 7 September 2026. Confirms that modern strength comes from an
  interacting search system, including continuation and correction histories, rather than a
  checklist of independent switches.
- [AlphaZero paper](https://arxiv.org/abs/1712.01815), Silver et al., 2017, and
  [Lc0 architecture overview](https://lczero.org/dev/overview/), Leela Chess Zero project,
  accessed 7 September 2026. Support the policy/value-network-plus-MCTS characterization and its
  mismatch with this event's CPU-only runtime.
- [Maia paper](https://arxiv.org/abs/2006.01855), McIlroy-Young et al., 2020. Supports the claim
  that Maia optimizes human move prediction rather than best-play engine strength.
- [Sunfish repository](https://github.com/thomasahle/sunfish), Thomas Ahle, accessed
  7 September 2026. Supports its educational/minimal design and documented tactical-speed
  limitation of its NNUE variant.
- Repository evidence: `docs/LEARNED_EVALUATOR.md`, the versioned backtest summaries under
  `benchmarks/runs`, and the supplied rated PGNs through round 55. Fixed-node game analysis used
  the team's local Stockfish 18 executable at 100,000 nodes and was development-only.

Research covered the live competition contract, the current V7 source and recorded experiments,
all supplied rated PGNs available locally, Stockfish's primary NNUE/search/testing documentation,
and the primary descriptions of AlphaZero, Lc0, Maia and Sunfish. It stopped when the consequential
options had direct supporting or disconfirming evidence and further architecture cataloguing was
unlikely to change the immediate mechanism-classification decision.

## Limitations

The official opening set and opponents' implementations are unknown. Local wall-clock tests do
not reproduce the platform's exact EPYC core. The new PGN review uses a fixed-node teacher and no
platform search-depth logs, so it cannot by itself distinguish evaluation from selectivity. That
is precisely why the next step is the equal-node mechanism classification rather than another
uncontrolled candidate.
