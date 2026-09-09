# Closing the gap: what to learn from Toby Coad, and what to build next

Research date: 9 September 2026. This is a short-horizon engineering decision for the
qualifier ending at 11:00 London time on 11 September, not a generic chess-engine roadmap.

## Executive decision

We should **not reproduce Toby Coad's engine feature by feature**. We should reproduce the
parts of his method that transfer:

1. measure the actual bottleneck before coding;
2. make large evaluator/representation experiments as well as small search experiments;
3. preserve a frozen champion and put every behavioural change through paired testing;
4. keep rejected experiments rejected;
5. validate a release at the real clock and in the real package.

The best immediate bet is not a book, tablebases, opponent modelling, or a wholesale search
rewrite. It is to determine whether the remaining 25% handcrafted-evaluation blend is both
hurting decisions and halving evaluation throughput. The already prepared Round 79 matrix does
exactly that. Its result decides between two serious lanes:

- If pure KingNet fixes the critical choice, test **90% and 100% KingNet** immediately. This is
  the only remaining change that may deliver a large speed gain without new search code or new
  weights.
- If HCE fixes it and pure KingNet does not, the network is the binding weakness. Spend the main
  effort on a better mixed dataset and a modestly stronger KingNet, while leaving the uploaded
  75% build safe.
- If only no-LMR or no-null fixes it, build one conservative search bundle behind independent
  switches, then gate the bundle. Do not infer this from one position alone: confirm it on the
  full platform-loss suite.

The current uploaded KingNet75 plus qcache remains the release baseline until a candidate passes
those gates. Two ladder losses do not reverse a 61-pair SPRT result; neither do they prove that the
candidate is globally strong enough.

## Why copying Toby literally is the wrong strategy

### It is prohibited at the implementation level

The competition permits ordinary published ideas and engine-labelled training, but prohibits
shipping another engine, a port or translation of one, or another team's network. The submitted
source must also be explainable to judges. Therefore Toby's public repository is evidence about
promising hypotheses, not a source tree or weight file for us to transplant. See the canonical
[competition rules](https://aichessathon.com/docs/rules.md) and
[agent contract](https://aichessathon.com/docs/agent-contract.md).

### His visible engine includes failures as well as wins

The inspected public snapshot is commit
[`917d4fac`](https://github.com/TobyCoad/aichessathon-starter/tree/917d4fac9f1005ad07ead9a9e5bcf7bbd83d2151).
It contains many disabled or rejected experiments alongside active features. His own journal
records, among other examples, a 1024-wide predecessor that improved an offline metric but was
about 35 Elo worse than its 256-wide control, repeatedly rejected staged move generation, and
book experiments that were difficult to measure. This is valuable negative evidence: the lesson
is not "use a 1024-wide net" or "add every pruning technique". The lesson is to run the control.

Search mechanisms are coupled to board representation, evaluator scale, move ordering, and other
pruning. Current Stockfish search likewise combines TT information, static evaluation, history,
node type, improving state, and pruning conditions rather than treating each feature as a
standalone checklist ([official search source](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)).
Our previous no-LMR, no-null, ordering, and qsearch experiments already demonstrate that a useful
idea in another engine need not transfer with the same constants.

### His durable advantage is the iteration system

The public repository has an automated experiment loop, immutable challengers, crash gates,
paired openings, sequential tests, full-clock checks, and a journal of negative results. That is
the most transferable advantage. Stockfish's own testing guidance says to keep patches focused
and test one idea at a time, while its FAQ recommends short-time-control screening followed by a
second stage and warns about selection bias and large error bars
([test guide](https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html),
[FAQ](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-FAQ.html)).

## What Toby actually did that mattered

The following is reconstructed from the public snapshot and its self-reported experiment notes.
The numbers are Toby's measurements, not independently reproduced results.

### 1. Representation and training produced the step changes

His current evaluator uses mirrored king-conditioned sparse inputs, a much wider accumulator, a
pairwise dense head, dual activations, and material-dependent output buckets. More importantly,
his network lineage moved from engine-heavy data to mixed human/engine training with disjoint
validation and repeated checkpoint selection. His v12 notes report a mixed-data improvement; v15
changes the evaluator/data while leaving search unchanged; v16f changes the learned head and
output treatment again.

This aligns with the purpose of NNUE: sparse position features can be updated incrementally so a
CPU alpha-beta search can afford a learned evaluation at very high frequency. The official NNUE
documentation stresses the trade-off between network capacity and inference cost and supports
training in WDL/probability space, including interpolation between teacher evaluation and game
result ([official NNUE documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)).

### 2. He compiled the board/search hot path and profiled leaf cost

Toby's current engine has a compiled board/search path and handwritten Numba/NumPy inference. His
notes explicitly measure evaluation kernels and reject optimisations that are slower in Python,
even when they are conventional in C++. This matters because an `int8` idea or accumulator cache
is not automatically faster under Numba on the competition CPU.

Our own checked-in diagnostics expose the comparable bottleneck: older measurements put the
50:50 blended evaluator near 165k nodes/s, while each evaluator alone was near 287--289k nodes/s,
and roughly four fifths of nodes were qsearch nodes. Current KingNet is 128-wide with a 32-unit
head and still evaluates KingNet plus HCE on every qcache miss. The precise current margin must be
remeasured, but the structural double-evaluation cost remains.

### 3. He built a selectivity system, not isolated buzzwords

The active snapshot includes dynamic null-move logic, reverse futility, SEE pruning, richer
history, more aggressive/contextual LMR, bucketed TT behaviour, qsearch TT/eval reuse, aspiration
windows, and endgame/search guards. It also leaves several fashionable mechanisms off. We should
interpret that as a sequence of interacting, tested policies—not permission to paste ProbCut,
singular extensions, and a clustered TT into one untestable patch.

## Our current position

| Area | Current champion | Practical implication |
|---|---|---|
| Evaluator | 16 king buckets, 128 accumulator, 256→32→1 head | King-conditioned, but far smaller/simpler than Toby's current representation |
| Blend | 75% KingNet, 25% HCE | Better in the completed match, but pays for two evaluators and can combine conflicting opinions |
| Model size | 6.3 MB | Plenty of the 50 MB archive budget remains; CPU cost, not file size alone, is the constraint |
| Dataset | up to 4M accepted Fishnet rows; about 500k held out | Too narrow for a final evaluator programme and weakly stratified |
| Targets | teacher centipawns mapped to probability | No game-result target or source/material-aware validation |
| Filters | non-check, non-terminal, teacher move non-capture | Sensible quiet filter, but no tactical-stability score or near-duplicate control |
| Search | PVS, aspiration, direct-mapped TT, qsearch, SEE ordering, killers/history, fixed LMR, fixed R=2 NMP | Sound intermediate search, not a modern coordinated selectivity system |
| Testing | paired backtester, pentanomial SPRT, fixed-node diagnostics, loss suites | Good foundation; the main risk is reacting to tiny samples or changing too many axes |

The key limitation in the current data pipeline is concrete. `pack_nnue_data.py` partitions by
Parquet row group, stores only sparse features, side to move, and a clamped teacher score, and
stops after a requested row count. It does not store source game identity, result, material band,
or stability. `train_king_factored.py` trains one shuffled dataset with teacher-probability MSE and
selects the best epoch on that one validation set. Increasing `--train-target` alone would make a
larger version of the same experiment, not a Toby-quality training programme.

## The decision experiment already in front of us

Run the Round 79 `f3` matrix at 100k, 300k, and 1M nodes for:

1. current KingNet75;
2. pure KingNet;
3. HCE only;
4. KingNet75 without LMR;
5. KingNet75 without null move.

Interpret it as a routing probe, not a promotion test:

| Result | Meaning | Next action |
|---|---|---|
| KingNet100 selects `Na4`, especially by 300k/1M | HCE disagreement or double-eval depth is implicated | Materialise 90 and 100 blends; fixed-node loss-suite comparison, then paired games |
| HCE0 selects `Na4`, KingNet100 does not | Learned evaluation is the likely weakness | Keep 75 uploaded; begin data/representation lane |
| no-LMR or no-null alone selects `Na4` | Selectivity is hiding the move | Repeat on every stable platform regression before coding the replacement |
| All profiles find `Na4` only at 1M | Primarily a depth/throughput problem | Prefer cheaper evaluation or exact speed; avoid more expensive nets |
| None finds it | The 500k teacher label may be unstable, or both eval/search lack the concept | Relabel deeper and with a second legal teacher before using it as a hard target |

One position cannot establish Elo. A hypothesis advances only if it also improves the stable loss
suite without breaking previously solved cases.

## Recommended work until the qualifier lock

### P0 — protect the uploaded champion

Do not alter `current/` while experimenting. Keep the uploaded KingNet75+qcache build reproducible.
Before another upload, require zero technical failures, a two-colour real-clock smoke, `make gate`,
archive inspection, and model/source provenance recorded. Reliability has infinite practical Elo
when a crash is a loss.

### P1 — exploit the blend opportunity first

If the routing matrix supports it, test 90 and 100 using the same code and weights. This experiment
has unusually good economics:

- no training uncertainty;
- minimal implementation risk;
- removes some or all HCE work from qcache misses;
- directly tests whether evaluator disagreement caused a real ladder error.

Gate it in this order: fixed-node critical suite, clean single-process throughput with reversed
run order, 20--25 pair disaster screen, then SPRT against the frozen current champion. Use an
independent opening split before promotion. Do not choose 90 or 100 because one produces the
highest score in a ten-game run.

### P2 — build one search candidate only if the diagnostic supports search

The highest-value coherent candidate for our engine is:

1. retain current PVS/TT/qsearch;
2. dynamic null-move reduction based on depth and evaluation margin, with verification at deeper
   nodes and conservative endgame guards;
3. moderate contextual LMR based on depth, move index, PV/check/killer/history status;
4. shallow reverse futility and late-quiet/SEE pruning with wide initial margins;
5. history bonuses **and maluses**, including capture history if the board encoding permits it
   cheaply.

Each mechanism must have a switch and invariant tests. First diagnose them separately on the
critical suite; because time is short and small effects cannot resolve individually, combine at
most a pre-declared, coherent search bundle for the expensive match gate. Do not add ProbCut,
singular extensions, a new TT layout, and new LMR simultaneously. Those are post-qualifier work
unless the first search bundle wins decisively.

### P3 — prepare, but do not rush, the next evaluator

The best serious next evaluator is not simply "our current model with more rows" and not a blind
1024-wide replica. Build the data contract first:

- stable game/source identity and exact/near-duplicate control;
- human/game-result information where licensed and available;
- teacher settings and label stability;
- piece count, phase, evaluation band, quiet/tactical category;
- deliberately balanced 9--16-piece coverage, because our platform failures concentrate there;
- separate held-out human, engine, and material-band reports.

Then train in this order:

1. a same-architecture control on the improved mixture;
2. the same representation with cheap material output buckets;
3. a 256-wide or 512-wide candidate only after benchmarking untrained/random-weight inference;
4. mirrored/finer king conditioning as a separate architectural axis;
5. pairwise/dual-activation head only after a clean control exists.

Use WDL/probability targets with a tunable teacher/result mixture. Keep several checkpoints;
offline validation eliminates bad nets, but engine games select the winner. Toby's own width
failure is exactly why 1024 must not be assumed superior.

With less than two days, only ship a newly trained net if its verifier passes, its critical
behaviour is sane, and it wins a meaningful paired screen. Otherwise finish this programme after
the qualifier rather than replacing a proven model with an unvalidated one.

## Work we should stop doing now

- **Do not chase the latest ladder PGN one move at a time.** Add stable failures to a diagnostic
  suite, but never tune a rule specifically to a named opponent or FEN.
- **Do not try to "beat Stockfish" as the development objective.** A fixed low-node Stockfish is a
  useful opponent family, not proof against the ladder distribution.
- **Do not prioritise 3/4-man tablebases.** They improve cleanup, not the 9--16-piece mistakes that
  dominate our evidence.
- **Do not build a book without measured coverage and value.** Starts are curated and unpublished;
  Toby's own notes show that book hits can be sparse and harmful.
- **Do not spend another long match resolving a 1--5% speed tweak.** At this horizon, exact speed
  work needs roughly 10% repeatable NPS or it belongs in a later bundle.
- **Do not use ACPL as the promotion metric.** It depends heavily on teacher settings, game phase,
  and which positions each engine reaches. Paired score is the target; stable critical-position
  analysis explains the result.
- **Do not infer strength from unfinished tests or tiny samples.** Sequential testing exists to
  prevent exactly that error.

## Concrete 36-hour schedule

### First 3 hours

- Complete and inspect the five-profile Round 79 matrix.
- Check the same profiles on the existing stable platform-loss suite at 300k nodes.
- Freeze the resulting hypothesis before looking at games.

### Hours 3--12

- If evaluator/blend wins the diagnosis: build 90/100 same-weight challengers and run their
  fixed-node plus 20--25-pair screens concurrently on genuinely free cores.
- If search wins: implement the smallest diagnostic-supported search mechanism, not the entire
  modern-search list.
- In parallel, extend the dataset manifest/schema and mixture builder; do not start a huge run
  against the old data contract.

### Overnight

- Run one or two SPRTs only: the leading blend/evaluator candidate and, if ready, the leading
  search candidate.
- Stop candidates on technical failure or clear lower-bound rejection.
- Preserve all summaries and exact candidate hashes.

### Final day

- Independent opening split and different opponent family for the winner.
- Real-clock two-colour smoke and clock-floor inspection.
- `make gate`, package, inspect size/content, upload with enough time to recover from validation.
- No architectural change after the release cutoff chosen by the team.

## Promotion thresholds

A change is eligible to consume a long match only if it meets at least one of these:

- exact and repeatable throughput gain around 10% or more;
- fixes multiple independently labelled platform failures without new regressions;
- exceeds 55% in a pre-declared 50-game screen with zero failures;
- delivers a material, source-consistent validation improvement for a new evaluator **and** does
  not lose search depth catastrophically.

Promotion still requires paired evidence against the frozen champion. Fishtest's warning is
important here: repeated tries create selection bias, individual Elo estimates have large error
bars, and passing estimates are themselves biased upward. Treat four or five variants of one
idea as the exploration limit, not an invitation to keep rolling until one looks lucky.

## Bottom line

Toby's engine is not ahead because it contains a secret named feature. It is ahead because it
combined a better learned representation, much broader data work, a faster compiled hot path, a
coherent search system, and vastly more disciplined experiment throughput. We can adopt that
method immediately. We cannot safely recreate the accumulated implementation in 36 hours.

Our best chance of another large jump before lock is therefore:

1. route Round 79 correctly;
2. exploit a pure/near-pure KingNet blend if the evidence supports it;
3. test one coherent search candidate only if selectivity is implicated;
4. protect the current uploaded release;
5. build the next evaluator from a better data contract rather than merely a larger row count.

That is more likely to close real Elo than copying Toby's current list of switches.

## Sources

- [AI Chessathon competition rules](https://aichessathon.com/docs/rules.md)
- [AI Chessathon agent contract](https://aichessathon.com/docs/agent-contract.md)
- [Toby Coad public engine, inspected commit](https://github.com/TobyCoad/aichessathon-starter/tree/917d4fac9f1005ad07ead9a9e5bcf7bbd83d2151)
- [Toby Coad continuous experiment notes](https://github.com/TobyCoad/aichessathon-starter/blob/917d4fac9f1005ad07ead9a9e5bcf7bbd83d2151/overnight/continuous/NOTES.md)
- [Toby Coad experiment journal](https://github.com/TobyCoad/aichessathon-starter/blob/917d4fac9f1005ad07ead9a9e5bcf7bbd83d2151/overnight/JOURNAL.md)
- [Toby Coad v17 evaluator report](https://github.com/TobyCoad/aichessathon-starter/blob/917d4fac9f1005ad07ead9a9e5bcf7bbd83d2151/overnight/eval/v17/eval.md)
- [Stockfish search source](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)
- [Official NNUE architecture and training documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
- [Official Fishtest contributor testing guide](https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html)
- [Official Fishtest FAQ](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-FAQ.html)
