# Round 4 diagnosis and competitive recovery plan

Date: 4 September 2026  
Audience: AI Chessathon team  
Decision horizon: trial improvements now; locked build by 11 September 11:00 London

## Executive decision

Do not replace the current reliable submission with another hand-tuned Python heuristic or a
last-minute neural evaluator. Preserve it as the champion while building a team-written Numba
bitboard challenger. Use three distinct test levels:

1. the current submission for paired regression and promotion;
2. the repaired Alpha Gambit example as a peer/style opponent;
3. Stockfish 18 at fixed node limits as an offline, progressively stronger benchmark.

The fourth loss is not evidence that the agent is broken. It is evidence that its current search
and evaluation ceiling is below the competitive field. The game was lost by a systematic
misvaluation beginning with `12...Rxc3`, not by a crash, illegal move, time loss, or forced bad
opening.

## Round 4 forensic result

The PGN begins from:

```text
r1bqkb1r/pp3ppp/2np4/1N1Pp3/8/8/PPP2PPP/R1BQKB1R b KQkq - 0 8
```

Clock use and local reproduction identify our agent as Black. Stockfish 18 was used only as an
offline diagnostic at 200,000 nodes per position; neither it nor any of its code or weights is in
the submission.

| Black move | Approximate loss | Position after move | Interpretation |
|---|---:|---:|---|
| `8...Ne7` | 0.04 pawn | -0.43 | Fine |
| `9...Bd7` | 0.00 | -0.10 | Best move |
| `10...Qa5` | 0.02 | -0.14 | Fine |
| `11...Rc8` | 0.72 | -0.68 | First warning; `...f5` was stronger |
| `12...Rxc3` | 1.55 | -1.77 | Principal losing decision |
| `13...Nxd5` | 1.18 | -3.00 | Compensation overestimated |
| `14...Qc3` | 1.19 | -4.37 | Fails to stabilize |
| `16...Nxc4` | 0.95 | -5.91 | Further material/coordination loss |
| `17...h5` | 1.66 | -7.36 | Development and king safety ignored |
| `20...f6` | 2.24 | -9.70 | Decisive; queen retreat was required |

Negative scores are from Black's perspective. Values are approximate and engine-depth dependent,
but the error sequence is too large to be ambiguous.

The current local champion reproduced `Rxc3`, `Nxd5`, `Qc3`, `Nxc4`, `h5`, and `f6` from fresh
probes. Therefore this was not merely an outdated upload.

## Why the engine chose the sacrifice

The critical position before `12...Rxc3` is:

```text
2r1kb1r/pp1bnppp/3p4/q2Pp3/8/2NB1Q2/PPP2PPP/R1B2RK1 b k - 8 12
```

Search scaling on the unmodified champion produced:

| Search limit | Completed depth | Nodes | Move |
|---:|---:|---:|---|
| 1 s | 3 | 12,224 | `Rxc3` |
| 2 s | 3 | 23,584 | `Rxc3` |
| 4 s | 4 | 29,887 | `Rxc3` |
| 8 s | 4 | 29,689 | `Rxc3` |
| 16 s | 5 | 163,456 | `b5` |

The 8-second run ended early because the iterative-deepening completion predictor declined to
start the next iteration. More importantly, `b5` is not an objective cure: a deeper Stockfish
probe scored `Ng6` around -0.3 pawn, `Rxc3` around -1.7, and `b5` around -1.8. Our evaluator/search
ranked `Rxc3` near -0.2 through depth 4 and still could not distinguish the bad alternatives at
depth 5.

This establishes two independent deficiencies:

- **throughput:** the Python-object search reaches only depth 3-4 on the live decision budget;
- **evaluation/selectivity:** it badly overvalues the structural compensation and does not
  understand the long-term rook/minor imbalance, development, and coordination.

A 60-centipawn exchange-imbalance correction removed `Rxc3` but substituted `b5`. It scored only
11-10-9 (53.3%) against the champion, below the 55% promotion gate, and was rejected. This is why
the live position must be a regression test rather than a position to hardcode or hand-tune.

## Benchmark hierarchy

### Champion: regression authority

Every challenger plays paired colors from identical positions against the exact archived
`submission.zip`. Promotion requires:

- no crashes, illegal moves, flags, or state corruption;
- at least 55% in a fast screen;
- confirmation over at least 100 paired games and a near-real-clock sample;
- no serious regression on the four supplied live-game tactical positions.

### Alpha Gambit: peer, not boss

The supplied Alpha Gambit archive did not initially compile because a future import appeared after
ordinary imports. A temporary benchmark copy was repaired by moving that import only; no engine
logic was changed or copied into our submission.

Our agent scored 12 wins, 5 draws, and 13 losses (48.3%) over 30 paired games against it. Alpha
Gambit did choose `12...Ng6` in the round-4 position, so it is useful as a stylistically different
peer and tactical regression opponent. It is not a solid winning benchmark.

### Fixed-node Stockfish: strength ladder

`tools/stockfish_arena.py` runs an official local Stockfish executable through `python-chess`, one
thread, with a fixed number of nodes per move. It is development-only and cannot be packaged by
the root-only submission builder. The complete 15-position baseline at 500 nodes per move gave our
agent 3 wins, 13 draws, and 14 losses (31.7%) over 30 paired games, with no technical failures.
This is the desired kind of uphill benchmark: clearly stronger, but not so dominant that progress
is invisible. All 30 PGNs are retained under `benchmarks/stockfish500/full/`.

Use progressive levels:

| Tier | Nodes/move | Purpose |
|---|---:|---|
| SF-500 | 500 | Current winning benchmark; target at least 45% over a larger holdout |
| SF-2K | 2,000 | Add after the SF-500 target is repeatable |
| SF-10K | 10,000 | Strong tactical/positional stress |
| SF-50K | 50,000 | Long-term ceiling/reference, not an immediate pass gate |

Fixed nodes reduce hardware and scheduling noise. Game score still needs paired openings and many
games; a 30-game result remains directional, not an Elo estimate. The benchmark follows the official
Stockfish design's own use of node-limited search and the broader champion/challenger principle
used by [Fishtest](https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html).

## Engineering plan

### Phase 1: test infrastructure and frozen champion — complete

- Preserve the validated pure-Python champion and exact archive hash.
- Keep every supplied live PGN as a holdout regression source.
- Retain all SF-500 losses under `benchmarks/stockfish500/`.
- Record W/D/L, colors, position pair, termination, and PGN for every experiment.
- Never promote because one target position changed move.

### Phase 2: compiled legal board core — highest priority

Create a team-written Numba engine behind the existing safe `get_move` boundary:

- 12 piece bitboards, occupancy, side, castling, en-passant, rule-50 clock;
- packed integer moves and fixed NumPy move buffers;
- precomputed pawn, knight, king, and sliding attacks;
- make/unmake with an incremental Zobrist key and reversible state stack;
- pseudo-legal generation followed by king-safety filtering initially;
- FEN import and root validation through `python-chess`.

Correctness gates before search:

- standard perft positions through practical depths;
- exact legal-move-set comparison against `python-chess` on at least 10,000 random positions;
- make/unmake restores every state field and hash;
- explicit castling, en-passant, promotion, check/evasion, repetition, and rule-50 tests;
- import-time warm-up compiles every JIT signature within the current 90-second allowance.

The live [agent contract](https://aichessathon.com/docs/agent-contract.md) provides one core, Numba,
90 seconds of initialization, persistent per-game memory, and no native binaries. That makes a
fully JIT-compiled Python source engine the clearest legal path to a step change.

### Phase 3: simple compiled search before clever pruning

Implement, in order:

1. iterative deepening negamax and hard stop flag;
2. transposition table with depth/bound/move/generation;
3. PVS and aspiration windows;
4. capture/promotion quiescence with correct in-check evasions;
5. TT move, good captures, killers, and quiet history ordering;
6. tapered incremental material/PST evaluation;
7. repetition and rule-50 handling.

Only after this beats the champion should we test null-move pruning, futility, SEE/delta pruning,
LMR refinements, check extensions, and continuation histories individually. The official
[Stockfish search source](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)
shows that mature strength comes from interacting search, TT, histories, correction histories,
selectivity, and time management—not from one isolated trick. Its own comments warn that parameters
tuned at one time control require verification at longer controls.

### Phase 4: trained evaluation

Do not begin with Kaggle move imitation. Generate or sample diverse legal positions, label them
offline with Stockfish, and train team-owned weights. The rules allow engine-labelled training but
forbid shipping an engine, a published chess network, or a runtime database of engine answers.
See the current [competition rules](https://aichessathon.com/docs/rules.md).

First train a linear/tapered HCE over explicit features because it is fast and interpretable. Split
by source game, reserve all competition PGNs as holdout, and report both prediction loss and game
strength. Test a small NNUE only after the compiled board can update its accumulator incrementally;
the official [NNUE documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
emphasizes sparse, incrementally updated features and the evaluation-quality/search-speed trade-off.

### Phase 5: clock and pondering

The current engine left substantial time unused, but spending more cannot repair a wrong evaluator.
After the compiled core is stable:

- measure completed depth rather than only nodes/s;
- use soft and hard deadlines with partial-root results only when rigorously tested;
- retain TT/history between moves;
- add pondering last, with a generation token, cancellation, and stress tests so only one search
  owns mutable state and the single core.

## Exact decision rules

Reject a candidate immediately if it:

- fails legality, perft, time, memory, or extracted-zip tests;
- improves a live position by hardcoding or competition-position lookup;
- scores below 50% in an initial paired champion screen;
- loses more than five percentage points at SF-500 without a compensating, independently confirmed
  champion gain;
- relies on a static-evaluation metric without a playing-strength gain.

Promote only when it:

- clears all reliability gates;
- scores at least 55% against the champion in screening and remains positive in confirmation;
- improves or preserves SF-500 score on a disjoint opening set;
- avoids the major four-game blunders at representative time without special-case code;
- remains within package, dependency, import, memory, and clock constraints.

## Git Bash commands

Run Alpha Gambit as a peer only after repairing its misplaced future import in a temporary copy.
The stronger reproducible benchmark is:

```bash
cd /c/Users/adirj/OneDrive/Documents/GitHub/aichessathon

./.venv/Scripts/python.exe -m tools.stockfish_arena \
  --candidate . \
  --engine "C:/Users/adirj/AppData/Local/Temp/stockfish18-analysis/stockfish/stockfish-windows-x86-64-avx2.exe" \
  --nodes 500 \
  --base-ms 10000 \
  --increment-ms 100 \
  --pgn-dir benchmarks/stockfish500/full
```

Raise `--nodes` only when the lower tier produces enough wins and draws to distinguish candidates.

## Immediate next action

Start the Numba board/move-generation core now while the existing `submission.zip` remains the
active safety build. The first deliverable is not a stronger move—it is a correct compiled board
that passes perft and random differential tests. Once that exists, the search can reach the depth
at which evaluation improvements become meaningful. Until then, more ad hoc HCE terms are likely
to exchange one ladder blunder for another.

## Limitations

- Thirty Alpha Gambit games and the initial ten SF-500 games have wide uncertainty.
- Stockfish evaluations vary with depth, but 1-2 pawn errors are large enough that the diagnosis is
  robust.
- Public competition PGNs are a tiny and selected sample; they are a holdout regression suite, not
  representative training data.
- A Numba rewrite has the highest ceiling and the highest correctness risk. The champion must stay
  available until the compiled challenger passes both legality and playing-strength gates.
