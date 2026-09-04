# Internal research source: AI Chessathon competitive engine

Status: synthesis source for `docs/HYPERCOMPETITIVE_ROADMAP.md`. This is an internal
research ledger, not the user-facing deliverable. Facts were rechecked on 2026-09-04.

## Decision

Build and test two tracks:

1. Keep the current pure-Python `agent.py` and its known-good archive as the safety champion for
   the next trial.
2. Build a challenger whose hot path is a team-written Numba bitboard engine, using alpha-beta,
   a compact general opening oracle, a tuned classical evaluation first, and only then a small
   team-trained NNUE if it wins measured games.

Do not bet the submission on AlphaZero/MCTS, a large ONNX model, copied engine code or borrowed
weights. Do not replace the champion merely because a challenger looks stronger in a few games.

## Competition facts and implications

Primary source: https://aichessathon.com/docs

- Zip: at most 50 MB unzipped; `agent.py` at root; readable Python source; weights, books and
  tablebases may accompany it.
- API: `get_move(fen, time_left_ms) -> str`, returning legal UCI.
- Runtime: Python 3.12; torch 2.13 CPU, numpy 2.5.2, python-chess 1.11.2,
  onnxruntime 1.29 and numba 0.67; no install step or network; one CPU core; 2 GB RAM;
  read-only filesystem except 256 MB `/tmp`.
- Lifecycle: 60 seconds to import; one process per game; module state survives; the process retains
  its core after a move and pondering is allowed.
- Clock: 120 seconds plus 0.5 seconds per move; an illegal output, exception, OOM, init timeout or
  flag is a loss. Games stop at 300 plies and are adjudicated by material.
- Openings: curated positions close to level; public games show ordinary named openings several
  moves in, so a book that only knows the initial position is inadequate.
- Third-party runtime engines are forbidden, but engine-annotated training data is permitted.
  Any shipped model must be trained by the team. Books and tablebases are expressly permitted.
- Operations: six uploads per team/day; the latest passing upload plays; upload close is
  2026-09-11 11:00 London. The ladder only seeds the 13-round qualification Swiss. Final tiebreaks
  include earlier submission.

Primary rules: https://aichessathon.com/terms (version 2026-08-31.v3). Do not access hidden match
data or misrepresent authorship. Public concepts can be independently reimplemented, but code from
other engines should not be copied into the submission. Ask the organizer before deliberately
turning revealed competition positions into a targeted oracle; a broad public opening book is
already explicitly permitted.

## Current field context

Primary source: https://aichessathon.com/leaderboard

On the research snapshot the ladder had 158 teams and only one completed round, so the ordering and
1662/1500 ratings had essentially no discriminating value. More useful are the house-bot anchors:
Loki 3.0 is shown as CCRL 2428 and Zagreus 5.0 as CCRL 2168. Public games already show entrants
playing near-perfect games against strong house bots. This is evidence that merely beating the
starter two-ply minimax is not a competitive target. It is not enough evidence to estimate any
entrant's true rating.

## What comparable engines teach

- Stockfish source (`src/search.cpp`, `src/position.cpp`) is the strongest primary reference for
  architecture: iterative deepening, PVS, aspiration, array-based TT, history-guided ordering,
  qsearch, mate-distance pruning, selective extensions/reductions and guarded pruning. Use the
  ideas; do not copy code. Sources:
  https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp and
  https://github.com/official-stockfish/Stockfish/blob/master/src/position.cpp
- Boychesser, winner-level work from the 636-entry Tiny Chess Bot Challenge, is a compact proof that
  the useful subset is small: PVS, qsearch, TT, aspiration, null/reverse-futility/delta/futility/
  late-move pruning, LMR/history/IIR, check extension, strong move ordering, tapered evaluation and
  soft/hard time limits. The author estimates roughly 2750 CCRL blitz. This is evidence about
  feature priority, not a portable Elo guarantee. Sources:
  https://github.com/analog-hors/Boychesser and
  https://github.com/SebLague/Tiny-Chess-Bot-Challenge-Results
- Sunfish demonstrates how far a compact Python engine can go with MTD-bi, incremental PST and
  transposition tables, while its own improvement notes point toward faster board representations.
  Source: https://github.com/thomasahle/sunfish
- Black Numba demonstrates that a fully jitted team-written Python bitboard core is feasible. Its
  repository reports a move-generation increase from thousands to millions of nodes per second and
  uses packed moves, magic bitboards and import-time JIT compilation. This is not playing-strength
  evidence, but it directly de-risks the platform architecture.
  Source: https://github.com/Avo-k/black_numba
- Numbfish suggests that a small incrementally updated NNUE can compensate for lower search NPS,
  but its borrowed Stockfish weights, TFLite dependency and admitted chess-state bugs make it
  unsuitable to ship here. Use only the architectural lesson.
  Source: https://github.com/dimdano/numbfish
- Stockfish's official NNUE documentation explains sparse inputs, incrementally updated
  accumulators, clipped low-precision layers and the speed/quality trade-off. A HalfKP
  40,960x128 int16 first layer is about 10 MiB before minor layers; 256 units is about 20 MiB.
  Source: https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md
- AlphaZero is the wrong deployment shape. The published system couples a policy/value network to
  MCTS and was evaluated with accelerator-heavy compute, unlike one CPU core and a 120+0.5 clock.
  Source: https://arxiv.org/abs/1712.01815 and
  https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/

## Data assets

- The official Lichess database is CC0 and exposes hundreds of millions of engine-evaluated
  positions with FEN, evaluation and PV. It is legal training/label data under the Chessathon rules.
  Prefer the highest-depth evaluation per position, deduplicate by normalized position, remove
  near-duplicates by game, and split by game rather than by position.
  Source: https://database.lichess.org/
- Stockfish's official book repository provides millions of diverse opening positions, including
  UHO_Lichess and popular-position sets. Use them as neutral paired test positions. Do not confuse
  an EPD test-position set with a move book.
  Source: https://github.com/official-stockfish/books/blob/master/README.md
- `python-chess` has a memory-mapped Polyglot reader keyed by Zobrist hash and returns the
  highest-weight entry. This is deployment-safe in the fixed environment.
  Source: https://python-chess.readthedocs.io/en/latest/polyglot.html
- Syzygy WDL/DTZ is supported, but tablebase bytes compete with book/network bytes and only help a
  small portion of games. It is a later optional feature, not the first 50 MB allocation.
  Source: https://github.com/niklasf/python-chess/blob/master/docs/syzygy.rst

## Testing method

Stockfish Fishtest uses short time-control then long time-control SPRT rather than accepting a patch
from a tiny match. Its regression measurements use 60,000 games, and its FAQ warns about error bars,
selection bias and unfinished tests. We cannot reproduce that scale before the deadline, but the
method transfers:

- champion/challenger, identical build except one patch;
- paired colors from identical openings;
- quick rejection at fast time controls, then confirmation closer to 120+0.5;
- retain the complete W/D/L and paired outcomes; calculate confidence intervals/SPRT;
- never promote on a handful of games or a single leaderboard result.

Sources:
https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html,
https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-FAQ.html,
https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html and
https://official-stockfish.github.io/docs/stockfish-wiki/Regression-Tests.html

## Local evidence

The current 517-line agent is a useful safe champion: iterative deepening PVS, aspiration, LMR,
TT, qsearch, tapered hand evaluation, time reserve and an exception-safe legal fallback. Previous
local stress testing covered randomized and special positions and exact-zip games without a
technical failure. Its bottleneck is architectural: profiling put most time in Python evaluation,
qsearch, `piece_map()`, check detection and move sorting. Representative local measurements reached
only depth 3-4 in complex positions over several seconds, with roughly 12k search visits/s there.
These numbers are machine-specific, but the conclusion is robust: adding more Python heuristics
will eventually reduce rather than increase effective depth.

An environment probe also found that Python wall-clock functions do not compile inside Numba 0.67
nopython mode. The Numba challenger therefore needs either (a) a Python timer thread controlling a
shared stop flag while the `nogil=True` search polls it, or (b) conservative calibrated node budgets.
The first design must be stress-tested because shared-memory visibility and shutdown correctness are
part of the engine's clock safety. Official Numba documentation confirms efficient fixed NumPy array
access and `nogil=True`, but the stop-flag design remains our implementation inference:
https://numba.readthedocs.io/en/stable/reference/numpysupported.html and
https://numba.readthedocs.io/en/stable/user/jit.html

## Evidence gaps and how to close them

| Question | Current evidence | Confidence | Required experiment |
|---|---|---:|---|
| Does a general book hit curated starts? | Public games show common named openings at roughly ply 11-16; books allowed. | Medium | Replay every public FEN against candidate book and report coverage, without targeting hidden data. |
| Will Numba beat the Python champion? | Architecture proof and local Python profile. | High on speed, unknown on Elo | Perft first, then fixed-node tactical and paired game benchmarks. |
| Is NNUE better than tuned HCE here? | Strong external architecture, no team-trained net yet. | Low | Train both, compare NPS and paired Elo on the same compiled core. |
| Does pondering add useful depth? | Explicitly allowed; state persists. | Medium | Measure ponder-hit rate, extra completed depth and clock safety over 1,000 games. |
| Which pruning margins win? | Strong-engine precedents only. | Low | One-parameter/one-patch paired SPRT; include zugzwang tests for null-move changes. |
| Are small tablebases worth package space? | Mechanically supported, likely rare. | Low | Measure occurrence and conversion failures in representative games before allocation. |

## Recommended order

1. Upload/freeze safe champion; archive hash and validation log.
2. Create the real test lab and benchmark against Loki/Rustic/Zagreus-class local opponents.
3. Add broad book coverage and low-risk persistence/time improvements to the champion only if they
   pass reliability and paired strength gates.
4. Build a fully jitted bitboard core behind python-chess root validation.
5. Add search features one at a time.
6. Tune a compact classical evaluation from data.
7. Train and test a small NNUE; ship only if net Elo after its NPS cost is positive.
8. Add pondering; consider tiny tablebases last.

## 2026-09-04 live-game and recovery addendum

The three supplied PGNs identify our side through the observed clock spend: Black in rounds 1 and
3, White in round 2. Stockfish 18 at 100,000 nodes per position found repeated 100-450 centipawn
losses in the middlegame; none of the games was lost by flag, crash, illegal output, or opening
failure. Important examples were 12...O-O-O and 14...a5 in round 1, 27.exf5/28.Qc5/33.Rf7 in
round 2, and 24...Rfe6 plus later passivity in round 3. Clocks retained about 45-70 seconds late.

Archive audit findings:

- `alpha-gambit-main.zip` does not compile because its future import is not at the beginning of
  `agent.py`.
- The evolved `aichessathon-starter-main.zip` learned-evaluator champion lost 21-4-7 to our prior
  agent at 3+0.05 locally. Its own 98-experiment history records repeated failures of static model
  improvements to convert into playing strength. Its one retained late feature was cheap mobility.
- Direct Lichess evaluated positions are preferable to Kaggle mirrors. The page exposed
  394,669,566 evaluated positions on this date and documented FEN/evaluation/depth/PV JSONL.

Recovery experiments against the exact accepted artifact (SHA-256 prefix `5A019AE9`) rejected a
mobility scalar at 50.0% and first-ply quiet-check quiescence at 46.7%. The retained complete build
combined redundant-work removal, 32-node clock polling, a bounded evaluation cache, a bounded
persistent depth-preferred TT, and more assertive use of clock above 60 seconds. It scored 15-7-8
(61.7%) over 30 paired games against the accepted artifact and 13-7-0 (82.5%) over 20 games against
starter minimax, with no technical failures. Final archive SHA-256:
`391082CCF4E635D65E17C26603064D5AA58727373EF308155BF43BA9489EED61`.

## 2026-09-04 round-4 addendum

Source PGN: `aichessathon-round-4-malvoo.pgn`. The agent was Black. Stockfish 18 at 200,000 nodes
per position scored 12...Rxc3 as a 155 cp loss, followed by 13...Nxd5 (118 cp), 14...Qc3
(119 cp), 16...Nxc4 (95 cp), 17...h5 (166 cp), and 20...f6 (224 cp). The current local champion
reproduced all of these important decisions from fresh probes.

At the critical pre-12...Rxc3 FEN, the champion chose Rxc3 at completed depths 3 and 4 over 1-8
second limits. At 16 seconds it completed depth 5 and chose b5. Stockfish 18 at one million nodes
scored Ng6 about -31 cp for Black, Rxc3 about -172 cp, and b5 about -180 cp. Thus added depth changed
the move without resolving the evaluator/search ranking. A general 60 cp rook/minor imbalance term
replaced Rxc3 with b5 and scored 11-10-9 (53.3%) against the champion; rejected and reverted.

The supplied Alpha Gambit archive required moving its misplaced future import before it compiled.
With no logic changes, our champion scored 12-5-13 (48.3%) over 30 paired games. Alpha Gambit is a
peer/style benchmark, though it selected Ng6 in the critical round-4 position. A new development-only
fixed-node Stockfish harness produced a full SF-500 result of 3-13-14 (31.7%) over 30 paired games
from 15 positions, with no technical failures. This is the new uphill benchmark. Stockfish is never
included in the submission; current Chessathon
rules permit engine-labelled offline work but prohibit third-party engines, published chess nets,
or runtime databases of engine answers in the submitted artifact.
