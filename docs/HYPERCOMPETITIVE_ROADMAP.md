# Hypercompetitive AI Chessathon engine roadmap

> Historical planning snapshot from 4 September. The pure-Python champion and dated schedule have
> been superseded. See `CURRENT_STATE.md` and `docs/EXPERIMENT_LEDGER.md` before acting on it.

Research snapshot: 4 September 2026  
Objective: submit a reliable trial build immediately, then maximize strength before the
11 September 11:00 London lock.

## Executive verdict

The best realistic architecture is a **team-written Numba bitboard alpha-beta engine**, backed by
a compact general opening oracle and a data-tuned evaluation. Start with a tuned classical
evaluation; add a small team-trained NNUE only if its playing-strength gain exceeds the search
speed it costs. Keep `python-chess` at the public boundary to parse the supplied FEN and validate
the chosen root move, even after the internal board becomes custom. This hybrid makes the fast
core ambitious without allowing a move-generation defect to become an automatic illegal-move
loss.

The current pure-Python `agent.py` should be frozen as the **champion and emergency submission**.
It is much better than a placeholder, but profiling shows that complex positions reach only about
depth 3-4 in several seconds on the development machine. Further decorating that Python object
search with sophisticated pruning will have diminishing returns. The route to a step change is to
compile the whole hot path—board, make/unmake, move generation, evaluation and search—not merely
one evaluation function.

For the next trial, reliability beats an unfinished rewrite. The immediate build should therefore
remain the current champion unless a small, isolated improvement clears the gates in this report.
The Numba core is the post-trial challenger.

This conclusion follows from the actual competition shape: one core, no GPU or network, 2 GB RAM,
a 50 MB package, a generous 90-second import window for JIT warm-up, persistent per-game state and
explicit permission to ponder. Books and tablebases are allowed, engine-annotated training data is
allowed, but third-party engines and borrowed runtime models are not. The canonical details are in
the live [AI Chessathon documentation](https://aichessathon.com/docs) and the dated
[competition rules](https://aichessathon.com/terms).

## 1. What game we are actually optimizing

### The hard constraints

| Constraint | Engineering consequence |
|---|---|
| `agent.py` at zip root; ≤50 MB expanded | Keep code readable; allocate package bytes deliberately between book, model and optional tablebases. |
| `get_move(fen, time_left_ms) -> UCI` | There is no opponent metadata or move history input; reconstruct history only from successive FENs within the game. |
| Python 3.12; Numba 0.67, NumPy 2.5.2, python-chess 1.11.2 | A native-speed engine is feasible if its complete hot path uses Numba-compatible primitive arrays. |
| One dedicated CPU core, no GPU/network | Large policy networks, hosted inference, root parallel search and AlphaZero-style MCTS are poor fits. |
| 90-second import budget | Precompute attacks and compile every Numba signature before the chess clock starts. Cold-JIT paths during a game are unacceptable. |
| 120 seconds + 0.5 seconds/move | Use iterative deepening with separate soft and hard limits, a clock reserve and an always-legal completed-iteration move. |
| Process persists for one game | Preserve TT/history, reconstruct repetition state and prepare a ponder search while the opponent thinks. |
| Core remains available after returning | Pondering is free compute if only one search owns the core at a time and can be stopped safely. |
| Illegal/crash/OOM/flag/init failure loses | Reliability is part of Elo. Root move validation and a deterministic legal fallback are non-negotiable. |
| Curated, near-level starting positions | Test from diverse middlegame-adjacent opening positions, not just the initial board. |
| 600-ply material adjudication | When adjudication is near, material becomes the literal objective; avoid shuffling while materially behind and simplify while ahead. |
| Ten uploads/day; latest valid one plays | Maintain an immutable champion. Never let an unproven upload displace it late in the day. |

The site currently says hourly ladder games only seed a 13-round Swiss, with points, Buchholz,
head-to-head and earlier final submission as tiebreaks. That makes variance control and an early
stable final build more important than chasing a noisy ladder number. Recheck the
[live documentation](https://aichessathon.com/docs) before every upload because the repository
instructions correctly treat it as canonical.

### What the visible field says—and does not say

The [live leaderboard](https://aichessathon.com/leaderboard) showed 158 teams at this research
snapshot, with only one completed round and the next still running. Ratings after one game carry
almost no information. Do not tune strategy around that ordering.

The house bots are more useful anchors: the site labels Loki 3.0 at CCRL 2428 and Zagreus 5.0 at
CCRL 2168. Public games already include entrants producing extremely accurate games against these
bots. The competitive target is consequently not “beats the supplied minimax”; it is “scores
credibly against the 2200-2400-class house anchors without technical losses.” Public games also
show mainstream positions such as the Petroff, French Winawer, Caro-Kann Advance, English
Symmetrical and London System, often around 6-8 moves into the game. This supports a broad opening
oracle. It does **not** prove the hidden position distribution or justify overfitting to a handful
of revealed games.

### What AI-assisted competitors will converge on

Most teams using a coding model can quickly produce the same surface checklist:

- python-chess negamax/alpha-beta;
- iterative deepening, a Python dictionary TT and capture-first ordering;
- qsearch, piece-square tables, killer/history moves and elementary LMR;
- a timeout check and broad exception handler.

Those features are necessary, but they are no longer differentiators. They also create a dangerous
illusion: the code looks like a mature engine while python-chess object construction, legality,
evaluation and sorting keep the searched depth low. Our durable edges should instead be:

1. a completely compiled search path;
2. better training and opening data rather than hand-guessed constants;
3. persistent state and safe pondering, both unusually valuable under this contract;
4. a professional champion/challenger test process;
5. zero forfeits.

## 2. Lessons from comparable systems

### Strong compact classical engines

[Boychesser](https://github.com/analog-hors/Boychesser), built for the 636-entry
[Tiny Chess Bot Challenge](https://github.com/SebLague/Tiny-Chess-Bot-Challenge-Results), is the
closest compact-engine precedent. Its published feature set includes PVS, qsearch, transposition
tables, aspiration windows, several forms of guarded pruning and reduction, history-based move
ordering, phased evaluation and soft/hard clock limits. Its author estimates about 2750 CCRL blitz.
That number does not transfer to this hardware or time control, but the architecture is highly
relevant: a small, well-integrated classical engine can be far stronger than its code size implies.

[Sunfish](https://github.com/thomasahle/sunfish) shows that compact Python can play respectable
chess with incremental piece-square evaluation, transposition tables and a best-first alpha-beta
variant. Its own notes also expose the ceiling: a mutable compact board and dedicated attack/move
logic are routes to much greater speed.

[Black Numba](https://github.com/Avo-k/black_numba) is a particularly useful feasibility proof for
this contest. It uses Numba, bitboards, packed moves and precomputed sliding attacks; the project
reports a move-generation improvement from thousands to millions of nodes per second. That is not
an Elo result and its source must not be copied, but it strongly supports building our own compiled
core in the exact language/runtime available here.

The lesson is not to copy these engines. Chessathon requires our moves to come from code we wrote,
and finalists may need to explain the build. The lesson is which ideas repeatedly survive severe
resource or code-size constraints.

### Modern alpha-beta priorities

The current [Stockfish search source](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)
is a primary reference for the mature shape of alpha-beta: iterative deepening, PVS, aspiration,
TT cutoffs, history-based ordering, qsearch, mate-distance pruning, null-move search with guards and
verification, futility/razoring, extensions and late-move reductions. Its
[position source](https://github.com/official-stockfish/Stockfish/blob/master/src/position.cpp)
shows why a fast engine maintains compact incremental state and an incrementally updated hash
during make/unmake.

We should copy none of its code and very few of its current numeric constants. Its value here is
dependency ordering:

```text
correct compact board
  -> fast legal make/unmake and Zobrist state
  -> iterative deepening + TT + move ordering
  -> qsearch and stable evaluation
  -> safe reductions/pruning/extensions
  -> trained evaluation and finely tuned margins
```

Putting aggressive pruning before correctness, ordering and evaluation will amplify tactical
errors. Putting a slow NNUE before fast incremental board state will trade away most of the search.

### Why not AlphaZero/MCTS

[AlphaZero](https://arxiv.org/abs/1712.01815) combines policy/value inference, Monte Carlo tree
search and enormous self-play training. The published evaluation used accelerator-heavy hardware;
DeepMind's own [system summary](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/)
describes four first-generation TPUs and 44 CPU cores for AlphaZero's side in that comparison.
Our deployment has one CPU core and no GPU. A tiny policy net could eventually help move ordering,
but making MCTS the main search would surrender the tactical efficiency and mature pruning of
alpha-beta in this exact setting.

## 3. Target architecture

### Boundary layer: keep the contract boring

`agent.py` should expose only a thin, exception-safe boundary:

1. parse the FEN with `python-chess`;
2. obtain the root legal moves;
3. consult the opening oracle or compiled search;
4. validate the returned packed move against the root legal set;
5. if anything is wrong or the hard deadline is close, return a deterministic legal fallback.

This layer should never trust the custom engine enough to emit an unchecked move. Terminal
positions are not normally queried, but handle an empty legal set explicitly so the failure mode
is intelligible rather than accidental.

### Compiled board and search core

Use fixed-size NumPy arrays and integer scalars that Numba lowers directly:

- 12 piece bitboards plus white/black/all occupancy;
- side, castling rights, en-passant square, halfmove clock and ply;
- an incremental 64-bit Zobrist key;
- packed 32-bit moves containing from, to, promotion and flags;
- a fixed state stack for captured piece, prior rights, EP, clock and hash;
- precomputed knight, king and pawn attacks;
- sliding attacks from a team-written lookup scheme or simple ray attacks first, then optimize;
- fixed-size move buffers per ply—no Python lists in the hot path.

Start with pseudo-legal generation plus make-and-king-safety filtering. Optimize evasions, pinned
pieces and staged generation only after exact perft passes. Correctness is worth more than the last
factor of two during the first implementation.

The search target is:

- iterative deepening;
- PVS/negamax alpha-beta;
- aspiration windows with widening re-search;
- fixed-array transposition table with key fragment, move, score, depth, bound and generation;
- qsearch with stand-pat, ordered captures/promotions and later SEE/delta pruning;
- ordering by TT move, good captures, killers/countermove and quiet history;
- mate-distance scores normalized when storing/probing TT;
- explicit repetition, fifty-move and insufficient-material handling;
- soft deadline between completed iterations and hard abort polled inside the tree.

Then add, one independently tested patch at a time:

1. history gravity and maluses for quiet moves that failed to cut off;
2. late-move reductions with full-depth verification on promising reduced moves;
3. reverse futility and shallow futility pruning;
4. static exchange evaluation and bad-capture deferral;
5. delta pruning in qsearch;
6. guarded null-move pruning with verification and zugzwang exclusions;
7. late-move pruning, internal iterative reduction and limited check/singular extensions only if
   the simpler engine has stabilized.

This order is informed by mature engine practice and Boychesser's compact feature set, but every
margin is an empirical parameter for our engine—not inherited folklore.

### Timing under Numba

An important local probe found that `time.monotonic()`, `time.perf_counter()` and `time.time()` do
not compile in Numba 0.67 nopython mode. Therefore the compiled recursive function cannot simply
read a Python clock.

The preferred design is:

- compile the search with `@njit(nogil=True)`;
- pass a one-element NumPy stop array into the search;
- start a lightweight Python timer thread that sets the flag at the hard deadline;
- poll the flag every small power-of-two number of nodes;
- let the root Python frame enforce a second deadline and retain the last completed legal move;
- verify in a dedicated stress test that the compiled loop observes flag changes promptly on the
  platform version.

Numba documents that fixed NumPy array access lowers efficiently and that `nogil=True` releases the
GIL, but it also warns that normal concurrent-programming hazards apply
([array support](https://numba.readthedocs.io/en/stable/reference/numpysupported.html),
[JIT options](https://numba.readthedocs.io/en/stable/user/jit.html)). The stop-flag mechanism is our
implementation inference and must earn its reliability through testing. A calibrated node budget
is a safe fallback if visibility is unreliable, though it uses the clock less precisely.

### Persistent transposition and pondering

The TT and history tables should survive between our moves in the same game. Clear or age them at
a detected new game, not on every `get_move` call. Preserve the principal variation after moving.

After returning a move, begin one background ponder search on the most likely reply from the PV.
When the next FEN arrives:

- if the prediction matched, reuse the deeper PV/TT work;
- if it did not, stop the ponder search immediately and retain only safe TT information;
- never allow ponder and on-move search to compete for the single core;
- never return while an old timer can later stop a new search—use a search generation/token;
- join or positively quiesce the prior worker before starting another owner of mutable state.

Pondering is potentially one of the largest contest-specific advantages because it converts the
opponent's clock into our depth. It is also a race-condition risk, so implement it after the
non-ponder engine has passed long stress games.

### Evaluation: HCE first, NNUE second

The first compiled evaluation should be a tapered middlegame/endgame model with incrementally
maintained material and piece-square terms. Add cheap global terms only after measuring their cost:

- pawn structure: isolated/doubled/passed/connected pawns and pawn shield;
- mobility by piece class;
- bishop pair, rook files/seventh rank and minor-piece outposts;
- king safety via attack units and virtual mobility;
- space, tempo and simple threats;
- endgame king activity and scaling for drawish material.

Fit the weights rather than hand-adjusting dozens of constants. A logistic/Texel-style objective
can learn middlegame/endgame weights from team-generated features and game outcomes or deep engine
labels. Split data by game to avoid nearly identical positions leaking into train and validation.
Then play games: lower prediction loss is not the same as higher Elo.

A small NNUE is the high-ceiling follow-up. The official
[Stockfish NNUE documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
explains why sparse king-relative features and incrementally updated accumulators make neural
evaluation compatible with alpha-beta. A HalfKP-like 40,960-by-128 int16 first layer is roughly
10 MiB; 256 units is roughly 20 MiB, leaving package room for a book. Use integer inference inside
Numba, test accumulator updates against a full refresh after every special move, and ship only
weights the team trained. A smaller net that preserves search depth can be stronger than a more
accurate but slower net.

### Opening oracle and package budget

Books are explicitly permitted. `python-chess` supplies a
[memory-mapped Polyglot reader](https://python-chess.readthedocs.io/en/latest/polyglot.html) that
looks positions up by Zobrist hash and returns weighted moves. The official
[Lichess open database](https://database.lichess.org/) provides CC0 games and hundreds of millions
of engine-evaluated positions; Stockfish's official
[book repository](https://github.com/official-stockfish/books/blob/master/README.md) provides large,
diverse opening-position sets for testing.

The oracle should be broad rather than a brittle move-one repertoire:

- cover common balanced positions through roughly ply 20-30;
- store only legal, stable moves with adequate evaluation depth and a small evaluation gap;
- retain two near-equal moves in common nodes if predictability becomes exploitable;
- exclude tactical positions where source evaluations disagree or are shallow;
- verify every stored move by replay before packaging;
- use a checksum and gracefully fall through to search on absence/corruption.

A standard Polyglot record is convenient, but a team-written sorted record of one 64-bit key plus
one packed move can use package bytes more efficiently. Whether that added format complexity is
worth it should be measured. Do not silently scrape or infer hidden organiser data. Public match
records can inform general testing; ask the organisers for written clarification before building a
competition-specific oracle from revealed starts because the rules forbid accessing hidden match
data even though published results are public.

Suggested expanded-package allocation:

| Component | Initial HCE build | Later NNUE build |
|---|---:|---:|
| Python source/constants | <1 MB | <1 MB |
| General opening oracle | 30-45 MB | 18-30 MB |
| Team-trained network | 0 | 10-20 MB |
| Optional tablebase subset | 0 | 0-5 MB only if measured useful |
| Headroom | ≥4 MB | ≥4 MB |

Syzygy is supported by python-chess, but comprehensive tablebases do not fit. A tiny subset could
improve exact conversion, yet it is lower expected value than opening coverage or evaluation. Add
it only after representative games show recurring, mishandled covered endings.

## 4. Phased delivery plan

### Phase 0 — freeze a valid trial champion (now, 30-60 minutes)

Goal: guarantee that tomorrow has a serious, legal entry even if later work fails.

- Keep the current `agent.py` unchanged as the champion.
- Build a new archive from exactly that file; inspect archive entries, expanded size and SHA-256.
- Run the local gate in the correct project environment and upload early enough to inspect the
  platform's two-color validation log.
- Archive the exact accepted zip and log under a versioned name outside the build output.
- Do not let exploratory code overwrite this file or become the latest passing upload.

Exit gate: correct root layout, platform validation passes both colors, import below 75s, no
unexplained warnings, hash recorded.

### Phase 1 — build the measurement laboratory (today, 3-5 hours)

Goal: make “better” measurable before spending the week tuning noise.

- Extend the paired arena to accept an EPD/FEN opening suite and play both colors from each start.
- Create at least three suites: broad balanced openings, tactical/sharp positions, and endgames.
- Add local **opponents only** from published Loki/Rustic/Zagreus-class engines when licensing and
  platform compatibility permit. They must never enter the submission zip.
- Record result, opening ID, color, termination reason, wall time, nodes, completed depth, qnodes,
  TT hit/cutoff rates and maximum memory.
- Add deterministic seeds and machine metadata.
- Preserve each champion binary/source and complete PGN set.

Use Stockfish's testing discipline as the model. Its
[Fishtest workflow](https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html)
first screens at short time control and confirms at longer time control with SPRT. Its
[FAQ](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-FAQ.html) explicitly warns
about large error bars and selection bias; official regression estimates use tens of thousands of
games ([regression tests](https://official-stockfish.github.io/docs/stockfish-wiki/Regression-Tests.html)).
We have less compute, so reject obvious failures quickly, but do not call a 20-game fluctuation an
improvement.

Exit gate: one command can reproduce paired results and technical-failure statistics from a pinned
suite and two pinned agents.

### Phase 2 — low-risk trial-plus improvements (today/tomorrow, 2-4 hours)

Goal: extract safe strength without destabilizing the champion.

Work on a copy, one patch at a time:

1. retain TT entries and history information across successive moves, with generation aging;
2. measure a broad, licensed opening book against the public and neutral test suites, then add it if
   coverage is material and every move is validated;
3. refine clock allocation based on legal-move count, increment, game phase and score stability;
4. add only a well-tested search feature such as history maluses or conservative delta pruning if
   paired games show a gain.

Do not attempt custom move generation, NNUE or background pondering in the build needed tomorrow.

Promotion gate: zero failures in at least 1,000 accelerated games/fuzzed calls, no time-control
regression, and a clear paired improvement over the frozen champion. If evidence is ambiguous, ship
the champion.

### Phase 3 — correct Numba board core (1-2 focused days)

Goal: remove the Python node-throughput ceiling without yet chasing Elo.

Milestones:

1. FEN parse/serialize and attack detection.
2. Pseudo-legal moves and make/unmake for ordinary moves.
3. Castling, en passant, all promotions, checks/evasions and pins.
4. Incremental hash, repetition key and halfmove clock.
5. Exact perft for canonical positions and randomized differential checks against python-chess.
6. Import-time compilation of every production signature.
7. Root conversion back to a python-chess-validated UCI move.

Correctness gates:

- exact perft counts on the standard multi-position suite through the deepest practical depth;
- for at least 100,000 randomized legal positions, the internal legal-move set equals
  python-chess's set;
- make then unmake restores every bitboard, scalar, hash and evaluation accumulator bit-for-bit;
- targeted cases for illegal EP exposing check, Chess960-like castling fields rejected/handled as
  appropriate, underpromotions, double check, stalemate, mate, repetition and fifty-move state;
- JIT import comfortably below 75 seconds in a fresh cache-less environment.

Performance gate: at least a 10x increase over the current complex-position search throughput on
the same machine, while all correctness gates remain exact. Raw perft NPS alone does not qualify.

### Phase 4 — compiled search strength (1-2 days)

Goal: turn speed into selective depth.

Implement the baseline stack—ID, PVS, aspiration, fixed TT, qsearch, TT/capture/killer/history
ordering—and establish a new stable champion. Then add the pruning/reduction list from section 3 in
isolated patches.

Every patch follows:

```text
unit/perft/fuzz -> deterministic bench -> fast paired screen
               -> longer paired confirmation -> promotion or deletion
```

For null-move or verification-search changes, include zugzwang positions; Stockfish's own test
guidance specifically calls for a Zugzwang suite when verification search changes. For TT changes,
test mate-score normalization, halfmove-rule interactions and collision replacement. Search
results must be deterministic at fixed node limits so semantic bugs are distinguishable from
timing noise.

Exit gate: the Numba challenger scores at least 55% against the Python champion in a sufficiently
large paired suite, has zero technical losses in stress play, and confirms the direction against a
strong external local opponent.

### Phase 5 — train the classical evaluation (about 1 day plus games)

Goal: improve positional judgment without cutting search speed sharply.

- Sample a balanced mixture of quiet, tactical and endgame positions from independent games.
- Generate deep labels offline; this is allowed because only the trained model/weights ship.
- Deduplicate normalized positions and cap positions per source game.
- Fit tapered feature weights with regularization; keep a game-disjoint validation set.
- Export integer constants directly into the engine.
- Measure evaluation throughput, prediction loss, tactical regressions and paired Elo.

Promotion gate: no material slowdown, improved held-out loss, no systematic endgame/tactical
regression, and a positive paired-game result. Prediction loss alone is insufficient.

### Phase 6 — optional small NNUE (2-4 days, parallel to game testing)

Goal: raise evaluation ceiling if the team has enough training compute and data engineering time.

- Begin with a 128-unit king-relative sparse accumulator, not a maximal network.
- Train the weights ourselves from legal external labels/results and retain provenance scripts,
  dataset hashes and checkpoints for finalist explanation.
- Quantize with calibrated integer ranges; assert no accumulator/output overflow.
- Implement full-refresh inference first, then incremental updates.
- Property-test incremental versus full refresh after millions of random moves, especially king
  moves, captures, castling, EP and promotions.
- Compare net accuracy **and** end-to-end nodes per second against tuned HCE.

Promotion gate: clear paired Elo gain at relevant time controls after accounting for lower NPS;
package remains below 50 MB; import remains below 75s; all accumulator property tests pass. If not,
ship tuned HCE. “Neural” has no value on the scoreboard by itself.

### Phase 7 — pondering, final operations and freeze (last 1-2 days)

Goal: exploit the lifecycle safely and arrive early with the best proven build.

- Add single-owner ponder state with generation-safe cancellation.
- Measure prediction hit rate, reused depth/nodes and flags over long games.
- Replay every public game for evaluation diagnostics, not for conclusions from individual results.
- Use at most a few deliberate uploads: champion, clearly stronger challenger, final stable build.
- Stop strength changes early enough to run the complete reliability battery.
- Submit the final proven build before the deadline; the published tiebreak rewards earlier final
  submission.

Ponder promotion gate: positive paired result, measurable hit/reuse rate, clean shutdown on every
miss, no extra flags or races in at least 1,000 stress games.

### Calendar allocation

Assuming work begins on 4 September and uploads lock at 11:00 on 11 September:

| Window | Primary deliverable | Stop condition |
|---|---|---|
| 4 Sep evening | Accepted safe champion; reproducible arena and opening suites | Do not touch the accepted archive after validation. |
| 5 Sep trial | Observe technical behavior and house-bot gap; test only isolated low-risk challenger changes | Revert immediately on any validation/time anomaly. |
| 5-6 Sep | Correct Numba board, perft, differential fuzz, root validation | No advanced search until move sets and make/unmake are exact. |
| 6-7 Sep | Compiled ID/PVS/TT/qsearch engine; fast paired testing | Keep Python champion if compiled challenger is not clearly stronger. |
| 7-8 Sep | Search ordering/pruning patches; broad opening oracle | Delete non-winning patches rather than accumulating complexity. |
| 8-9 Sep | Tuned HCE; optional small NNUE only if training pipeline is already healthy | Abandon NNUE if it threatens time needed for reliability. |
| 9-10 Sep | Pondering only after stable non-ponder build; full actual-clock confirmation | Disable pondering on any race or flag regression. |
| 10 Sep night | Feature freeze, exact-zip endurance, checksum and platform validation | Only critical bug fixes after freeze. |
| 11 Sep morning | Submit the already-proven final artifact well before 11:00 | No speculative last-minute upload. |

This schedule deliberately gives the compiled engine two chances to lose: first to correctness
gates, then to paired strength. If either happens, the safe champion remains a valid tournament
entry while work continues.

## 5. Promotion gates and test matrix

### Reliability gate—must be absolute

- archive contains `agent.py` at root and only intended files;
- expanded size below 50 MB with explicit headroom;
- fresh-container import below 75 seconds, leaving 15 seconds contingency;
- peak RSS below 1.5 GB;
- no writes outside `/tmp`, network or external processes;
- all dependencies belong to the fixed platform set;
- 10,000 randomized API calls always return a legal UCI move within budget;
- 1,000 accelerated games: zero illegal moves, exceptions, OOMs, init failures and flags;
- emergency 1/10/50/100 ms calls return legal moves;
- exact-zip tests import and run the extracted artifact, not the working tree;
- corrupt/missing book or weights cause search fallback, not process failure.

### Strength gate—statistics, not vibes

- always pair colors from the identical starting FEN;
- use a broad fixed opening suite and a separate holdout suite;
- change one conceptual feature per challenger;
- screen obvious losses at fast controls, confirm survivors near the real control;
- require at least 55% against the current champion before a major architectural promotion;
- report W/D/L, paired outcomes and an interval/test statistic, not only percentage;
- retest the final candidate against at least two distinct opponent styles;
- never tune on the holdout or on a tiny set of public Chessathon games.

Fishtest's scale demonstrates why small results are deceptive; our threshold is a practical minimum,
not statistical magic. A 52-48 result over 100 games should normally be treated as “unknown.”

### Performance gate

Track on pinned positions:

- total nodes/s and qnodes/s;
- completed depth and selective depth;
- branching factor by iteration;
- TT hit, usable-hit and cutoff rates;
- time in move generation, make/unmake, evaluation, sorting and stop checks;
- book hit rate and evaluation disagreement;
- ponder prediction/reuse rate;
- import/JIT time and peak memory.

Optimize end-to-end completed depth and Elo, not a synthetic NPS number. An expensive evaluation or
SEE can be worthwhile if it reduces the tree more than it costs.

## 6. Main failure modes and controls

| Failure | Why it is dangerous | Control |
|---|---|---|
| Custom movegen bug | One illegal root move is an immediate loss; internal omissions also silently weaken search. | Differential legal-move sets, make/unmake invariants and python-chess root validation. |
| Cold JIT during a move | Consumes chess clock and can flag. | Compile every signature at import using production dtypes and paths; test fresh cache. |
| Timer/ponder race | Old worker can corrupt TT or stop a new search. | Single owner, generation tokens, explicit stop/join, stress tests. |
| Aggressive null/futility pruning | Misses tactics, mates or zugzwangs. | Conservative guards, verification and targeted suites; one-patch paired tests. |
| TT semantic bug | Produces wrong cutoffs across plies, rule-50 states or collisions. | Bound/depth tests, mate-score normalization, generation replacement and deterministic fixed-node traces. |
| NNUE accumulator drift | Evaluation becomes position-history dependent and silently wrong. | Incremental-vs-refresh property test on every move type. |
| Book poisoning/overfit | A single bad forced move can lose before search; revealed-position tuning may raise fairness questions. | Deep stable labels, legal replay, broad sources, holdout coverage and organizer clarification. |
| Large Python objects | Search exhausts time or memory despite algorithmic sophistication. | Fixed arrays and measurements; keep Python outside the node loop. |
| Noisy arena decisions | Repeatedly promotes lucky regressions. | Paired openings, champion/challenger isolation, intervals/SPRT and a locked holdout. |
| Last-minute upload | Latest passing build displaces a safer agent and earlier submission is a tiebreak. | Freeze early and upload only proven artifacts. |

## 7. What we should do next

The highest-value immediate sequence is:

1. Rebuild, inspect and upload the current safe champion for the trial; preserve its accepted hash
   and validation log.
2. Build the paired multi-opening arena and reliability runner before modifying search again.
3. Measure a broad permitted book's hit rate on neutral/public positions; add it to a challenger
   only with legal validation and a clean fallback.
4. Create `engine_numba.py` as a development module while keeping the deliverable champion intact.
   Finish board correctness and perft before writing advanced search.
5. Once the compiled core passes, add the baseline PVS/TT/qsearch stack, benchmark it, and only then
   begin pruning and evaluation training.

The strategic principle is simple: **ship the strongest engine we have proved, while building the
engine with the highest ceiling in parallel as a challenger.** The tournament will punish one crash
more reliably than it rewards one clever but untested heuristic.

## Source notes

Core primary and first-party sources used in this synthesis:

- [AI Chessathon technical documentation](https://aichessathon.com/docs)
- [AI Chessathon rules, version 2026-08-31.v3](https://aichessathon.com/terms)
- [AI Chessathon live leaderboard](https://aichessathon.com/leaderboard)
- [Stockfish search source](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)
- [Stockfish position source](https://github.com/official-stockfish/Stockfish/blob/master/src/position.cpp)
- [Stockfish Fishtest workflow](https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html)
- [Stockfish Fishtest FAQ](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-FAQ.html)
- [Stockfish Fishtest mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
- [Stockfish regression-testing record](https://official-stockfish.github.io/docs/stockfish-wiki/Regression-Tests.html)
- [Stockfish NNUE documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
- [Stockfish test books](https://github.com/official-stockfish/books/blob/master/README.md)
- [Lichess open database](https://database.lichess.org/)
- [python-chess Polyglot documentation](https://python-chess.readthedocs.io/en/latest/polyglot.html)
- [python-chess Syzygy implementation/docs](https://github.com/niklasf/python-chess/blob/master/docs/syzygy.rst)
- [Numba NumPy support](https://numba.readthedocs.io/en/stable/reference/numpysupported.html)
- [Numba JIT/nogil documentation](https://numba.readthedocs.io/en/stable/user/jit.html)
- [Boychesser](https://github.com/analog-hors/Boychesser)
- [Tiny Chess Bot Challenge results](https://github.com/SebLague/Tiny-Chess-Bot-Challenge-Results)
- [Sunfish](https://github.com/thomasahle/sunfish)
- [Black Numba](https://github.com/Avo-k/black_numba)
- [Numbfish](https://github.com/dimdano/numbfish)
- [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [Deep Blue paper summary at IBM Research](https://research.ibm.com/publications/deep-blue)
- [Zobrist's hashing paper](https://journals.sagepub.com/doi/10.3233/ICG-1990-13203)
