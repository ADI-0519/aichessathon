# AI Chessathon finals: evidence, priorities, and operating plan

Date: 11 September 2026

## Executive decision

### Finals-day initialization change

The organisers announced a 30-second initialization budget at the live final.
This newer communicated constraint supersedes the still-public 90-second value.
The synchronous V16/V18 warm-up is therefore not deployable unchanged.

Two local adapters preserve the respective Core and SearchMax engines while
starting compilation on one daemon thread. Import returns after at most 20
seconds; the first move joins any unfinished compilation and subtracts that
wall time from the value passed to the time manager. The protocol thread does
no engine work until the join completes, so search state remains single-owner.
A cold local protocol test reached ready in 20.074 seconds and returned legal
`e2e4` 18.553 seconds into the first request. This is a reliability trade: it
spends first-move clock rather than losing automatically during initialization.

The adapters are:

- `challengers/exp_finals_v18_core_init30`
- `challengers/exp_finals_v18_searchmax_init30`

All engine, evaluator, model and time-manager files in each adapter are
byte-identical to its non-adapted parent. Only `agent.py` and explanatory
documentation differ.

The public documentation available on the morning of 12 September still says
that uploads closed at 11:00 on 11 September and the last valid build froze for
the final qualification Swiss.[^1] It does not document a finals-day upload
window. Prepare candidates locally, but do not assume one can play unless the
organisers provide a newer written finals instruction at the venue.

The correct plan is therefore not to rewrite the engine overnight. Freeze the
submitted V16 as the rollback build and prepare one finals candidate combining
two changes with unusually strong evidence:

1. a substantially longer time horizon, while retaining the existing hard
   deadline and emergency fallback; and
2. the already-developed deferred move generation, but only after exact-tree
   parity is demonstrated against V16.

Time management addresses the repeated long-game collapse. Deferred generation
buys more search without changing chess semantics. These may be combined for
the final candidate because their effects are largely orthogonal, but each must
also exist as a separate challenger so a failure can be isolated quickly.

Do not ship V17 SearchMax merely because it scored slightly above V16 internally.
Its deeper pruning and stronger LMR deliberately change the tree and remain a
larger tactical risk. Do not start another network training run: the 200M-position
BigNet was a large verified improvement within our lineage, whereas the later
V11-256 experiment improved offline loss and still lost timed games.

## What the public finals rules say

- The live final is on 12 September at Encode Club, London.[^1]
- The public page does not currently grant a post-Swiss upload window. A newer
  finalist briefing, if supplied by the organisers, would supersede this report.
- The bracket is seeded from the final Swiss. Higher seeds receive the first
  round byes. A normal tie uses two positions, each played with both colours;
  semifinals use three positions and the final four.[^1]
- A level knockout match advances the team with the better final-Swiss standing.
  As seed 51, AIY should assume a drawn match is a loss against nearly every
  opponent. This makes real winning strength more valuable than a draw-oriented
  patch.[^1]
- Under the currently published conditions the engine receives 120 seconds plus
  0.5 seconds per move, a 90-second import budget, one EPYC core, 2 GB RAM, no
  network and no GPU. The package is at most 50 MB uncompressed.[^1] Re-check the
  one announced condition before changing any clock or memory constant.
- Opening tables may answer positions through move 20; endgame tables may answer
  positions with at most seven pieces. Both count toward the 50 MB limit.
  `python-chess` Polyglot and Syzygy readers are installed.[^1]
- Each finalist must walk through the engine, and a team shipping a network must
  show how it trained that network.[^1]

The invitation email may contain private arrival, identity or venue instructions
that are absent from the public documentation. Treat that email as authoritative
for logistics and preserve a copy offline.

## What the available final-Swiss games say

The repository contains all thirteen relevant games, rounds 110--122. They score
8 wins, 1 draw and 4 losses (8.5/13).

I re-analysed every AIY move in those ten games with Stockfish at 100,000 nodes.
This is a triage tool, not an oracle: shallow engine scores are noisy in long
endgames and mate scores make ACPL especially unstable. The useful evidence is
the location and type of the large swings, checked together with the clocks.

| Round | Result | AIY ACPL | Main finding |
|---:|:---:|---:|---|
| 110 | W | 10.4 | Solid win with about 34 seconds left. |
| 111 | W | 14.7 | Won, but finished with about 5 seconds; `Qd2` and `Kxd4` each lose roughly 100 cp in the shallow replay. |
| 112 | W | 10.6 | Won with about 12 seconds left; another clock warning rather than a failed result. |
| 113 | L | 19.2 | `...h5` loses about 90 cp while ample time remains; the later conversion occurs with roughly 2--5 seconds and contains much larger errors. Mixed strategic and clock failure. |
| 114 | D | 10.0 | A 148-move survival. `Qh6+` is a large midgame miss; several later endgame misses occur near 5--15 seconds. Clock policy makes the defence harder but the engine still saves the draw. |
| 115 | L | 25.8 | Not a clock loss. `...Qg5` and especially `...Qxf4` produce large swings while roughly 77 and 48 seconds remain. This is a genuine evaluation/selectivity problem. |
| 116 | L | 25.5 | `Nxg6` gives away much of an apparent advantage with about 84 seconds left; `Qh3` and `f5` are later quiet-position errors. Low clock compounds the ending, but did not cause the initial failure. |
| 117 | W | 12.9 | Forcing tactical game won quickly, with about a minute left. This is V16's strongest regime. |
| 118 | W | 9.0 | Clean conversion; no 100-cp centipawn event in the scored moves. |
| 119 | W | 8.1 | Cleanest game in the sample, although the clock falls to about 8 seconds by the end. |
| 120 | W | 8.9 | Clean tactical/positional win with about 41 seconds left. |
| 121 | L | 128.9 | Extreme 245-ply endgame. The score is distorted by huge late queen-endgame swings, but the clock is only about 2--5 seconds during them. Clear horizon failure. |
| 122 | W | 15.1 | Won, but reaches about 10 seconds by AIY move 60 and under 5 seconds at the end. Another warning despite the result. |

Rounds 110--112 reinforce rather than reverse the diagnosis: all are wins, but
two finish with little clock. Across rounds 113--122, the five wins have weighted
ACPL about 11.4 and the draw 10.0. The four losses average 63.5, dominated by
round 121; excluding it, the other losses still average about 23.5. This supports
two distinct causes:

1. **Clock/horizon failure.** V16 plans only 40 moves in high material, then 34,
   32, 24 and finally 10 as material falls. Long, difficult endings routinely
   outlive those assumptions. It spends around 3--5 seconds per early move and
   then has sub-second freedom precisely when quiet accuracy matters most.
2. **Absolute chess-strength failure.** Rounds 115 and 116 contain important
   mistakes with tens of seconds available. More remaining time alone cannot
   repair the wrong static judgement, an over-aggressive reduction, or a horizon
   error in a quiet continuation.

The external test makes the second point unavoidable. The recorded V16 match
against Toby Coad's public commit `8456160d` finished 1 win, 9 draws and 34 losses:
12.5%, approximately -338 Elo, with no technical failures. V16 was therefore a
large improvement over our own ancestors, not yet close to the strongest public
lineage. Toby's public time-management change independently moved its planning
horizon from 56 to 80 moves and reported 56.2% in a 40-game test.[^2] That does
not prove the same constants will work in our engine, but it agrees strongly with
our clocks and should change the prior from “small tweak” to “urgent experiment.”

## What V16 actually is

The submitted V16 is not “just BigNet.” Its move path is:

```text
agent.get_move
  -> synchronise persistent board and repetition history
  -> convert python-chess FEN to the Numba bitboard position
  -> calculate soft / normal / hard time limits
  -> iterative deepening root search
       -> aspiration/PVS negamax
       -> two-slot TT, dynamic null move, futility/LMP/SEE pruning
       -> signed quiet history, capture history and countermoves
       -> contextual LMR and conservative singular extensions
       -> capture/check quiescence with a static-evaluation cache
       -> 75% BigNet + 25% handcrafted evaluation
  -> commit the chosen move to persistent state
  -> legal emergency fallback on any internal failure
```

The evaluator is a team-trained format-3 network: a 256-wide king-conditioned
accumulator, pairwise interaction features, dual activation and eight
material-dependent output heads. Its packaged SHA-256 is
`78931e8692e0ad3b90a6fa4614643aa9c2b9d8bba8731419c26eae2108b93647`.
The runtime uses incremental accumulator updates and rebuilds a perspective when
its king changes bucket. The model output is blended with the handcrafted
evaluation because that blend beat pure learned evaluation in the preceding
generation.

### Hot-path and code-quality audit

Generic Python optimization advice is misleading for this repository. V16 does
not search with `python-chess` objects: its board, move generation, make/unmake,
hashing, SEE, evaluation and recursive search are Numba-compiled over primitive
NumPy arrays. Per-ply move, undo, accumulator, ordering and SEE buffers are
allocated once at the root of each move and reused by recursion. Pawn, knight
and king attacks are precomputed; the transposition table is a fixed two-slot
array rather than a Python dictionary; and qsearch has a separate exact static
evaluation cache. `python-chess` remains only at the protocol boundary.

Consequently:

- PyPy is neither present in the fixed competition environment nor a sensible
  replacement for the Numba/NumPy execution model.
- `cProfile` can measure protocol and root setup, but it sees the compiled
  search as one opaque call. The existing fixed-node search counters and
  `tools.numba_search_scaling` are the correct first profiler for the hot path.
- Replacing the fixed TT with another generic cache, or merely “using
  bitboards,” would duplicate work already done.
- The remaining board-level optimization target is specifically sliding-ray
  attack/move generation, which still walks squares direction by direction.
  A ray-table or magic-style replacement is plausible but must pass differential
  legal-move tests and perft before any strength test.
- V16 already defers qsearch accumulator deltas until a move survives pruning,
  and safely propagates stale king-bucket state. A fully lazy delta chain could
  avoid still more work, but it is a materially different and correctness-
  sensitive implementation, not a one-line cache addition.

The audit also confirms that the apparent “hardcoded time” problem is real but
localized. Named safety constants and search-stability signals are reasonable;
the fragile part is V16's discontinuous material horizon. The V18 core replaces
that horizon without touching V16's search tree. Root search already uses the
previous completed iteration's best move, PVS, aspiration and move/score
stability. Missing features such as per-root-move score and effort histories
may be useful, but should not be described as if no root feedback exists.

Finally, V16 enables guarded root LMR, whereas Toby's inspected public snapshot
keeps its root-LMR switch disabled. That is a legitimate behavioral hypothesis
for quiet positions, but disabling it is not an exact optimization and requires
paired games; it should not be silently folded into the clock/deferred candidate.

This architecture is explainable, but “trained on 200M positions” is not enough
for the required walkthrough. Before the final, obtain from the teammate who ran
training:

- exact data sources, licenses and mixture weights;
- how positions were sampled, deduplicated and split;
- label engine/version and label search limit;
- network dimensions, features, material buckets, loss and optimizer;
- seed, epochs or samples seen, validation criterion and selected checkpoint;
- export/quantisation command and the source model hash.

Keep the training command, configuration, manifest and one small reproducible
sample available. The rules expressly require the team to show how a shipped
network was trained.[^1]

## Highest-value engineering plan

### P0: preserve a rollback

Never edit the submitted V16 directory. Copy it into named challengers, record
the source commit and model hash, and keep the accepted V16 zip. Any finals
candidate must retain the legal fallback, persistent game state, hard deadline,
single-thread settings and verified initialization path.

There is also an immediate packaging hazard: the repository's `current/`
directory still describes V14 Runtime, while the accepted qualifier build is
`challengers/exp_release_v16_search_bignet/`. Do not run an unqualified
`make zip` until the canonical champion has deliberately been migrated to V16,
or package with an explicit V16 root and inspect the archive. This is an
operational correction, not an engine experiment.

### P1: long-horizon time manager

Build `V16 + long_time` independently. Replace the current stepped 40/34/32/24
horizon with a continuous, deliberately conservative estimate that plans for
roughly 70--80 remaining moves early and never suddenly collapses merely because
piece count crosses a boundary. Retain material as a modifier, not the dominant
horizon. Preserve the low-clock reserve and hard cap.

The stopping rule should also use search information already available:

- spend less when the root move and score remain stable across completed depths;
- permit extra time, within the hard cap, when the root move changes, the score
  falls sharply, or the top moves remain close;
- spend almost nothing when only one legal move exists.

This is the same broad principle used by current Stockfish: it computes distinct
optimum and maximum budgets and adjusts stopping using falling evaluation, best
move instability and search effort.[^3][^4] We should implement the concept in
our own compact design, not transplant its tuned constants.

Gate it at the real announced time control. Fast 10+0.1 games distort the exact
trade-off this candidate changes. Use long-game/endgame starts, both colours,
and compare clock at engine moves 20, 40, 60 and 80 in addition to match score.

### P1: exact deferred move generation

Build `V16 + defer` separately from the teammate's already-tested deferred
generation work. The point is to avoid creating and ordering a full move list at
nodes that static pruning will return from. This should be semantic-preserving.
Require identical root move, score, completed depth, nodes and qnodes on a broad
fixed-node suite. Only then measure single-process NPS. If the tree differs, it
is a search challenger, not a speed optimization, and needs game testing.

### P1: combined finals candidate

Once both individual gates pass, build `V16 + long_time + defer`. This is the
only bundled candidate that should be preferred before tomorrow's announcement.
It attacks the demonstrated clock defect and converts saved CPU directly into
more depth without taking the unvalidated pruning risk of SearchMax.

Use three comparisons, in this order:

1. fixed-node parity between V16 and the deferred candidate;
2. official-clock paired games of `long_time` versus V16 on the long-game suite;
3. official-clock paired games of the combined candidate versus V16 on a balanced
   suite with disjoint offsets across machines.

Results from different machines may be aggregated for game score only when the
candidate/opponent commits, time control, suite, split, ordering seed and offsets
match and do not overlap. Do not aggregate NPS across machines.

### Finals candidate matrix

The local implementation now exposes four deliberately related candidates. This
permits Toby-style cheap screening without losing attribution when a large bundle
wins or fails:

| Candidate | Clock/deferred generation | SearchMax | BigNet blend |
|---|:---:|:---:|---:|
| `exp_finals_v18_core` | yes | no | 75% |
| `exp_finals_v18_eval50` | yes | no | 50% |
| `exp_finals_v18_searchmax` | yes | yes | 75% |
| `exp_finals_v18_max` | yes | yes | 50% |

The core candidate has exact move, score, depth, node and qnode parity with V16
on the four-position fixed-node critical suite, while its single-run NPS was
10--14% higher on each position. This is strong engineering evidence, although
repeat measurements are still required before quoting a stable throughput gain.

An evaluator calibration tool now measures HCE, BigNet and blends against the
Stockfish-labelled PGN corpus with positions deduplicated by FEN and whole games
assigned to a deterministic holdout. On 505 held-out positions, blend 50 had
RMSE 212.1 cp and blend 75 had RMSE 214.9 cp; their MAE values were 133.3 and
133.6 cp. Pure HCE and pure BigNet were materially worse. This rejects both a
blind pure-network switch and an invented per-phase scaling curve. It supports
testing 50 against 75 as a cheap evaluator lane.

The 15-position non-clock regression screen at 100,000 nodes is deliberately
not a promotion test. Core, evaluator-50, SearchMax and the maximum bundle match
the shallow teacher on 5, 6, 5 and 5 positions respectively. SearchMax reaches
greater mean depth (9.8 versus 9.33), but neither it nor the maximum bundle shows
a decisive tactical-suite advantage. This prevents a false claim that the large
bundle has already closed the external gap; paired games remain necessary.

Run short-control paired SPRTs first. Each machine receives a disjoint opening
offset, one single-threaded engine per game and as many independent workers as
physical cores can sustain without throttling. A candidate that cannot beat V16
at short control does not consume an official-clock gate. The survivors then
play the same offsets against the frozen public Toby build; only a bundle that
improves both internal and external score proceeds to the expensive match-clock
test. Time-management claims themselves must still be decided at match clock.

### P2: opening-book coverage, not blind book deployment

All ten observed Swiss starts begin on moves 5--9, leaving legal book coverage
through move 20. A broad book could save meaningful clock and reduce early
variance. First build a coverage report against every known platform start and
the development/validation starts. Then test every returned move for legality
and reject book entries outside the move-20 rule. Ship only if coverage is high
and paired games show no regression. A rushed narrow book overfitted to revealed
starts is unlikely to cover unseen finals positions.

### P2: small tablebases

Tablebases are legal and can prevent embarrassing four-piece errors, but they do
not repair the decisions that created the losing endings in rounds 113, 115 and
116. They also compete with the book and network for the 50 MB budget and add a
new runtime path. Add only a complete, package-tested subset with deterministic
legal move selection and correct fifty-move handling. This is below time and
deferred generation in expected value.

### After the final, if development continues

The durable strength project is not “copy everything Toby did.” It is to use the
external gap to prioritize independently implemented mechanisms and then test
them. The next generation should combine:

- a larger and more diverse team-trained evaluator with explicit platform and
  9--16-piece representation, plus phase/score calibration;
- safer quiet-search selectivity learned from the loss replays;
- dedicated tactical move generation and faster board attack code;
- stronger TT information and search feedback; and
- large, diverse external-opponent regression matches rather than only ancestral
  self-play.

The previous V11 result is a warning that lower offline loss does not guarantee
timed Elo. Every evaluator must pass throughput, score-distribution calibration
and paired games before promotion.

## Finals-day operating plan

1. Bring the accepted V16 zip and keep it as the rollback build.
2. Ask for the written finals technical briefing on arrival. Only package or
   upload a new candidate if that briefing explicitly reopens submissions.
3. If uploads reopen, read any changed match condition before selecting clock or
   memory constants. Record build identifiers and the active validated upload.
4. Use any practice games to catch technical failures and large regressions, not
   to infer Elo from one or two results.
5. Stop feature work with enough time for a clean package, archive listing, two
   colour smoke games and dashboard validation before the stated deadline.
6. Bring the accepted zip, source commit, model hash, training artefacts and the
   code walkthrough below on a laptop and offline backup.

## Five-minute code walkthrough

1. **Contract and reliability:** `agent.py` exposes `get_move`, maintains one
   process's board/repetition state, converts FEN, commits every chosen or
   fallback move, and returns legal UCI.
2. **Board representation:** twelve 64-bit piece bitboards, compact move encoding,
   Zobrist hash, Numba move generation/make/unmake, legal-check filtering and SEE.
3. **Search:** iterative deepening with aspiration/PVS; TT and move ordering;
   conservative pruning/reductions; quiescence for unstable tactical leaves;
   hard periodic deadline checks.
4. **Evaluation:** 75% incremental king-conditioned BigNet and 25% tapered
   handcrafted evaluation. Explain accumulator rebuilds on king-bucket changes,
   pairwise features and material heads.
5. **Training:** show provenance, sampling, labels, split, optimizer/checkpoint,
   export, quantisation verification and packaged hash.
6. **Evidence:** BigNet and V15 search each beat their frozen parent; V16 combined
   them and passed differential, initialization, packaging and paired-game gates.
   State honestly that the Toby match exposed a remaining absolute gap and that
   finals changes target observed long-game behaviour.

## Evidence limits

- All thirteen final-Swiss PGNs from rounds 110--122 are present.
- The Stockfish replay used 100,000 nodes per position; it identifies candidate
  failure points but does not establish exact best play.
- The public site does not contain private invitation logistics.
- Opponent implementations and submitted versions cannot be inferred reliably
  from ratings or team names alone.

## Sources

[^1]: [AI Chessathon official documentation](https://aichessathon.com/docs), accessed 11 September 2026. This is the canonical source for the final window, format, environment, permitted artefacts and walkthrough requirement.
[^2]: [Toby Coad, `TIME_V8` public commit](https://github.com/TobyCoad/aichessathon-starter/commit/8456160d93f537e28a763ece57af92d5678e25ab), 10 September 2026. Used as external engineering evidence, not as code to copy.
[^3]: [Stockfish `timeman.cpp`](https://github.com/official-stockfish/Stockfish/blob/master/src/timeman.cpp), accessed 11 September 2026.
[^4]: [Stockfish `search.cpp`](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp), accessed 11 September 2026.
[^5]: [AI Chessathon event archive](https://aichessathon.com/archive), accessed 11 September 2026. The archive states that the final record will be published after the event, so it currently adds no result or schedule detail beyond the live documentation.

### Research process

The conclusions combine the official rules, public primary-source engine changes,
the V16 source and model metadata, the recorded V16--Toby match summary, and a
fresh Stockfish 100k-node replay of all thirteen available final-Swiss PGNs. No
private invitation detail was inferred beyond the text supplied by the team.
