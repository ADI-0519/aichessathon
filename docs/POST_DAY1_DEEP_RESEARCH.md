# Post-Day-1 competitive engine plan

> Historical decision memo from 5 September. Its architecture research remains useful, but its
> priorities predate V5--V8 results. Current decisions live in `CURRENT_STATE.md` and
> `docs/EXPERIMENT_LEDGER.md`.

**Decision memo — 5 September 2026**

This document supersedes the implementation priorities in the earlier roadmap. It is based on the
current competition contract, a static audit of our V3 engine, our public-game evidence, and primary
material from major chess-engine projects. It deliberately distinguishes ideas we may implement
ourselves from third-party code and networks that the rules prohibit us from shipping.

## Executive decision

Our highest-probability route is:

> **Keep the custom Numba alpha-beta engine, replace guessed positional knowledge with a
> data-tuned evaluation, then add a small number of independently implemented search
> improvements under paired testing. Treat a team-trained compact NNUE as a gated stretch goal,
> not the main plan.**

V3 is now the immutable control. We should expect two substantive challenger generations before
the final build:

1. **V4: position understanding.** A richer, phase-aware classical evaluator whose weights are
   trained from licensed positions and engine labels. A narrowly scoped opening-book experiment
   may join V4 only if it has meaningful held-out coverage and wins games.
2. **V5: search selectivity and final integration.** Individually tested move-ordering and pruning
   changes, followed by official-clock reliability testing.
3. **V6 only if earned:** a compact network trained entirely by us, added only if end-to-end
   equal-time games beat the best classical build before the go/no-go deadline.

We should **not** spend the remaining week building AlphaZero/MCTS, reproducing Lc0, training a
Maia-style human imitator, adding pondering, or integrating a third-party engine/network. Those
paths either target the wrong objective, require much more compute and systems work, or conflict
with the rules.

No honest plan can guarantee first place against an unknown field. This plan maximises the amount
of reliable strength we can add per remaining engineering hour while preserving a submission that
cannot crash or flag.

## The live constraints change the architecture

The [current competition documentation](https://aichessathon.com/docs) was rechecked on 5
September. The relevant facts are:

- One AMD EPYC core, 2 GB RAM, no GPU and no network during a game.
- 120 seconds plus 0.5 seconds per move; a crash, illegal move, malformed response or flag is a
  loss.
- A 90-second import budget, a 50 MB uncompressed submission, and only the fixed Python/NumPy/
  Numba/Torch/ONNX Runtime/python-chess environment.
- The latest passing upload plays; there are ten uploads per team per day; uploads close at 11:00
  London time on 11 September. Earlier final submission is the last listed tie-break.
- Third-party engines and networks are forbidden. A network must be trained by our team. Training
  on positions labelled by an existing engine is explicitly allowed. Opening books and tablebases
  are explicitly allowed.
- **The live page now says our process is suspended while the opponent moves.** Older copies of
  the documentation said the process retained its core and could ponder. Pondering therefore has
  no place in the current plan.

This environment rewards fast, single-threaded, CPU-friendly evaluation inside alpha-beta. It is
almost the exact setting for which NNUE was designed, but the short deadline changes the risk:
training a network is permitted; correctly integrating an incrementally updated network into every
make/unmake path is still a substantial engineering project.

The competition structure matters too. Qualification is a 13-round Swiss, while the ladder only
seeds it. Reliability and consistent conversion against weaker engines are more valuable than a
volatile trap that wins a few ladder games and occasionally crashes.

## What V3 actually is

V3 is no longer a toy minimax bot. Its hot path is a team-written Numba bitboard engine with:

- legal move generation and make/unmake;
- incremental Zobrist hashing and draw handling;
- iterative deepening, aspiration windows, principal-variation search and quiescence search;
- a persistent transposition table;
- SEE-informed capture ordering, killers, quiet history, delta pruning and conservative late-move
  reductions;
- soft/hard time checks, a legal fallback and root-move validation.

That is a credible base. Replacing it would throw away the most expensive solved work.

The visible weakness is its static understanding. The evaluator is presently material plus
procedural piece-square terms, tapered phase, bishop pair, basic pawn structure, open/semi-open
rooks, a small king shield term and tempo. It lacks several interactions that distinguish quiet
good moves from quiet bad ones: mobility and safe mobility, king-ring pressure, threats and loose
pieces, richer pawn relations, outposts, space, development, and drawish/endgame scaling. Its
search also lacks several selective mechanisms, but aggressive pruning becomes safer only after
static evaluation and move ordering are trustworthy.

The independent Stockfish run in progress is useful evidence, not a final verdict. We must wait
for complete pairs and inspect the PGNs. A small match score cannot tell us which subsystem is at
fault, and its interval is much wider than the displayed percentage makes it feel.

## What the major engine families teach us

### Stockfish: the relevant reference architecture

Stockfish combines a CPU-efficient NNUE evaluation with highly selective alpha-beta/PVS search.
Its current [`search.cpp`](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp)
shows TT cutoffs, corrected static evaluation, improving-state logic, razoring, futility pruning,
verified null-move pruning, layered history signals, LMR and tablebase probes. The lesson is not to
copy its implementation or constants. It is that strength comes from the interaction of:

1. a useful static value;
2. excellent move ordering;
3. selective search guarded by node type, depth, check state, material and evaluation margins;
4. enormous empirical testing.

Stockfish's own development guidance says to keep changes focused, test one idea per patch, screen
at short time control and confirm at longer time control. It uses sequential tests that can run
thousands of games because small apparent gains are often noise. See the official
[Fishtest methodology](https://official-stockfish.github.io/docs/fishtest-wiki/Creating-my-first-test.html)
and [Fishtest mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html).
We cannot reproduce its scale, but we can reproduce its discipline.

NNUE was introduced into Stockfish as a CPU-friendly learned evaluator trained from millions of
moderate-depth engine-labelled positions, while alpha-beta remained the move-search mechanism. The
initial integration was reported at more than 80 Elo over the classical evaluator in large
Fishtest matches. That result demonstrates the potential of learned evaluation, not the expected
gain from our first small network. See Stockfish's [NNUE announcement](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/).

### NNUE: why it works, and why it is not a two-line change

The official [NNUE technical documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
explains the core trick: sparse piece-square features feed a first layer whose accumulator is
updated by subtracting features removed by a move and adding features introduced by it. King moves
can force a refresh. Later layers are small and dense. Integer quantisation lets CPUs process many
small values efficiently, with wider accumulators to avoid overflow.

That produces an important implementation requirement for us: a viable NNUE must live inside the
Numba search state. Calling Python, Torch or ONNX Runtime at every leaf would cross the compiled
boundary millions of times and destroy throughput. Recomputing a large sparse first layer at every
leaf would lose most of NNUE's advantage. We need accumulator state on the ply stack, exact update
and undo parity, quantised inference, and special handling for king moves, promotions, castling and
en passant.

The same documentation recommends mapping centipawn values to win/draw/loss space with a fitted
sigmoid and optionally blending engine evaluation targets with game results. This is directly
useful both for a small NNUE and for tuning our classical evaluator.

### Lc0 and AlphaZero: powerful, but mismatched to this event

AlphaZero learns a policy and value network from self-play and uses those outputs in tree search.
The original paper describes a deep network producing move probabilities and expected outcome,
then MCTS/PUCT choosing which branches to expand. See the
[AlphaZero paper](https://arxiv.org/abs/1712.01815) and Lc0's
[PUCT primer](https://lczero.org/dev/lc0/search/alphazero/).

Lc0's conventional networks use a residual tower with many channels and separate policy, value
and moves-left heads; its project supports GPU-oriented backends as well as CPU backends. See the
[Lc0 network topology](https://lczero.org/dev/old/nn/) and
[engine repository](https://github.com/LeelaChessZero/lc0). The competition gives us one CPU core,
no GPU, 50 MB, six development days and no existing self-play infrastructure. A new policy/value
network plus MCTS would require us to solve data generation, distributed training, inference,
search and time control at once. It is a research programme, not a competitive patch for this
deadline.

One Lc0 idea remains useful: learned policy can improve move ordering. If every higher-priority
track is complete, a tiny **root-only** policy experiment could be measured without replacing
alpha-beta. It remains below NNUE and classical search work in priority because iterative
deepening already gives the root good ordering after the first completed iteration.

### Maia: excellent research, wrong objective

Maia replaces self-play with human games to predict the moves made by players at particular rating
levels. Its published objective is granular human-behaviour prediction, not maximum playing
strength; the paper explicitly contrasts this with Stockfish and Leela. See the
[Maia paper](https://arxiv.org/abs/2006.01855) and
[project repository](https://github.com/CSSLab/maia-chess).

Human-game data is valuable for position diversity and opening frequencies. A Maia network is not
a strength shortcut, cannot be shipped under the rules, and would teach the bot to reproduce human
errors as well as human choices.

### Deep Blue: systems integration, not a portable recipe

IBM's Deep Blue combined custom chess hardware, massive parallelism, search extensions, a complex
evaluation and a grandmaster database. Its hardware strategy is unavailable here, but the broader
lesson survives: no single algorithm won by itself; move generation, ordering, evaluation, search
control and domain data were co-designed. See IBM Research's
[Deep Blue paper page](https://research.ibm.com/publications/deep-blue).

### Sunfish and modern independent engines: the practical bridge

[Sunfish](https://github.com/thomasahle/sunfish) demonstrates how much a compact Python engine can
do with iterative search, a transposition table and cheaply updated piece-square evaluation. It is
a teaching reference, not something we may port into the submission. Its own simplicity also
exposes the ceiling of shallow positional knowledge.

Modern independent engines reinforce the staged route:

- [Viridithas](https://github.com/cosmobobak/viridithas) documents a progression from material/
  piece-square evaluation, to a Texel-tuned handcrafted evaluator, to self-play data and NNUE. It
  also records thousands of paired games for release changes.
- [Seer](https://github.com/connormcmonigle/seer-nnue) combines a learned WDL evaluator with
  conventional PVS, TT, SEE, multiple history signals, null move, reverse futility, futility, LMR
  and aspirations.
- [Ethereal](https://github.com/AndyGrant/Ethereal) similarly pairs alpha-beta with a neural
  positional evaluator and uses the Fishtest-inspired OpenBench framework for development.

These are evidence for architecture and process. We must independently write and explain every
runtime component, and train every shipped weight ourselves.

## Architecture decision matrix

| Candidate | Upside | Deadline risk | One-core fit | Decision |
|---|---:|---:|---:|---|
| Richer, data-tuned classical evaluation | High | Low–medium | Excellent | **Primary V4 track** |
| Focused alpha-beta selectivity patches | High | Medium | Excellent | **Primary V5 track** |
| Broad, conservative opening book | Medium, coverage-dependent | Low | Excellent | One-day measured experiment |
| Small team-trained NNUE | Potentially very high | High | Good only with incremental integer inference | Conditional V6 |
| Root-only learned policy | Low–medium | Medium | Acceptable | Stretch after NNUE decision |
| Syzygy tablebases | Low in expected game frequency | Medium | Good | Skip for now |
| AlphaZero/Lc0-style MCTS | High long-term | Extreme | Poor without suitable accelerator | Reject for this event |
| Maia-style human imitation | Wrong objective | High | Poor | Reject |
| Pondering | None under live lifecycle | Medium/racy | Process is suspended | Reject |

The full five-piece Syzygy WDL and DTZ sets take 378 MB and 561 MB respectively, already far above
the entire submission allowance, according to the [Syzygy generator documentation](https://github.com/syzygy1/tb).
A hand-picked smaller subset would add complexity and very sparse coverage. We can revisit it only
after the main engine is frozen.

## V4: learn the evaluator before learning a network

A tuned classical evaluation is machine learning: the feature definitions are ours, while the
weights are estimated from data. It gives us most of the experimental pipeline an NNUE needs while
remaining transparent, cheap to execute and easy to debug.

### Feature families

Implement features as signed white-minus-black counts with separate middlegame and endgame
coefficients where appropriate:

1. **Mobility:** legal or pseudo-legal mobility by piece, preferably discounted for attacked
   destinations; trapped minor/rook indicators.
2. **Pawn structure:** isolated, doubled, backward, connected and protected pawns; passed and
   candidate passers by rank; pawn islands.
3. **King safety:** pawn shelter and storms; open/semi-open files near the king; weighted attacks
   into the king ring; attacker count. Avoid expensive legal-move generation inside evaluation.
4. **Threats:** attacked loose pieces, pawn attacks on pieces, minor-on-major pressure, and simple
   hanging-piece terms.
5. **Piece quality:** supported/outpost knights, bad bishops, bishop pair, rook activity and seventh
   rank, rook behind passer, queen development/exposure.
6. **Space and development:** controlled central/advanced squares and undeveloped minors while
   queens remain.
7. **Endgame scaling:** insufficient or nearly insufficient material, opposite-coloured bishops,
   pawnless winning-material cases and phase-aware king activity.

Do not add every feature in one patch. Start with mobility, king pressure, threats and richer pawn
structure because they address the largest missing concepts. Measure evaluation cost separately;
search strength depends on information per microsecond, not feature count.

### Data source

Use data with explicit provenance. The [Lichess open database](https://database.lichess.org/) is
CC0 and currently provides both game exports and hundreds of millions of Stockfish-evaluated
positions. The official competition rules allow engine-labelled training. This is preferable to
an arbitrary Kaggle mirror whose source, transformations and licence may be unclear.

Recommended first dataset:

- sample 100,000–300,000 labelled positions rather than downloading or processing the whole
  corpus;
- split by game or normalized position hash **before** fitting, so transpositions and adjacent
  plies do not leak into validation;
- deduplicate positions and cap positions per game;
- balance opening/middlegame/endgame, side to move, material, and evaluation bands;
- retain a separate tactical/quiet tag; either reject unstable tactical positions for the first
  fit or label them after quiescence;
- clamp or separately handle mate labels and extreme evaluations;
- record source URL, source checksum, extraction seed, filters and label convention.

The existing official Stockfish 8moves suite is a match-opening suite, not training data. Keep its
development, validation and holdout partitions clean.

### Fitting and integration

Build one reference feature extractor and test it against the Numba implementation on randomized
legal positions. Then:

1. Fit middlegame/endgame coefficients using regularised regression or a WDL-space loss.
2. Compare against the current evaluator on held-out label error, calibration and move-ranking
   agreement. These are diagnostics, not promotion criteria.
3. Round to bounded integer constants and recheck that Python/reference and Numba scores agree
   exactly.
4. Benchmark fixed positions and fixed depth to quantify nodes per second and branching effects.
5. Play paired games against V3. Equal-time games, not offline loss, decide promotion.

The official NNUE documentation's WDL transform is a useful model: fit the centipawn-to-outcome
scale on our data instead of copying a constant from another engine.

### Opening-book experiment

The book must be a genuine broad opening book, not a lookup database of engine answers. Build it
from licensed human games, key by full position identity, and store only a few well-supported moves
per position. Apply minimum game/rating and result-quality filters, then down-weight sharp moves
with weak support. Never return a book move without checking it is legal.

Before integrating it, measure exact-position hit rate on the development and validation opening
suites and play book-on versus book-off games. If coverage is low or score is neutral, remove it.
The [official Stockfish books repository](https://github.com/official-stockfish/books) is CC0 and
shows how paired engine testing uses broad, fixed opening sets; our checked-in 8moves sample comes
from that source. A test-opening suite and a runtime move book serve different purposes and must
not be conflated.

## V5: search improvements in the right order

Each item gets its own branch or commit, its own before/after benchmark, and its own paired match.
Constants are starting hypotheses, not values to copy from another engine.

1. **History maluses and countermove ordering.** Reward the quiet move that causes a cutoff and
   penalise quiets searched before it. Add a small previous-move-to-reply table. This improves
   ordering without deliberately dropping branches.
2. **Reverse futility pruning.** At shallow non-PV nodes, not in check, return when reliable static
   evaluation exceeds beta by a depth-scaled margin. Exclude mate-score regions.
3. **Guarded null-move pruning.** Only at non-PV nodes, not in check, with non-pawn material and a
   comfortably high static evaluation. Use a reduced null search and verification at higher depth;
   disable in likely zugzwang endgames.
4. **Shallow futility and late-move pruning.** Skip late quiet moves only at shallow depths when
   static evaluation plus a margin cannot raise alpha. Never prune checks, promotions, tactical
   captures or mate-score windows.
5. **Extensions and reduction refinement.** A tightly capped check extension and a log/depth/
   move-count LMR table are candidates. Re-search every reduced move that improves alpha.
6. **Time-management tuning.** Use completed-depth stability, score swings and root move changes
   to adjust the soft limit while retaining an unconditional hard reserve. Test this at the real
   120+0.5 clock.

Do not attempt singular extensions, correction histories, a pawn hash and multiple new pruning
schemes simultaneously. They have high interaction cost and are difficult to validate in six
days.

## Compact NNUE: explicit go/no-go gate

Begin an NNUE spike only after the V4 data and testing pipeline works. A reasonable prototype is a
small king-aware piece-square input, a 64- or 128-wide accumulator per perspective, one or two tiny
dense layers, clipped activations and integer weights. The exact topology is an experiment.

The spike must demonstrate all of the following by **09:00 on 9 September**:

- the weights were trained by us and the dataset/config/checkpoints are reproducible;
- incremental and full-refresh accumulators match on long randomized make/unmake sequences,
  including castling, en passant, promotion and king moves;
- quantised inference closely matches the floating-point reference without overflow;
- import plus warm-up stays comfortably below 90 seconds and the archive remains below 50 MB;
- fixed-depth/fixed-time benchmarks explain its speed cost;
- equal-time paired games beat the strongest classical challenger, with no failures.

If any gate fails, freeze NNUE work. A tuned classical V5 is a complete and competitive entry;
an unfinished network is a failure multiplier.

Tools such as Stockfish's trainer or [Bullet](https://github.com/jw1912/bullet) are valuable
references for data layout and training workflow, but using their runtime network or copying an
engine implementation is not allowed. Training may use extra local tools; only the submitted
runtime is constrained to the platform packages.

## Testing protocol for every challenger

### Invariants before games

- unit tests, perft, randomized make/unmake and Zobrist consistency;
- legal-move equivalence against python-chess on adversarial positions;
- search termination at tiny budgets and low remaining clocks;
- deterministic fixed-node benchmark with node count, depth, TT hit/cutoff rate, qsearch share,
  LMR reductions/researches and elapsed time;
- zip inspection, clean-process import, two smoke games and environment-compatible imports only.

### Match ladder

1. **Development screen:** 20–30 opening pairs at 10+0.1 against V3.
2. **Validation:** at least 50 fresh pairs for a promising patch. Do not tune on this split.
3. **External anchors:** Stockfish at fixed 500 and 2,000 nodes, plus Alpha Gambit or another strong
   independently run opponent where available. These diagnose absolute weaknesses; they do not
   replace candidate-versus-champion games.
4. **Official clock:** 8–12 fresh pairs at 120+0.5 for the final two candidates, with exact runtime
   logging.
5. **Holdout:** touch the reserved opening split only for a release candidate.

Always report complete opening pairs, pentanomial outcomes, score interval, colour split,
termination reasons and failures. A patch is not promoted because it is ahead after a handful of
games. If the result is ambiguous, either extend the test or keep the simpler champion.

### Regression corpus from losses

For every ladder or benchmark loss, record:

- first large evaluation swing;
- whether the cause was horizon/search, static evaluation, move ordering, draw handling or time;
- the pre-blunder FEN and tactical motif;
- expected safe candidate moves from offline analysis;
- whether the new build fixes the position without breaking a related position.

This corpus is for regression and diagnosis, not for shipping a table of memorized answers.

## How we differentiate from other AI-assisted teams

Most teams can ask an AI to produce minimax, piece values, a piece-square table, alpha-beta and a
few named pruning techniques. Those ingredients are not a moat. Our advantage must be execution:

1. **A compiled custom hot path.** V3 already searches with a Numba bitboard core rather than
   python-chess recursion.
2. **Competition-shaped measurement.** Paired openings, independent anchors, official clock,
   confidence intervals, failure accounting and a genuinely untouched holdout.
3. **Engine-specific learned knowledge.** Our evaluator or network is trained for our representation
   and search, with a reproducible lineage—not borrowed weights or generic tables.
4. **One-idea experiments.** We avoid the common failure mode where ten plausible heuristics are
   added together and nobody knows which one caused the regression.
5. **A loss-derived regression suite.** Every public failure becomes a durable test without becoming
   a prohibited runtime oracle.
6. **Reliability as strength.** Zero crashes, illegal moves and flags can be worth more Swiss points
   than a fragile theoretical Elo gain.

The differentiator is not that nobody else knows NNUE or null-move pruning. It is that we can prove
which version works in this exact container, time control and engine.

## Six-day execution schedule

### Night of 5 September — preserve the measurement

- Let the current SF500 then SF2000 chain finish without editing `agent.py`, `engine.py` or
  `search.py`.
- Preserve raw PGNs, manifest, command, engine hash and summary.
- V3 remains the rollback submission.

### 6 September — diagnosis and data foundation

- Review complete games by first evaluation swing, not only W/D/L.
- Add offline feature extraction and dataset validation.
- Produce a small end-to-end labelled dataset and fit the first linear/tapered evaluator.
- Benchmark the exact evaluator cost before starting games.

### 7 September — V4 evaluation candidate

- Finish the first high-value feature group and integer integration.
- Run development then validation matches against V3.
- Run a time-boxed book coverage/on-off experiment.
- Promote only the measured winner to `challengers/v4_eval`.

### 8 September — V5 search candidate

- Test history maluses/countermove first.
- Test reverse futility and guarded null move separately.
- Keep only passing patches; retest the union because interactions can reverse gains.

### 9 September — network decision and long games

- Apply the 09:00 NNUE gate. Continue only if the full integration is already correct and fast.
- Begin official-clock matches among V3, V4 and V5.
- Analyse failures and colour asymmetry.

### 10 September — release freeze

- No speculative features.
- Holdout pairs, real-clock gate, clean-process validation, zip size/root/import inspection.
- Build a versioned archive and submit the strongest validated build by the evening, leaving time
  for platform validation and preserving the earlier-submission tie-break.

### 11 September — buffer, not development

- Use the morning only for a confirmed validation/platform issue.
- Do not replace a passing build near 11:00 with an under-tested candidate.

## Immediate next development task

When the overnight chain finishes, the next code stage should be **the evaluation-data pipeline,
not another bundle of search heuristics**:

1. freeze and summarize both benchmark runs;
2. build a provenance-tracked position sampler and label schema;
3. implement a reference feature vector plus parity tests;
4. fit a small phase-aware model;
5. integrate one feature group into a challenger and measure it against V3.

That creates V4 and also lays the foundation for a legal team-trained NNUE. It is the shortest path
from today's credible tactical engine to a bot that understands the quiet positions where strong
engines earn their advantage.

