# Search upgrade: what was changed and what it was worth

Date: 4 September 2026
Baseline: `champion/agent.py`, a frozen copy of the uploaded agent (SHA-256
`391082CC…`, the artifact described in [TRIAL_RECOVERY_REPORT.md](TRIAL_RECOVERY_REPORT.md))

## Headline

The rebuilt search scores **64.7% against the uploaded champion** over 58 paired games at
4 s + 0.1 s: +30 =15 -13, Elo **+105** with a 95% interval of +29 to +192. No crash, illegal move,
flag or init failure in 134 games across all runs.

**Nearly all of that gain comes from Tier 1 — throughput and clock. The Tier 2 pruning work
bought three plies of depth and no measurable strength.** That result should change what we build
next; see [Where this leaves the roadmap](#where-this-leaves-the-roadmap).

## Where the time was going

Profiling the champion on the round-4 losing position, at a 3 s budget:

- ~12–20k nodes/sec, depth 3–4 on a live budget.
- **77% of all nodes were quiescence nodes**, and each one called `list(board.legal_moves)` —
  generating roughly thirty quiet moves and discarding them to keep the captures.
- Legal move generation was ~44% of search time; `evaluate` was another ~44%.
- `_piece_square` was called 262,000 times in three seconds, re-deriving a pure function of
  `(piece_type, square, colour)` that has only 768 distinct results.
- The iterative-deepening predictor left **25–35% of every move's clock unspent**: a 4 s budget
  finished in 2.95 s, an 8 s budget in 6.22 s.

## Tier 1 — throughput and clock

| Change | Why |
|---|---|
| `_PST`, a precomputed piece-square table | `_piece_square` is pure; the table is built inside the 90 s init budget |
| `evaluate` walks piece bitboards | avoids building the whole `piece_map()` dict per call |
| Quiescence generates captures directly | `generate_legal_captures()` plus quiet promotions, instead of all legal moves then filtering |
| Partial iteration results kept | a timed-out iteration used to be discarded whole; ordering searches the previous best first, so any root move that replaced it did so on a deeper search |
| Soft and hard deadlines | the soft one stops new iterations, the hard one cuts off a running one — the clock stops sitting idle |
| Transposition key bucketed on the halfmove clock | keying on the exact clock split each position across up to a hundred entries; exact discrimination is only kept from 80 plies, where the fifty-move rule is actually in view |
| Key hashed to an int, table raised to 900k entries | the tuple key dominated entry size; hashing made a much larger table affordable |
| Check extensions | never hand a position with the king attacked to a capture-only quiescence |

Node rate went from 19,180 to 34,900 nps. Against the champion: 65.0% over 30 games,
Elo +108 (95% CI +5 to +232).

## Tier 2 — selectivity

Null-move pruning, static exchange evaluation for quiescence pruning, delta pruning, futility
pruning at depth 1, and a time extension when the root move is still changing.

Depth reached at a 3 s soft / 7.5 s hard budget:

| Position | champion | Tier 1 | Tier 2 |
|---|---:|---:|---:|
| round-4 loss | 4 | 5 | **8** |
| kiwipete | 4 | 6 | 7 |
| start | 6 | 8 | 9 |
| pawn endgame | 11 | 12 | 14 |
| middlegame | 4 | 5 | 7 |

On the round-4 position the champion and Tier 1 both play `12...Rxc3`, the losing sacrifice.
Tier 2 reaches depth 8 and plays something else. (It plays `f7f5`; no engine probe of that move
exists in our records, so the claim is only that it stops walking into the known blunder.)

And yet, measured by games:

| Match | Games | Score | Elo (95% CI) | Verdict |
|---|---:|---:|---|---|
| Tier 2 vs Tier 1, 2 s + 0.05 s | 58 | 49.1% | −6 (−87 to +75) | not resolved |
| Tier 2 vs Tier 1, 4 s + 0.1 s | 46 | 52.2% | +15 (−70 to +103) | not resolved |
| Tier 2 vs champion, 4 s + 0.1 s | 58 | 64.7% | +105 (+29 to +192) | stronger |
| Tier 1 vs champion, 3 s + 0.1 s | 30 | 65.0% | +108 (+5 to +232) | stronger |

Tier 2 scores the same against the champion as Tier 1 does. Three extra plies, no strength.

## Where this leaves the roadmap

The evidence that depth is no longer the binding constraint is now threefold:

1. Three plies of extra depth produced no measurable gain (two independent runs).
2. The round-4 forensics already showed it: at depth 5 the engine abandoned `Rxc3` (−1.7) for
   `b5` (−1.8) — a *different* bad move, because the evaluation could not separate them.
3. `Ng6`, the move an engine scores at −0.3, is not found at any depth we can reach.

[HYPERCOMPETITIVE_ROADMAP.md](HYPERCOMPETITIVE_ROADMAP.md) schedules the Numba board core
(Phases 3–4, one to two days each) before training the evaluation (Phase 5). A Numba core buys
what Tier 2 just bought — more nodes, therefore more depth — and Tier 2 is direct evidence that
more depth is currently worth close to nothing. **Phase 5 should come first.**

What the evaluation is missing, concretely: it has no mobility term and no king-safety term
beyond a +9 bonus per pawn in front of the king. The round-4 game was lost to a mating attack the
evaluation had no way to see coming. A hand-guessed mobility term was already tried once and
scored 50.0% — which is the argument for tuning weights against labelled positions rather than
guessing them, not an argument that mobility does not matter.

Engine-labelled training data is explicitly allowed by the rules; only shipping engine code or
weights is not. The [Lichess open database](https://database.lichess.org/) provides CC0 games with
Stockfish evaluations.

## Reliability gate

Every check below was run on the build described here.

| Check | Result |
|---|---|
| `ruff check .` | clean |
| `mypy` (strict, 9 files) | clean |
| `unittest discover -s tests` | 14/14, including three new for capture-only quiescence and the soft/hard clock |
| Failed terminations across 134 arena games | none: no crash, illegal move, flag or init failure |
| One game at the rated 120 s + 0.5 s control | won by checkmate, no flag |
| `import agent` | 60 ms against a 90,000 ms budget |
| Worst-case table memory, both caps full | 177 MB, leaving roughly 1.7 GB against the 2 GB cap |
| `submission.zip` | `agent.py` alone at the root, 30,644 bytes unzipped |

The enlarged tables were the one change that needed measuring rather than assuming: `TT_MAX_SIZE`
went from 100,000 to 900,000 entries, which is what hashing the key to a machine int made
affordable. Filled to the cap alongside the eval cache, both together cost 177 MB.

## Reproducing any of this

See [PROMOTION_TESTING.md](PROMOTION_TESTING.md). In short:

```bash
uv run python -m tools.paired_arena --candidate . --opponent champion --base-ms 4000 --increment-ms 100 --extra-positions 20
```
