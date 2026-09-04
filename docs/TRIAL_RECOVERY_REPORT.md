# Trial recovery report

Date: 4 September 2026

## Outcome

The new `submission.zip` is a conservative but measurable upgrade over the exact previously
submitted artifact. It keeps the same readable pure-Python engine architecture and changes search
efficiency, state reuse, and early clock allocation. It does not add third-party code, an external
engine, or borrowed model weights.

Final artifact:

- contents: one root-level `agent.py`;
- expanded size: 19,558 bytes;
- SHA-256: `391082CCF4E635D65E17C26603064D5AA58727373EF308155BF43BA9489EED61`;
- Ruff and strict mypy: clean;
- unit/regression tests: 11/11 passed;
- extracted-archive smoke match: 1 win, 3 draws, 0 losses against starter minimax.

## What the three live games established

Clock timing identifies our side as Black in rounds 1 and 3 and White in round 2. The move-time
pattern matches the agent's budget function, while the opposing clock often used much less time.
Offline Stockfish 18 analysis was used only as a development diagnostic and is not part of the
submission.

| Game | Important errors | Diagnosis |
|---|---|---|
| Round 1, Black | `12...O-O-O`, `14...a5` | The opening was playable; consecutive middlegame choices allowed the position to collapse. |
| Round 2, White | `27.exf5`, `28.Qc5`, `33.Rf7` | A substantial advantage was lost through tactical horizon errors, culminating in a mating attack. |
| Round 3, Black | `11...Bd4`, `17...Bh6`, `24...Rfe6`, later passive moves | The first three moves were strong; search/evaluation errors accumulated in the middlegame. |

There was no crash, illegal UCI output, memory failure, or flag. Our clocks still held roughly
45-70 seconds late in the games. The immediate problem was therefore playing strength per node and
unused clock reserve, not API reliability or inability to beat the starter baselines.

## Changes retained

1. Poll the wall clock once per 32 visited nodes instead of at every node.
2. Generate legal moves once per node and use the empty list to score mate or stalemate, avoiding
   redundant `is_checkmate()` and `is_stalemate()` move generation.
3. Stop calling `gives_check()` for every move during ordering; call it only for quiet moves that
   are actual late-move-reduction candidates.
4. Cache static evaluations in a bounded 65,536-entry cache.
5. Reuse a bounded transposition table between moves in the same game, with depth-preferred
   replacement. Only completed searches write entries.
6. Spend more of the clock while at least 60 seconds remain. Below that threshold the original
   conservative schedule is retained. A pessimistic 300-move simulation with the official
   500-millisecond increment retains positive time.

The complete challenger scored 15 wins, 7 draws, and 8 losses (61.7%) against the exact accepted
champion over 30 color-swapped games from 15 positions. This clears the roadmap's 55% fast-screen
threshold, but 30 games still has a wide error bar; the result is evidence of an upgrade, not a
precise Elo estimate.

It then scored 13 wins, 7 draws, and 0 losses (82.5%) over 20 accelerated games against the starter
minimax. This confirms that the live opponents, not the visible starter bots, are the current
competitive gap.

## Changes rejected

| Candidate | Result | Decision |
|---|---:|---|
| Speed changes alone vs accepted champion | 11-10-9, 53.3% | Useful foundation, not decisive alone. |
| Small bitboard mobility term vs accepted champion | 10-10-10, 50.0% | Rejected; neutral aggregate result and a supplied-position regression. |
| Quiet checks on first quiescence ply vs speed/time control | 10-8-12, 46.7% | Rejected; increased tree cost and changed a correct live-position choice into a speculative sacrifice. |
| Persistent TT vs otherwise identical control | 11-9-10, 51.7% | Inconclusive alone; retained because the complete build cleared the champion gate and the mechanism reuses completed work. |

This experiment trail is also why a last-minute learned evaluator was not adopted. The other
provided repository's evolved learned-model champion lost 21-4-7 to our prior agent locally, and
its own experiment record showed that improved prediction metrics repeatedly failed to improve
games. A dataset is not a substitute for a fast search core and paired playing-strength tests.

## Data and ML decision

For later training, prefer the official [Lichess open database](https://database.lichess.org/) over
an arbitrary Kaggle mirror. It provides CC0 games, Stockfish-evaluated positions, and tactical
puzzles with clear provenance. Engine-labelled training is permitted by the
[AI Chessathon rules](https://aichessathon.com/docs/rules.md), but any shipped model must be trained
by the team and no third-party engine may run in the submission.

A small NNUE remains a later option, not tomorrow's shortcut. The official
[Stockfish NNUE documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
explains the key deployment trade-off: sparse incremental evaluation is valuable only when its
extra accuracy outweighs the search speed it costs. The next major engineering phase should first
move board representation, make/unmake, move generation, and search into a team-written Numba hot
path, then compare tuned HCE and small NNUE challengers by games.

## Next phases

1. Upload and preserve this exact artifact and its platform validation log.
2. Add a larger curated tactical regression suite from the three failures plus Lichess puzzles.
3. Build the Numba board/move-generation core behind python-chess root validation; require exact
   perft and make/unmake invariants before search.
4. Add iterative deepening, TT, PVS, quiescence, and conservative ordering to the compiled core.
5. Train evaluation weights from direct Lichess data and team-generated Stockfish labels; split by
   game, measure NPS, and promote only through paired games.
6. Test safe pondering only after the non-ponder engine survives long clock and race-condition
   stress tests.

The canonical contract remains the live
[agent documentation](https://aichessathon.com/docs/agent-contract.md). Recheck it before each
upload because limits and competition details may change.
