# exp_v14_bignet

V14's search carrying the 200M-position evaluator. Assembled 2026-09-11.

Two gains that had only ever been measured separately:

| build | search | evaluator | vs external_a |
|---|---|---|---|
| `adi_v14` | V14 | format 2, 128 acc, 1 head, ~4M positions | 34.4% |
| `bignet256` | v11_big | format 3, 256 acc, 8 heads, 200M positions | 31.2% |
| this | V14 | format 3, 256 acc, 8 heads, 200M positions | — |

The net is worth +112 Elo under v11_big's search (+28 =7 -13, 65.6% over 48
paired games, 95% interval 56.9%-74.4%, against the evaluator in `current/`).
Whether that survives a stronger search is what this build exists to answer.

## How it was assembled

`engine.py`, `search.py`, `agent.py`, `time_manager.py` are adi_v14's. Only one
line of adi_v14's `engine.py` differs from `exp_kingnet_v11_big`'s (a
`perft(position, 1)` warmup call), so the engine layers were already
interchangeable; the search and clock are where V14's work lives.

`nnue.py` is `exp_kingnet_v11_big`'s, because adi_v14's hardcodes
`FORMAT_VERSION = 2` and rejects anything else on load. It needed one function
ported forward.

### The `update_after_move` port

V14 delays the accumulator delta until it knows a child will actually be
searched -- roughly half of ordered qsearch moves are rejected first -- so it
calls `nnue.update_after_move()` *after* `make_move`, reading the moved and
captured pieces out of the populated undo row instead of looking them up.

The v11_big `nnue.py` had no such entry point, and the obvious port is wrong.
`update_for_move()` opens with `refresh(pieces, parent)` for format 3, which is
correct only while `pieces` still describes the parent. After `make_move` it
describes the child, so refreshing the parent there rebuilds the parent's
accumulator from the wrong board.

So `update_after_move()` does not refresh. If the parent is not fresh its
bucket holds the STALE sentinel, and using that as a feature offset reads
garbage weights -- it hands the staleness to the child instead and lets
`evaluate()` rebuild from the child's own position. That costs a full rebuild
rather than an incremental update, but only in the rare case, and it is correct
by construction rather than by assumption.

`update_for_move()` was split so both entry points share one body
(`_update_from_metadata`), which derives the side to move from the moving
piece's colour rather than from `state`, since post-move `state` has flipped.

### Init-budget fix

Also carries the numba fix from `ee665e5`: a pinned signature on
`is_square_attacked` (twelve call sites pass a constant castling square, and
numba was compiling thirteen copies) and `np.int64()` on the `ply` and
no-TT-move constants entering `_negamax`/`_order_moves`.

## Measured

    verify_kingnet        maximum_incremental_error   0.0  over 406 positions
                          116 stale-bucket transitions exercised
                          median 0.57 cp, p99 4.27 cp, max 7.00 cp
                          1,302,336 evaluations/second
    update_after_move     max accumulator delta         0  over 464 positions
                          differential against full rebuild, 5 stale deferred

Cold import, interleaved three rounds on an idle machine, best of each:

    current                36.2 s    1.00x
    exp_v14_bignet         41.8 s    1.15x
    adi_v14                61.4 s    1.70x

adi_v14 does not carry the init fix. At ~1.4x local-to-platform (pre-fix
`current` measured 61.4s local against 88.5s on rated round 98) that puts it
near 88s of a 90s budget, which is worth fixing there regardless of what this
build measures.
