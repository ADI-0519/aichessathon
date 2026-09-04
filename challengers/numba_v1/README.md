# Numba challenger

This directory is deliberately isolated from the production `agent.py`. It now contains a complete
first playable compiled challenger, but is not included by the current submission packager and has
not passed the full promotion gate.

## Implemented scope

- twelve `uint64` piece bitboards using the python-chess square convention (`a1 == 0`);
- side to move, standard castling rights, en-passant square, halfmove clock, and fullmove number;
- packed 32-bit moves with from, to, promotion kind, and explicit special-move flags;
- precomputed pawn, knight, and king attacks;
- compiled sliding attacks without Python objects or allocations;
- complete standard-chess pseudo-legal generation;
- legal filtering through reversible make/check/unmake;
- castling transit safety, en passant, all four promotions, rights and clock updates;
- fixed-size caller-owned move and undo buffers;
- deterministic incremental Zobrist hashing with canonical legal-en-passant identity;
- repetition, rule-50, and insufficient-material draw state;
- a fully compiled recursive perft driver;
- iterative deepening, PVS, aspiration windows, a fixed-array transposition table, quiescence,
  and TT/capture/killer/history ordering;
- a tapered material, piece-square, pawn-structure, rook-file, and king-shield evaluation;
- deterministic fixed-node searches and a wall-clock stop flag observed by a `nogil` search;
- persistent per-game TT/history plus python-chess root validation and emergency fallback.

The search is intentionally conservative. It does not yet contain LMR, null-move pruning, SEE,
futility pruning, trained evaluation weights, an opening oracle, or pondering. Each belongs in an
isolated challenger only after the baseline's correctness and strength are measured.

## Representation

`pieces[color * 6 + kind]` stores a bitboard. Colors are white `0`, black `1`; kinds are pawn,
knight, bishop, rook, queen, king (`0..5`). `state` is a five-element `int64` array:

| Slot | Meaning |
|---:|---|
| 0 | side to move |
| 1 | KQkq castling bit mask |
| 2 | en-passant target or `-1` |
| 3 | halfmove clock |
| 4 | fullmove number |

`Position.key` is a one-element `uint64` array so compiled make/unmake can update it in place. The
key includes pieces, side, castling rights, and en-passant only when a legal en-passant capture
exists. Halfmove/fullmove counters are deliberately excluded; TT probes separately require an
equal capped halfmove clock.

Packed moves use:

| Bits | Meaning |
|---|---|
| 0–5 | from square |
| 6–11 | destination square |
| 12–14 | promotion kind |
| 15–19 | capture, double pawn, en passant, castling, promotion flags |

An undo record stores only the prior irreversible scalar state, captured piece/square, and original
moving piece. Ordinary and special moves are reversed from the packed move plus that record; the
twelve bitboards are not copied at each node.

## Invariants

- Hot-path functions accept primitive NumPy arrays and scalar integers only.
- Search will allocate one legal buffer, pseudo buffer, and undo record per ply; move generation
  itself allocates nothing.
- `generate_legal_moves` returns with the position byte-for-byte unchanged.
- Every successful `make_move` followed by `unmake_move` restores both arrays and the Zobrist key
  exactly.
- Castling is standard chess only. Chess960 is outside the competition contract.
- Root FEN parsing and eventual emitted-move validation remain with python-chess.
- All decorators use `cache=False`; `agent.py` calls `search.warmup()` during the 90-second import
  allowance so no compilation lands on the chess clock.

## Verified results

As of 4 September 2026:

- starting perft depths 0–5: `1, 20, 400, 8,902, 197,281, 4,865,609`;
- CPW position 3 depths 0–5: `1, 14, 191, 2,812, 43,238, 674,624`;
- exact legal-move agreement on 1,000 deterministic random positions in the unit suite;
- exact complete attack maps on 200 deterministic random positions;
- 500 random make/unmake transitions matched python-chess FEN and restored every field;
- a separate 100,000-position campaign crossed 519 games with exact legal moves, complete sampled
  attack maps, FEN transitions, incremental hashes, and byte-exact undo restoration;
- explicit castling, en-passant, promotion-capture, and rook-rights transitions;
- canonical hashing tests for legal, phantom, and pinned en passant, plus castling/turn identity;
- deterministic fixed-node results, mate-score TT round trips, timed interruption, and varied-FEN
  legal-root tests;
- strict mypy and Ruff clean;
- warm perft approximately 3.7–4.7 million leaf nodes/second on the development machine.

Fresh-process search warm-up measured approximately 21–34 seconds. On the deterministic scaling
run, starting-position searches sustained roughly 145,000 total nodes/second: 50,000 nodes
completed depth 6 in 0.34 seconds and one million nodes completed depth 7 in 6.85 seconds. These
figures are machine-specific.

In the Round 4 `12...Rxc3` regression position, the challenger still selected `Rxc3` through depth
5 at 200,000 nodes, but changed to `f5` after completing depth 6 at one million nodes. That removes
the sacrifice only at an impractical current move budget and identifies qsearch selectivity and
deeper tactical search as the next strength bottleneck; it is not counted as a solved regression.

The initial playing screens were deliberately small:

- `+2 =0 -0` against starter minimax over one paired opening;
- `+5 =1 -0` against the frozen Python champion over three paired openings at 2,000+50 ms;
- `+0 =1 -1` against Stockfish at 500 nodes/move over one paired opening at the comparable
  10,000+100 ms control.

The champion result justifies continued development. The Stockfish result is not a promotion pass;
the full previous-champion baseline was 31.7% over 30 games, and the compiled challenger needs a
larger comparable-control confirmation after search-strength improvements.

Run the standard gate:

```bash
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Run the longer deterministic differential campaign:

```bash
./.venv/Scripts/python.exe -m tools.fuzz_numba_core --positions 100000
```

Run the challenger through the unmodified platform harness:

```bash
./.venv/Scripts/python.exe -m harness.play \
  --white challengers/numba_v1 --black baselines/minimax
```

Perft does no evaluation, transposition lookup, ordering, or clock polling and must not be reported
as search NPS.

## Next engineering boundary

The next changes should improve selective depth one feature at a time. Because qsearch consumed
roughly 74–90% of nodes in the measured positions, begin with exact SEE, bad-capture deferral, and
conservative delta pruning. Then add LMR with full-depth verification and guarded main-search
pruning. Every change must pass unit/perft/fuzz, deterministic positions, a paired champion screen,
and external confirmation. The frozen root submission remains the active safety build until a much
larger promotion match.
