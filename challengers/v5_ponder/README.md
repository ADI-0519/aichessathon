# V5 plus pondering

`v5_nnue` with a background search that runs on the opponent's clock. The platform
keeps our core while they think, and `AGENTS.md` permits using it.

It does not guess their reply. It searches the position *they* are about to move in,
so alpha-beta concentrates on their best move while still touching the alternatives.
Whatever they play, the shared transposition table is already warm for that subtree,
and there is no ponder-miss cliff.

Two changes from `v5_nnue`:

- `search.py`: `search_position` accepts an optional `stop` flag instead of always
  creating its own, so another thread can interrupt it. Nothing compiled changed.
- `agent.py`: `_stop_ponder()` is the first statement of `_choose_move`, so the
  background thread is always joined before the main thread touches `_memory` or
  `_game_board`. A ponder starts just before the move is returned, and is stopped
  on the exception path too.

## Measured

- Stop latency **0.4-1.3 ms** against a 2000 ms join timeout. The tree polls the flag
  every 256 nodes, so this cannot cost a flag.
- Time to reach depth 8 on the position actually faced, when the opponent plays the
  expected move: **-53% to -76%**, roughly a threefold speedup.
- Games against `v5_nnue`: **53.3%** (+25 =14 -21) over 60 paired games at 20s+0.5s,
  95% interval 46.8% to 59.9%. Not resolved. A 240-game run is the arbiter.

The three figures are consistent: a roughly 60% prediction hit rate against two thirds
of the time saved is about +25 Elo, which is what the 60-game point estimate says.
Sixty games cannot resolve an effect that size.
