# V12: a clock that knows how long the game has left

`v11_full` with the time manager repaired. Nothing else differs.

## What was wrong

`estimated_moves_remaining` floored at twelve, so past move fifty it assumed the
game was nearly over however long it ran. The budget is the usable clock divided
by that number, so it *rose* to the 4500 ms cap as a game went long. Rounds 80
and 82 reached moves 86 and 87 spending the maximum from move 40 with more than
forty moves still to play, and finished under seven seconds against opponents
holding twenty-five. Both were losses.

The audit of rounds 77 to 82 priced the consequence over 342 scored moves:

| think time | moves | mean cp loss | losses of 200 cp or more |
| --- | --- | --- | --- |
| under 1.0s | 49 | 35.4 | 2 |
| 1.0 to 2.0s | 83 | 18.9 | 1 |
| 2.0 to 3.0s | 96 | 27.2 | 3 |
| 3.0s and over | 114 | 16.5 | 0 |

Moves given three seconds sat at a median fullmove of 16; moves given under a
second sat at 68. The clock was being spent on openings and the endings played
on what was left.

## What it does now

**Estimates remaining moves from material rather than the move number.** Medians
over 3,439 positions from our own rated games: 26-32 pieces leaves about 43 of
our moves, 14-19 about 33, 8-13 about 23, and 7 or fewer about 8. A grinding
ending still has twenty moves in it, which is exactly where the old estimate was
worst. `agent.py` passes `chess.popcount(board.occupied)`.

**Banks time on settled positions.** When the root move has held for four
completed depths and the score is steady past depth six, the search stops and
leaves the rest for a move that needs it. This is the half of `v9_time` that was
worth keeping; its extension half is deliberately absent, because these games
show the engine short of time rather than short of thinking.

**Spends the increment.** The old credit topped out at 342 ms of a 500 ms
increment.

## Measured on the real games

Replaying each of the six games' own piece-count trajectory through both clocks:

| round | our moves | V7 last 30% | V12 last 30% | V7 min | V12 min |
| --- | --- | --- | --- | --- | --- |
| 77 | 66 | 0.91s | 1.56s | 0.65s | 0.90s |
| 78 | 43 | 1.95s | 1.83s | 1.63s | 1.59s |
| 79 | 60 | 1.15s | 1.67s | 0.78s | 1.36s |
| 80 | 85 | 0.58s | 1.01s | 0.52s | 0.81s |
| 81 | 64 | 1.05s | 1.43s | 0.70s | 1.14s |
| 82 | 87 | 0.61s | 1.00s | 0.52s | 0.67s |

Round 78 is the cost: it ended abruptly at move 43 and this build would finish
it with 38.5s unused. No estimate can foresee that, and by the table above
under-spending costs about 2 cp a move where starving costs about 17.

Unmeasured in games. The trajectories above are arithmetic on a fixed budget
function, not evidence of Elo.
