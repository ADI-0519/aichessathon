# Rounds 77 to 82, 9 September 2026

Six rated games, 2 wins, 3 losses, 1 draw. Refereed by Stockfish 19 at 1M nodes
per position.

## Move quality has improved a great deal

| set | games | mean ACPL |
| --- | --- | --- |
| rounds 27 to 67 (V5 and V7 era) | 15 | 75.7 |
| rounds 77 to 82 | 6 | **24.4** |

Wins average 12.9, the draw 25.2, losses 31.7. Round 77 was played at 10.5.
Whatever is running now is roughly three times more accurate per move than the
engine that played the earlier archive, and the gap between our wins and our
losses has narrowed from 34.5 against 85.1 to 12.9 against 31.7.

## The clock, not the evaluation, is now the problem

Every one of the six games ended with us poorer on time than the opponent, in
five of them by a factor of two and a half to four.

| round | result | plies | our final clock | theirs |
| --- | --- | --- | --- | --- |
| 77 | win | 132 | 7.6s | 11.5s |
| 78 | loss | 86 | 22.7s | 52.1s |
| 79 | win | 120 | 9.3s | 24.9s |
| 80 | loss | 171 | 8.6s | 25.3s |
| 81 | draw | 128 | 6.5s | 24.9s |
| 82 | loss | 174 | 6.4s | 19.3s |

The two longest games were both losses, played out at around seven seconds.

Sorting all 342 scored moves by how long they took:

| think time | moves | mean cp loss | losses of 200 cp or more |
| --- | --- | --- | --- |
| under 1.0s | 49 | 35.4 | 2 |
| 1.0 to 2.0s | 83 | 18.9 | 1 |
| 2.0 to 3.0s | 96 | 27.2 | 3 |
| 3.0s and over | 114 | 16.5 | 0 |

Moves given three seconds or more sit at a median fullmove of 16. Moves given
under a second sit between fullmove 50 and 83, median 68. We spend the clock on
the opening and play the ending on what is left.

## Why

`time_manager.estimated_moves_remaining` floors at 12, so from move 50 onward it
assumes twelve moves remain however long the game actually runs. The budget is
the usable clock divided by that estimate, so it *rises* as a game goes long.
For round 82, which reached move 87:

| at move | assumed remaining | actually remaining | budget |
| --- | --- | --- | --- |
| 20 | 24 | 67 | 2739 ms |
| 40 | 16 | 47 | 3948 ms |
| 50 | 12 | 37 | 4500 ms |
| 70 | 12 | 17 | 4500 ms |

From move 40 the engine spends its maximum on every move with more than forty
still to play.

## This reverses an earlier conclusion

Rounds 66 and 67 were lost while we still held 51.5s and 20.1s, which argued we
were not thinking long enough at critical moments, and `challengers/v9_time` was
built to extend on an unstable search. These six games show the opposite failure
and V9 would deepen it, since every mechanism in it spends more. The floor in
the moves-remaining estimate is the thing to fix, and it is a smaller change.

The difference between the two sets of games is length: rounds 66 and 67 ended
by move 37 and 54, and these ran to 87.
