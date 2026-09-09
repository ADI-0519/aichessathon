# Where the evaluation is wrong, 10 September 2026

600 positions drawn from our own rated games, 120 in each material bucket, our
static evaluation against Stockfish 19 at 200k nodes. `tools/eval_calibration.py`.

| material | slope | MAE vs Stockfish | our mean abs | Stockfish mean abs |
| --- | --- | --- | --- | --- |
| 32-26 pieces | 0.63 | 93.8 cp | 106 | 141 |
| 25-20 | 0.50 | 166.2 | 168 | 266 |
| 19-14 | 0.63 | 213.1 | 278 | 369 |
| 13-8 | 0.83 | 235.6 | 388 | 479 |
| 7-3 | 0.75 | 345.6 | 1292 | 1366 |
| all | 0.74 | 210.9 | | |

Slope is `sum(ours * ref) / sum(ref^2)`: 1.00 is correctly scaled, above 1 shouts,
below 1 whispers.

## The evaluation whispers, by about a quarter

Every fixed-centipawn pruning margin compares a static score against a constant:
reverse futility clears beta by `85 * depth`, razoring gives up at `300 * depth`
below alpha, futility at `110 * depth`, and the null-move guard tests the static
against beta. A slope of 0.74 means all four fire **less** often than their
constants intend, so the build is slower and safer than designed rather than
pruning real moves.

That matters for any future rescale. The margins were chosen against a
compressed scale, so multiplying the output head to reach slope 1.0 would tighten
all of them at once and make the search markedly more aggressive overnight. A
rescale and a margin retune are one change, not two.

## The endgame is where the evaluation fails

Error grows 3.7x from opening to bare ending, 93.8 cp to 345.6 cp, while the
positions themselves get sharper. Rounds 80 and 82, the two longest games in the
77-82 set, were both endgame losses, and the audit of those games put our worst
moves at fullmove 68 and beyond.

Search depth does not fix an evaluation that is wrong by 300 cp. The levers are a
better-trained net, endgame-dense training data, tablebases for the smallest
buckets, or blending toward material when material is thin.
