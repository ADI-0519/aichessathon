# What four builds score against one strong external engine

Every run below is 48 paired games at the real 120s + 0.5s control, from the
same 24 positions, against the same public engine used purely as a measuring
stick. No code from it is in this repository.

| build | what it adds | W-D-L | draws | score |
| --- | --- | --- | --- | --- |
| `v12_clock` | seven selectivity techniques plus a phase-aware clock | 8-5-35 | 10% | 21.9% |
| `submission_v11.zip` | the uploaded build | 6-7-27 | 18% | 23.8% |
| `exp_release_v12` | signed history and quiet-SEE pruning | 9-6-31 | 13% | 26.1% |
| `v13_contempt` | material-conditional contempt | 8-7-33 | 15% | 24.0% |

Every interval overlaps every other. The spread from best to worst is 4.2
points where the standard error on each is about 5.

## What that means

Four builds carrying four different ideas -- selectivity, clock, history and
ordering, draw preference -- land in the same band, roughly 180 to 220 Elo
behind. None of the search-layer work separates from any other.

The one variable none of them changed is the network. All four evaluate
positions with the same net trained on about four million positions. The engine
they are all losing to trains on 581 million.

## Contempt specifically

Draws had always scored exactly zero, so the engine repeated whenever the
alternative evaluated at or below nothing. V13 made a draw worth +40 when more
than a minor piece behind and -40 when ahead, conditioned on the position
because the contract gives no way to know the opponent.

It changed nothing measurable. The likely reason is that against an engine this
much stronger, positions offering a repetition usually evaluate far below -40,
so the draw already won on merit and the bias never cast a deciding vote. It
only bites between -40 and 0. A larger value, 100 to 150, would actually change
decisions, but it would also decline play in positions where we are only
slightly worse, and this result gives no reason to expect that trade to pay.
