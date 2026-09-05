# Where the strength comes from

A model is optional. Material plus piece-square tables is a legal entry, and the winning
shape is usually a search that calls a small evaluation, learned or not. This is what tends to
matter, roughly in order.

## Search

Negamax with alpha-beta is the whole game. The gap between the `minimax` baseline and something
respectable is mostly move ordering, because alpha-beta only pays off when good moves come first.

- Order captures before quiet moves, and order captures by MVV-LVA.
- Keep a transposition table. Even a plain dict keyed on `board._transposition_key()` and cleared
  between moves is worth a ply.
- Iterative deepening: search depth 1, then 2, then 3, keeping the best move from each pass. It
  gives you ordering for free and, more importantly, it gives you something to return when time
  runs out.
- Quiescence search at the leaves, captures only. Without it your evaluation is measured in
  positions that are mid-exchange and it will be wrong.

You are on one core in Python, so node counts are small: expect thousands, not millions. That
changes the trade. Depth is expensive, so evaluation quality and ordering buy more than they
would in a C engine. numba closes most of that gap: the platform preinstalls it, and a jitted
movegen and evaluation reach node counts pure Python cannot. Warm every jitted function once at
import so compilation happens inside the init budget, and warm it with the argument types the
real calls use, since numba compiles per signature. `baselines/numba` shows the pattern. Note
that it scores barely better than `baselines/minimax`, because jitting a two-ply search wins
nothing on its own. The gain is the depth the speed lets you afford.

## Evaluation

Material plus piece-square tables is a real evaluation and it beats both baselines. It is also
the thing to build first, because it gives you a reference to measure a model against.

The base image ships torch and onnxruntime, so a small network is practical. Export to ONNX and
run it with onnxruntime: startup is faster than torch and inference on one core is competitive.
Keep it small. A net you can evaluate thousands of times per move is worth more than a better net
you can evaluate fifty times.

Batching helps: collect the leaf positions of a search pass and evaluate them in one call rather
than one at a time.

Size a learned evaluation like the CPU engine nets. An NNUE-style net quantised to int8 or int16
lands between 1 and 40 MB and is fast enough to search with. A deep convolutional net at fp32
manages a few hundred evaluations a second on one core, which is a policy model's budget, not a
search evaluation's.

## Training data

You have no network at runtime, so everything ships in the zip inside the 50 MB cap. Data
gathering happens on your machine, before you upload. Public game databases and self-play against
your own earlier versions are both reasonable starting points, and labelling positions with an
existing engine is explicitly allowed: the ban covers what ships inside the zip, not what you
learn from. Whatever you train on, the model has to be one you trained.

## Time management

120 seconds plus 0.5 per move. A flag is a loss unless the other side cannot mate, and it is
the most common self-inflicted one.

- Budget per move from the clock you were handed, not from a constant. Something like
  `time_left_ms / max(20, expected_moves_left)` is enough to start.
- Check the clock inside your search, not only between moves, and return the best move you have
  when the budget is gone. Iterative deepening makes that easy.
- Leave a margin. The referee measures wall time, and the watchdog does not forgive.

## Things the position alone does not tell you

The process stays alive between your moves, so you can keep state. Two things are worth keeping:

- The positions you have been asked about. The referee claims threefold repetition automatically,
  so if you are winning and shuffling, you can draw a won game without ever being told.
- Your own search results. A transposition table that survives across moves is a real gain.

An opening book is worth less here than it looks. Rated games start from curated positions
rather than the standard start, so a book keyed on move one is often already out of book. Test
with `make play FEN=...` from positions you have not prepared, and spend the effort on the search
instead.

## Measuring a change

Two games tell you nothing. Alternate colours, fix the opponent, and play enough games that the
score means something: a change worth 3% needs hundreds of games to see, and `make arena` at a
fast time control is how you get them. Keep the previous version around as an opponent, because
"better than my last one" is the only comparison that matters.

## What loses games for free

- Flagging. See above.
- Crashing on an edge case: no legal moves, a promotion, an en passant capture. Play a few hundred
  games against a random baseline and the rare paths show up.
- Blowing the 60 second import budget loading weights.
- Writing anywhere but `/tmp`. Everything else is read-only.
- More threads than cores. `torch.set_num_threads(1)`.
