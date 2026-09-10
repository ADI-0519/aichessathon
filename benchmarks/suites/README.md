# Benchmark suites

These positions are development inputs for paired engine tests. The submission packager does not
include this directory.

## Pinned opening suite

`openings_8moves_v3_500.epd` is a deterministic 500-position sample of the 34,700-line
`8moves_v3.pgn` corpus maintained by the official Stockfish project. Each source line reaches the
position after eight moves by both sides, which resembles the competition's curated opening starts
more closely than repeated games from the initial position.

Provenance and integrity:

- source: <https://github.com/official-stockfish/books/blob/master/8moves_v3.pgn.zip>
- source archive SHA-256: `7e1e9dd118b4bb97d8a8b5b8a790c86e21f8509d59a27d2883767d94477be02e`
- extracted PGN SHA-256: `5835239f88cc2c7511b177c32392a69f3ede21819cf0616f80a7f907cd21d17e`
- generated EPD SHA-256: `e637babf9a0c83ab14ccbc39e4d493e69bd4482e0c68d367f9b480c2cb4e4e32`
- license: CC0 1.0; the upstream text is retained in `STOCKFISH_BOOKS_LICENSE.txt`

The adjacent manifest records the selection algorithm, seeds, hashes, suite digest, and split
counts. The fixed split contains 302 development, 105 validation, and 93 locked holdout positions.

## Rebuilding the sample

Download and extract the source into the ignored `benchmarks/suites/sources/` directory, verify the
archive hash above, then run:

```bash
./.venv/Scripts/python.exe -m tools.build_suite \
  --source benchmarks/suites/sources/8moves_v3.pgn \
  --output benchmarks/suites/openings_8moves_v3_500.epd \
  --count 500 \
  --source-url https://github.com/official-stockfish/books/blob/master/8moves_v3.pgn.zip \
  --force
```

The hash-ranked selection is independent of source ordering. The EPD retains halfmove and fullmove
counters, and `tools.backtest` recomputes the stable development/validation/holdout assignment from
each normalized FEN.
