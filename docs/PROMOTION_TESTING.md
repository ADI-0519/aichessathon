# Testing a challenger before it goes up

The rule: nothing is uploaded because it looks better. It is uploaded because it beat the
artifact currently playing, over enough games that the result means something.

## The champion

`champions/python_v2/agent.py` is the previous uploaded agent, and `champions/` holds every
frozen build worth playing against. The most recently uploaded one is the opponent
every challenger has to beat. It is not edited; it is *replaced*, and only by a challenger that
cleared the gate below.

```bash
mkdir champions/<name> && cp agent.py champions/<name>/agent.py
```

Do that only at the moment you promote, and note the date and the score it won by.

## Running the match

`tools/paired_arena.py` plays each position twice, once with each colour, so a candidate cannot
win on the luck of the draw.

```bash
uv run python -m tools.paired_arena --candidate . --opponent champions/python_v2 --base-ms 2000 --increment-ms 50 --extra-positions 20
```

- `--extra-positions N` adds N random positions to the nine fixed ones. Every position is two
  games, so `N=20` is 58 games. This is the knob for sample size.
- `--seed` changes which random positions are drawn. Re-running with a different seed is the
  honest way to check a result you like.
- `--base-ms` / `--increment-ms` set the time control. Faster buys games; the rated control is
  120000 / 500, and a change that only helps at 2 s per game is not necessarily a change that
  helps at 120 s.

The run ends with a score, a 95% interval and an Elo estimate:

```
+34 =12 -12, score 68.9% ...
Elo +138 (95% CI +51 to +236)
verdict: candidate is stronger
```

`verdict: NOT RESOLVED at this sample size` means exactly that. It is not a weak yes. It means
run more games or accept that you cannot tell.

## Why 30 games is not enough

At 30 games the 95% interval on a 50% score runs from 35% to 65% — about ±105 Elo. Almost any
change looks like it might have helped. The previous 61.7% over 30 games in
[TRIAL_RECOVERY_REPORT.md](TRIAL_RECOVERY_REPORT.md) was consistent with anything from "slightly
worse" to "much better".

Rough guide, for telling a real gain from noise:

| Games | Resolves a gain of about |
|---:|---|
| 30 | 150 Elo |
| 60 | 100 Elo |
| 200 | 55 Elo |
| 800 | 28 Elo |

## The gate

A challenger is promoted when all of these hold.

1. `make gate` passes: ruff, mypy strict, and two games that finish cleanly.
2. `uv run python -m unittest discover -s tests` passes.
3. No `crash`, `illegal`, `flag`, or `init` termination appears in any arena game. The arena
   exits non-zero if one does; that is a hard stop, not a note.
4. The paired arena verdict against the current upload is `stronger`, not `NOT RESOLVED`.
5. `uv run python -m harness.play` completes one game at the real 120 s + 0.5 s control without
   a flag.

## Diagnosing rather than guessing

- `tools/search_scaling.py` — what the agent plays at 1, 2, 4, 8, 16 seconds on one position, and
  the depth it reached. Use it on positions from games you lost.
- `tools/probe_pgn_positions.py` — replay a PGN and ask the agent what it would have played.
- `tools/analyze_pgn_stockfish.py` and `tools/stockfish_arena.py` — offline benchmarking against a
  local engine. Development only; nothing from it ships in `submission.zip`.
