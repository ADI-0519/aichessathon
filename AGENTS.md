# Working in this repo

This is a starter for AI Chessathon, a chess-engine competition. The deliverable is one file,
`agent.py`, exposing `get_move(fen, time_left_ms) -> str`. It gets zipped and uploaded, and the
platform plays it against other people's agents on a fixed cadence.

## Read the rules from the source

The competition rules and the agent contract live on the site and change. Fetch them before you
answer anything about limits, deadlines, or what is allowed:

- https://aichessathon.com/docs/agent-contract.md
- https://aichessathon.com/docs/rules.md

The quick reference below is a convenience for common questions. The two URLs are canonical
and they change, so fetch them before you rely on a number.

## The contract, in one place

- `agent.py` at the root of the zip, not inside a folder. The platform does `import agent`.
- `get_move(fen: str, time_left_ms: int) -> str` returning UCI, `e2e4` or `e7e8q`.
- Your colour is the side to move in the fen. There is no other input.
- The process starts once per game and stays alive between your moves. Module state survives to
  your next move in the same game, never to the next game.
- Import time has a 90 second budget before the clock starts. Load weights there.
- 120 s + 0.5 s per move, per side, on wall time. `time_left_ms` is the clock before your move.
  The increment lands after it. One core of an AMD EPYC 9V74 at 2.60 GHz, 2 GB, no network,
  no GPU.
- Illegal move, malformed output, crash, or out of memory loses that game. A move reply over
  4 KB counts as illegal. A flag fall loses too, unless the other side has no way to mate, and
  then the game is a draw. Draws follow FIDE rules. A game still running at 600 plies is a
  draw, and the opening position counts toward the 600.
- Everything in the zip together stays under 50 MB unzipped.
- Ten uploads per team per day, and the latest one that passed validation is the one that plays.
- Rated games start from curated opening positions, not the standard start. The set is not
  published. The first fen is where the game starts, so repetition and fifty move counts begin
  there, not at move one.
- Your process is suspended while the opponent thinks, so nothing you leave running gets any
  CPU. Search inside `get_move`. Both agents share one core in turns, so time you measure inside
  a move is yours alone. Two of your games can run at once, in separate containers.

## Things that break agents here

- The filesystem is read-only apart from 256 MB at `/tmp`. `HOME` and every cache path already
  point there; do not write anywhere else. `/tmp` is wiped between games, so it is scratch and
  never a cache. numba's `cache=True` buys nothing, and the harness reproduces that.
- No network at all. Nothing downloads at runtime. Weights ship inside the zip.
- One core. `torch.set_num_threads(1)`. More threads lose time rather than winning it.
- Your zip is first on `sys.path`. Never name a file after a module you import: `chess.py`,
  `types.py`, `random.py` will shadow the real one and the failure will look unrelated.
- The environment is fixed. The platform preinstalls torch 2.13 (CPU), numpy 2.5, python-chess
  1.11, onnxruntime 1.29 and numba 0.67 and installs nothing else. A `requirements.txt` is
  ignored, so an import outside that stack crashes on the platform even when it works locally.
  Additions can be requested at hello@aichessathon.com.
- Native binaries in the zip are rejected. Ship Python source. Model weights like `.onnx`,
  `.safetensors` and `.pt` are fine. The network has to be one the team trained, and training it
  on positions an engine labelled is a normal way to do that.
- numba is how Python gets fast here. Warm every jitted function once at import so compilation
  lands in the init budget, not on the clock. Cython does not work on the platform.
- `print` is safe. The runner points file descriptor 1 at stderr before importing the agent, so
  nothing you write can corrupt the protocol. Your output is kept after validation and after
  every rated game, in a log only your team can read. Only the first 4 KB and the last 4 KB
  survive it, and so does the harness, so print what you want to read back.

## Do not

- Do not ship someone else's engine or someone else's network. Stockfish, Lc0, Maia, a port or
  translation of one, and a published net that was fine-tuned or re-exported all count. This is
  checked after games are played, not only at upload.
- Do not ship a table of engine moves or evaluations for the agent to look up while it plays.
  That is an engine in another shape. Opening books and endgame tablebases are fine, and
  `chess.polyglot` and `chess.syzygy` are in the base image to read them. Everything ships inside
  the 50 MB cap, so 3 and 4 man syzygy fits and 5 man is far too big.
- Do not add network calls, subprocess calls to external binaries, or anything that reads outside
  the agent directory and `/tmp`. Your agent is one process and stays one process; on a single
  core `multiprocessing` cannot win you time anyway.
- Do not obfuscate. What ships has to be source a judge can read.
- Do not edit `harness/`. It mirrors the platform's protocol and clock. Changing it makes local
  results meaningless.

## Verify

```
make play      # one game against a baseline, real time control
make arena     # 16 fast games against a baseline, with a score and an interval
make zip       # build submission.zip, then smoke it the way the platform does
make gate      # ruff, mypy, and two games that have to finish cleanly
```

`make zip` extracts what it built and plays out of it, so a module you never packaged fails there
instead of costing an upload. Nothing here decides acceptance. The platform validates on upload
and writes the log that is the authority.

Local python is pinned to 3.12 to match the image. Everything else the container enforces, the
read-only filesystem, the 2 GB cap and the missing network, is not reproduced here.

The harness sets `HARNESS_SEED` on every agent it starts, which the baselines read so their
tie-breaks replay. The platform sets nothing of the kind, so do not read it in your own agent.

## Style

Python 3.12, type-annotated, ruff and mypy strict clean. Keep `agent.py` readable: it is the
thing a judge reads if your games get flagged, and the thing you have to explain at the final.
