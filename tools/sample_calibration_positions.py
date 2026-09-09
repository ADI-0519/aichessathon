"""Collect positions across piece counts from our own rated games."""
import chess, chess.pgn, pathlib, json, random
rng = random.Random(20260910)
buckets = {"32-26": [], "25-20": [], "19-14": [], "13-8": [], "7-3": []}
def bucket(n):
    if n >= 26: return "32-26"
    if n >= 20: return "25-20"
    if n >= 14: return "19-14"
    if n >= 8:  return "13-8"
    return "7-3"
for d in ("games/rated-77-82", "games/rated"):
    for path in sorted(pathlib.Path(d).glob("*.pgn")):
        g = chess.pgn.read_game(path.open(encoding="utf-8"))
        if g is None: continue
        b = g.board()
        for mv in g.mainline_moves():
            b.push(mv)
            if b.is_game_over(): break
            n = chess.popcount(b.occupied)
            k = bucket(n)
            if len(buckets[k]) < 400 and not b.is_check():
                buckets[k].append(b.fen())
out = []
for k, v in buckets.items():
    rng.shuffle(v)
    out.extend((k, f) for f in v[:120])
pathlib.Path("benchmarks/suites/eval_calibration.json").write_text(json.dumps(out, indent=1))
print({k: len(v[:120]) for k, v in buckets.items()}, "total", len(out))
