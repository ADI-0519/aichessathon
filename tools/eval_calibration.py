"""Compare our static evaluation against Stockfish, bucketed by material.

Two questions, both of which decide real things:
  slope  -- does our evaluation shout or whisper? Every fixed-centipawn pruning
            margin (razoring, futility, reverse futility, the null-move margin)
            compares a static score against a constant, so a mis-scaled eval
            mistunes all of them at once.
  error  -- where is the evaluation actually wrong? If the endgame is far worse
            than the middlegame, no amount of search depth fixes it.
"""
import json, pathlib, sys, chess, chess.engine
sys.path.insert(0, sys.argv[1])
import engine, nnue, search
search.warmup()

cases = json.loads(pathlib.Path("benchmarks/suites/eval_calibration.json").read_text())
sf = chess.engine.SimpleEngine.popen_uci(pathlib.Path(".stockfish_path").read_text().strip())
rows = []
for bucket, fen in cases:
    board = chess.Board(fen)
    pos = engine.position_from_board(board)
    acc = nnue.new_accumulator() if hasattr(nnue, "new_accumulator") else None
    import numpy as np
    acc = np.empty((2, nnue.ACCUMULATOR_ROW if hasattr(nnue, "ACCUMULATOR_ROW") else nnue.ACCUMULATOR_SIZE), dtype=np.int32)
    nnue.rebuild(pos.pieces, acc)
    ours = int(search.evaluate(pos.pieces, pos.state, acc))
    info = sf.analyse(board, chess.engine.Limit(nodes=200_000))
    ref = info["score"].pov(board.turn).score(mate_score=2000)
    rows.append({"bucket": bucket, "ours": ours, "ref": ref})
sf.quit()
pathlib.Path(sys.argv[2]).write_text(json.dumps(rows))
print(f"scored {len(rows)} positions")
