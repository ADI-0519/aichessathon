from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import chess
import numpy as np
from numba import njit
from numpy.typing import NDArray

from challengers.numba_v1 import engine, search

# perft counts leaves, the search counts nodes (differ by the branching factor)
POSITIONS = (
    ("start", chess.STARTING_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("round4", "r2q1rk1/1Qp1bppp/2np4/p7/2BPn3/5N2/PP3PPP/R1B2RK1 b - - 0 12"),
    ("endgame", "8/5pk1/6p1/3p4/3P4/5KP1/5P2/8 w - - 0 1"),
)


@njit(cache=False)
def _bench_generate(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: NDArray[np.uint64],
    legal: NDArray[np.int32],
    pseudo: NDArray[np.int32],
    undo: NDArray[np.int64],
    undo_key: NDArray[np.uint64],
    iterations: int,
) -> int:
    total = 0
    for _ in range(iterations):
        total += engine.generate_legal_moves(pieces, state, key, legal, pseudo, undo, undo_key)
    return total


@njit(cache=False)
def _bench_pseudo(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    pseudo: NDArray[np.int32],
    iterations: int,
) -> int:
    total = 0
    for _ in range(iterations):
        total += engine.generate_pseudo_legal_moves(pieces, state, pseudo)
    return total


@njit(cache=False)
def _bench_in_check(pieces: NDArray[np.uint64], side: int, iterations: int) -> int:
    total = 0
    for _ in range(iterations):
        if engine.is_in_check(pieces, side):
            total += 1
    return total


@njit(cache=False)
def _bench_make_unmake(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: NDArray[np.uint64],
    moves: NDArray[np.int32],
    count: int,
    undo: NDArray[np.int64],
    undo_key: NDArray[np.uint64],
    iterations: int,
) -> int:
    total = 0
    for _ in range(iterations):
        for index in range(count):
            move = int(moves[index])
            if engine.make_move(pieces, state, key, move, undo, undo_key):
                total += 1
                engine.unmake_move(pieces, state, key, move, undo, undo_key)
    return total


@njit(cache=False)
def _bench_evaluate(
    pieces: NDArray[np.uint64], state: NDArray[np.int64], iterations: int
) -> int:
    total = 0
    for _ in range(iterations):
        total += search.evaluate(pieces, state)
    return total


@njit(cache=False)
def _bench_order(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    side: int,
    moves: NDArray[np.int32],
    count: int,
    killers: NDArray[np.int32],
    quiet_history: NDArray[np.int32],
    scores: NDArray[np.int32],
    see_gains: NDArray[np.int32],
    scratch: NDArray[np.int32],
    iterations: int,
) -> int:
    total = 0
    for _ in range(iterations):
        # ordering sorts in place, each repetition needs the original order back
        for index in range(count):
            scratch[index] = moves[index]
        search._order_moves(
            pieces, state, side, scratch, count, 0, 0, killers, quiet_history, scores, see_gains
        )
        total += int(scratch[0])
    return total


@njit(cache=False)
def _bench_see(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    moves: NDArray[np.int32],
    count: int,
    see_gains: NDArray[np.int32],
    iterations: int,
) -> int:
    total = 0
    for _ in range(iterations):
        for index in range(count):
            total += search.static_exchange_eval(pieces, state, int(moves[index]), see_gains)
    return total


@njit(cache=False)
def _bench_rule_draw(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: NDArray[np.uint64],
    history: NDArray[np.uint64],
    history_count: int,
    iterations: int,
) -> int:
    total = 0
    for _ in range(iterations):
        if engine.has_rule_draw(pieces, state, key[0], history, history_count, history_count):
            total += 1
    return total


@njit(cache=False)
def _bare_negamax(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: NDArray[np.uint64],
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    legal_stack: NDArray[np.int32],
    pseudo_stack: NDArray[np.int32],
    undo_stack: NDArray[np.int64],
    undo_key_stack: NDArray[np.uint64],
    nodes: NDArray[np.int64],
) -> int:
    # the floor a node cannot go below
    nodes[0] += 1
    if depth <= 0:
        return search.evaluate(pieces, state)
    count = engine.generate_legal_moves(
        pieces,
        state,
        key,
        legal_stack[ply],
        pseudo_stack[ply],
        undo_stack[ply],
        undo_key_stack[ply],
    )
    if count == 0:
        return -search.MATE_SCORE + ply if engine.is_in_check(pieces, int(state[0])) else 0
    best = -search.INFINITY
    for index in range(count):
        move = int(legal_stack[ply, index])
        engine.make_move(pieces, state, key, move, undo_stack[ply], undo_key_stack[ply])
        score = -_bare_negamax(
            pieces,
            state,
            key,
            depth - 1,
            -beta,
            -alpha,
            ply + 1,
            legal_stack,
            pseudo_stack,
            undo_stack,
            undo_key_stack,
            nodes,
        )
        engine.unmake_move(pieces, state, key, move, undo_stack[ply], undo_key_stack[ply])
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


def _repetitions(seconds: float, per_call_guess_s: float) -> int:
    return max(64, int(seconds / per_call_guess_s))


def _timed(function: Callable[..., int], arguments: tuple[object, ...], reps: int) -> float:
    function(*arguments, 1)
    started = time.perf_counter()
    function(*arguments, reps)
    return time.perf_counter() - started


def _profile(name: str, fen: str, seconds: float) -> None:
    board = chess.Board(fen)
    position = engine.position_from_board(board)
    pieces, state, key = position.pieces, position.state, position.key
    side = int(state[engine.STATE_SIDE])

    legal = np.empty((engine.MAX_MOVES,), dtype=np.int32)
    pseudo = np.empty((engine.MAX_MOVES,), dtype=np.int32)
    scratch = np.empty((engine.MAX_MOVES,), dtype=np.int32)
    scores = np.empty((engine.MAX_MOVES,), dtype=np.int32)
    undo = np.empty((engine.UNDO_SIZE,), dtype=np.int64)
    undo_key = np.empty((1,), dtype=np.uint64)
    see_gains = np.empty((search.SEE_MAX_EXCHANGES,), dtype=np.int32)
    killers = np.zeros((search.MAX_PLY, 2), dtype=np.int32)
    quiet_history = np.zeros((2, 64, 64), dtype=np.int32)
    history = np.zeros(search.MAX_HISTORY, dtype=np.uint64)
    history[0] = key[0]

    count = engine.generate_legal_moves(pieces, state, key, legal, pseudo, undo, undo_key)
    pseudo_count = engine.generate_pseudo_legal_moves(pieces, state, pseudo)
    captures = np.array(
        [
            legal[index]
            for index in range(count)
            if engine.move_flags(int(legal[index]))
            & (engine.FLAG_CAPTURE | engine.FLAG_PROMOTION)
        ],
        dtype=np.int32,
    )
    capture_count = len(captures)
    if capture_count == 0:
        captures = np.zeros(1, dtype=np.int32)

    print(f"\n=== {name}: {fen}")
    print(f"{count} legal from {pseudo_count} pseudo, {capture_count} captures/promotions")

    def micros(
        function: Callable[..., int],
        arguments: tuple[object, ...],
        guess: float,
        per: int = 1,
    ) -> float:
        reps = _repetitions(seconds, guess)
        return _timed(function, arguments, reps) / (reps * per) * 1e6

    generate_us = micros(_bench_generate, (pieces, state, key, legal, pseudo, undo, undo_key), 6e-6)
    pseudo_us = micros(_bench_pseudo, (pieces, state, pseudo), 1e-6)
    in_check_us = micros(_bench_in_check, (pieces, side), 2e-7)
    make_unmake_us = micros(
        _bench_make_unmake, (pieces, state, key, legal, count, undo, undo_key), 6e-6, count
    )
    evaluate_us = micros(_bench_evaluate, (pieces, state), 1e-6)
    order_us = micros(
        _bench_order,
        (pieces, state, side, legal, count, killers, quiet_history, scores, see_gains, scratch),
        2e-5,
    )
    see_us = micros(
        _bench_see, (pieces, state, captures, len(captures), see_gains), 4e-6, len(captures)
    )
    rule_draw_us = micros(_bench_rule_draw, (pieces, state, key, history, 1), 5e-7)

    print(f"  generate_legal_moves        {generate_us:8.3f} us")
    print(f"    of which pseudo-legal     {pseudo_us:8.3f} us")
    print(f"    of which make/check/unmake{generate_us - pseudo_us:8.3f} us")
    print(f"  is_in_check                 {in_check_us:8.3f} us")
    print(f"  make + unmake (per move)    {make_unmake_us:8.3f} us")
    print(f"  evaluate                    {evaluate_us:8.3f} us")
    print(f"  order_moves (whole list)    {order_us:8.3f} us")
    print(f"    static_exchange_eval each {see_us:8.3f} us x {capture_count}")
    print(f"  has_rule_draw               {rule_draw_us:8.3f} us")
    print(
        f"  accounted per interior node "
        f"{generate_us + order_us + rule_draw_us + make_unmake_us:8.3f} us"
    )


def _floor(depth: int) -> None:
    position = engine.position_from_board(chess.Board())
    legal_stack = np.empty((search.MAX_PLY, engine.MAX_MOVES), dtype=np.int32)
    pseudo_stack = np.empty((search.MAX_PLY, engine.MAX_MOVES), dtype=np.int32)
    undo_stack = np.empty((search.MAX_PLY, engine.UNDO_SIZE), dtype=np.int64)
    undo_key_stack = np.empty((search.MAX_PLY, 1), dtype=np.uint64)
    nodes = np.zeros(1, dtype=np.int64)

    started = time.perf_counter()
    _bare_negamax(
        position.pieces,
        position.state,
        position.key,
        depth,
        -search.INFINITY,
        search.INFINITY,
        0,
        legal_stack,
        pseudo_stack,
        undo_stack,
        undo_key_stack,
        nodes,
    )
    elapsed = time.perf_counter() - started
    visited = int(nodes[0])
    print(
        f"\nbare alpha-beta floor, depth {depth}: {visited:,} nodes in {elapsed:.3f}s, "
        f"{visited / elapsed:,.0f} nodes/s, {elapsed / visited * 1e6:.3f} us/node"
    )


def _real_search(node_limit: int) -> None:
    result = search.search_position(
        engine.position_from_board(chess.Board()),
        search.SearchMemory.create(),
        node_limit=node_limit,
    )
    per_node = result.elapsed_s / result.nodes * 1e6 if result.nodes else 0.0
    nps = result.nodes / result.elapsed_s if result.elapsed_s else 0.0
    print(
        f"real search, {node_limit:,} node budget: depth {result.depth}, "
        f"{result.nodes:,} nodes ({result.qnodes / max(1, result.nodes):.0%} q), "
        f"{nps:,.0f} nodes/s, {per_node:.3f} us/node"
    )


def _perft(depth: int) -> None:
    position = engine.position_from_board(chess.Board())
    started = time.perf_counter()
    leaves = engine.perft(position, depth)
    elapsed = time.perf_counter() - started
    # generation runs at every node above leaves, which is the fair unit
    generations = sum((1, 20, 400, 8_902, 197_281, 4_865_609)[:depth])
    print(
        f"perft {depth}: {leaves:,} leaves in {elapsed:.3f}s, {leaves / elapsed:,.0f} leaves/s; "
        f"{generations:,} generate calls, {elapsed / generations * 1e6:.3f} us per generate"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="account for the compiled challenger's per-node search cost"
    )
    parser.add_argument("--seconds", type=float, default=0.4, help="per micro-benchmark")
    parser.add_argument("--floor-depth", type=int, default=5)
    parser.add_argument("--perft-depth", type=int, default=5, choices=range(1, 7))
    parser.add_argument("--nodes", type=int, default=200_000)
    arguments = parser.parse_args()

    started = time.perf_counter()
    search.warmup()
    print(f"warmup {time.perf_counter() - started:.1f}s")

    _perft(arguments.perft_depth)
    _floor(arguments.floor_depth)
    _real_search(arguments.nodes)
    for name, fen in POSITIONS:
        _profile(name, fen, arguments.seconds)


if __name__ == "__main__":
    main()
