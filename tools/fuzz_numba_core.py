"""Long deterministic differential campaign for the Numba board core."""

from __future__ import annotations

import argparse
import random
import time

import chess
import numpy as np

from challengers.numba_v1 import engine


def fail_move_set(board: chess.Board, actual: set[str], expected: set[str]) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise AssertionError(
        f"legal move mismatch\nfen={board.fen(en_passant='fen')}\n"
        f"missing={missing}\nextra={extra}"
    )


def check_transition(board: chess.Board, rng: random.Random) -> chess.Move:
    position = engine.position_from_board(board)
    packed_moves = engine.legal_moves(position)
    packed = int(packed_moves[rng.randrange(len(packed_moves))])
    uci = engine.move_to_uci(packed)
    move = chess.Move.from_uci(uci)

    original_pieces = position.pieces.copy()
    original_state = position.state.copy()
    original_key = position.key.copy()
    undo = np.empty(engine.UNDO_SIZE, dtype=np.int64)
    undo_key = np.empty(1, dtype=np.uint64)
    if not engine.make_move(
        position.pieces, position.state, position.key, packed, undo, undo_key
    ):
        raise AssertionError(f"make_move rejected generated move {uci} in {board.fen()}")

    expected = board.copy(stack=False)
    expected.push(move)
    rebuilt = engine.board_from_position(position)
    if rebuilt.fen(en_passant="fen") != expected.fen(en_passant="fen"):
        raise AssertionError(
            f"make mismatch for {uci}\nbefore={board.fen(en_passant='fen')}\n"
            f"expected={expected.fen(en_passant='fen')}\n"
            f"actual={rebuilt.fen(en_passant='fen')}"
        )
    recomputed_key = engine.recompute_zobrist(position.pieces, position.state)
    if position.key[0] != recomputed_key:
        raise AssertionError(
            f"incremental hash mismatch after {uci} in {board.fen(en_passant='fen')}"
        )

    engine.unmake_move(
        position.pieces, position.state, position.key, packed, undo, undo_key
    )
    if not np.array_equal(position.pieces, original_pieces):
        raise AssertionError(f"piece bitboards not restored after {uci} in {board.fen()}")
    if not np.array_equal(position.state, original_state):
        raise AssertionError(f"state not restored after {uci} in {board.fen()}")
    if not np.array_equal(position.key, original_key):
        raise AssertionError(f"hash not restored after {uci} in {board.fen()}")
    return move


def check_captures(board: chess.Board) -> None:
    packed, any_legal = engine.legal_captures(engine.position_from_board(board))
    actual = {engine.move_to_uci(int(move)) for move in packed}
    expected = {
        move.uci()
        for move in board.legal_moves
        if board.is_capture(move) or move.promotion is not None
    }
    if actual != expected:
        fail_move_set(board, actual, expected)
    if any_legal != any(board.legal_moves):
        raise AssertionError(
            f"stalemate sentinel disagrees: fen={board.fen(en_passant='fen')} "
            f"any_legal={any_legal}"
        )


def check_attacks(board: chess.Board) -> None:
    position = engine.position_from_board(board)
    for square in range(64):
        for internal_color, chess_color in (
            (engine.WHITE, chess.WHITE),
            (engine.BLACK, chess.BLACK),
        ):
            actual = engine.is_square_attacked(position.pieces, square, internal_color)
            expected = board.is_attacked_by(chess_color, square)
            if actual != expected:
                raise AssertionError(
                    f"attack mismatch: fen={board.fen()} square={chess.square_name(square)} "
                    f"color={chess.COLOR_NAMES[chess_color]} actual={actual} expected={expected}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=10_000)
    parser.add_argument("--attack-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026090404)
    args = parser.parse_args()
    if args.positions <= 0:
        parser.error("--positions must be positive")
    if args.attack_interval <= 0:
        parser.error("--attack-interval must be positive")

    compile_started = time.perf_counter()
    engine.warmup()
    compile_seconds = time.perf_counter() - compile_started

    rng = random.Random(args.seed)
    board = chess.Board()
    games = 1
    campaign_started = time.perf_counter()
    for index in range(args.positions):
        if board.is_game_over() or len(board.move_stack) >= 200:
            board.reset()
            games += 1

        expected = {move.uci() for move in board.legal_moves}
        actual = engine.legal_moves_uci(engine.position_from_board(board))
        if actual != expected:
            fail_move_set(board, actual, expected)
        check_captures(board)
        if index % args.attack_interval == 0:
            check_attacks(board)

        move = check_transition(board, rng)
        board.push(move)

    campaign_seconds = time.perf_counter() - campaign_started
    print(
        f"PASS positions={args.positions} games={games} seed={args.seed} "
        f"compile={compile_seconds:.3f}s campaign={campaign_seconds:.3f}s "
        f"positions_per_second={args.positions / campaign_seconds:.1f}"
    )


if __name__ == "__main__":
    main()
