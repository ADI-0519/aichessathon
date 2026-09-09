from __future__ import annotations

import os
import time

# NumPy and Numba inspect these variables when they are first imported.  The
# match container supplies one core, so larger native pools only waste memory
# and can exhaust thread resources during repeated local arena startups.
for _variable in (
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_variable] = "1"

import chess  # noqa: E402 - thread limits must precede native-library imports
import engine  # noqa: E402
import numpy as np  # noqa: E402
import search  # noqa: E402
from time_manager import hard_budget_ms as _hard_budget_ms  # noqa: E402
from time_manager import move_budget_ms as _move_budget_ms  # noqa: E402

_memory = search.SearchMemory.create()
_game_board: chess.Board | None = None
_position_history: list[np.uint64] = []


def _canonical_fen(board: chess.Board) -> str:
    return board.fen(en_passant="legal")


def _key_for(board: chess.Board) -> np.uint64:
    return np.uint64(engine.position_from_board(board).key[0])


def _reset_game(board: chess.Board) -> chess.Board:
    global _game_board
    _game_board = board
    _position_history.clear()
    _position_history.append(_key_for(board))
    _memory.clear()
    return board


def _sync_board(fen: str) -> chess.Board:
    """Advance persistent state through the opponent move or reset safely."""
    global _game_board
    incoming = chess.Board(fen)
    incoming_fen = _canonical_fen(incoming)
    if _game_board is None:
        return _reset_game(incoming)
    if _canonical_fen(_game_board) == incoming_fen:
        return _game_board

    for move in list(_game_board.legal_moves):
        _game_board.push(move)
        if _canonical_fen(_game_board) == incoming_fen:
            _position_history.append(_key_for(_game_board))
            return _game_board
        _game_board.pop()
    return _reset_game(incoming)


def _choose_move(fen: str, time_left_ms: int) -> str:
    global _game_board
    if time_left_ms <= 100:
        board = chess.Board(fen)
        try:
            move = next(iter(board.legal_moves))
        except StopIteration:
            return "0000"
        return move.uci()

    board = _sync_board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "0000"
    fallback = legal_moves[0]
    if len(legal_moves) == 1:
        chosen = fallback
    else:
        budget_ms = _move_budget_ms(time_left_ms, board.fullmove_number)
        hard_ms = _hard_budget_ms(time_left_ms, board.fullmove_number)
        chosen = fallback
        if budget_ms > 0:
            position = engine.position_from_board(board)
            prior_history = np.asarray(_position_history, dtype=np.uint64)
            result = search.search_position(
                position,
                _memory,
                time_limit_s=budget_ms / 1_000.0,
                hard_limit_s=hard_ms / 1_000.0,
                clock_left_ms=int(time_left_ms),
                prior_history=prior_history,
            )
            candidate = chess.Move.from_uci(engine.move_to_uci(result.move))
            if candidate in board.legal_moves:
                chosen = candidate

    if chosen not in board.legal_moves:
        chosen = fallback
    board.push(chosen)
    _position_history.append(_key_for(board))
    _game_board = board
    return chosen.uci()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return legal UCI, retaining a fresh-board fallback for every failure."""
    try:
        return _choose_move(fen, time_left_ms)
    except Exception as error:
        print(f"compiled challenger failed, using fallback: {type(error).__name__}: {error}")
        board = chess.Board(fen)
        try:
            move = next(iter(board.legal_moves))
        except StopIteration:
            return "0000"
        return move.uci()


_warmup_started = time.perf_counter()
search.warmup()
_warmup_elapsed_s = time.perf_counter() - _warmup_started
