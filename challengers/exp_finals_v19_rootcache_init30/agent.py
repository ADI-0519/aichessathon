from __future__ import annotations

import os
import threading
import time

_IMPORT_STARTED_S = time.perf_counter()

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
from time_manager import move_time_limits as _move_time_limits  # noqa: E402

_memory = search.SearchMemory.create()
_game_board: chess.Board | None = None
_position_history: list[np.uint64] = []
EMERGENCY_CLOCK_MS = 100
# Leave substantial scheduling and runner overhead below the 30-second finals
# deadline. Any unfinished compilation is joined, and explicitly charged, on
# the first move.
INIT_READY_TARGET_S = 20.0

_warmup_done = threading.Event()
_warmup_error: BaseException | None = None


def _warmup_worker() -> None:
    """Compile the engine once, recording failure for the protocol thread."""
    global _warmup_error
    try:
        search.warmup()
    except BaseException as error:
        _warmup_error = error
    finally:
        _warmup_done.set()


def _finish_warmup(time_left_ms: int) -> int:
    """Join deferred compilation and charge its wall time to this move."""
    started = time.perf_counter()
    _warmup_done.wait()
    waited_ms = int((time.perf_counter() - started) * 1_000.0) + 1
    if _warmup_error is not None:
        raise RuntimeError("engine warm-up failed") from _warmup_error
    return max(0, time_left_ms - waited_ms)


def _canonical_fen(board: chess.Board) -> str:
    return board.fen(en_passant="legal")


def _key_for(board: chess.Board) -> np.uint64:
    return np.uint64(engine.position_from_board(board).key[0])


def _reset_game(board: chess.Board, *, clear_memory: bool = True) -> chess.Board:
    global _game_board
    _game_board = board
    _position_history.clear()
    _position_history.append(_key_for(board))
    if clear_memory:
        _memory.clear()
    return board


def _commit_move(board: chess.Board, move: chess.Move) -> str:
    """Persist the position before and after every returned legal move."""
    global _game_board
    current_key = _key_for(board)
    if not _position_history or _position_history[-1] != current_key:
        _position_history.append(current_key)
    board.push(move)
    _position_history.append(_key_for(board))
    _game_board = board
    return move.uci()


def _fallback_move(fen: str) -> str:
    """Return a legal move and retain continuity whenever bookkeeping permits."""
    board = chess.Board(fen)
    try:
        move = next(iter(board.legal_moves))
    except StopIteration:
        return "0000"
    try:
        return _commit_move(board, move)
    except Exception as error:
        print(f"fallback state update failed: {type(error).__name__}: {error}")
        return move.uci()


def _sync_board(fen: str) -> chess.Board:
    """Advance persistent state through the opponent move or reset safely."""
    global _game_board
    incoming = chess.Board(fen)
    incoming_fen = _canonical_fen(incoming)
    if _game_board is None:
        # SearchMemory.create() is already zero-initialized. Avoid touching the
        # complete TT again on the first clocked move of a fresh process.
        return _reset_game(incoming, clear_memory=False)
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
    if time_left_ms <= EMERGENCY_CLOCK_MS:
        return _fallback_move(fen)

    board = _sync_board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "0000"
    fallback = legal_moves[0]
    if len(legal_moves) == 1:
        chosen = fallback
    else:
        limits = _move_time_limits(
            time_left_ms,
            board.fullmove_number,
            int(board.occupied).bit_count(),
        )
        chosen = fallback
        if limits.hard_ms > 0:
            position = engine.position_from_board(board)
            prior_history = np.asarray(_position_history, dtype=np.uint64)
            result = search.search_position(
                position,
                _memory,
                soft_time_limit_s=limits.soft_ms / 1_000.0,
                normal_time_limit_s=limits.normal_ms / 1_000.0,
                time_limit_s=limits.hard_ms / 1_000.0,
                prior_history=prior_history,
            )
            candidate = chess.Move.from_uci(engine.move_to_uci(result.move))
            if candidate in board.legal_moves:
                chosen = candidate

    if chosen not in board.legal_moves:
        chosen = fallback
    return _commit_move(board, chosen)


def get_move(fen: str, time_left_ms: int) -> str:
    """Return legal UCI, retaining a fresh-board fallback for every failure."""
    try:
        adjusted_time_ms = _finish_warmup(time_left_ms)
        return _choose_move(fen, adjusted_time_ms)
    except Exception as error:
        print(f"compiled challenger failed, using fallback: {type(error).__name__}: {error}")
        return _fallback_move(fen)


_warmup_thread = threading.Thread(
    target=_warmup_worker,
    name="engine-warmup",
    daemon=True,
)
_warmup_thread.start()
_ready_wait_s = max(0.0, INIT_READY_TARGET_S - (time.perf_counter() - _IMPORT_STARTED_S))
_warmup_done.wait(_ready_wait_s)
_warmup_elapsed_s = time.perf_counter() - _IMPORT_STARTED_S
