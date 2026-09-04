"""AI Chessathon entry point: a safe, time-bounded classical chess engine."""

from __future__ import annotations

import time
from collections.abc import Hashable
from dataclasses import dataclass
from functools import lru_cache

import chess

MATE_SCORE = 100_000
MATE_BOUND = 90_000
INFINITY = 1_000_000
MAX_DEPTH = 16
MAX_QUIESCENCE_PLY = 10
MAX_CHECK_QUIESCENCE_PLY = 14
TIME_CHECK_MASK = 31
EVAL_CACHE_SIZE = 65_536
TT_MAX_SIZE = 100_000

EXACT = 0
LOWER = 1
UPPER = 2

MG_VALUE = (0, 100, 320, 330, 500, 900, 0)
EG_VALUE = (0, 120, 310, 335, 525, 900, 0)
PHASE_VALUE = (0, 0, 1, 1, 2, 4, 0)
MAX_PHASE = 24


@dataclass(slots=True)
class TableEntry:
    depth: int
    score: int
    flag: int
    move: chess.Move | None


class SearchTimeout(Exception):
    """Raised inside the tree when the hard move deadline expires."""


_game_board: chess.Board | None = None
_history: dict[int, int] = {}
_eval_cache: dict[Hashable, int] = {}
_transposition_table: dict[Hashable, TableEntry] = {}


def _move_key(move: chess.Move) -> int:
    return (move.from_square << 10) | (move.to_square << 4) | (move.promotion or 0)


def _relative_rank(square: chess.Square, color: chess.Color) -> int:
    rank = chess.square_rank(square)
    return rank if color == chess.WHITE else 7 - rank


def _piece_square(
    piece_type: chess.PieceType, square: chess.Square, color: chess.Color
) -> tuple[int, int]:
    """Return compact, symmetric middlegame/endgame positional bonuses."""
    file_index = chess.square_file(square)
    rank = _relative_rank(square, color)
    center_distance = abs(2 * file_index - 7) + abs(2 * rank - 7)
    center = 14 - center_distance

    if piece_type == chess.PAWN:
        central_file = 4 - abs(2 * file_index - 7)
        return rank * 7 + central_file * 2, rank * 12 + central_file
    if piece_type == chess.KNIGHT:
        return center * 4, center * 3
    if piece_type == chess.BISHOP:
        return center * 2 + rank * 2, center * 2
    if piece_type == chess.ROOK:
        seventh = 22 if rank == 6 else 0
        return seventh + rank, seventh + rank * 2
    if piece_type == chess.QUEEN:
        return center - rank * 2, center * 2
    if piece_type == chess.KING:
        home_safety = 28 if rank == 0 and file_index in (2, 6) else 0
        return home_safety - center * 5, center * 5
    return 0, 0


@lru_cache(maxsize=32_768)
def _pawn_structure(
    pawn_mask: int, enemy_pawn_mask: int, color: chess.Color
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    pawns = tuple(chess.scan_forward(pawn_mask))
    enemy_pawns = tuple(chess.scan_forward(enemy_pawn_mask))
    file_counts = [0] * 8
    enemy_file_counts = [0] * 8
    for square in pawns:
        file_counts[chess.square_file(square)] += 1
    for square in enemy_pawns:
        enemy_file_counts[chess.square_file(square)] += 1

    middlegame = 0
    endgame = 0
    for square in pawns:
        file_index = chess.square_file(square)
        rank = chess.square_rank(square)
        relative_rank = _relative_rank(square, color)
        if file_counts[file_index] > 1:
            middlegame -= 11
            endgame -= 14
        neighbors = file_counts[file_index - 1] if file_index > 0 else 0
        neighbors += file_counts[file_index + 1] if file_index < 7 else 0
        if neighbors == 0:
            middlegame -= 10
            endgame -= 8

        passed = True
        for enemy_square in enemy_pawns:
            enemy_file = chess.square_file(enemy_square)
            if abs(enemy_file - file_index) > 1:
                continue
            enemy_rank = chess.square_rank(enemy_square)
            if (color == chess.WHITE and enemy_rank > rank) or (
                color == chess.BLACK and enemy_rank < rank
            ):
                passed = False
                break
        if passed:
            middlegame += relative_rank * 7
            endgame += relative_rank * relative_rank * 5

    return middlegame, endgame, tuple(file_counts), tuple(enemy_file_counts)


def _pawn_and_rook_features(board: chess.Board, color: chess.Color) -> tuple[int, int]:
    pawn_mask = board.pieces_mask(chess.PAWN, color)
    enemy_pawn_mask = board.pieces_mask(chess.PAWN, not color)
    middlegame, endgame, file_counts, enemy_file_counts = _pawn_structure(
        pawn_mask, enemy_pawn_mask, color
    )
    all_pawn_files = tuple(
        file_counts[index] + enemy_file_counts[index] for index in range(8)
    )
    for square in board.pieces(chess.ROOK, color):
        file_index = chess.square_file(square)
        if file_counts[file_index] == 0:
            middlegame += 12
            endgame += 8
            if all_pawn_files[file_index] == 0:
                middlegame += 10
                endgame += 6

    king_square = board.king(color)
    if king_square is not None:
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        shield_rank = king_rank + (1 if color == chess.WHITE else -1)
        if 0 <= shield_rank < 8:
            for file_index in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
                shield_square = chess.square(file_index, shield_rank)
                if pawn_mask & chess.BB_SQUARES[shield_square]:
                    middlegame += 9

    return middlegame, endgame


def evaluate(board: chess.Board) -> int:
    """Tapered evaluation, always from the side-to-move's perspective."""
    key = board._transposition_key()
    cached = _eval_cache.get(key)
    if cached is not None:
        return cached

    middlegame = 0
    endgame = 0
    phase = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        mg_square, eg_square = _piece_square(piece.piece_type, square, piece.color)
        middlegame += sign * (MG_VALUE[piece.piece_type] + mg_square)
        endgame += sign * (EG_VALUE[piece.piece_type] + eg_square)
        phase += PHASE_VALUE[piece.piece_type]

    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        mg_features, eg_features = _pawn_and_rook_features(board, color)
        middlegame += sign * mg_features
        endgame += sign * eg_features
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            middlegame += sign * 32
            endgame += sign * 42

    phase = min(phase, MAX_PHASE)
    score = (middlegame * phase + endgame * (MAX_PHASE - phase)) // MAX_PHASE
    score += 10 if board.turn == chess.WHITE else -10
    result = score if board.turn == chess.WHITE else -score
    if len(_eval_cache) >= EVAL_CACHE_SIZE:
        _eval_cache.clear()
    _eval_cache[key] = result
    return result


class Searcher:
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        if len(_transposition_table) >= TT_MAX_SIZE:
            _transposition_table.clear()
        self.table = _transposition_table
        self.killers: dict[int, tuple[chess.Move | None, chess.Move | None]] = {}
        self.nodes = 0
        self.qnodes = 0

    def check_time(self) -> None:
        self.nodes += 1
        if self.nodes & TIME_CHECK_MASK == 0 and time.monotonic() >= self.deadline:
            raise SearchTimeout

    @staticmethod
    def draw_score(board: chess.Board, ply: int) -> int | None:
        """Detect draws that are not discovered by an empty legal move list."""
        if board.is_insufficient_material():
            return 0
        if board.halfmove_clock >= 100:
            return 0
        if ply >= 4 and board.halfmove_clock >= 8 and board.is_repetition(3):
            return 0
        return None

    @staticmethod
    def _table_score(score: int, ply: int) -> int:
        if score >= MATE_BOUND:
            return score + ply
        if score <= -MATE_BOUND:
            return score - ply
        return score

    @staticmethod
    def _search_score(score: int, ply: int) -> int:
        if score >= MATE_BOUND:
            return score - ply
        if score <= -MATE_BOUND:
            return score + ply
        return score

    @staticmethod
    def _position_key(board: chess.Board) -> Hashable:
        return board._transposition_key(), min(board.halfmove_clock, 100)

    def move_score(
        self, board: chess.Board, move: chess.Move, hash_move: chess.Move | None, ply: int
    ) -> int:
        if move == hash_move:
            return 20_000_000

        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            victim_value = MG_VALUE[chess.PAWN] if board.is_en_passant(move) else 0
            if victim is not None:
                victim_value = MG_VALUE[victim.piece_type]
            attacker_value = MG_VALUE[attacker.piece_type] if attacker is not None else 0
            score += 10_000_000 + 16 * victim_value - attacker_value
        elif move.promotion is None:
            first, second = self.killers.get(ply, (None, None))
            if move == first:
                score += 9_000_000
            elif move == second:
                score += 8_000_000
            score += _history.get(_move_key(move), 0)

        if move.promotion is not None:
            score += 12_000_000 + MG_VALUE[move.promotion]
        return score

    def ordered_moves(
        self, board: chess.Board, hash_move: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        moves = list(board.legal_moves)
        moves.sort(key=lambda move: self.move_score(board, move, hash_move, ply), reverse=True)
        return moves

    def quiescence(
        self, board: chess.Board, alpha: int, beta: int, ply: int, qply: int = 0
    ) -> int:
        self.check_time()
        self.qnodes += 1
        draw = self.draw_score(board, ply)
        if draw is not None:
            return draw

        in_check = board.is_check()
        legal_moves = self.ordered_moves(board, None, ply) if in_check else list(board.legal_moves)
        if not legal_moves:
            return -MATE_SCORE + ply if in_check else 0

        limit = MAX_CHECK_QUIESCENCE_PLY if in_check else MAX_QUIESCENCE_PLY
        if qply >= limit:
            return evaluate(board)

        if in_check:
            moves = legal_moves
            best = -INFINITY
        else:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return stand_pat
            alpha = max(alpha, stand_pat)
            best = stand_pat
            moves = [
                move
                for move in legal_moves
                if board.is_capture(move) or move.promotion is not None
            ]
            moves.sort(key=lambda move: self.move_score(board, move, None, ply), reverse=True)

        for move in moves:
            board.push(move)
            try:
                score = -self.quiescence(board, -beta, -alpha, ply + 1, qply + 1)
            finally:
                board.pop()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best

    def _record_cutoff(self, move: chess.Move, ply: int, depth: int) -> None:
        first, second = self.killers.get(ply, (None, None))
        if move != first:
            self.killers[ply] = (move, first if move != second else second)
        key = _move_key(move)
        _history[key] = min(1_000_000, _history.get(key, 0) + depth * depth)

    def search(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self.check_time()
        draw = self.draw_score(board, ply)
        if draw is not None:
            return draw
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply)

        key = self._position_key(board)
        entry = self.table.get(key)
        original_alpha = alpha
        original_beta = beta
        if entry is not None and entry.depth >= depth:
            table_score = self._search_score(entry.score, ply)
            if entry.flag == EXACT:
                return table_score
            if entry.flag == LOWER:
                alpha = max(alpha, table_score)
            else:
                beta = min(beta, table_score)
            if alpha >= beta:
                return table_score

        in_check = board.is_check()
        best = -INFINITY
        best_move: chess.Move | None = None
        moves = self.ordered_moves(board, entry.move if entry is not None else None, ply)
        if not moves:
            return -MATE_SCORE + ply if in_check else 0
        for index, move in enumerate(moves):
            quiet = not board.is_capture(move) and move.promotion is None
            reduce_quiet = depth >= 3 and index >= 4 and quiet and not in_check
            if reduce_quiet and board.gives_check(move):
                reduce_quiet = False
            board.push(move)
            try:
                if index == 0:
                    score = -self.search(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    reduced_depth = depth - 1
                    if reduce_quiet:
                        reduced_depth = depth - 2
                    score = -self.search(board, reduced_depth, -alpha - 1, -alpha, ply + 1)
                    if score > alpha and reduced_depth != depth - 1:
                        score = -self.search(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if alpha < score < beta:
                        score = -self.search(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best:
                best = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if quiet:
                    self._record_cutoff(move, ply, depth)
                break

        flag = UPPER if best <= original_alpha else LOWER if best >= original_beta else EXACT
        previous = self.table.get(key)
        if previous is None or depth >= previous.depth:
            self.table[key] = TableEntry(
                depth, self._table_score(best, ply), flag, best_move
            )
        return best

    def search_root(
        self, board: chess.Board, depth: int, alpha: int, beta: int, preferred: chess.Move
    ) -> tuple[int, chess.Move]:
        best_score = -INFINITY
        best_move = preferred
        for index, move in enumerate(self.ordered_moves(board, preferred, 0)):
            board.push(move)
            try:
                if index == 0:
                    score = -self.search(board, depth - 1, -beta, -alpha, 1)
                else:
                    score = -self.search(board, depth - 1, -alpha - 1, -alpha, 1)
                    if alpha < score < beta:
                        score = -self.search(board, depth - 1, -beta, -alpha, 1)
            finally:
                board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return best_score, best_move

    def best_move(self, board: chess.Board, fallback: chess.Move) -> chess.Move:
        best_move = fallback
        previous_score = 0
        previous_iteration_s = 0.0

        for depth in range(1, MAX_DEPTH + 1):
            now = time.monotonic()
            if now >= self.deadline:
                break
            if previous_iteration_s and now + previous_iteration_s * 1.8 >= self.deadline:
                break
            iteration_started = now
            window = 45
            alpha = -INFINITY if depth == 1 else previous_score - window
            beta = INFINITY if depth == 1 else previous_score + window
            try:
                score, candidate = self.search_root(board, depth, alpha, beta, best_move)
                if score <= alpha or score >= beta:
                    score, candidate = self.search_root(
                        board, depth, -INFINITY, INFINITY, best_move
                    )
            except SearchTimeout:
                break
            best_move = candidate
            previous_score = score
            previous_iteration_s = time.monotonic() - iteration_started
        return best_move


def _sync_board(fen: str) -> chess.Board:
    """Advance the persistent game board by the opponent's move when possible."""
    global _game_board
    if _game_board is not None:
        if _game_board.fen() == fen:
            return _game_board
        for move in list(_game_board.legal_moves):
            _game_board.push(move)
            if _game_board.fen() == fen:
                return _game_board
            _game_board.pop()
    _game_board = chess.Board(fen)
    return _game_board


def _move_budget_ms(time_left_ms: int) -> int:
    reserve_ms = max(100, min(1_200, time_left_ms // 10))
    usable_ms = max(0, time_left_ms - reserve_ms)
    if time_left_ms >= 60_000:
        # Rated games add 500 ms per move. Spend more of the large opening reserve:
        # the old schedule left 45--70 seconds unused in all three supplied games.
        target_ms = min(4_000, time_left_ms // 35 + 300)
    else:
        target_ms = min(3_000, time_left_ms // 45 + 220)
    return max(0, min(target_ms, usable_ms))


def _choose_move(fen: str, time_left_ms: int) -> str:
    """Search for a move, keeping the board state needed between calls."""
    global _game_board

    # With no usable search time, avoid the persistent-board reconciliation and
    # materialising every legal move.  At sub-100 ms clocks even small amounts
    # of Python bookkeeping matter; the first generated move is already legal.
    if time_left_ms <= 100:
        board = chess.Board(fen)
        try:
            fallback = next(iter(board.legal_moves))
        except StopIteration:
            return "0000"
        move_uci = fallback.uci()
        board.push(fallback)
        _game_board = board
        return move_uci

    board = _sync_board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "0000"

    fallback = legal_moves[0]
    if len(legal_moves) == 1:
        board.push(fallback)
        return fallback.uci()

    budget_ms = _move_budget_ms(time_left_ms)
    chosen = fallback
    if budget_ms > 0:
        deadline = time.monotonic() + budget_ms / 1000.0
        try:
            chosen = Searcher(deadline).best_move(board, fallback)
        except Exception as error:
            print(f"search failed, using fallback: {type(error).__name__}: {error}")
            chosen = fallback

    if chosen not in board.legal_moves:
        chosen = fallback
    board.push(chosen)

    if _history and max(_history.values()) >= 900_000:
        for key in list(_history):
            _history[key] //= 2
    return chosen.uci()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move, with a fresh-board fallback on internal errors."""
    global _game_board

    try:
        return _choose_move(fen, time_left_ms)
    except Exception as error:
        print(f"agent failed, using emergency move: {type(error).__name__}: {error}")
        board = chess.Board(fen)
        try:
            fallback = next(iter(board.legal_moves))
        except StopIteration:
            return "0000"
        move_uci = fallback.uci()
        board.push(fallback)
        _game_board = board
        return move_uci
