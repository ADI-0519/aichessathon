"""Compiled alpha-beta search for the Numba board challenger.

The search is intentionally conservative: correctness, deterministic node
limits, and a safely interruptible completed iteration come before selective
pruning.  More aggressive features belong in separately measured changes.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import chess
import engine
import nnue
import numpy as np
from numba import njit
from numpy.typing import NDArray

INFINITY = 32_000
MATE_SCORE = 30_000
MATE_BOUND = 29_000
MAX_PLY = 96
MAX_DEPTH = 32
MAX_HISTORY = 512
STOP_POLL_MASK = 255
SEE_MAX_EXCHANGES = 32
DELTA_MARGIN = 120
STABLE_ITERATIONS_REQUIRED = 2
STABLE_SCORE_DELTA = 30
UNSTABLE_SCORE_DELTA = 75
NEXT_ITERATION_COST_MULTIPLIER = 1.5

# Percentage of the static evaluation supplied by the learned model.  Keep this
# as a source constant so every packaged challenger is reproducible.
NNUE_BLEND = 75

TT_EMPTY = 0
TT_EXACT = 1
TT_LOWER = 2
TT_UPPER = 3
TT_MOVE = 0
TT_SCORE = 1
TT_DEPTH = 2
TT_BOUND = 3
TT_GENERATION = 4
TT_HALFMOVE = 5
TT_FIELD_COUNT = 6
DEFAULT_TT_BITS = 22
TT_BUCKET_SIZE = 2
Q_EVAL_BITS = 16

LMR_TABLE_MAX = 64
LMR_TABLE = np.zeros((LMR_TABLE_MAX, LMR_TABLE_MAX), dtype=np.int32)
for _lmr_depth in range(1, LMR_TABLE_MAX):
    for _lmr_index in range(1, LMR_TABLE_MAX):
        _base_reduction = int(
            0.75
            + np.log(float(_lmr_depth))
            * np.log(float(_lmr_index))
            / 2.25
        )
        LMR_TABLE[_lmr_depth, _lmr_index] = max(1, _base_reduction)

ROOT_LMR_MIN_DEPTH = 6
ROOT_LMR_MIN_INDEX = 4
ROOT_LMR_DEEP_DEPTH = 9
ROOT_LMR_DEEP_INDEX = 10


# Search V10 plus the independently switchable S1 move-ordering layer.
ENABLE_V10_DYNAMIC_NMP = True
ENABLE_V10_REVERSE_FUTILITY = True
ENABLE_V10_LATE_MOVE_PRUNING = True
ENABLE_V10_QUIET_FUTILITY = True
ENABLE_V10_SEE_PRUNING = True
ENABLE_V10_CONTEXTUAL_LMR = True
ENABLE_HISTORY_V2 = True
ENABLE_QUIET_SEE_PRUNING = True
ENABLE_TT2 = True
ENABLE_CAPTURE_HISTORY = True
ENABLE_COUNTERMOVES = True
ENABLE_SINGULAR_EXTENSIONS = False
ACTIVE_PROFILE = "baseline"

_S1_PROFILES = {
    "baseline": (True, True, True, True, True, True, True, True),
    "v10": (True, True, True, True, True, True, False, False),
    "history-v2": (True, True, True, True, True, True, True, False),
    "quiet-see": (True, True, True, True, True, True, False, True),
    "current": (False, False, False, False, False, False, False, False),
}

# Conservative selective-search parameters. Values are intentionally modest:
# the goal is a material effective-depth gain without making the new KingNet
# evaluator carry Stockfish-scale pruning immediately.
V10_RFP_MAX_DEPTH = 4
V10_RFP_MARGIN_BASE = 80
V10_RFP_MARGIN_PER_DEPTH = 95
V10_QF_MAX_DEPTH = 3
V10_QF_MARGIN_BASE = 95
V10_QF_MARGIN_PER_DEPTH = 125
V10_LMP_MAX_DEPTH = 3
V10_SEE_MAX_DEPTH = 5
V10_SEE_MARGIN_PER_DEPTH = 70
V10_NMP_VERIFY_DEPTH = 8
V10_NMP_EVAL_DIVISOR = 220
V10_NMP_MAX_EVAL_BONUS = 2
V10_GOOD_HISTORY_THRESHOLD = 170

# Signed butterfly history uses a bounded gravity update, so recent evidence
# can move an old entry in either direction without clearing the table. The
# values are deliberately separate from the frozen V10 scale.
HISTORY_V2_LIMIT = 16_384
HISTORY_V2_DEPTH_SCALE = 16
HISTORY_V2_UPDATE_CAP = 2_048
HISTORY_V2_GOOD_THRESHOLD = 512
HISTORY_V2_BAD_THRESHOLD = -512

CAPTURE_HISTORY_LIMIT = 16_384
CAPTURE_HISTORY_DEPTH_SCALE = 16
CAPTURE_HISTORY_UPDATE_CAP = 2_048
COUNTERMOVE_ORDER_SCORE = 7_000_000
SINGULAR_MIN_DEPTH = 7
SINGULAR_TT_DEPTH_SLACK = 3
SINGULAR_MARGIN_PER_DEPTH = 2

# Quiet SEE is a conservative shallow non-PV pruning rule. Checks, killers,
# high-history moves and the first ordered move are always preserved.
QUIET_SEE_MAX_DEPTH = 6
QUIET_SEE_MARGIN_PER_DEPTH_SQUARED = 30

STAT_NODES = 0
STAT_QNODES = 1
STAT_TT_PROBES = 2
STAT_TT_HITS = 3
STAT_TT_CUTOFFS = 4
STAT_BETA_CUTOFFS = 5
STAT_LMR_REDUCTIONS = 6
STAT_LMR_RESEARCHES = 7
STAT_Q_EVAL_PROBES = 8
STAT_Q_EVAL_HITS = 9
STAT_TT_COLLISIONS = 10
STAT_TT_REPLACEMENTS = 11
STAT_CAPTURE_HISTORY_UPDATES = 12
STAT_SINGULAR_ATTEMPTS = 13
STAT_SINGULAR_EXTENSIONS = 14
STAT_COUNT = 15

MG_VALUE = np.array((100, 320, 330, 500, 900, 0), dtype=np.int32)
EG_VALUE = np.array((120, 310, 335, 525, 900, 0), dtype=np.int32)
PHASE_VALUE = np.array((0, 1, 1, 2, 4, 0), dtype=np.int32)
MAX_PHASE = 24


def _relative_rank(square: int, color: int) -> int:
    rank = square >> 3
    return rank if color == engine.WHITE else 7 - rank


def _piece_square(kind: int, square: int, color: int) -> tuple[int, int]:
    file_index = square & 7
    rank = _relative_rank(square, color)
    center_distance = abs(2 * file_index - 7) + abs(2 * rank - 7)
    center = 14 - center_distance
    if kind == engine.PAWN:
        central_file = 4 - abs(2 * file_index - 7)
        return rank * 7 + central_file * 2, rank * 12 + central_file
    if kind == engine.KNIGHT:
        return center * 4, center * 3
    if kind == engine.BISHOP:
        return center * 2 + rank * 2, center * 2
    if kind == engine.ROOK:
        seventh = 22 if rank == 6 else 0
        return seventh + rank, seventh + rank * 2
    if kind == engine.QUEEN:
        return center - rank * 2, center * 2
    if kind == engine.KING:
        home_safety = 28 if rank == 0 and file_index in (2, 6) else 0
        return home_safety - center * 5, center * 5
    return 0, 0


MG_PST = np.zeros((engine.PIECE_BITBOARD_COUNT, 64), dtype=np.int32)
EG_PST = np.zeros((engine.PIECE_BITBOARD_COUNT, 64), dtype=np.int32)
for _color in (engine.WHITE, engine.BLACK):
    for _kind in range(engine.PIECE_KIND_COUNT):
        for _square in range(64):
            _mg, _eg = _piece_square(_kind, _square, _color)
            _index = _color * engine.PIECE_KIND_COUNT + _kind
            MG_PST[_index, _square] = _mg
            EG_PST[_index, _square] = _eg

FILE_MASKS = np.zeros(8, dtype=np.uint64)
PASSED_MASKS = np.zeros((2, 64), dtype=np.uint64)
for _file in range(8):
    for _rank in range(8):
        FILE_MASKS[_file] |= np.uint64(1) << np.uint64(_rank * 8 + _file)
for _color in (engine.WHITE, engine.BLACK):
    for _square in range(64):
        _file = _square & 7
        _rank = _square >> 3
        _mask = 0
        for _target_file in range(max(0, _file - 1), min(7, _file + 1) + 1):
            _target_ranks = (
                range(_rank + 1, 8) if _color == engine.WHITE else range(0, _rank)
            )
            for _target_rank in _target_ranks:
                _mask |= 1 << (_target_rank * 8 + _target_file)
        PASSED_MASKS[_color, _square] = np.uint64(_mask)


@dataclass(slots=True)
class SearchMemory:
    """Persistent transposition and history tables owned by one game process."""

    tt_keys: NDArray[np.uint64]
    tt_data: NDArray[np.int32]
    quiet_history: NDArray[np.int32]
    capture_history: NDArray[np.int32]
    countermoves: NDArray[np.int32]
    q_eval_keys: NDArray[np.uint64]
    q_eval_scores: NDArray[np.int32]
    q_eval_valid: NDArray[np.uint8]
    generation: int = 0

    @classmethod
    def create(cls, tt_bits: int = DEFAULT_TT_BITS) -> SearchMemory:
        if not 10 <= tt_bits <= 24:
            raise ValueError("tt_bits must be between 10 and 24")
        size = 1 << tt_bits
        q_eval_size = 1 << Q_EVAL_BITS
        return cls(
            np.zeros(size, dtype=np.uint64),
            np.zeros((size, TT_FIELD_COUNT), dtype=np.int32),
            np.zeros((2, 64, 64), dtype=np.int32),
            np.zeros((2, engine.PIECE_KIND_COUNT, 64, engine.PIECE_KIND_COUNT), dtype=np.int32),
            np.zeros((2, 64, 64), dtype=np.int32),
            np.zeros(q_eval_size, dtype=np.uint64),
            np.zeros(q_eval_size, dtype=np.int32),
            np.zeros(q_eval_size, dtype=np.uint8),
        )

    def next_generation(self) -> int:
        self.generation = self.generation % 2_000_000_000 + 1
        return self.generation

    def clear(self) -> None:
        self.tt_keys.fill(0)
        self.tt_data.fill(0)
        self.quiet_history.fill(0)
        self.capture_history.fill(0)
        self.countermoves.fill(0)
        self.q_eval_valid.fill(0)
        self.generation = 0


@dataclass(frozen=True, slots=True)
class SearchResult:
    move: int
    score: int
    depth: int
    nodes: int
    qnodes: int
    elapsed_s: float
    stopped: bool
    tt_hits: int
    tt_cutoffs: int
    beta_cutoffs: int
    lmr_reductions: int
    lmr_researches: int
    q_eval_probes: int
    q_eval_hits: int
    tt_collisions: int = 0
    tt_replacements: int = 0
    capture_history_updates: int = 0
    singular_attempts: int = 0
    singular_extensions: int = 0


@njit(cache=False)
def handcrafted_evaluate(
    pieces: NDArray[np.uint64], state: NDArray[np.int64]
) -> int:
    """Tapered handcrafted evaluation from the side-to-move perspective."""
    middlegame = 0
    endgame = 0
    phase = 0
    for color in range(2):
        sign = 1 if color == engine.WHITE else -1
        for kind in range(engine.PIECE_KIND_COUNT):
            index = engine.piece_index(color, kind)
            occupied = pieces[index]
            count = engine.popcount(occupied)
            middlegame += sign * count * int(MG_VALUE[kind])
            endgame += sign * count * int(EG_VALUE[kind])
            phase += count * int(PHASE_VALUE[kind])
            while occupied:
                square = engine.lsb_square(occupied)
                occupied ^= engine.bit(square)
                middlegame += sign * int(MG_PST[index, square])
                endgame += sign * int(EG_PST[index, square])

        if engine.popcount(pieces[engine.piece_index(color, engine.BISHOP)]) >= 2:
            middlegame += sign * 32
            endgame += sign * 42

        pawns = pieces[engine.piece_index(color, engine.PAWN)]
        enemy_pawns = pieces[
            engine.piece_index(engine.BLACK if color == engine.WHITE else engine.WHITE, engine.PAWN)
        ]
        pawn_scan = pawns
        while pawn_scan:
            square = engine.lsb_square(pawn_scan)
            pawn_scan ^= engine.bit(square)
            file_index = square & 7
            relative_rank = _relative_rank_numba(square, color)
            if engine.popcount(pawns & FILE_MASKS[file_index]) > 1:
                middlegame -= sign * 11
                endgame -= sign * 14
            neighbor_pawns = np.uint64(0)
            if file_index > 0:
                neighbor_pawns |= pawns & FILE_MASKS[file_index - 1]
            if file_index < 7:
                neighbor_pawns |= pawns & FILE_MASKS[file_index + 1]
            if neighbor_pawns == 0:
                middlegame -= sign * 10
                endgame -= sign * 8
            if enemy_pawns & PASSED_MASKS[color, square] == 0:
                middlegame += sign * relative_rank * 7
                endgame += sign * relative_rank * relative_rank * 5

        rooks = pieces[engine.piece_index(color, engine.ROOK)]
        while rooks:
            square = engine.lsb_square(rooks)
            rooks ^= engine.bit(square)
            file_index = square & 7
            if pawns & FILE_MASKS[file_index] == 0:
                middlegame += sign * 12
                endgame += sign * 8
                if enemy_pawns & FILE_MASKS[file_index] == 0:
                    middlegame += sign * 10
                    endgame += sign * 6

        king = pieces[engine.piece_index(color, engine.KING)]
        if king:
            king_square = engine.lsb_square(king)
            king_file = king_square & 7
            king_rank = king_square >> 3
            shield_rank = king_rank + (1 if color == engine.WHITE else -1)
            if 0 <= shield_rank < 8:
                for shield_file in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
                    if pawns & engine.bit(shield_rank * 8 + shield_file):
                        middlegame += sign * 9

    phase = min(phase, MAX_PHASE)
    score = (middlegame * phase + endgame * (MAX_PHASE - phase)) // MAX_PHASE
    score += 10 if int(state[engine.STATE_SIDE]) == engine.WHITE else -10
    return score if int(state[engine.STATE_SIDE]) == engine.WHITE else -score


@njit(cache=False, inline="always")
def evaluate(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    accumulators: NDArray[np.int32],
) -> int:
    """Blend V4's static evaluator with the team's learned evaluator."""
    if NNUE_BLEND <= 0:
        return handcrafted_evaluate(pieces, state)
    learned = nnue.evaluate(pieces, accumulators, int(state[engine.STATE_SIDE]))
    if NNUE_BLEND >= 100:
        return learned
    handcrafted = handcrafted_evaluate(pieces, state)
    return (NNUE_BLEND * learned + (100 - NNUE_BLEND) * handcrafted) // 100


@njit(cache=False, inline="always")
def _cached_qeval(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: np.uint64,
    accumulators: NDArray[np.int32],
    q_eval_keys: NDArray[np.uint64],
    q_eval_scores: NDArray[np.int32],
    q_eval_valid: NDArray[np.uint8],
    stats: NDArray[np.int64],
) -> int:
    """Return the exact blended evaluation through a direct-mapped cache."""
    stats[STAT_Q_EVAL_PROBES] += 1
    index = int(key & np.uint64(len(q_eval_keys) - 1))
    if q_eval_valid[index] != 0 and q_eval_keys[index] == key:
        stats[STAT_Q_EVAL_HITS] += 1
        return int(q_eval_scores[index])

    score = evaluate(pieces, state, accumulators)
    q_eval_keys[index] = key
    q_eval_scores[index] = np.int32(score)
    q_eval_valid[index] = np.uint8(1)
    return score


@njit(cache=False, inline="always")
def _relative_rank_numba(square: int, color: int) -> int:
    rank = square >> 3
    return rank if color == engine.WHITE else 7 - rank


@njit(cache=False, inline="always")
def _visit_node(
    stats: NDArray[np.int64], stop: NDArray[np.uint8], node_limit: int, quiescence: bool
) -> bool:
    stats[STAT_NODES] += 1
    if quiescence:
        stats[STAT_QNODES] += 1
    nodes = int(stats[STAT_NODES])
    if node_limit > 0 and nodes > node_limit:
        return True
    return nodes & STOP_POLL_MASK == 0 and stop[0] != 0


@njit(cache=False, inline="always")
def _score_to_table(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


@njit(cache=False, inline="always")
def _score_from_table(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


@njit(cache=False)
def _attackers_to(
    pieces: NDArray[np.uint64], square: int, color: int, occupied: np.uint64
) -> np.uint64:
    """Return attackers of ``color`` under an explicit hypothetical occupancy."""
    enemy = engine.BLACK if color == engine.WHITE else engine.WHITE
    attackers = pieces[engine.piece_index(color, engine.PAWN)] & engine.PAWN_ATTACKS[
        enemy, square
    ]
    attackers |= pieces[engine.piece_index(color, engine.KNIGHT)] & engine.KNIGHT_ATTACKS[
        square
    ]
    attackers |= pieces[engine.piece_index(color, engine.KING)] & engine.KING_ATTACKS[square]
    square_file = square & 7
    square_rank = square >> 3

    for direction in range(4):
        file_index = square_file + int(engine.ROOK_FILE_DELTAS[direction])
        rank = square_rank + int(engine.ROOK_RANK_DELTAS[direction])
        while 0 <= file_index < 8 and 0 <= rank < 8:
            target = rank * 8 + file_index
            target_bit = engine.bit(target)
            if occupied & target_bit:
                sliders = pieces[engine.piece_index(color, engine.ROOK)] | pieces[
                    engine.piece_index(color, engine.QUEEN)
                ]
                if sliders & target_bit:
                    attackers |= target_bit
                break
            file_index += int(engine.ROOK_FILE_DELTAS[direction])
            rank += int(engine.ROOK_RANK_DELTAS[direction])

    for direction in range(4):
        file_index = square_file + int(engine.BISHOP_FILE_DELTAS[direction])
        rank = square_rank + int(engine.BISHOP_RANK_DELTAS[direction])
        while 0 <= file_index < 8 and 0 <= rank < 8:
            target = rank * 8 + file_index
            target_bit = engine.bit(target)
            if occupied & target_bit:
                sliders = pieces[engine.piece_index(color, engine.BISHOP)] | pieces[
                    engine.piece_index(color, engine.QUEEN)
                ]
                if sliders & target_bit:
                    attackers |= target_bit
                break
            file_index += int(engine.BISHOP_FILE_DELTAS[direction])
            rank += int(engine.BISHOP_RANK_DELTAS[direction])
    return np.uint64(attackers & occupied)


@njit(cache=False)
def _least_legal_attacker(
    pieces: NDArray[np.uint64], target: int, occupied: np.uint64, color: int
) -> tuple[int, int]:
    attackers = _attackers_to(pieces, target, color, occupied)
    enemy = engine.BLACK if color == engine.WHITE else engine.WHITE
    king = pieces[engine.piece_index(color, engine.KING)] & occupied
    for kind in range(engine.PIECE_KIND_COUNT):
        candidates = attackers & occupied & pieces[engine.piece_index(color, kind)]
        while candidates:
            square = engine.lsb_square(candidates)
            candidates ^= engine.bit(square)
            next_occupied = occupied & ~engine.bit(square)
            if kind == engine.KING:
                if _attackers_to(pieces, target, enemy, next_occupied):
                    continue
            elif king:
                king_square = engine.lsb_square(king)
                checks = _attackers_to(pieces, king_square, enemy, next_occupied)
                # The stale original piece type at the exchange square has
                # just been captured. The new occupant still blocks that
                # square, but cannot attack its own king.
                if checks & ~engine.bit(target):
                    continue
            return kind, square
    return engine.NO_PIECE, engine.NO_PIECE


@njit(cache=False, inline="always")
def _immediate_material_gain(
    pieces: NDArray[np.uint64], state: NDArray[np.int64], move: int
) -> int:
    side = int(state[engine.STATE_SIDE])
    enemy = engine.BLACK if side == engine.WHITE else engine.WHITE
    flags = engine.move_flags(move)
    gain = 0
    if flags & engine.FLAG_CAPTURE:
        if flags & engine.FLAG_EN_PASSANT:
            gain = int(MG_VALUE[engine.PAWN])
        else:
            captured = engine.piece_at(
                pieces,
                engine.move_to(move),
                enemy * engine.PIECE_KIND_COUNT,
                (enemy + 1) * engine.PIECE_KIND_COUNT,
            )
            if captured != engine.NO_PIECE:
                gain = int(MG_VALUE[captured % engine.PIECE_KIND_COUNT])
    if flags & engine.FLAG_PROMOTION:
        gain += int(MG_VALUE[engine.move_promotion(move)]) - int(MG_VALUE[engine.PAWN])
    return gain


@njit(cache=False)
def static_exchange_eval(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    move: int,
    gains: NDArray[np.int32],
) -> int:
    """Evaluate optimal material exchanges on the move's destination square."""
    side = int(state[engine.STATE_SIDE])
    enemy = engine.BLACK if side == engine.WHITE else engine.WHITE
    from_square = engine.move_from(move)
    target = engine.move_to(move)
    flags = engine.move_flags(move)
    moving_piece = engine.piece_at(
        pieces,
        from_square,
        side * engine.PIECE_KIND_COUNT,
        (side + 1) * engine.PIECE_KIND_COUNT,
    )
    if moving_piece == engine.NO_PIECE:
        return -INFINITY

    gains[0] = np.int32(_immediate_material_gain(pieces, state, move))
    moving_kind = moving_piece % engine.PIECE_KIND_COUNT
    occupant_kind = engine.move_promotion(move) if flags & engine.FLAG_PROMOTION else moving_kind
    occupant_value = int(MG_VALUE[occupant_kind])
    occupied = engine.all_occupancy(pieces) & ~engine.bit(from_square)
    if flags & engine.FLAG_EN_PASSANT:
        captured_square = target - 8 if side == engine.WHITE else target + 8
        occupied &= ~engine.bit(captured_square)
    occupied |= engine.bit(target)

    depth = 0
    exchange_side = enemy
    while depth + 1 < min(len(gains), SEE_MAX_EXCHANGES):
        kind, attacker_square = _least_legal_attacker(
            pieces, target, occupied, exchange_side
        )
        if kind == engine.NO_PIECE:
            break

        next_occupied = occupied & ~engine.bit(attacker_square)
        depth += 1
        gains[depth] = np.int32(occupant_value - int(gains[depth - 1]))
        promotes = kind == engine.PAWN and (
            (exchange_side == engine.WHITE and target >> 3 == 7)
            or (exchange_side == engine.BLACK and target >> 3 == 0)
        )
        if promotes:
            promotion_gain = int(MG_VALUE[engine.QUEEN]) - int(MG_VALUE[engine.PAWN])
            gains[depth] += np.int32(promotion_gain)
            occupant_value = int(MG_VALUE[engine.QUEEN])
        else:
            occupant_value = int(MG_VALUE[kind])
        occupied = next_occupied
        exchange_side = engine.BLACK if exchange_side == engine.WHITE else engine.WHITE

    while depth > 0:
        gains[depth - 1] = np.int32(
            -max(-int(gains[depth - 1]), int(gains[depth]))
        )
        depth -= 1
    return int(gains[0])


@njit(cache=False, inline="always")
def _move_order_score(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    side: int,
    move: int,
    tt_move: int,
    ply: int,
    killers: NDArray[np.int32],
    quiet_history: NDArray[np.int32],
    capture_history: NDArray[np.int32],
    countermoves: NDArray[np.int32],
    previous_move: int,
    see_gains: NDArray[np.int32],
) -> int:
    if move == tt_move:
        return 20_000_000
    flags = engine.move_flags(move)
    from_square = engine.move_from(move)
    to_square = engine.move_to(move)
    if flags & engine.FLAG_PROMOTION:
        see = static_exchange_eval(pieces, state, move, see_gains)
        return 12_000_000 + see
    if flags & engine.FLAG_CAPTURE:
        attacker = engine.piece_at(
            pieces,
            from_square,
            side * engine.PIECE_KIND_COUNT,
            (side + 1) * engine.PIECE_KIND_COUNT,
        )
        victim_value = int(MG_VALUE[engine.PAWN])
        victim = engine.NO_PIECE
        if flags & engine.FLAG_EN_PASSANT == 0:
            enemy = engine.BLACK if side == engine.WHITE else engine.WHITE
            victim = engine.piece_at(
                pieces,
                to_square,
                enemy * engine.PIECE_KIND_COUNT,
                (enemy + 1) * engine.PIECE_KIND_COUNT,
            )
            if victim != engine.NO_PIECE:
                victim_value = int(MG_VALUE[victim % engine.PIECE_KIND_COUNT])
        attacker_kind = (
            attacker % engine.PIECE_KIND_COUNT
            if attacker != engine.NO_PIECE
            else engine.PAWN
        )
        victim_kind = engine.PAWN
        if flags & engine.FLAG_EN_PASSANT == 0 and victim != engine.NO_PIECE:
            victim_kind = victim % engine.PIECE_KIND_COUNT
        attacker_value = (
            int(MG_VALUE[attacker % engine.PIECE_KIND_COUNT])
            if attacker != engine.NO_PIECE
            else 0
        )
        see = static_exchange_eval(pieces, state, move, see_gains)
        category = 10_000_000 if see >= 0 else 1_000_000
        history_score = (
            int(capture_history[side, attacker_kind, to_square, victim_kind])
            if ENABLE_CAPTURE_HISTORY
            else 0
        )
        return category + 16 * victim_value - attacker_value + see + history_score
    if ply < MAX_PLY:
        if move == int(killers[ply, 0]):
            return 9_000_000
        if move == int(killers[ply, 1]):
            return 8_000_000
    if ENABLE_COUNTERMOVES and previous_move != 0:
        previous_from = engine.move_from(previous_move)
        previous_to = engine.move_to(previous_move)
        if move == int(countermoves[side, previous_from, previous_to]):
            return COUNTERMOVE_ORDER_SCORE
    return int(quiet_history[side, from_square, to_square])


@njit(cache=False)
def _order_moves(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    side: int,
    moves: NDArray[np.int32],
    count: int,
    tt_move: int,
    ply: int,
    killers: NDArray[np.int32],
    quiet_history: NDArray[np.int32],
    capture_history: NDArray[np.int32],
    countermoves: NDArray[np.int32],
    previous_move: int,
    scores: NDArray[np.int32],
    see_gains: NDArray[np.int32],
) -> None:
    for index in range(count):
        scores[index] = _move_order_score(
            pieces,
            state,
            side,
            int(moves[index]),
            tt_move,
            ply,
            killers,
            quiet_history,
            capture_history,
            countermoves,
            previous_move,
            see_gains,
        )
    for index in range(1, count):
        move = moves[index]
        score = scores[index]
        insertion = index
        while insertion > 0 and scores[insertion - 1] < score:
            moves[insertion] = moves[insertion - 1]
            scores[insertion] = scores[insertion - 1]
            insertion -= 1
        moves[insertion] = move
        scores[insertion] = score


@njit(cache=False, inline="always")
def _bounded_history_update(
    side: int,
    move: int,
    delta: int,
    quiet_history: NDArray[np.int32],
) -> None:
    """Apply a signed gravity update while keeping history in fixed bounds."""
    from_square = engine.move_from(move)
    to_square = engine.move_to(move)
    bounded_delta = max(
        -HISTORY_V2_UPDATE_CAP,
        min(HISTORY_V2_UPDATE_CAP, delta),
    )
    previous = int(quiet_history[side, from_square, to_square])
    update = bounded_delta - (
        previous * abs(bounded_delta) // HISTORY_V2_LIMIT
    )
    quiet_history[side, from_square, to_square] = np.int32(
        max(-HISTORY_V2_LIMIT, min(HISTORY_V2_LIMIT, previous + update))
    )


@njit(cache=False, inline="always")
def _record_quiet_cutoff(
    side: int,
    move: int,
    ply: int,
    depth: int,
    killers: NDArray[np.int32],
    quiet_history: NDArray[np.int32],
    countermoves: NDArray[np.int32],
    previous_move: int,
) -> None:
    if ply < MAX_PLY and move != int(killers[ply, 0]):
        killers[ply, 1] = killers[ply, 0]
        killers[ply, 0] = np.int32(move)
    if ENABLE_HISTORY_V2:
        bonus = HISTORY_V2_DEPTH_SCALE * depth * depth
        _bounded_history_update(side, move, bonus, quiet_history)
    else:
        from_square = engine.move_from(move)
        to_square = engine.move_to(move)
        bonus = depth * depth
        previous = int(quiet_history[side, from_square, to_square])
        quiet_history[side, from_square, to_square] = min(
            1_000_000, previous + bonus
        )
    if ENABLE_COUNTERMOVES and previous_move != 0:
        countermoves[
            side,
            engine.move_from(previous_move),
            engine.move_to(previous_move),
        ] = np.int32(move)


@njit(cache=False, inline="always")
def _record_quiet_failures(
    side: int,
    depth: int,
    moves: NDArray[np.int32],
    searched_quiet: NDArray[np.int32],
    cutoff_index: int,
    quiet_history: NDArray[np.int32],
) -> None:
    """Penalize quiet alternatives actually searched before a quiet cutoff."""
    malus = -(HISTORY_V2_DEPTH_SCALE * depth * depth)
    for index in range(cutoff_index):
        if searched_quiet[index] == 1:
            _bounded_history_update(side, int(moves[index]), malus, quiet_history)


@njit(cache=False, inline="always")
def _capture_kinds(
    pieces: NDArray[np.uint64],
    side: int,
    move: int,
) -> tuple[int, int]:
    from_square = engine.move_from(move)
    to_square = engine.move_to(move)
    attacker = engine.piece_at(
        pieces,
        from_square,
        side * engine.PIECE_KIND_COUNT,
        (side + 1) * engine.PIECE_KIND_COUNT,
    )
    attacker_kind = (
        attacker % engine.PIECE_KIND_COUNT
        if attacker != engine.NO_PIECE
        else engine.PAWN
    )
    if engine.move_flags(move) & engine.FLAG_EN_PASSANT:
        return attacker_kind, engine.PAWN
    enemy = engine.BLACK if side == engine.WHITE else engine.WHITE
    victim = engine.piece_at(
        pieces,
        to_square,
        enemy * engine.PIECE_KIND_COUNT,
        (enemy + 1) * engine.PIECE_KIND_COUNT,
    )
    victim_kind = (
        victim % engine.PIECE_KIND_COUNT
        if victim != engine.NO_PIECE
        else engine.PAWN
    )
    return attacker_kind, victim_kind


@njit(cache=False, inline="always")
def _bounded_capture_history_update(
    pieces: NDArray[np.uint64],
    side: int,
    move: int,
    delta: int,
    capture_history: NDArray[np.int32],
) -> None:
    attacker_kind, victim_kind = _capture_kinds(pieces, side, move)
    to_square = engine.move_to(move)
    bounded_delta = max(
        -CAPTURE_HISTORY_UPDATE_CAP,
        min(CAPTURE_HISTORY_UPDATE_CAP, delta),
    )
    previous = int(capture_history[side, attacker_kind, to_square, victim_kind])
    update = bounded_delta - previous * abs(bounded_delta) // CAPTURE_HISTORY_LIMIT
    capture_history[side, attacker_kind, to_square, victim_kind] = np.int32(
        max(-CAPTURE_HISTORY_LIMIT, min(CAPTURE_HISTORY_LIMIT, previous + update))
    )


@njit(cache=False)
def _record_capture_cutoff(
    pieces: NDArray[np.uint64],
    side: int,
    move: int,
    depth: int,
    moves: NDArray[np.int32],
    searched_moves: NDArray[np.int32],
    cutoff_index: int,
    capture_history: NDArray[np.int32],
    stats: NDArray[np.int64],
) -> None:
    bonus = CAPTURE_HISTORY_DEPTH_SCALE * depth * depth
    _bounded_capture_history_update(pieces, side, move, bonus, capture_history)
    stats[STAT_CAPTURE_HISTORY_UPDATES] += 1
    for index in range(cutoff_index):
        if searched_moves[index] == 2:
            _bounded_capture_history_update(
                pieces,
                side,
                int(moves[index]),
                -bonus,
                capture_history,
            )
            stats[STAT_CAPTURE_HISTORY_UPDATES] += 1


@njit(cache=False)
def _quiescence(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: NDArray[np.uint64],
    alpha: int,
    beta: int,
    ply: int,
    history: NDArray[np.uint64],
    history_count: int,
    root_history_count: int,
    legal_stack: NDArray[np.int32],
    pseudo_stack: NDArray[np.int32],
    undo_stack: NDArray[np.int64],
    undo_key_stack: NDArray[np.uint64],
    accumulator_stack: NDArray[np.int32],
    score_stack: NDArray[np.int32],
    see_gain_stack: NDArray[np.int32],
    killers: NDArray[np.int32],
    quiet_history: NDArray[np.int32],
    capture_history: NDArray[np.int32],
    countermoves: NDArray[np.int32],
    move_stack: NDArray[np.int32],
    q_eval_keys: NDArray[np.uint64],
    q_eval_scores: NDArray[np.int32],
    q_eval_valid: NDArray[np.uint8],
    stop: NDArray[np.uint8],
    node_limit: int,
    stats: NDArray[np.int64],
) -> tuple[int, bool]:
    if _visit_node(stats, stop, node_limit, True):
        return 0, True
    if ply >= MAX_PLY - 1 or history_count >= len(history):
        return (
            _cached_qeval(
                pieces,
                state,
                key[0],
                accumulator_stack[ply],
                q_eval_keys,
                q_eval_scores,
                q_eval_valid,
                stats,
            ),
            False,
        )

    side = int(state[engine.STATE_SIDE])
    in_check = engine.is_in_check(pieces, side)
    if in_check:
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
            return -MATE_SCORE + ply, False
    else:
        count = engine.generate_legal_captures(
            pieces,
            state,
            key,
            legal_stack[ply],
            pseudo_stack[ply],
            undo_stack[ply],
            undo_key_stack[ply],
        )
        if count < 0:
            return 0, False
    if engine.has_rule_draw(
        pieces, state, key[0], history, history_count, root_history_count
    ):
        return 0, False

    stand_pat = -INFINITY
    if in_check:
        best = -INFINITY
    else:
        stand_pat = _cached_qeval(
            pieces,
            state,
            key[0],
            accumulator_stack[ply],
            q_eval_keys,
            q_eval_scores,
            q_eval_valid,
            stats,
        )
        if stand_pat >= beta:
            return stand_pat, False
        if stand_pat > alpha:
            alpha = stand_pat
        best = stand_pat

    _order_moves(
        pieces,
        state,
        side,
        legal_stack[ply],
        count,
        0,
        ply,
        killers,
        quiet_history,
        capture_history,
        countermoves,
        int(move_stack[ply - 1]) if ply > 0 else 0,
        score_stack[ply],
        see_gain_stack[ply],
    )
    for index in range(count):
        move = int(legal_stack[ply, index])
        flags = engine.move_flags(move)
        material_gain = 0
        if not in_check:
            material_gain = _immediate_material_gain(pieces, state, move)
        # A qeval-cache hit can bypass NNUE evaluation and leave a row stale
        # after a king-bucket transition. Refresh before mutating the board so
        # a subsequently searched child always derives from the right parent.
        nnue.refresh(pieces, accumulator_stack[ply])
        engine.make_move(
            pieces, state, key, move, undo_stack[ply], undo_key_stack[ply]
        )
        move_stack[ply] = np.int32(move)
        gives_check = engine.is_in_check(pieces, int(state[engine.STATE_SIDE]))
        if not in_check and not gives_check:
            losing_non_promotion = (
                flags & engine.FLAG_PROMOTION == 0
                and int(score_stack[ply, index]) < 2_000_000
            )
            delta_fails = (
                alpha > -MATE_BOUND
                and stand_pat + material_gain + DELTA_MARGIN < alpha
            )
            if losing_non_promotion or delta_fails:
                engine.unmake_move(
                    pieces, state, key, move, undo_stack[ply], undo_key_stack[ply]
                )
                continue
        # Roughly half of ordered qsearch moves fail the checks above. Delay
        # the feature deltas until we know the child will actually be searched.
        nnue.update_after_move(
            move,
            undo_stack[ply],
            accumulator_stack[ply],
            accumulator_stack[ply + 1],
        )
        history[history_count] = key[0]
        child_score, aborted = _quiescence(
            pieces,
            state,
            key,
            -beta,
            -alpha,
            ply + 1,
            history,
            history_count + 1,
            root_history_count,
            legal_stack,
            pseudo_stack,
            undo_stack,
            undo_key_stack,
            accumulator_stack,
            score_stack,
            see_gain_stack,
            killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
            q_eval_scores,
            q_eval_valid,
            stop,
            node_limit,
            stats,
        )
        engine.unmake_move(
            pieces, state, key, move, undo_stack[ply], undo_key_stack[ply]
        )
        if aborted:
            return 0, True
        score = -child_score
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            stats[STAT_BETA_CUTOFFS] += 1
            break
    return best, False


@njit(cache=False, inline="always")
def _has_non_pawn_material(pieces: NDArray[np.uint64], side: int) -> bool:
    occupied = np.uint64(0)
    for kind in range(engine.KNIGHT, engine.QUEEN + 1):
        occupied |= pieces[engine.piece_index(side, kind)]
    return bool(occupied != np.uint64(0))


@njit(cache=False, inline="always")
def _reverse_futility_allowed(
    excluded_move: int,
    have_static_eval: bool,
    depth: int,
    has_non_pawn_material: bool,
) -> bool:
    """Return whether RFP may terminate this node without searching moves.

    An excluded-move search must inspect the alternatives to its excluded move.
    Letting reverse futility return from that verification can falsely classify
    the transposition-table move as singular.
    """
    return bool(
        ENABLE_V10_REVERSE_FUTILITY
        and excluded_move == 0
        and have_static_eval
        and depth <= V10_RFP_MAX_DEPTH
        and has_non_pawn_material
    )


@njit(cache=False, inline="always")
def _tt_bucket_start(key: np.uint64, table_size: int) -> int:
    index = int(key & np.uint64(table_size - 1))
    return index & ~(TT_BUCKET_SIZE - 1) if ENABLE_TT2 else index


@njit(cache=False, inline="always")
def _tt_probe_index(
    key: np.uint64,
    tt_keys: NDArray[np.uint64],
    tt_data: NDArray[np.int32],
) -> int:
    start = _tt_bucket_start(key, len(tt_keys))
    slots = TT_BUCKET_SIZE if ENABLE_TT2 else 1
    best_index = -1
    best_depth = -1
    for offset in range(slots):
        index = start + offset
        if (
            tt_keys[index] == key
            and int(tt_data[index, TT_BOUND]) != TT_EMPTY
            and int(tt_data[index, TT_DEPTH]) > best_depth
        ):
            best_index = index
            best_depth = int(tt_data[index, TT_DEPTH])
    return best_index


@njit(cache=False, inline="always")
def _tt_replacement_index(
    key: np.uint64,
    depth: int,
    generation: int,
    tt_keys: NDArray[np.uint64],
    tt_data: NDArray[np.int32],
    stats: NDArray[np.int64],
) -> int:
    start = _tt_bucket_start(key, len(tt_keys))
    slots = TT_BUCKET_SIZE if ENABLE_TT2 else 1
    replacement = start
    replacement_priority = 1_000_000_000
    collision = False
    for offset in range(slots):
        index = start + offset
        bound = int(tt_data[index, TT_BOUND])
        if tt_keys[index] == key and bound != TT_EMPTY:
            old_depth = int(tt_data[index, TT_DEPTH])
            old_generation = int(tt_data[index, TT_GENERATION])
            if old_generation == generation and depth + 2 < old_depth:
                return -1
            return index
        if bound == TT_EMPTY:
            return index
        collision = True
        entry_depth = int(tt_data[index, TT_DEPTH])
        entry_generation = int(tt_data[index, TT_GENERATION])
        stale_penalty = -1_000_000 if entry_generation != generation else 0
        exact_bonus = 2 if int(tt_data[index, TT_BOUND]) == TT_EXACT else 0
        priority = stale_penalty + entry_depth * 4 + exact_bonus
        if priority < replacement_priority:
            replacement = index
            replacement_priority = priority
    if collision:
        stats[STAT_TT_COLLISIONS] += 1
    old_depth = int(tt_data[replacement, TT_DEPTH])
    old_generation = int(tt_data[replacement, TT_GENERATION])
    if old_generation == generation and depth + 2 < old_depth:
        return -1
    stats[STAT_TT_REPLACEMENTS] += 1
    return replacement


@njit(cache=False)
def _negamax(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: NDArray[np.uint64],
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    allow_null: bool,
    history: NDArray[np.uint64],
    history_count: int,
    root_history_count: int,
    legal_stack: NDArray[np.int32],
    pseudo_stack: NDArray[np.int32],
    undo_stack: NDArray[np.int64],
    undo_key_stack: NDArray[np.uint64],
    accumulator_stack: NDArray[np.int32],
    score_stack: NDArray[np.int32],
    see_gain_stack: NDArray[np.int32],
    killers: NDArray[np.int32],
    quiet_history: NDArray[np.int32],
    capture_history: NDArray[np.int32],
    countermoves: NDArray[np.int32],
    move_stack: NDArray[np.int32],
    q_eval_keys: NDArray[np.uint64],
    q_eval_scores: NDArray[np.int32],
    q_eval_valid: NDArray[np.uint8],
    tt_keys: NDArray[np.uint64],
    tt_data: NDArray[np.int32],
    generation: int,
    excluded_move: int,
    stop: NDArray[np.uint8],
    node_limit: int,
    stats: NDArray[np.int64],
) -> tuple[int, bool]:
    if depth <= 0:
        return _quiescence(
            pieces,
            state,
            key,
            alpha,
            beta,
            ply,
            history,
            history_count,
            root_history_count,
            legal_stack,
            pseudo_stack,
            undo_stack,
            undo_key_stack,
            accumulator_stack,
            score_stack,
            see_gain_stack,
            killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
            q_eval_scores,
            q_eval_valid,
            stop,
            node_limit,
            stats,
        )
    if _visit_node(stats, stop, node_limit, False):
        return 0, True
    if ply >= MAX_PLY - 1 or history_count >= len(history):
        return evaluate(pieces, state, accumulator_stack[ply]), False

    side = int(state[engine.STATE_SIDE])
    in_check = engine.is_in_check(pieces, side)
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
        return (-MATE_SCORE + ply if in_check else 0), False
    if engine.has_rule_draw(
        pieces, state, key[0], history, history_count, root_history_count
    ):
        return 0, False

    original_alpha = alpha
    original_beta = beta
    tt_index = _tt_probe_index(key[0], tt_keys, tt_data)
    tt_move = 0
    tt_score = 0
    tt_bound = TT_EMPTY
    tt_depth = -1
    stats[STAT_TT_PROBES] += 1
    if excluded_move == 0 and tt_index >= 0:
        stats[STAT_TT_HITS] += 1
        tt_move = int(tt_data[tt_index, TT_MOVE])
        tt_score = _score_from_table(int(tt_data[tt_index, TT_SCORE]), ply)
        tt_bound = int(tt_data[tt_index, TT_BOUND])
        tt_depth = int(tt_data[tt_index, TT_DEPTH])
        if (
            tt_depth >= depth
            and int(tt_data[tt_index, TT_HALFMOVE]) == min(
                100, int(state[engine.STATE_HALFMOVE])
            )
        ):
            if tt_bound == TT_EXACT:
                return tt_score, False
            if tt_bound == TT_LOWER and tt_score > alpha:
                alpha = tt_score
            elif tt_bound == TT_UPPER and tt_score < beta:
                beta = tt_score
            if alpha >= beta:
                stats[STAT_TT_CUTOFFS] += 1
                return tt_score, False

    singular_move = 0
    if (
        ENABLE_SINGULAR_EXTENSIONS
        and excluded_move == 0
        and depth >= SINGULAR_MIN_DEPTH
        and tt_move != 0
        and tt_depth >= depth - SINGULAR_TT_DEPTH_SLACK
        and tt_bound in (TT_EXACT, TT_LOWER)
        and abs(tt_score) < MATE_BOUND
    ):
        stats[STAT_SINGULAR_ATTEMPTS] += 1
        singular_beta = tt_score - SINGULAR_MARGIN_PER_DEPTH * depth
        verification_depth = max(1, (depth - 1) // 2)
        verification_score, aborted = _negamax(
            pieces,
            state,
            key,
            verification_depth,
            singular_beta - 1,
            singular_beta,
            ply,
            False,
            history,
            history_count,
            root_history_count,
            legal_stack,
            pseudo_stack,
            undo_stack,
            undo_key_stack,
            accumulator_stack,
            score_stack,
            see_gain_stack,
            killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
            q_eval_scores,
            q_eval_valid,
            tt_keys,
            tt_data,
            generation,
            tt_move,
            stop,
            node_limit,
            stats,
        )
        if aborted:
            return 0, True
        if verification_score < singular_beta:
            singular_move = tt_move
            stats[STAT_SINGULAR_EXTENSIONS] += 1
        # The excluded search shares per-ply move buffers with this node.
        # Restore the current node's legal list before normal move ordering.
        count = engine.generate_legal_moves(
            pieces,
            state,
            key,
            legal_stack[ply],
            pseudo_stack[ply],
            undo_stack[ply],
            undo_key_stack[ply],
        )

    # Search V10: cache the exact static evaluation once when a selective
    # mechanism needs it. Reusing the existing exact q-eval cache avoids
    # paying KingNet repeatedly for transposed positions.
    static_eval = 0
    have_static_eval = False
    v10_selective_enabled = (
        ENABLE_V10_DYNAMIC_NMP
        or ENABLE_V10_REVERSE_FUTILITY
        or ENABLE_V10_LATE_MOVE_PRUNING
        or ENABLE_V10_QUIET_FUTILITY
        or ENABLE_V10_SEE_PRUNING
        or ENABLE_V10_CONTEXTUAL_LMR
        or ENABLE_HISTORY_V2
        or ENABLE_QUIET_SEE_PRUNING
    )
    # V10 classifies node type from the entry window, before a TT bound can
    # narrow a PV window to width one. The all-off profile deliberately keeps
    # the frozen champion's historical behaviour for exact parity diagnostics.
    non_pv = (
        original_beta - original_alpha == 1
        if v10_selective_enabled
        else beta - alpha == 1
    )
    null_move_candidate = (
        excluded_move == 0
        and allow_null
        and depth >= 3
        and not in_check
        and non_pv
        and abs(beta) < MATE_BOUND
        and _has_non_pawn_material(pieces, side)
    )
    if (
        not in_check
        and abs(beta) < MATE_BOUND
        and non_pv
        and (
            (
                ENABLE_V10_REVERSE_FUTILITY
                and excluded_move == 0
                and depth <= V10_RFP_MAX_DEPTH
            )
            or (ENABLE_V10_QUIET_FUTILITY and depth <= V10_QF_MAX_DEPTH)
            or (ENABLE_V10_DYNAMIC_NMP and null_move_candidate)
        )
    ):
        static_eval = _cached_qeval(
            pieces,
            state,
            key[0],
            accumulator_stack[ply],
            q_eval_keys,
            q_eval_scores,
            q_eval_valid,
            stats,
        )
        have_static_eval = True

    # Reverse futility pruning. Keep this deliberately shallow, non-PV, out
    # of check, and away from pawn-only endings where static evaluation is a
    # less trustworthy substitute for search.
    if (
        _reverse_futility_allowed(
            excluded_move,
            have_static_eval,
            depth,
            _has_non_pawn_material(pieces, side),
        )
    ):
        rfp_margin = V10_RFP_MARGIN_BASE + V10_RFP_MARGIN_PER_DEPTH * depth
        if static_eval - rfp_margin >= beta:
            return static_eval, False

    # Dynamic null-move pruning. The old engine always used R=2. V10 grows
    # the reduction with depth and with the amount by which static evaluation
    # exceeds beta, while retaining the existing zugzwang/material guards.
    if null_move_candidate:
        null_static_eval = (
            static_eval
            if ENABLE_V10_DYNAMIC_NMP and have_static_eval
            else evaluate(pieces, state, accumulator_stack[ply])
        )
    else:
        null_static_eval = 0
    if null_move_candidate and null_static_eval >= beta:
        reduction = 2
        if ENABLE_V10_DYNAMIC_NMP:
            eval_bonus = max(0, null_static_eval - beta) // V10_NMP_EVAL_DIVISOR
            eval_bonus = min(V10_NMP_MAX_EVAL_BONUS, eval_bonus)
            reduction += depth // 6 + eval_bonus
            reduction = min(depth - 1, max(2, reduction))
        null_depth = max(0, depth - 1 - reduction)

        for perspective in range(2):
            for column in range(nnue.ACCUMULATOR_ROW):
                accumulator_stack[ply + 1, perspective, column] = accumulator_stack[
                    ply, perspective, column
                ]
        engine.make_null_move(
            pieces, state, key, undo_stack[ply], undo_key_stack[ply]
        )
        move_stack[ply] = np.int32(0)
        null_score, aborted = _negamax(
            pieces,
            state,
            key,
            null_depth,
            -beta,
            -beta + 1,
            ply + 1,
            False,
            history,
            history_count,
            root_history_count,
            legal_stack,
            pseudo_stack,
            undo_stack,
            undo_key_stack,
            accumulator_stack,
            score_stack,
            see_gain_stack,
            killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
            q_eval_scores,
            q_eval_valid,
            tt_keys,
            tt_data,
            generation,
            0,
            stop,
            node_limit,
            stats,
        )
        engine.unmake_null_move(state, key, undo_stack[ply], undo_key_stack[ply])
        if aborted:
            return 0, True
        null_value = -null_score
        if null_value >= beta:
            # At high depth, verify the cutoff from the real position with null
            # move disabled. This protects against deep zugzwang / tactical
            # false positives without paying verification at ordinary depths.
            if ENABLE_V10_DYNAMIC_NMP and depth >= V10_NMP_VERIFY_DEPTH:
                verify_depth = max(1, depth - reduction)
                verify_score, aborted = _negamax(
                    pieces,
                    state,
                    key,
                    verify_depth,
                    beta - 1,
                    beta,
                    ply,
                    False,
                    history,
                    history_count,
                    root_history_count,
                    legal_stack,
                    pseudo_stack,
                    undo_stack,
                    undo_key_stack,
                    accumulator_stack,
                    score_stack,
                    see_gain_stack,
                    killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
                    q_eval_scores,
                    q_eval_valid,
                    tt_keys,
                    tt_data,
            generation,
            0,
            stop,
                    node_limit,
                    stats,
                )
                if aborted:
                    return 0, True
                if verify_score < beta:
                    null_value = verify_score
                else:
                    return beta if null_value >= MATE_BOUND else null_value, False
            else:
                # A mate seen after giving the opponent a free move is not a
                # mate score we can safely claim.
                return beta if null_value >= MATE_BOUND else null_value, False

    _order_moves(
        pieces,
        state,
        side,
        legal_stack[ply],
        count,
        tt_move,
        ply,
        killers,
        quiet_history,
        capture_history,
        countermoves,
        int(move_stack[ply - 1]) if ply > 0 else 0,
        score_stack[ply],
        see_gain_stack[ply],
    )
    if ENABLE_HISTORY_V2:
        # The ordering scores are dead after sorting. Reuse this per-ply row as
        # an exact record of quiet moves that were actually searched, excluding
        # moves skipped by a pruning rule.
        for index in range(count):
            score_stack[ply, index] = 0
    best = -INFINITY
    best_move = 0
    for index in range(count):
        move = int(legal_stack[ply, index])
        if move == excluded_move:
            continue
        flags = engine.move_flags(move)
        quiet = flags & (engine.FLAG_CAPTURE | engine.FLAG_PROMOTION) == 0
        quiet_hist = 0
        is_killer = False
        good_history_threshold = V10_GOOD_HISTORY_THRESHOLD
        if quiet:
            quiet_from = engine.move_from(move)
            quiet_to = engine.move_to(move)
            quiet_hist = int(quiet_history[side, quiet_from, quiet_to])
            if ply < MAX_PLY:
                is_killer = (
                    move == int(killers[ply, 0])
                    or move == int(killers[ply, 1])
                )
            if ENABLE_HISTORY_V2:
                good_history_threshold = HISTORY_V2_GOOD_THRESHOLD
        capture_see = 0
        if (
            ENABLE_V10_SEE_PRUNING
            and not quiet
            and flags & engine.FLAG_PROMOTION == 0
            and depth <= V10_SEE_MAX_DEPTH
            and index > 0
        ):
            # SEE must be evaluated on the pre-move position.
            capture_see = static_exchange_eval(
                pieces, state, move, see_gain_stack[ply]
            )
        quiet_see = 0
        if (
            ENABLE_QUIET_SEE_PRUNING
            and quiet
            and flags & engine.FLAG_CASTLING == 0
            and not in_check
            and not is_killer
            and quiet_hist < good_history_threshold
            and depth <= QUIET_SEE_MAX_DEPTH
            and index > 0
        ):
            # Like capture SEE, quiet SEE must inspect the pre-move position.
            # It measures whether the moved piece can survive on its destination.
            quiet_see = static_exchange_eval(
                pieces, state, move, see_gain_stack[ply]
            )
        nnue.refresh(pieces, accumulator_stack[ply])

        engine.make_move(
            pieces, state, key, move, undo_stack[ply], undo_key_stack[ply]
        )
        history[history_count] = key[0]
        move_stack[ply] = np.int32(move)
        gives_check = engine.is_in_check(pieces, int(state[engine.STATE_SIDE]))

        # Search V10 shallow move pruning. These rules apply only to late,
        # non-checking moves at non-PV nodes. They deliberately preserve the
        # first moves, checks, promotions, and all in-check evasions.
        prune_move = False
        if (
            excluded_move == 0
            and non_pv
            and not in_check
            and not gives_check
            and index > 0
        ):
            if quiet and excluded_move == 0:
                if ENABLE_V10_LATE_MOVE_PRUNING and depth <= V10_LMP_MAX_DEPTH:
                    # More moves survive as depth grows. With zero-based index,
                    # this begins pruning only after 4/8/14 ordered moves at
                    # depths 1/2/3 respectively.
                    lmp_limit = 2 + depth * depth + depth
                    if (
                        index >= lmp_limit
                        and not is_killer
                        and quiet_hist < good_history_threshold
                    ):
                        prune_move = True
                if (
                    not prune_move
                    and ENABLE_V10_QUIET_FUTILITY
                    and depth <= V10_QF_MAX_DEPTH
                    and have_static_eval
                    and index >= 2
                ):
                    qf_margin = V10_QF_MARGIN_BASE + V10_QF_MARGIN_PER_DEPTH * depth
                    if (
                        static_eval + qf_margin <= alpha
                        and not is_killer
                        and quiet_hist < good_history_threshold
                    ):
                        prune_move = True
                if (
                    not prune_move
                    and ENABLE_QUIET_SEE_PRUNING
                    and depth <= QUIET_SEE_MAX_DEPTH
                    and index > 0
                    and not is_killer
                    and quiet_hist < good_history_threshold
                    and quiet_see
                    < -QUIET_SEE_MARGIN_PER_DEPTH_SQUARED * depth * depth
                ):
                    prune_move = True
            elif (
                ENABLE_V10_SEE_PRUNING
                and depth <= V10_SEE_MAX_DEPTH
                and flags & engine.FLAG_PROMOTION == 0
            ):
                if capture_see < -V10_SEE_MARGIN_PER_DEPTH * depth:
                    prune_move = True

        if prune_move:
            engine.unmake_move(
                pieces, state, key, move, undo_stack[ply], undo_key_stack[ply]
            )
            continue

        # Same accumulator transition as update_for_move(), except the populated
        # undo row now tells us exactly which piece moved/captured.
        nnue.update_after_move(
            move,
            undo_stack[ply],
            accumulator_stack[ply],
            accumulator_stack[ply + 1],
        )

        reduction = 0
        lmr_start_index = 3 if ENABLE_V10_CONTEXTUAL_LMR else 4
        if (
            excluded_move == 0
            and depth >= 3
            and index >= lmr_start_index
            and quiet
            and not in_check
            and not gives_check
        ):
            reduction = int(
                LMR_TABLE[
                    min(depth, LMR_TABLE_MAX - 1),
                    min(index, LMR_TABLE_MAX - 1),
                ]
            )
            if ENABLE_V10_CONTEXTUAL_LMR:
                if not non_pv:
                    reduction -= 1
                if quiet_hist >= good_history_threshold:
                    reduction -= 1
                elif ENABLE_HISTORY_V2 and quiet_hist <= HISTORY_V2_BAD_THRESHOLD:
                    reduction += 1
                if is_killer:
                    reduction -= 1
            reduction = max(0, min(reduction, depth - 2))

        reduced = reduction > 0
        extension = 1 if move == singular_move else 0
        child_depth = depth - 1 + extension - reduction
        if reduced:
            stats[STAT_LMR_REDUCTIONS] += 1
        if index == 0:
            child_score, aborted = _negamax(
                pieces,
                state,
                key,
                child_depth,
                -beta,
                -alpha,
                ply + 1,
                True,
                history,
                history_count + 1,
                root_history_count,
                legal_stack,
                pseudo_stack,
                undo_stack,
                undo_key_stack,
                accumulator_stack,
                score_stack,
                see_gain_stack,
                killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
                q_eval_scores,
                q_eval_valid,
                tt_keys,
                tt_data,
            generation,
            0,
            stop,
                node_limit,
                stats,
            )
        else:
            child_score, aborted = _negamax(
                pieces,
                state,
                key,
                child_depth,
                -alpha - 1,
                -alpha,
                ply + 1,
                True,
                history,
                history_count + 1,
                root_history_count,
                legal_stack,
                pseudo_stack,
                undo_stack,
                undo_key_stack,
                accumulator_stack,
                score_stack,
                see_gain_stack,
                killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
                q_eval_scores,
                q_eval_valid,
                tt_keys,
                tt_data,
            generation,
            0,
            stop,
                node_limit,
                stats,
            )
            if not aborted and reduced and -child_score > alpha:
                stats[STAT_LMR_RESEARCHES] += 1
                child_score, aborted = _negamax(
                    pieces,
                    state,
                    key,
                    depth - 1 + extension,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    True,
                    history,
                    history_count + 1,
                    root_history_count,
                    legal_stack,
                    pseudo_stack,
                    undo_stack,
                    undo_key_stack,
                    accumulator_stack,
                    score_stack,
                    see_gain_stack,
                    killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
                    q_eval_scores,
                    q_eval_valid,
                    tt_keys,
                    tt_data,
            generation,
            0,
            stop,
                    node_limit,
                    stats,
                )
            if not aborted and -child_score > alpha and -child_score < beta:
                child_score, aborted = _negamax(
                    pieces,
                    state,
                    key,
                    depth - 1 + extension,
                    -beta,
                    -alpha,
                    ply + 1,
                    True,
                    history,
                    history_count + 1,
                    root_history_count,
                    legal_stack,
                    pseudo_stack,
                    undo_stack,
                    undo_key_stack,
                    accumulator_stack,
                    score_stack,
                    see_gain_stack,
                    killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
                    q_eval_scores,
                    q_eval_valid,
                    tt_keys,
                    tt_data,
            generation,
            0,
            stop,
                    node_limit,
                    stats,
                )
        engine.unmake_move(
            pieces, state, key, move, undo_stack[ply], undo_key_stack[ply]
        )
        if aborted:
            return 0, True
        if ENABLE_HISTORY_V2 and quiet:
            score_stack[ply, index] = 1
        elif (
            ENABLE_CAPTURE_HISTORY
            and flags & engine.FLAG_CAPTURE
            and flags & engine.FLAG_PROMOTION == 0
        ):
            score_stack[ply, index] = 2
        score = -child_score
        if score > best:
            best = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            stats[STAT_BETA_CUTOFFS] += 1
            if quiet and excluded_move == 0:
                _record_quiet_cutoff(
                    side,
                    move,
                    ply,
                    depth,
                    killers,
                    quiet_history,
                    countermoves,
                    int(move_stack[ply - 1]) if ply > 0 else 0,
                )
                if ENABLE_HISTORY_V2:
                    _record_quiet_failures(
                        side,
                        depth,
                        legal_stack[ply],
                        score_stack[ply],
                        index,
                        quiet_history,
                    )
            elif (
                ENABLE_CAPTURE_HISTORY
                and excluded_move == 0
                and flags & engine.FLAG_CAPTURE
                and flags & engine.FLAG_PROMOTION == 0
            ):
                _record_capture_cutoff(
                    pieces,
                    side,
                    move,
                    depth,
                    legal_stack[ply],
                    score_stack[ply],
                    index,
                    capture_history,
                    stats,
                )
            break

    bound = TT_UPPER if best <= original_alpha else TT_LOWER if best >= original_beta else TT_EXACT
    tt_index = (
        _tt_replacement_index(key[0], depth, generation, tt_keys, tt_data, stats)
        if excluded_move == 0
        else -1
    )
    if tt_index >= 0:
        tt_keys[tt_index] = key[0]
        tt_data[tt_index, TT_MOVE] = np.int32(best_move)
        tt_data[tt_index, TT_SCORE] = np.int32(_score_to_table(best, ply))
        tt_data[tt_index, TT_DEPTH] = np.int32(depth)
        tt_data[tt_index, TT_BOUND] = np.int32(bound)
        tt_data[tt_index, TT_GENERATION] = np.int32(generation)
        tt_data[tt_index, TT_HALFMOVE] = np.int32(
            min(100, int(state[engine.STATE_HALFMOVE]))
        )
    return best, False


@njit(cache=False, nogil=True)
def _search_root(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: NDArray[np.uint64],
    depth: int,
    alpha: int,
    beta: int,
    preferred_move: int,
    history: NDArray[np.uint64],
    history_count: int,
    root_history_count: int,
    legal_stack: NDArray[np.int32],
    pseudo_stack: NDArray[np.int32],
    undo_stack: NDArray[np.int64],
    undo_key_stack: NDArray[np.uint64],
    accumulator_stack: NDArray[np.int32],
    score_stack: NDArray[np.int32],
    see_gain_stack: NDArray[np.int32],
    killers: NDArray[np.int32],
    quiet_history: NDArray[np.int32],
    capture_history: NDArray[np.int32],
    countermoves: NDArray[np.int32],
    move_stack: NDArray[np.int32],
    q_eval_keys: NDArray[np.uint64],
    q_eval_scores: NDArray[np.int32],
    q_eval_valid: NDArray[np.uint8],
    tt_keys: NDArray[np.uint64],
    tt_data: NDArray[np.int32],
    generation: int,
    stop: NDArray[np.uint8],
    node_limit: int,
    stats: NDArray[np.int64],
) -> tuple[int, int, bool, int]:
    if _visit_node(stats, stop, node_limit, False):
        return 0, preferred_move, True, 0
    side = int(state[engine.STATE_SIDE])
    root_in_check = engine.is_in_check(pieces, side)
    count = engine.generate_legal_moves(
        pieces,
        state,
        key,
        legal_stack[0],
        pseudo_stack[0],
        undo_stack[0],
        undo_key_stack[0],
    )
    if count == 0:
        score = -MATE_SCORE if engine.is_in_check(pieces, side) else 0
        return score, 0, False, 0

    tt_index = _tt_probe_index(key[0], tt_keys, tt_data)
    tt_move = preferred_move
    if tt_index >= 0 and int(tt_data[tt_index, TT_MOVE]) != 0:
        tt_move = int(tt_data[tt_index, TT_MOVE])
    _order_moves(
        pieces,
        state,
        side,
        legal_stack[0],
        count,
        tt_move,
        0,
        killers,
        quiet_history,
        capture_history,
        countermoves,
        0,
        score_stack[0],
        see_gain_stack[0],
    )

    best = -INFINITY
    best_move = preferred_move if preferred_move != 0 else int(legal_stack[0, 0])
    improved_move = 0
    for index in range(count):
        move = int(legal_stack[0, index])
        flags = engine.move_flags(move)
        quiet = flags & (engine.FLAG_CAPTURE | engine.FLAG_PROMOTION) == 0
        quiet_hist = 0
        if quiet:
            quiet_hist = int(
                quiet_history[
                    side,
                    engine.move_from(move),
                    engine.move_to(move),
                ]
            )
        nnue.update_for_move(
            pieces,
            state,
            move,
            accumulator_stack[0],
            accumulator_stack[1],
        )
        engine.make_move(pieces, state, key, move, undo_stack[0], undo_key_stack[0])
        move_stack[0] = np.int32(move)
        history[history_count] = key[0]
        gives_check = engine.is_in_check(pieces, int(state[engine.STATE_SIDE]))
        good_root_history = (
            HISTORY_V2_GOOD_THRESHOLD
            if ENABLE_HISTORY_V2
            else V10_GOOD_HISTORY_THRESHOLD
        )
        root_reduction = 0
        if (
            index > 0
            and depth >= ROOT_LMR_MIN_DEPTH
            and index >= ROOT_LMR_MIN_INDEX
            and not root_in_check
            and quiet
            and flags & engine.FLAG_CASTLING == 0
            and not gives_check
            and quiet_hist < good_root_history
        ):
            root_reduction = 1
            if depth >= ROOT_LMR_DEEP_DEPTH and index >= ROOT_LMR_DEEP_INDEX:
                root_reduction = 2
            root_reduction = min(root_reduction, max(0, depth - 2))
        child_depth = depth - 1 - root_reduction
        if index == 0:
            child_score, aborted = _negamax(
                pieces,
                state,
                key,
                child_depth,
                -beta,
                -alpha,
                1,
                True,
                history,
                history_count + 1,
                root_history_count,
                legal_stack,
                pseudo_stack,
                undo_stack,
                undo_key_stack,
                accumulator_stack,
                score_stack,
                see_gain_stack,
                killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
                q_eval_scores,
                q_eval_valid,
                tt_keys,
                tt_data,
            generation,
            0,
            stop,
                node_limit,
                stats,
            )
        else:
            child_score, aborted = _negamax(
                pieces,
                state,
                key,
                child_depth,
                -alpha - 1,
                -alpha,
                1,
                True,
                history,
                history_count + 1,
                root_history_count,
                legal_stack,
                pseudo_stack,
                undo_stack,
                undo_key_stack,
                accumulator_stack,
                score_stack,
                see_gain_stack,
                killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
                q_eval_scores,
                q_eval_valid,
                tt_keys,
                tt_data,
            generation,
            0,
            stop,
                node_limit,
                stats,
            )
            if (
                not aborted
                and root_reduction > 0
                and -child_score > alpha
            ):
                child_score, aborted = _negamax(
                    pieces,
                    state,
                    key,
                    depth - 1,
                    -alpha - 1,
                    -alpha,
                    1,
                    True,
                    history,
                    history_count + 1,
                    root_history_count,
                    legal_stack,
                    pseudo_stack,
                    undo_stack,
                    undo_key_stack,
                    accumulator_stack,
                    score_stack,
                    see_gain_stack,
                    killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
                    q_eval_scores,
                    q_eval_valid,
                    tt_keys,
                    tt_data,
            generation,
            0,
            stop,
                    node_limit,
                    stats,
                )
            if not aborted and -child_score > alpha and -child_score < beta:
                child_score, aborted = _negamax(
                    pieces,
                    state,
                    key,
                    depth - 1,
                    -beta,
                    -alpha,
                    1,
                    True,
                    history,
                    history_count + 1,
                    root_history_count,
                    legal_stack,
                    pseudo_stack,
                    undo_stack,
                    undo_key_stack,
                    accumulator_stack,
                    score_stack,
                    see_gain_stack,
                    killers,
            quiet_history,
            capture_history,
            countermoves,
            move_stack,
            q_eval_keys,
                    q_eval_scores,
                    q_eval_valid,
                    tt_keys,
                    tt_data,
            generation,
            0,
            stop,
                    node_limit,
                    stats,
                )
        engine.unmake_move(pieces, state, key, move, undo_stack[0], undo_key_stack[0])
        if aborted:
            return 0, best_move, True, improved_move
        score = -child_score
        if score > best:
            best = score
            best_move = move
        if score > alpha:
            alpha = score
            # only a raised alpha has full-window score behind it
            if index > 0:
                improved_move = move
        if alpha >= beta:
            break
    return best, best_move, False, 0


def _history_buffer(
    position: engine.Position, prior_history: NDArray[np.uint64] | None
) -> tuple[NDArray[np.uint64], int]:
    history = np.zeros(MAX_HISTORY, dtype=np.uint64)
    if prior_history is None or len(prior_history) == 0:
        history[0] = position.key[0]
        return history, 1
    retained = prior_history[-(MAX_HISTORY - MAX_PLY) :]
    count = len(retained)
    history[:count] = retained
    if history[count - 1] != position.key[0]:
        history[count] = position.key[0]
        count += 1
    return history, count


def _should_stop_after_iteration(
    *,
    elapsed_s: float,
    soft_limit_s: float,
    normal_limit_s: float,
    hard_limit_s: float,
    stable_iterations: int,
    unstable: bool,
    last_iteration_s: float,
) -> bool:
    """Stop only between completed iterations while preserving the hard limit."""
    if (
        elapsed_s >= soft_limit_s
        and stable_iterations >= STABLE_ITERATIONS_REQUIRED
    ):
        return True
    if elapsed_s >= normal_limit_s and not unstable:
        return True
    predicted_next_s = last_iteration_s * NEXT_ITERATION_COST_MULTIPLIER
    return hard_limit_s - elapsed_s <= predicted_next_s


def search_position(
    position: engine.Position,
    memory: SearchMemory,
    *,
    soft_time_limit_s: float | None = None,
    normal_time_limit_s: float | None = None,
    time_limit_s: float | None = None,
    node_limit: int = 0,
    max_depth: int = MAX_DEPTH,
    prior_history: NDArray[np.uint64] | None = None,
) -> SearchResult:
    if time_limit_s is None and node_limit <= 0:
        raise ValueError("a positive time or node limit is required")
    if time_limit_s is not None and time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive")
    if (soft_time_limit_s is None) != (normal_time_limit_s is None):
        raise ValueError("soft and normal time limits must be provided together")
    if soft_time_limit_s is not None and normal_time_limit_s is not None:
        if time_limit_s is None:
            raise ValueError("adaptive time limits require a hard time limit")
        if not 0 < soft_time_limit_s <= normal_time_limit_s <= time_limit_s:
            raise ValueError("time limits must satisfy 0 < soft <= normal <= hard")
    if not 1 <= max_depth <= MAX_DEPTH:
        raise ValueError(f"max_depth must be between 1 and {MAX_DEPTH}")

    working = position.copy()
    root_moves = engine.legal_moves(working)
    if len(root_moves) == 0:
        return SearchResult(0, 0, 0, 0, 0, 0.0, False, 0, 0, 0, 0, 0, 0, 0)
    fallback = int(root_moves[0])
    if len(root_moves) == 1:
        return SearchResult(fallback, 0, 0, 0, 0, 0.0, False, 0, 0, 0, 0, 0, 0, 0)

    history, history_count = _history_buffer(working, prior_history)
    root_history_count = history_count
    legal_stack = np.empty((MAX_PLY, engine.MAX_MOVES), dtype=np.int32)
    pseudo_stack = np.empty((MAX_PLY, engine.MAX_MOVES), dtype=np.int32)
    undo_stack = np.empty((MAX_PLY, engine.UNDO_SIZE), dtype=np.int64)
    undo_key_stack = np.empty((MAX_PLY, 1), dtype=np.uint64)
    accumulator_stack = np.empty(
        (MAX_PLY, 2, nnue.ACCUMULATOR_ROW), dtype=np.int32
    )
    nnue.rebuild(working.pieces, accumulator_stack[0])
    score_stack = np.empty((MAX_PLY, engine.MAX_MOVES), dtype=np.int32)
    see_gain_stack = np.empty((MAX_PLY, SEE_MAX_EXCHANGES), dtype=np.int32)
    killers = np.zeros((MAX_PLY, 2), dtype=np.int32)
    move_stack = np.zeros(MAX_PLY, dtype=np.int32)
    stop = np.zeros(1, dtype=np.uint8)
    stats = np.zeros(STAT_COUNT, dtype=np.int64)
    generation = memory.next_generation()

    started = time.perf_counter()
    timer: threading.Timer | None = None
    if time_limit_s is not None:
        timer = threading.Timer(time_limit_s, stop.__setitem__, args=(0, np.uint8(1)))
        timer.daemon = True
        timer.start()

    best_move = fallback
    best_score = 0
    completed_depth = 0
    stopped = False
    stable_iterations = 0
    last_iteration_s = 0.0
    try:
        for depth in range(1, max_depth + 1):
            if time_limit_s is not None and time.perf_counter() - started >= time_limit_s:
                stopped = True
                break
            iteration_started = time.perf_counter()
            window = 45
            alpha = -INFINITY if depth <= 2 else best_score - window
            beta = INFINITY if depth <= 2 else best_score + window
            score, move, aborted, _partial_move = _search_root(
                working.pieces,
                working.state,
                working.key,
                depth,
                alpha,
                beta,
                best_move,
                history,
                history_count,
                root_history_count,
                legal_stack,
                pseudo_stack,
                undo_stack,
                undo_key_stack,
                accumulator_stack,
                score_stack,
                see_gain_stack,
                killers,
                memory.quiet_history,
                memory.capture_history,
                memory.countermoves,
                move_stack,
                memory.q_eval_keys,
                memory.q_eval_scores,
                memory.q_eval_valid,
                memory.tt_keys,
                memory.tt_data,
                generation,
                stop,
                node_limit,
                stats,
            )
            if aborted:
                # Keep only a fully completed deeper root move that already
                # raised alpha; otherwise retain the last completed-depth move.
                if _partial_move != 0:
                    best_move = _partial_move
                stopped = True
                break
            aspiration_failed = score <= alpha or score >= beta
            if aspiration_failed:
                score, move, aborted, _partial_move = _search_root(
                    working.pieces,
                    working.state,
                    working.key,
                    depth,
                    -INFINITY,
                    INFINITY,
                    best_move,
                    history,
                    history_count,
                    root_history_count,
                    legal_stack,
                    pseudo_stack,
                    undo_stack,
                    undo_key_stack,
                    accumulator_stack,
                    score_stack,
                    see_gain_stack,
                    killers,
                    memory.quiet_history,
                    memory.capture_history,
                    memory.countermoves,
                    move_stack,
                    memory.q_eval_keys,
                    memory.q_eval_scores,
                    memory.q_eval_valid,
                    memory.tt_keys,
                    memory.tt_data,
                    generation,
                    stop,
                    node_limit,
                    stats,
                )
                if aborted:
                    if _partial_move != 0:
                        best_move = _partial_move
                    stopped = True
                    break
            previous_move = best_move
            previous_score = best_score
            had_previous_iteration = completed_depth > 0
            move_changed = had_previous_iteration and move != previous_move
            score_delta = abs(score - previous_score) if had_previous_iteration else 0
            if (
                had_previous_iteration
                and not move_changed
                and score_delta <= STABLE_SCORE_DELTA
                and not aspiration_failed
            ):
                stable_iterations += 1
            else:
                stable_iterations = 0
            best_move = move
            best_score = score
            completed_depth = depth
            last_iteration_s = time.perf_counter() - iteration_started
            if abs(score) >= MATE_BOUND:
                break
            if (
                soft_time_limit_s is not None
                and normal_time_limit_s is not None
                and time_limit_s is not None
            ):
                elapsed_s = time.perf_counter() - started
                unstable = (
                    move_changed
                    or score_delta >= UNSTABLE_SCORE_DELTA
                    or aspiration_failed
                )
                if _should_stop_after_iteration(
                    elapsed_s=elapsed_s,
                    soft_limit_s=soft_time_limit_s,
                    normal_limit_s=normal_time_limit_s,
                    hard_limit_s=time_limit_s,
                    stable_iterations=stable_iterations,
                    unstable=unstable,
                    last_iteration_s=last_iteration_s,
                ):
                    break
    finally:
        stop[0] = np.uint8(1)
        if timer is not None:
            timer.cancel()
            timer.join()

    elapsed_s = time.perf_counter() - started
    return SearchResult(
        best_move,
        best_score,
        completed_depth,
        int(stats[STAT_NODES]),
        int(stats[STAT_QNODES]),
        elapsed_s,
        stopped,
        int(stats[STAT_TT_HITS]),
        int(stats[STAT_TT_CUTOFFS]),
        int(stats[STAT_BETA_CUTOFFS]),
        int(stats[STAT_LMR_REDUCTIONS]),
        int(stats[STAT_LMR_RESEARCHES]),
        int(stats[STAT_Q_EVAL_PROBES]),
        int(stats[STAT_Q_EVAL_HITS]),
        int(stats[STAT_TT_COLLISIONS]),
        int(stats[STAT_TT_REPLACEMENTS]),
        int(stats[STAT_CAPTURE_HISTORY_UPDATES]),
        int(stats[STAT_SINGULAR_ATTEMPTS]),
        int(stats[STAT_SINGULAR_EXTENSIONS]),
    )


def available_profiles() -> tuple[str, ...]:
    """Return the compile-time profiles supported by this challenger."""
    return tuple(_S1_PROFILES)


def configure_experiment(name: str) -> None:
    """Select an S1 profile before Numba compiles the recursive search."""
    global ACTIVE_PROFILE
    global ENABLE_V10_CONTEXTUAL_LMR
    global ENABLE_V10_DYNAMIC_NMP
    global ENABLE_V10_LATE_MOVE_PRUNING
    global ENABLE_V10_QUIET_FUTILITY
    global ENABLE_V10_REVERSE_FUTILITY
    global ENABLE_V10_SEE_PRUNING
    global ENABLE_HISTORY_V2
    global ENABLE_QUIET_SEE_PRUNING

    if name not in _S1_PROFILES:
        choices = ", ".join(available_profiles())
        raise ValueError(f"unknown search profile {name!r}; choose from {choices}")
    if _negamax.signatures or _quiescence.signatures:
        raise RuntimeError("search profile must be selected before warmup")
    (
        ENABLE_V10_DYNAMIC_NMP,
        ENABLE_V10_REVERSE_FUTILITY,
        ENABLE_V10_LATE_MOVE_PRUNING,
        ENABLE_V10_QUIET_FUTILITY,
        ENABLE_V10_SEE_PRUNING,
        ENABLE_V10_CONTEXTUAL_LMR,
        ENABLE_HISTORY_V2,
        ENABLE_QUIET_SEE_PRUNING,
    ) = _S1_PROFILES[name]
    ACTIVE_PROFILE = name


def warmup() -> None:
    """Compile all board, evaluation, and search signatures before the clock."""
    engine.warmup()
    memory = SearchMemory.create(10)
    position = engine.position_from_board(chess.Board())
    search_position(position, memory, node_limit=256, max_depth=2)
