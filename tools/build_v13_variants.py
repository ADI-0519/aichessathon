#!/usr/bin/env python3
# Build aggressive V13 candidates from challengers/exp_release_v12.
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "challengers" / "exp_release_v12"
CORE = ROOT / "challengers" / "exp_release_v13_core"
SEARCH = ROOT / "challengers" / "exp_release_v13_search"
NUCLEAR = ROOT / "challengers" / "exp_release_v13_nuclear"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def fresh_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


MATERIAL_TIME_MANAGER = """\
# Material-aware adaptive clock allocation for V13.
from __future__ import annotations

from dataclasses import dataclass

MAX_NORMAL_BUDGET_MS = 4_500
MAX_HARD_BUDGET_MS = 8_000
MIN_RESERVE_MS = 250
MAX_RESERVE_MS = 2_000
SOFT_BUDGET_NUMERATOR = 3
SOFT_BUDGET_DENOMINATOR = 4
HARD_BUDGET_NUMERATOR = 9
HARD_BUDGET_DENOMINATOR = 5


@dataclass(frozen=True, slots=True)
class MoveTimeLimits:
    soft_ms: int
    normal_ms: int
    hard_ms: int


def _move_number_horizon(fullmove_number: int) -> int:
    move_number = max(1, fullmove_number)
    return max(12, min(32, 32 - (2 * move_number) // 5))


def _material_horizon(piece_count: int) -> int:
    pieces = max(2, min(32, piece_count))
    if pieces >= 26:
        return 40
    if pieces >= 20:
        return 34
    if pieces >= 14:
        return 32
    if pieces >= 8:
        return 24
    return 10


def estimated_moves_remaining(fullmove_number: int, piece_count: int = 32) -> int:
    material = _material_horizon(piece_count)
    move_number = _move_number_horizon(fullmove_number)
    if piece_count >= 14:
        return max(material, move_number)
    if fullmove_number < 35:
        return max(material, move_number)
    return material


def move_time_limits(
    time_left_ms: int,
    fullmove_number: int,
    piece_count: int = 32,
) -> MoveTimeLimits:
    if time_left_ms <= 0:
        return MoveTimeLimits(0, 0, 0)

    reserve_ms = max(
        MIN_RESERVE_MS,
        min(MAX_RESERVE_MS, time_left_ms // 16),
    )
    usable_ms = max(0, time_left_ms - reserve_ms)
    if usable_ms == 0:
        return MoveTimeLimits(0, 0, 0)

    moves_remaining = estimated_moves_remaining(fullmove_number, piece_count)
    increment_credit_ms = 350 * time_left_ms // (time_left_ms + 5_000)
    target_ms = usable_ms // moves_remaining + increment_credit_ms

    normal_ms = max(1, min(MAX_NORMAL_BUDGET_MS, usable_ms, target_ms))
    soft_ms = max(
        1,
        normal_ms * SOFT_BUDGET_NUMERATOR // SOFT_BUDGET_DENOMINATOR,
    )
    hard_ms = min(
        MAX_HARD_BUDGET_MS,
        usable_ms,
        normal_ms * HARD_BUDGET_NUMERATOR // HARD_BUDGET_DENOMINATOR,
    )
    return MoveTimeLimits(soft_ms, normal_ms, max(normal_ms, hard_ms))


def move_budget_ms(
    time_left_ms: int,
    fullmove_number: int,
    piece_count: int = 32,
) -> int:
    return move_time_limits(
        time_left_ms,
        fullmove_number,
        piece_count,
    ).normal_ms
"""


def patch_material_clock(candidate: Path) -> None:
    (candidate / "time_manager.py").write_text(MATERIAL_TIME_MANAGER, encoding="utf-8")
    agent_path = candidate / "agent.py"
    text = agent_path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "        limits = _move_time_limits(time_left_ms, board.fullmove_number)\n",
        "        limits = _move_time_limits(\n"
        "            time_left_ms,\n"
        "            board.fullmove_number,\n"
        "            int(board.occupied).bit_count(),\n"
        "        )\n",
        label="agent material-aware time call",
    )
    agent_path.write_text(text, encoding="utf-8")


def patch_tt20_and_partial_root(candidate: Path) -> None:
    path = candidate / "search.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "DEFAULT_TT_BITS = 18",
        "DEFAULT_TT_BITS = 20",
        label="TT20",
    )

    old = """            if aborted:
                # A root move from an interrupted iteration has not been
                # compared against every legal alternative.  Keep the result
                # from the last fully completed depth instead.
                stopped = True
                break
"""
    new = """            if aborted:
                # Keep only a fully completed deeper root move that already
                # raised alpha; otherwise retain the last completed-depth move.
                if _partial_move != 0:
                    best_move = _partial_move
                stopped = True
                break
"""
    text = replace_once(text, old, new, label="partial root recovery")

    old_retry = """                if aborted:
                    stopped = True
                    break
            previous_move = best_move
"""
    new_retry = """                if aborted:
                    if _partial_move != 0:
                        best_move = _partial_move
                    stopped = True
                    break
            previous_move = best_move
"""
    text = replace_once(
        text,
        old_retry,
        new_retry,
        label="partial root recovery after aspiration retry",
    )
    path.write_text(text, encoding="utf-8")


def patch_log_lmr(candidate: Path) -> None:
    path = candidate / "search.py"
    text = path.read_text(encoding="utf-8")

    anchor = "Q_EVAL_BITS = 16\n"
    addition = """Q_EVAL_BITS = 16

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

"""
    text = replace_once(text, anchor, addition, label="LMR table insertion")

    start_token = (
        "        reduction = 0\n"
        "        lmr_start_index = 3 if ENABLE_V10_CONTEXTUAL_LMR else 4\n"
    )
    end_token = "        reduced = reduction > 0\n"
    start = text.find(start_token)
    if start == -1:
        raise RuntimeError("interior LMR block start not found")
    end = text.find(end_token, start)
    if end == -1:
        raise RuntimeError("interior LMR block end not found")

    replacement = """        reduction = 0
        lmr_start_index = 3 if ENABLE_V10_CONTEXTUAL_LMR else 4
        if (
            depth >= 3
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

"""
    text = text[:start] + replacement + text[end:]
    path.write_text(text, encoding="utf-8")


def patch_root_lmr(candidate: Path) -> None:
    path = candidate / "search.py"
    text = path.read_text(encoding="utf-8")

    constants_anchor = "LMR_TABLE[_lmr_depth, _lmr_index] = max(1, _base_reduction)\n\n"
    constants = """LMR_TABLE[_lmr_depth, _lmr_index] = max(1, _base_reduction)

ROOT_LMR_MIN_DEPTH = 6
ROOT_LMR_MIN_INDEX = 4
ROOT_LMR_DEEP_DEPTH = 9
ROOT_LMR_DEEP_INDEX = 10

"""
    text = replace_once(text, constants_anchor, constants, label="root LMR constants")

    root_start = text.find("@njit(cache=False, nogil=True)\ndef _search_root(")
    root_end = text.find("\ndef _history_buffer(", root_start)
    if root_start == -1 or root_end == -1:
        raise RuntimeError("_search_root region not found")
    root = text[root_start:root_end]

    root = replace_once(
        root,
        "    side = int(state[engine.STATE_SIDE])\n"
        "    count = engine.generate_legal_moves(\n",
        "    side = int(state[engine.STATE_SIDE])\n"
        "    root_in_check = engine.is_in_check(pieces, side)\n"
        "    count = engine.generate_legal_moves(\n",
        label="root in-check state",
    )

    root = replace_once(
        root,
        "    for index in range(count):\n"
        "        move = int(legal_stack[0, index])\n"
        "        nnue.update_for_move(\n",
        "    for index in range(count):\n"
        "        move = int(legal_stack[0, index])\n"
        "        flags = engine.move_flags(move)\n"
        "        quiet = flags & (engine.FLAG_CAPTURE | engine.FLAG_PROMOTION) == 0\n"
        "        quiet_hist = 0\n"
        "        if quiet:\n"
        "            quiet_hist = int(\n"
        "                quiet_history[\n"
        "                    side,\n"
        "                    engine.move_from(move),\n"
        "                    engine.move_to(move),\n"
        "                ]\n"
        "            )\n"
        "        nnue.update_for_move(\n",
        label="root move classification",
    )

    root = replace_once(
        root,
        "        engine.make_move(pieces, state, key, move, undo_stack[0], undo_key_stack[0])\n"
        "        history[history_count] = key[0]\n"
        "        if index == 0:\n",
        "        engine.make_move(pieces, state, key, move, undo_stack[0], undo_key_stack[0])\n"
        "        history[history_count] = key[0]\n"
        "        gives_check = engine.is_in_check(pieces, int(state[engine.STATE_SIDE]))\n"
        "        good_root_history = (\n"
        "            HISTORY_V2_GOOD_THRESHOLD\n"
        "            if ENABLE_HISTORY_V2\n"
        "            else V10_GOOD_HISTORY_THRESHOLD\n"
        "        )\n"
        "        root_reduction = 0\n"
        "        if (\n"
        "            index > 0\n"
        "            and depth >= ROOT_LMR_MIN_DEPTH\n"
        "            and index >= ROOT_LMR_MIN_INDEX\n"
        "            and not root_in_check\n"
        "            and quiet\n"
        "            and flags & engine.FLAG_CASTLING == 0\n"
        "            and not gives_check\n"
        "            and quiet_hist < good_root_history\n"
        "        ):\n"
        "            root_reduction = 1\n"
        "            if depth >= ROOT_LMR_DEEP_DEPTH and index >= ROOT_LMR_DEEP_INDEX:\n"
        "                root_reduction = 2\n"
        "            root_reduction = min(root_reduction, max(0, depth - 2))\n"
        "        child_depth = depth - 1 - root_reduction\n"
        "        if index == 0:\n",
        label="root reduction decision",
    )

    root = replace_once(
        root,
        "                depth - 1,\n"
        "                -beta,\n"
        "                -alpha,\n",
        "                child_depth,\n"
        "                -beta,\n"
        "                -alpha,\n",
        label="root first child depth",
    )
    root = replace_once(
        root,
        "                depth - 1,\n"
        "                -alpha - 1,\n"
        "                -alpha,\n",
        "                child_depth,\n"
        "                -alpha - 1,\n"
        "                -alpha,\n",
        label="root reduced child depth",
    )

    marker = "            if not aborted and -child_score > alpha and -child_score < beta:\n"
    if root.count(marker) != 1:
        raise RuntimeError(f"root PVS marker expected once, found {root.count(marker)}")

    verification = """            if (
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
                    q_eval_keys,
                    q_eval_scores,
                    q_eval_valid,
                    tt_keys,
                    tt_data,
                    generation,
                    stop,
                    node_limit,
                    stats,
                )
            if not aborted and -child_score > alpha and -child_score < beta:
"""
    root = root.replace(marker, verification, 1)

    text = text[:root_start] + root + text[root_end:]
    path.write_text(text, encoding="utf-8")


def patch_nuclear_rfp(candidate: Path) -> None:
    path = candidate / "search.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "V10_RFP_MAX_DEPTH = 4",
        "V10_RFP_MAX_DEPTH = 6",
        label="nuclear RFP depth",
    )
    text = replace_once(
        text,
        "V10_RFP_MARGIN_BASE = 80",
        "V10_RFP_MARGIN_BASE = 40",
        label="nuclear RFP base",
    )
    text = replace_once(
        text,
        "V10_RFP_MARGIN_PER_DEPTH = 95",
        "V10_RFP_MARGIN_PER_DEPTH = 80",
        label="nuclear RFP slope",
    )
    path.write_text(text, encoding="utf-8")


def write_readme(candidate: Path, title: str, changes: list[str]) -> None:
    lines = [f"# {title}", "", "Forked from `exp_release_v12`.", "", "Adds:"]
    lines.extend(f"- {change}" for change in changes)
    lines.extend(
        [
            "",
            "KingNet75, stale-accumulator repair, q-eval caching, post-pruning "
            "qsearch updates, HISTORY_V2 and quiet-SEE remain inherited from V12.",
            "",
        ]
    )
    (candidate / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not SOURCE.is_dir():
        raise SystemExit(f"missing {SOURCE}; run after placing this file in tools/")

    fresh_copy(SOURCE, CORE)
    patch_material_clock(CORE)
    patch_tt20_and_partial_root(CORE)
    write_readme(
        CORE,
        "V13 core",
        [
            "material-aware time horizon with V12's existing hard cap",
            "TT size 2^20",
            "partial-root recovery for fully completed alpha-raising moves",
        ],
    )

    fresh_copy(CORE, SEARCH)
    patch_log_lmr(SEARCH)
    patch_root_lmr(SEARCH)
    write_readme(
        SEARCH,
        "V13 search",
        [
            "all V13-core changes",
            "log-log interior LMR with V12 context corrections",
            "guarded root LMR with full-depth verification",
        ],
    )

    fresh_copy(SEARCH, NUCLEAR)
    patch_nuclear_rfp(NUCLEAR)
    write_readme(
        NUCLEAR,
        "V13 nuclear",
        [
            "all V13-search changes",
            "RFP extended through depth 6 with 40 + 80*d cp margin",
        ],
    )

    print("Created:")
    print(f"  {CORE.relative_to(ROOT)}")
    print(f"  {SEARCH.relative_to(ROOT)}")
    print(f"  {NUCLEAR.relative_to(ROOT)}")
    print("Unchanged:")
    print("  current/")
    print("  challengers/exp_release_v12/")


if __name__ == "__main__":
    main()
