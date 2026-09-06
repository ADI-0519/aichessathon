from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import torch
from numpy.typing import NDArray

import search
from tools import eval_features as features

PHASE_BUCKETS = ((0, 8, "endgame"), (9, 17, "middlegame"), (18, 24, "opening"))

Dataset = dict[str, NDArray[np.int64]]


def _load(source: Path, cache: Path | None) -> Dataset:
    if cache is not None and cache.exists():
        with np.load(cache) as stored:
            return {name: stored[name] for name in stored.files}

    rows, phases, sides, labels, games = [], [], [], [], []
    for line in source.open():
        record = json.loads(line)
        board = chess.Board(record["fen"])
        vector, phase = features.extract(board)
        rows.append(vector)
        phases.append(min(phase, search.MAX_PHASE))
        sides.append(1 if board.turn == chess.WHITE else -1)
        labels.append(record["cp"])
        games.append(record["game"])
    data = {
        "features": np.array(rows, dtype=np.int16),
        "phase": np.array(phases, dtype=np.int64),
        "side": np.array(sides, dtype=np.int64),
        "label": np.array(labels, dtype=np.int64),
        "game": np.array(games, dtype=np.int64),
    }
    if cache is not None:
        np.savez_compressed(cache, **data)
    return data


def _scores(
    data: Dataset,
    middlegame: NDArray[np.int64],
    endgame: NDArray[np.int64],
    tempo: int,
) -> NDArray[np.int64]:
    # white-relative and integer, matching evaluate() before its final flip
    vectors = data["features"].astype(np.int64)
    phase = data["phase"]
    tapered = vectors @ middlegame * phase + vectors @ endgame * (search.MAX_PHASE - phase)
    return tapered // search.MAX_PHASE + tempo * data["side"]


def _loss(scores: NDArray[np.int64], labels: NDArray[np.int64], k: float) -> float:
    predicted = 1.0 / (1.0 + np.exp(-scores / k))
    target = 1.0 / (1.0 + np.exp(-labels / k))
    return float(np.mean((predicted - target) ** 2))


def _report(name: str, scores: NDArray[np.int64], data: Dataset, k: float) -> None:
    error = (scores - data["label"]) * data["side"]
    print(
        f"{name:<10} loss {_loss(scores, data['label'], k):.5f}  "
        f"bias {error.mean():+7.1f} cp  absolute {np.abs(error).mean():6.1f} cp  "
        f"optimistic {np.mean(error > 0):.1%}"
    )
    for low, high, label in PHASE_BUCKETS:
        mask = (data["phase"] >= low) & (data["phase"] <= high)
        if mask.any():
            print(
                f"           {label:<11} n {mask.sum():>6}  bias {error[mask].mean():+7.1f} cp  "
                f"absolute {np.abs(error[mask]).mean():6.1f} cp"
            )


def _fit(
    data: Dataset,
    train: NDArray[np.bool_],
    penalty: float,
    steps: int,
    k: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    base_mg, base_eg, base_tempo = features.baseline_weights()
    vectors = torch.from_numpy(data["features"][train].astype(np.float32))
    phase = torch.from_numpy(data["phase"][train].astype(np.float32))
    side = torch.from_numpy(data["side"][train].astype(np.float32))
    target = torch.sigmoid(torch.from_numpy(data["label"][train].astype(np.float32)) / k)
    remaining = search.MAX_PHASE - phase

    mg_mask = torch.ones(features.FEATURE_COUNT)
    eg_mask = torch.ones(features.FEATURE_COUNT)
    mg_mask[list(features.MG_PINNED)] = 0.0
    eg_mask[list(features.EG_PINNED)] = 0.0

    # fit the move away from current weights, so penalty shrinks towards them
    mg_delta = torch.zeros(features.FEATURE_COUNT, requires_grad=True)
    eg_delta = torch.zeros(features.FEATURE_COUNT, requires_grad=True)
    tempo_delta = torch.zeros(1, requires_grad=True)
    mg_base = torch.from_numpy(base_mg.astype(np.float32))
    eg_base = torch.from_numpy(base_eg.astype(np.float32))

    optimiser = torch.optim.Adam([mg_delta, eg_delta, tempo_delta], lr=1.0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps, eta_min=0.02)
    for _ in range(steps):
        optimiser.zero_grad()
        mg = (mg_base + mg_delta) * mg_mask
        eg = (eg_base + eg_delta) * eg_mask
        scores = (vectors @ mg * phase + vectors @ eg * remaining) / search.MAX_PHASE
        scores = scores + (base_tempo + tempo_delta) * side
        loss = torch.mean((torch.sigmoid(scores / k) - target) ** 2)
        loss = loss + penalty * (torch.mean(mg_delta**2) + torch.mean(eg_delta**2))
        loss.backward()
        optimiser.step()
        schedule.step()

    with torch.no_grad():
        mg = np.rint(((mg_base + mg_delta) * mg_mask).numpy()).astype(np.int64)
        eg = np.rint(((eg_base + eg_delta) * eg_mask).numpy()).astype(np.int64)
        return mg, eg, round(base_tempo + float(tempo_delta))


def main() -> None:
    parser = argparse.ArgumentParser(description="fit evaluate()'s tapered weights to deep labels")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--penalty", type=float, nargs="+", default=[1e-5])
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--k", type=float, default=200.0)
    parser.add_argument("--holdout", type=int, default=7, help="every nth game validates")
    parser.add_argument("--emit-weights", type=Path)
    arguments = parser.parse_args()

    data = _load(arguments.source, arguments.cache)
    # split by game, or the same position reached twice lands on both sides of split
    validate = data["game"] % arguments.holdout == 0
    train = ~validate
    training = {name: column[train] for name, column in data.items()}
    validation = {name: column[validate] for name, column in data.items()}
    print(
        f"{len(data['label'])} positions, {len(np.unique(data['game']))} games, "
        f"{train.sum()} train / {validate.sum()} validation, k {arguments.k}"
    )

    baseline = features.baseline_weights()
    print("\ntraining set")
    _report("baseline", _scores(training, *baseline), training, arguments.k)
    print("\nvalidation set")
    _report("baseline", _scores(validation, *baseline), validation, arguments.k)

    candidates = []
    for penalty in arguments.penalty:
        fitted = _fit(data, train, penalty, arguments.steps, arguments.k)
        scores = _scores(validation, *fitted)
        print(f"\npenalty {penalty:g}")
        _report("train", _scores(training, *fitted), training, arguments.k)
        _report("validation", scores, validation, arguments.k)
        candidates.append((_loss(scores, validation["label"], arguments.k), penalty, fitted))

    loss, penalty, (middlegame, endgame, tempo) = min(candidates, key=lambda row: row[0])
    print(f"\nbest penalty {penalty:g}, validation loss {loss:.5f}")
    for feature, name in (
        (features.F_BISHOP_PAIR, "bishop pair"),
        (features.F_DOUBLED, "doubled"),
        (features.F_ISOLATED, "isolated"),
        (features.F_PASSED_RANK, "passed x rank"),
        (features.F_PASSED_RANK_SQUARED, "passed x rank squared"),
        (features.F_ROOK_SEMI_OPEN, "rook, no own pawn"),
        (features.F_ROOK_OPEN, "rook, no pawn at all"),
        (features.F_KING_SHIELD, "king shield pawn"),
    ):
        print(f"{name:<22} mg {int(middlegame[feature]):>5}  eg {int(endgame[feature]):>5}")
    print(f"{'tempo':<22} {tempo:>8}")
    if arguments.emit_weights:
        np.savez(
            arguments.emit_weights,
            middlegame=middlegame,
            endgame=endgame,
            tempo=np.array(tempo),
        )
        print(f"\nweights written to {arguments.emit_weights}; tools/eval_apply.py installs them")


if __name__ == "__main__":
    main()
