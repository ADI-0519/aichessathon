"""Train and export the team's sparse neural chess evaluator.

The model is deliberately small enough to call at every search leaf.  It sums
piece-square embeddings from each player's perspective, orders the two sums as
side-to-move/opponent, and predicts a centipawn-scaled win-probability logit.
Only the exported ``.npz`` is intended for a future challenger submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from tools.backtest_core import atomic_write_text
from tools.cli import positive_int
from tools.nnue_features import FEATURE_COUNT, MAX_PIECES, PADDING_INDEX
from tools.pack_nnue_data import PACKED_DTYPE

CP_SCALE = 400.0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    accumulator: int = 128
    hidden: int = 32


class SparseEvaluator(nn.Module):
    """Colour-symmetric sparse accumulator network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.EmbeddingBag(
            FEATURE_COUNT + 1,
            config.accumulator,
            mode="sum",
            padding_idx=PADDING_INDEX,
        )
        self.accumulator_bias = nn.Parameter(torch.zeros(config.accumulator))
        self.hidden = nn.Linear(2 * config.accumulator, config.hidden)
        self.output = nn.Linear(config.hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.embedding.weight[PADDING_INDEX].zero_()
        nn.init.zeros_(self.accumulator_bias)
        nn.init.kaiming_uniform_(self.hidden.weight, a=math.sqrt(5))
        nn.init.zeros_(self.hidden.bias)
        nn.init.uniform_(self.output.weight, -0.05, 0.05)
        nn.init.zeros_(self.output.bias)

    def _accumulate(self, indices: Tensor) -> Tensor:
        if indices.ndim != 2 or indices.shape[1] != MAX_PIECES:
            raise ValueError(f"indices must have shape (batch, {MAX_PIECES})")
        offsets = torch.arange(
            0,
            indices.numel(),
            MAX_PIECES,
            dtype=torch.long,
            device=indices.device,
        )
        return self.embedding(indices.reshape(-1), offsets) + self.accumulator_bias  # type: ignore[no-any-return]

    def forward(self, canonical: Tensor, white_to_move: Tensor) -> Tensor:
        white = self._accumulate(canonical)
        black_indices = black_perspective(canonical)
        black = self._accumulate(black_indices)
        selector = white_to_move.bool().unsqueeze(1)
        own = torch.where(selector, white, black)
        opponent = torch.where(selector, black, white)
        accumulators = torch.cat((own, opponent), dim=1).clamp_(0.0, 1.0)
        hidden = self.hidden(accumulators).clamp_(0.0, 1.0)
        return self.output(hidden).squeeze(1)  # type: ignore[no-any-return]


def black_perspective(canonical: Tensor) -> Tensor:
    """Vectorised colour swap and vertical reflection for packed indices."""
    padding = canonical == PADDING_INDEX
    piece_slot = torch.div(canonical, 64, rounding_mode="floor")
    square = canonical.remainder(64)
    oriented = (piece_slot + 6).remainder(12) * 64 + torch.bitwise_xor(square, 56)
    return torch.where(padding, PADDING_INDEX, oriented)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(path: Path) -> NDArray[np.void]:
    records = cast(NDArray[np.void], np.load(path, mmap_mode="r", allow_pickle=False))
    if records.dtype != PACKED_DTYPE or records.ndim != 1 or len(records) == 0:
        raise ValueError(f"{path} is not a non-empty packed evaluator dataset")
    return records


def _batch(
    records: NDArray[np.void], rows: NDArray[np.int64], device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    selected = records[rows]
    indices = torch.from_numpy(
        np.asarray(selected["indices"], dtype=np.int64)
    ).to(device)
    white_to_move = torch.from_numpy(
        np.asarray(selected["stm"], dtype=np.bool_)
    ).to(device)
    cp_white = torch.from_numpy(
        np.asarray(selected["cp"], dtype=np.float32)
    ).to(device)
    cp_side_to_move = torch.where(white_to_move, cp_white, -cp_white)
    return indices, white_to_move, cp_side_to_move


def probability_loss(predicted_logit: Tensor, target_cp: Tensor) -> Tensor:
    target_probability = torch.sigmoid(target_cp / CP_SCALE)
    return torch.mean((torch.sigmoid(predicted_logit) - target_probability) ** 2)


@torch.no_grad()
def evaluate_model(
    model: SparseEvaluator,
    records: NDArray[np.void],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    squared_error = absolute_cp_error = probability_error = 0.0
    count = 0
    for start in range(0, len(records), batch_size):
        rows = np.arange(start, min(start + batch_size, len(records)), dtype=np.int64)
        indices, stm, target_cp = _batch(records, rows, device)
        prediction = model(indices, stm)
        predicted_cp = prediction * CP_SCALE
        difference = predicted_cp - target_cp
        squared_error += float(torch.sum(difference * difference))
        absolute_cp_error += float(torch.sum(torch.abs(difference)))
        probability_error += float(
            torch.sum(
                (torch.sigmoid(prediction) - torch.sigmoid(target_cp / CP_SCALE))
                ** 2
            )
        )
        count += len(rows)
    return {
        "count": float(count),
        "rmse_cp": math.sqrt(squared_error / count),
        "mae_cp": absolute_cp_error / count,
        "probability_mse": probability_error / count,
    }


def export_model(model: SparseEvaluator, path: Path) -> None:
    """Write simple float32 arrays that NumPy and Numba can load directly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    np.savez(
        path,
        format_version=np.asarray(1, dtype=np.int32),
        cp_scale=np.asarray(CP_SCALE, dtype=np.float32),
        feature_weights=state["embedding.weight"][:FEATURE_COUNT].cpu().numpy().astype(np.float32),
        accumulator_bias=state["accumulator_bias"].cpu().numpy().astype(np.float32),
        hidden_weights=state["hidden.weight"].cpu().numpy().astype(np.float32),
        hidden_bias=state["hidden.bias"].cpu().numpy().astype(np.float32),
        output_weights=state["output.weight"].cpu().numpy().astype(np.float32),
        output_bias=state["output.bias"].cpu().numpy().astype(np.float32),
    )


def train(
    train_path: Path,
    validation_path: Path,
    output: Path,
    manifest: Path,
    *,
    config: ModelConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    requested_device: str,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this PyTorch build cannot use it")

    train_records = _load_records(train_path)
    validation_records = _load_records(validation_path)
    model = SparseEvaluator(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    generator = np.random.default_rng(seed)
    best_loss = math.inf
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float | int]] = []

    print(
        f"training {len(train_records):,} positions on {device} "
        f"({config.accumulator}x{config.hidden})",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        order = generator.permutation(len(train_records))
        running_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            indices, stm, target_cp = _batch(train_records, rows, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(indices, stm)
            loss = probability_loss(prediction, target_cp)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            running_loss += float(loss.detach()) * len(rows)
            seen += len(rows)

        metrics = evaluate_model(
            model,
            validation_records,
            batch_size=batch_size,
            device=device,
        )
        train_loss = running_loss / seen
        history.append({"epoch": epoch, "train_probability_mse": train_loss, **metrics})
        print(
            f"epoch {epoch}: train MSE {train_loss:.6f}, "
            f"validation MSE {metrics['probability_mse']:.6f}, "
            f"MAE {metrics['mae_cp']:.1f} cp",
            flush=True,
        )
        if metrics["probability_mse"] < best_loss:
            best_loss = metrics["probability_mse"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training completed without a model checkpoint")
    model.load_state_dict(best_state)
    model.cpu()
    export_model(model, output)
    metadata = {
        "schema_version": 1,
        "architecture": "symmetric_sparse_piece_square_accumulator",
        "model": asdict(config),
        "cp_scale": CP_SCALE,
        "train_data": train_path.as_posix(),
        "train_sha256": _sha256(train_path),
        "validation_data": validation_path.as_posix(),
        "validation_sha256": _sha256(validation_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "best_validation_probability_mse": best_loss,
        "history": history,
        "output": output.as_posix(),
        "output_sha256": _sha256(output),
    }
    atomic_write_text(manifest, json.dumps(metadata, indent=2) + "\n")
    print(f"exported {output} ({output.stat().st_size / 1024:.1f} KiB)")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accumulator", type=positive_int, default=128)
    parser.add_argument("--hidden", type=positive_int, default=32)
    parser.add_argument("--epochs", type=positive_int, default=8)
    parser.add_argument("--batch-size", type=positive_int, default=8_192)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    for path in (args.train, args.validation):
        if not path.is_file():
            parser.error(f"packed dataset not found: {path}")
    train(
        args.train,
        args.validation,
        args.output,
        args.manifest,
        config=ModelConfig(args.accumulator, args.hidden),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        requested_device=args.device,
    )


if __name__ == "__main__":
    main()
