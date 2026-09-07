"""Train the team's king-conditioned sparse chess evaluator.

The packed V5 data is reused directly.  Each absolute piece-square input is
conditioned on the friendly king square inside the model, so no second large
dataset is needed.  The sparse embedding uses SparseAdam; updating the entire
49,152 x accumulator matrix with a dense optimizer every batch would waste most
of the GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from tools.backtest_core import atomic_write_text
from tools.cli import nonnegative_int, positive_int
from tools.halfkp_features import FEATURE_COUNT, PADDING_INDEX, PIECE_BUCKETS
from tools.nnue_features import MAX_PIECES
from tools.nnue_features import PADDING_INDEX as BASE_PADDING_INDEX
from tools.pack_nnue_data import PACKED_DTYPE
from tools.train_nnue import CP_SCALE, probability_loss

INPUT_SCALE = 2_048
WEIGHT_SCALE = 2_048
FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class ModelConfig:
    accumulator: int = 256
    hidden: int = 32


def black_perspective(canonical: Tensor) -> Tensor:
    """Orient packed absolute-colour indices from Black's perspective."""
    padding = canonical == BASE_PADDING_INDEX
    piece_slot = torch.div(canonical, 64, rounding_mode="floor")
    square = canonical.remainder(64)
    oriented = (piece_slot + 6).remainder(12) * 64 + torch.bitwise_xor(square, 56)
    return torch.where(padding, BASE_PADDING_INDEX, oriented)


def king_conditioned_indices(canonical: Tensor, *, black: bool) -> Tensor:
    """Transform a packed batch into friendly-king-conditioned indices."""
    if canonical.ndim != 2 or canonical.shape[1] != MAX_PIECES:
        raise ValueError(f"indices must have shape (batch, {MAX_PIECES})")
    oriented = black_perspective(canonical) if black else canonical
    padding = oriented == BASE_PADDING_INDEX
    piece_slot = torch.div(oriented, 64, rounding_mode="floor")
    square = oriented.remainder(64)
    king_locations = torch.argmax((piece_slot == 5).to(torch.int64), dim=1)
    king_square = square.gather(1, king_locations.unsqueeze(1))
    conditioned = king_square * (PIECE_BUCKETS * 64) + oriented
    return torch.where(padding, PADDING_INDEX, conditioned)


class KingConditionedEvaluator(nn.Module):
    """Two-perspective sparse accumulator conditioned on each friendly king."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.EmbeddingBag(
            FEATURE_COUNT + 1,
            config.accumulator,
            mode="sum",
            padding_idx=PADDING_INDEX,
            sparse=True,
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

    def _accumulate(self, canonical: Tensor, *, black: bool) -> Tensor:
        indices = king_conditioned_indices(canonical, black=black)
        offsets = torch.arange(
            0,
            indices.numel(),
            MAX_PIECES,
            dtype=torch.long,
            device=indices.device,
        )
        return self.embedding(indices.reshape(-1), offsets) + self.accumulator_bias  # type: ignore[no-any-return]

    def forward(self, canonical: Tensor, white_to_move: Tensor) -> Tensor:
        white = self._accumulate(canonical, black=False)
        black = self._accumulate(canonical, black=True)
        selector = white_to_move.bool().unsqueeze(1)
        own = torch.where(selector, white, black)
        opponent = torch.where(selector, black, white)
        inputs = torch.cat((own, opponent), dim=1).clamp_(0.0, 1.0)
        hidden = self.hidden(inputs).clamp_(0.0, 1.0)
        return self.output(hidden).squeeze(1)  # type: ignore[no-any-return]


def initialise_from_v5(model: KingConditionedEvaluator, path: Path) -> None:
    """Lift a V5 model exactly into every king context and extra channel."""
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["format_version"]) != 1:
            raise ValueError("the initial model is not a V5 format-1 export")
        feature_weights = np.asarray(archive["feature_weights"], dtype=np.float32)
        accumulator_bias = np.asarray(archive["accumulator_bias"], dtype=np.float32)
        hidden_weights = np.asarray(archive["hidden_weights"], dtype=np.float32)
        hidden_bias = np.asarray(archive["hidden_bias"], dtype=np.float32)
        output_weights = np.asarray(archive["output_weights"], dtype=np.float32)
        output_bias = np.asarray(archive["output_bias"], dtype=np.float32)

    old_accumulator = accumulator_bias.shape[0]
    old_hidden = hidden_bias.shape[0]
    expected = {
        "feature_weights": (PIECE_BUCKETS * 64, old_accumulator),
        "hidden_weights": (old_hidden, 2 * old_accumulator),
        "output_weights": (1, old_hidden),
        "output_bias": (1,),
    }
    actual = {
        "feature_weights": feature_weights.shape,
        "hidden_weights": hidden_weights.shape,
        "output_weights": output_weights.shape,
        "output_bias": output_bias.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"invalid V5 {name} shape: {actual[name]}, expected {shape}")
    if model.config.accumulator < old_accumulator:
        raise ValueError("new accumulator cannot be narrower than the V5 accumulator")
    if model.config.hidden != old_hidden:
        raise ValueError("new hidden width must equal the V5 hidden width for exact lifting")

    with torch.no_grad():
        repeated = np.tile(feature_weights, (64, 1))
        model.embedding.weight[:FEATURE_COUNT, :old_accumulator].copy_(
            torch.from_numpy(repeated)
        )
        model.accumulator_bias.zero_()
        model.accumulator_bias[:old_accumulator].copy_(
            torch.from_numpy(accumulator_bias)
        )
        # The new accumulator channels retain their small random feature
        # weights, while their outgoing dense columns start at zero.  They
        # therefore contribute exactly zero at epoch zero.  Once the dense
        # head is unfrozen, its columns receive gradients from the non-zero
        # accumulator activations and open a trainable path.  Zeroing both
        # sides would make the added capacity permanently dead.
        model.hidden.weight.zero_()
        model.hidden.weight[:, :old_accumulator].copy_(
            torch.from_numpy(hidden_weights[:, :old_accumulator])
        )
        offset = model.config.accumulator
        model.hidden.weight[:, offset : offset + old_accumulator].copy_(
            torch.from_numpy(hidden_weights[:, old_accumulator:])
        )
        model.hidden.bias.copy_(torch.from_numpy(hidden_bias))
        model.output.weight.copy_(torch.from_numpy(output_weights))
        model.output.bias.copy_(torch.from_numpy(output_bias))


def _quantize(
    values: NDArray[np.float32], scale: int, dtype: Any, name: str
) -> NDArray[Any]:
    rounded = np.rint(values.astype(np.float64) * scale)
    bounds = np.iinfo(dtype)
    if rounded.size and (float(rounded.min()) < bounds.min or float(rounded.max()) > bounds.max):
        raise ValueError(f"{name} exceeds {np.dtype(dtype).name} after quantization")
    return np.ascontiguousarray(rounded.astype(dtype))


def export_model(model: KingConditionedEvaluator, path: Path) -> None:
    """Export only integer runtime arrays to stay inside the submission limit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    feature_weights = state["embedding.weight"][:FEATURE_COUNT].cpu().numpy()
    accumulator_bias = state["accumulator_bias"].cpu().numpy()
    hidden_weights = state["hidden.weight"].cpu().numpy()
    hidden_bias = state["hidden.bias"].cpu().numpy()
    output_weights = state["output.weight"].cpu().numpy()
    output_bias = state["output.bias"].cpu().numpy()
    np.savez(
        path,
        format_version=np.asarray(FORMAT_VERSION, dtype=np.int32),
        cp_scale=np.asarray(CP_SCALE, dtype=np.float32),
        input_scale=np.asarray(INPUT_SCALE, dtype=np.int32),
        weight_scale=np.asarray(WEIGHT_SCALE, dtype=np.int32),
        feature_weights_q=_quantize(feature_weights, INPUT_SCALE, np.int16, "feature_weights"),
        accumulator_bias_q=_quantize(accumulator_bias, INPUT_SCALE, np.int32, "accumulator_bias"),
        hidden_weights_q=_quantize(hidden_weights, WEIGHT_SCALE, np.int16, "hidden_weights"),
        hidden_bias_q=_quantize(
            hidden_bias, INPUT_SCALE * WEIGHT_SCALE, np.int32, "hidden_bias"
        ),
        output_weights_q=_quantize(output_weights, WEIGHT_SCALE, np.int16, "output_weights"),
        output_bias_q=_quantize(
            output_bias, INPUT_SCALE * WEIGHT_SCALE, np.int64, "output_bias"
        ),
    )


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
    indices = torch.from_numpy(np.asarray(selected["indices"], dtype=np.int64)).to(device)
    white_to_move = torch.from_numpy(np.asarray(selected["stm"], dtype=np.bool_)).to(device)
    cp_white = torch.from_numpy(np.asarray(selected["cp"], dtype=np.float32)).to(device)
    target = torch.where(white_to_move, cp_white, -cp_white)
    return indices, white_to_move, target


@torch.no_grad()
def evaluate_model(
    model: KingConditionedEvaluator,
    records: NDArray[np.void],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    squared_error = absolute_error = probability_error = 0.0
    count = 0
    for start in range(0, len(records), batch_size):
        rows = np.arange(start, min(start + batch_size, len(records)), dtype=np.int64)
        indices, side, target = _batch(records, rows, device)
        prediction = model(indices, side)
        difference = prediction * CP_SCALE - target
        squared_error += float(torch.sum(difference * difference))
        absolute_error += float(torch.sum(torch.abs(difference)))
        probability_error += float(
            torch.sum(
                (torch.sigmoid(prediction) - torch.sigmoid(target / CP_SCALE)) ** 2
            )
        )
        count += len(rows)
    return {
        "count": float(count),
        "rmse_cp": math.sqrt(squared_error / count),
        "mae_cp": absolute_error / count,
        "probability_mse": probability_error / count,
    }


def train(
    train_path: Path,
    validation_path: Path,
    initial_model: Path,
    output: Path,
    manifest: Path,
    *,
    config: ModelConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    freeze_dense_epochs: int,
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
    model = KingConditionedEvaluator(config)
    initialise_from_v5(model, initial_model)
    model.to(device)
    sparse_optimizer = torch.optim.SparseAdam(
        (model.embedding.weight,), lr=learning_rate
    )
    dense_parameters = (
        model.accumulator_bias,
        *model.hidden.parameters(),
        *model.output.parameters(),
    )
    dense_optimizer = torch.optim.AdamW(
        dense_parameters, lr=learning_rate, weight_decay=1e-5
    )
    generator = np.random.default_rng(seed)
    initial_metrics = evaluate_model(
        model, validation_records, batch_size=batch_size, device=device
    )
    best_loss = initial_metrics["probability_mse"]
    best_epoch = 0
    best_state: dict[str, Tensor] | None = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    history: list[dict[str, float | int]] = []

    print(
        f"training {len(train_records):,} positions on {device} "
        f"({config.accumulator}x{config.hidden}, king-conditioned)",
        flush=True,
    )
    print(
        f"initial V5 lift: validation MSE "
        f"{initial_metrics['probability_mse']:.6f}, "
        f"MAE {initial_metrics['mae_cp']:.1f} cp",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        train_dense = epoch > freeze_dense_epochs
        for parameter in dense_parameters:
            parameter.requires_grad_(train_dense)
        order = generator.permutation(len(train_records))
        running_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            indices, side, target = _batch(train_records, rows, device)
            sparse_optimizer.zero_grad(set_to_none=True)
            if train_dense:
                dense_optimizer.zero_grad(set_to_none=True)
            prediction = model(indices, side)
            loss = probability_loss(prediction, target)
            loss.backward()  # type: ignore[no-untyped-call]
            sparse_optimizer.step()  # type: ignore[no-untyped-call]
            if train_dense:
                dense_optimizer.step()
            running_loss += float(loss.detach()) * len(rows)
            seen += len(rows)

        metrics = evaluate_model(
            model, validation_records, batch_size=batch_size, device=device
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
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training completed without a model checkpoint")
    model.load_state_dict(best_state)
    model.cpu()
    export_model(model, output)
    print(
        f"selected epoch {best_epoch} with validation MSE {best_loss:.6f}",
        flush=True,
    )
    metadata = {
        "schema_version": 1,
        "architecture": "friendly_king_conditioned_12_piece_accumulator",
        "model": asdict(config),
        "feature_count": FEATURE_COUNT,
        "cp_scale": CP_SCALE,
        "initial_model": initial_model.as_posix(),
        "initial_model_sha256": _sha256(initial_model),
        "train_data": train_path.as_posix(),
        "train_sha256": _sha256(train_path),
        "validation_data": validation_path.as_posix(),
        "validation_sha256": _sha256(validation_path),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "freeze_dense_epochs": freeze_dense_epochs,
        "seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "initial_validation": initial_metrics,
        "best_epoch": best_epoch,
        "best_validation_probability_mse": best_loss,
        "history": history,
        "output": output.as_posix(),
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
    }
    atomic_write_text(manifest, json.dumps(metadata, indent=2) + "\n")
    print(f"exported {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=Path("challengers/v5_nnue/weights/model.npz"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accumulator", type=positive_int, default=256)
    parser.add_argument("--hidden", type=positive_int, default=32)
    parser.add_argument("--epochs", type=positive_int, default=6)
    parser.add_argument("--batch-size", type=positive_int, default=8_192)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-4)
    parser.add_argument("--freeze-dense-epochs", type=nonnegative_int, default=2)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    for path in (args.train, args.validation, args.initial_model):
        if not path.is_file():
            parser.error(f"input file not found: {path}")
    train(
        args.train,
        args.validation,
        args.initial_model,
        args.output,
        args.manifest,
        config=ModelConfig(args.accumulator, args.hidden),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        freeze_dense_epochs=args.freeze_dense_epochs,
        seed=args.seed,
        requested_device=args.device,
    )


if __name__ == "__main__":
    main()
