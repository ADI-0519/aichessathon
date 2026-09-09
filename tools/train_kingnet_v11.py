"""Train KingNet on a reproducible mixture of packed evaluator shards.

This trainer keeps the deployed V9 KingNet architecture unchanged while fixing the
training-side limitations of ``tools.train_king_factored``:

* multiple memory-mapped training shards with explicit sampling weights;
* explicit material-band sampling instead of implicit corpus frequencies;
* optional horizontal-mirror augmentation in feature space;
* separate named validation sets and per-material-band metrics;
* cosine learning-rate decay with warmup;
* atomic best-checkpoint recovery and complete data/model provenance.

The JSON config is the experiment contract.  No dataset path, month, sampling
mixture, or material weighting is hard-coded in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from tools.backtest_core import atomic_write_text
from tools.king_features import (
    BASE_FEATURE_COUNT,
    FEATURE_COUNT,
    KING_BUCKET_COUNT,
    PADDING_INDEX as KING_PADDING_INDEX,
)
from tools.nnue_features import MAX_PIECES
from tools.nnue_features import PADDING_INDEX as BASE_PADDING_INDEX
from tools.pack_nnue_data import PACKED_DTYPE
from tools.train_king_factored import ModelConfig, SparseEvaluator, export_model

CP_SCALE = 400.0


@dataclass(frozen=True, slots=True)
class PieceBand:
    name: str
    min_pieces: int
    max_pieces: int
    weight: float


@dataclass(frozen=True, slots=True)
class ShardSpec:
    name: str
    path: Path
    weight: float
    kind: str


@dataclass(frozen=True, slots=True)
class ValidationSpec:
    name: str
    path: Path
    weight: float
    kind: str


@dataclass(frozen=True, slots=True)
class TrainSettings:
    epochs: int
    samples_per_epoch: int
    batch_size: int
    learning_rate: float
    min_learning_rate: float
    warmup_fraction: float
    weight_decay: float
    mirror_probability: float
    seed: int
    device: str
    gradient_clip_norm: float | None
    init_model: Path | None


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: ModelConfig
    settings: TrainSettings
    bands: tuple[PieceBand, ...]
    train_shards: tuple[ShardSpec, ...]
    validation_sets: tuple[ValidationSpec, ...]


@dataclass(slots=True)
class LoadedShard:
    spec: ShardSpec
    records: NDArray[np.void]
    band_rows: tuple[NDArray[np.int32], ...]


@dataclass(frozen=True, slots=True)
class SamplingComponent:
    shard_index: int
    band_index: int
    probability: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _parse_bands(raw: list[dict[str, Any]]) -> tuple[PieceBand, ...]:
    bands: list[PieceBand] = []
    occupied: set[int] = set()
    for item in raw:
        band = PieceBand(
            name=str(item["name"]),
            min_pieces=_positive_int(item["min_pieces"], "min_pieces"),
            max_pieces=_positive_int(item["max_pieces"], "max_pieces"),
            weight=_positive_float(item["weight"], "piece band weight"),
        )
        if band.min_pieces > band.max_pieces or band.max_pieces > MAX_PIECES:
            raise ValueError(
                f"invalid piece band {band.name}: "
                f"{band.min_pieces}..{band.max_pieces}"
            )
        overlap = occupied.intersection(range(band.min_pieces, band.max_pieces + 1))
        if overlap:
            raise ValueError(f"piece band {band.name} overlaps existing counts: {sorted(overlap)}")
        occupied.update(range(band.min_pieces, band.max_pieces + 1))
        bands.append(band)
    if not bands:
        raise ValueError("at least one piece band is required")
    return tuple(bands)


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment config, resolving paths relative to it."""
    raw = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    base = path.parent.resolve()

    model_raw = cast(dict[str, Any], raw.get("model", {}))
    model = ModelConfig(
        accumulator=_positive_int(model_raw.get("accumulator", 128), "accumulator"),
        hidden=_positive_int(model_raw.get("hidden", 32), "hidden"),
    )

    training = cast(dict[str, Any], raw["training"])
    init_value = training.get("init_model")
    settings = TrainSettings(
        epochs=_positive_int(training["epochs"], "epochs"),
        samples_per_epoch=_positive_int(training["samples_per_epoch"], "samples_per_epoch"),
        batch_size=_positive_int(training["batch_size"], "batch_size"),
        learning_rate=_positive_float(training["learning_rate"], "learning_rate"),
        min_learning_rate=_nonnegative_float(
            training.get("min_learning_rate", 0.0), "min_learning_rate"
        ),
        warmup_fraction=_nonnegative_float(training.get("warmup_fraction", 0.0), "warmup_fraction"),
        weight_decay=_nonnegative_float(training.get("weight_decay", 1e-5), "weight_decay"),
        mirror_probability=float(training.get("mirror_probability", 0.0)),
        seed=int(training.get("seed", 20260909)),
        device=str(training.get("device", "auto")),
        gradient_clip_norm=(
            None
            if training.get("gradient_clip_norm") is None
            else _positive_float(training["gradient_clip_norm"], "gradient_clip_norm")
        ),
        init_model=None if init_value is None else _resolve(base, str(init_value)),
    )
    if not 0.0 <= settings.mirror_probability <= 1.0:
        raise ValueError("mirror_probability must be in [0, 1]")
    if not 0.0 <= settings.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if settings.min_learning_rate > settings.learning_rate:
        raise ValueError("min_learning_rate cannot exceed learning_rate")
    if settings.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of auto/cpu/cuda")

    bands = _parse_bands(cast(list[dict[str, Any]], raw["piece_bands"]))

    train_specs: list[ShardSpec] = []
    for item in cast(list[dict[str, Any]], raw["train_shards"]):
        train_specs.append(
            ShardSpec(
                name=str(item["name"]),
                path=_resolve(base, str(item["path"])),
                weight=_positive_float(item.get("weight", 1.0), "training shard weight"),
                kind=str(item.get("kind", "unspecified")),
            )
        )
    if not train_specs:
        raise ValueError("at least one training shard is required")

    validation_specs: list[ValidationSpec] = []
    for item in cast(list[dict[str, Any]], raw["validation_sets"]):
        validation_specs.append(
            ValidationSpec(
                name=str(item["name"]),
                path=_resolve(base, str(item["path"])),
                weight=_positive_float(item.get("weight", 1.0), "validation weight"),
                kind=str(item.get("kind", "unspecified")),
            )
        )
    if not validation_specs:
        raise ValueError("at least one validation set is required")

    names = [spec.name for spec in train_specs] + [spec.name for spec in validation_specs]
    if len(set(names)) != len(names):
        raise ValueError("training and validation dataset names must be unique")

    for dataset in (*train_specs, *validation_specs):
        if not dataset.path.is_file():
            raise ValueError(f"dataset not found: {dataset.path}")
    if settings.init_model is not None and not settings.init_model.is_file():
        raise ValueError(f"init_model not found: {settings.init_model}")

    return ExperimentConfig(
        model=model,
        settings=settings,
        bands=bands,
        train_shards=tuple(train_specs),
        validation_sets=tuple(validation_specs),
    )


def _load_records(path: Path) -> NDArray[np.void]:
    records = cast(NDArray[np.void], np.load(path, mmap_mode="r", allow_pickle=False))
    if records.dtype != PACKED_DTYPE or records.ndim != 1 or len(records) == 0:
        raise ValueError(f"{path} is not a non-empty packed evaluator dataset")
    return records


def _build_loaded_shards(config: ExperimentConfig) -> tuple[LoadedShard, ...]:
    loaded: list[LoadedShard] = []
    for spec in config.train_shards:
        records = _load_records(spec.path)
        piece_counts = np.asarray(records["count"], dtype=np.uint8)
        band_rows: list[NDArray[np.int32]] = []
        for band in config.bands:
            rows = np.flatnonzero(
                (piece_counts >= band.min_pieces) & (piece_counts <= band.max_pieces)
            ).astype(np.int32, copy=False)
            band_rows.append(rows)
        loaded.append(LoadedShard(spec=spec, records=records, band_rows=tuple(band_rows)))
    return tuple(loaded)


def _sampling_components(
    shards: tuple[LoadedShard, ...],
    bands: tuple[PieceBand, ...],
) -> tuple[SamplingComponent, ...]:
    raw: list[tuple[int, int, float]] = []
    total = 0.0
    for shard_index, shard in enumerate(shards):
        for band_index, band in enumerate(bands):
            if len(shard.band_rows[band_index]) == 0:
                continue
            weight = shard.spec.weight * band.weight
            raw.append((shard_index, band_index, weight))
            total += weight
    if total <= 0.0:
        raise ValueError("sampling mixture has no non-empty shard/band components")
    return tuple(
        SamplingComponent(shard_index=s, band_index=b, probability=w / total)
        for s, b, w in raw
    )


def _horizontal_mirror(indices: NDArray[np.int64], rows: NDArray[np.bool_]) -> None:
    """Mirror selected canonical feature rows across files, preserving padding."""
    if not np.any(rows):
        return
    selected = indices[rows]
    non_padding = selected != BASE_PADDING_INDEX
    piece_slots = selected // 64
    squares = selected % 64
    mirrored = piece_slots * 64 + np.bitwise_xor(squares, 7)
    selected[non_padding] = mirrored[non_padding]
    indices[rows] = selected


def _sample_batch(
    shards: tuple[LoadedShard, ...],
    components: tuple[SamplingComponent, ...],
    batch_size: int,
    mirror_probability: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.bool_], NDArray[np.float32]]:
    probabilities = np.asarray(
        [component.probability for component in components], dtype=np.float64
    )
    allocations = rng.multinomial(batch_size, probabilities)

    index_parts: list[NDArray[np.uint16]] = []
    stm_parts: list[NDArray[np.uint8]] = []
    cp_parts: list[NDArray[np.int16]] = []
    for component, count in zip(components, allocations, strict=True):
        if count == 0:
            continue
        shard = shards[component.shard_index]
        pool = shard.band_rows[component.band_index]
        picked = pool[rng.integers(0, len(pool), size=count)]
        selected = shard.records[picked]
        index_parts.append(np.asarray(selected["indices"], dtype=np.uint16))
        stm_parts.append(np.asarray(selected["stm"], dtype=np.uint8))
        cp_parts.append(np.asarray(selected["cp"], dtype=np.int16))

    indices = np.concatenate(index_parts, axis=0).astype(np.int64, copy=True)
    stm = np.concatenate(stm_parts, axis=0).astype(np.bool_, copy=False)
    cp_white = np.concatenate(cp_parts, axis=0).astype(np.float32, copy=False)
    if len(indices) != batch_size:
        raise RuntimeError(f"sampled {len(indices)} rows for requested batch of {batch_size}")

    if mirror_probability > 0.0:
        mirror_rows = rng.random(batch_size) < mirror_probability
        _horizontal_mirror(indices, mirror_rows)

    order = rng.permutation(batch_size)
    return indices[order], stm[order], cp_white[order]


def _batch_to_device(
    indices: NDArray[np.int64],
    stm: NDArray[np.bool_],
    cp_white: NDArray[np.float32],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    index_tensor = torch.from_numpy(indices).to(device)
    stm_tensor = torch.from_numpy(stm).to(device)
    cp_tensor = torch.from_numpy(cp_white).to(device)
    cp_side_to_move = torch.where(stm_tensor, cp_tensor, -cp_tensor)
    return index_tensor, stm_tensor, cp_side_to_move


def _probability_loss(predicted_logit: Tensor, target_cp: Tensor) -> Tensor:
    target_probability = torch.sigmoid(target_cp / CP_SCALE)
    return torch.mean((torch.sigmoid(predicted_logit) - target_probability) ** 2)


def _load_exported_model(model: SparseEvaluator, path: Path) -> None:
    """Exactly lift an exported format-v2 KingNet into the factored training form."""
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["format_version"]) != 2:
            raise ValueError(f"{path} is not a format-v2 KingNet export")
        feature = np.asarray(archive["feature_weights"], dtype=np.float32)
        accumulator_bias = np.asarray(archive["accumulator_bias"], dtype=np.float32)
        hidden_weights = np.asarray(archive["hidden_weights"], dtype=np.float32)
        hidden_bias = np.asarray(archive["hidden_bias"], dtype=np.float32)
        output_weights = np.asarray(archive["output_weights"], dtype=np.float32)
        output_bias = np.asarray(archive["output_bias"], dtype=np.float32)

    accumulator = model.config.accumulator
    hidden = model.config.hidden
    expected = {
        "feature_weights": (FEATURE_COUNT, accumulator),
        "accumulator_bias": (accumulator,),
        "hidden_weights": (hidden, 2 * accumulator),
        "hidden_bias": (hidden,),
        "output_weights": (1, hidden),
        "output_bias": (1,),
    }
    actual = {
        "feature_weights": feature.shape,
        "accumulator_bias": accumulator_bias.shape,
        "hidden_weights": hidden_weights.shape,
        "hidden_bias": hidden_bias.shape,
        "output_weights": output_weights.shape,
        "output_bias": output_bias.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{path}: {name} has shape {actual[name]}, expected {shape}")

    bucketed = feature.reshape(KING_BUCKET_COUNT, BASE_FEATURE_COUNT, accumulator)
    shared = bucketed.mean(axis=0, dtype=np.float32)
    residual = bucketed - shared[None, :, :]

    with torch.no_grad():
        model.embedding.weight.zero_()
        model.factor.weight.zero_()
        model.embedding.weight[:FEATURE_COUNT].copy_(
            torch.from_numpy(residual.reshape(FEATURE_COUNT, accumulator))
        )
        model.factor.weight[:BASE_FEATURE_COUNT].copy_(torch.from_numpy(shared))
        model.accumulator_bias.copy_(torch.from_numpy(accumulator_bias))
        model.hidden.weight.copy_(torch.from_numpy(hidden_weights))
        model.hidden.bias.copy_(torch.from_numpy(hidden_bias))
        model.output.weight.copy_(torch.from_numpy(output_weights))
        model.output.bias.copy_(torch.from_numpy(output_bias))
        model.embedding.weight[KING_PADDING_INDEX].zero_()
        model.factor.weight[BASE_PADDING_INDEX].zero_()

    reconstructed = (
        model.embedding.weight[:FEATURE_COUNT].detach().cpu().numpy().reshape(
            KING_BUCKET_COUNT, BASE_FEATURE_COUNT, accumulator
        )
        + model.factor.weight[:BASE_FEATURE_COUNT].detach().cpu().numpy()[None, :, :]
    )
    max_error = float(np.max(np.abs(reconstructed - bucketed)))
    if max_error > 2e-6:
        raise RuntimeError(f"warm-start reconstruction error {max_error:.3e} is too large")


def _learning_rate(
    step: int,
    total_steps: int,
    base: float,
    minimum: float,
    warmup_steps: int,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base * float(step + 1) / float(warmup_steps)
    remaining = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (base - minimum) * cosine


@torch.no_grad()
def _evaluate_records(
    model: SparseEvaluator,
    records: NDArray[np.void],
    *,
    batch_size: int,
    device: torch.device,
    bands: tuple[PieceBand, ...],
) -> dict[str, Any]:
    model.eval()
    aggregate = {
        "count": 0,
        "squared_cp": 0.0,
        "absolute_cp": 0.0,
        "probability": 0.0,
    }
    band_acc = {
        band.name: {"count": 0, "squared_cp": 0.0, "absolute_cp": 0.0, "probability": 0.0}
        for band in bands
    }

    for start in range(0, len(records), batch_size):
        selected = records[start : min(start + batch_size, len(records))]
        indices = np.asarray(selected["indices"], dtype=np.int64)
        stm = np.asarray(selected["stm"], dtype=np.bool_)
        cp_white = np.asarray(selected["cp"], dtype=np.float32)
        piece_count = np.asarray(selected["count"], dtype=np.int16)
        index_tensor, stm_tensor, target_cp = _batch_to_device(indices, stm, cp_white, device)
        prediction = model(index_tensor, stm_tensor)
        predicted_cp = prediction * CP_SCALE
        difference = predicted_cp - target_cp
        probability_error = (
            torch.sigmoid(prediction) - torch.sigmoid(target_cp / CP_SCALE)
        ) ** 2

        squared = (difference * difference).detach().cpu().numpy()
        absolute = torch.abs(difference).detach().cpu().numpy()
        probability = probability_error.detach().cpu().numpy()
        aggregate["count"] += len(selected)
        aggregate["squared_cp"] += float(np.sum(squared))
        aggregate["absolute_cp"] += float(np.sum(absolute))
        aggregate["probability"] += float(np.sum(probability))

        for band in bands:
            mask = (piece_count >= band.min_pieces) & (piece_count <= band.max_pieces)
            if not np.any(mask):
                continue
            stats = band_acc[band.name]
            stats["count"] += int(np.sum(mask))
            stats["squared_cp"] += float(np.sum(squared[mask]))
            stats["absolute_cp"] += float(np.sum(absolute[mask]))
            stats["probability"] += float(np.sum(probability[mask]))

    def finish(stats: dict[str, float | int]) -> dict[str, float | int | None]:
        count = int(stats["count"])
        if count == 0:
            return {"count": 0, "rmse_cp": None, "mae_cp": None, "probability_mse": None}
        return {
            "count": count,
            "rmse_cp": math.sqrt(float(stats["squared_cp"]) / count),
            "mae_cp": float(stats["absolute_cp"]) / count,
            "probability_mse": float(stats["probability"]) / count,
        }

    return {
        "overall": finish(aggregate),
        "by_piece_band": {name: finish(stats) for name, stats in band_acc.items()},
    }


def _atomic_torch_save(state: dict[str, Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(
    config_path: Path,
    output: Path,
    manifest: Path,
    checkpoint: Path,
) -> None:
    config = load_config(config_path)
    settings = config.settings

    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)

    requested = settings.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    shards = _build_loaded_shards(config)
    components = _sampling_components(shards, config.bands)
    validation_records = {
        spec.name: _load_records(spec.path) for spec in config.validation_sets
    }

    model = SparseEvaluator(config.model)
    if settings.init_model is not None:
        _load_exported_model(model, settings.init_model)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    rng = np.random.default_rng(settings.seed)

    steps_per_epoch = math.ceil(settings.samples_per_epoch / settings.batch_size)
    total_steps = settings.epochs * steps_per_epoch
    warmup_steps = int(round(settings.warmup_fraction * total_steps))
    global_step = 0

    best_objective = math.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []

    component_manifest = [
        {
            "shard": shards[item.shard_index].spec.name,
            "band": config.bands[item.band_index].name,
            "probability": item.probability,
            "rows": len(shards[item.shard_index].band_rows[item.band_index]),
        }
        for item in components
    ]

    print(
        f"KingNet V11: {settings.epochs} epochs, "
        f"{settings.samples_per_epoch:,} samples/epoch, {steps_per_epoch:,} steps/epoch",
        flush=True,
    )
    print(f"device={device}; components={len(components)}", flush=True)

    for epoch in range(1, settings.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for _ in range(steps_per_epoch):
            current_batch = min(settings.batch_size, settings.samples_per_epoch - seen)
            if current_batch <= 0:
                break
            indices, stm, cp_white = _sample_batch(
                shards,
                components,
                current_batch,
                settings.mirror_probability,
                rng,
            )
            index_tensor, stm_tensor, target_cp = _batch_to_device(
                indices, stm, cp_white, device
            )
            learning_rate = _learning_rate(
                global_step,
                total_steps,
                settings.learning_rate,
                settings.min_learning_rate,
                warmup_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)
            prediction = model(index_tensor, stm_tensor)
            loss = _probability_loss(prediction, target_cp)
            loss.backward()  # type: ignore[no-untyped-call]
            if settings.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings.gradient_clip_norm
                )
            optimizer.step()

            running += float(loss.detach()) * current_batch
            seen += current_batch
            global_step += 1

        validation: dict[str, Any] = {}
        objective_numerator = 0.0
        objective_denominator = 0.0
        for spec in config.validation_sets:
            metrics = _evaluate_records(
                model,
                validation_records[spec.name],
                batch_size=settings.batch_size,
                device=device,
                bands=config.bands,
            )
            validation[spec.name] = metrics
            probability_mse = metrics["overall"]["probability_mse"]
            if probability_mse is None:
                raise RuntimeError(f"validation set {spec.name} is empty")
            objective_numerator += spec.weight * float(probability_mse)
            objective_denominator += spec.weight

        objective = objective_numerator / objective_denominator
        epoch_record = {
            "epoch": epoch,
            "train_probability_mse": running / seen,
            "validation_objective": objective,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation,
        }
        history.append(epoch_record)

        print(
            f"epoch {epoch:02d}: train={running / seen:.6f} "
            f"val={objective:.6f} lr={optimizer.param_groups[0]['lr']:.3g}",
            flush=True,
        )

        if objective < best_objective:
            best_objective = objective
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            _atomic_torch_save(best_state, checkpoint)

    if best_epoch == 0 or not checkpoint.is_file():
        raise RuntimeError("training completed without a best checkpoint")

    state = cast(dict[str, Tensor], torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.cpu()
    model.load_state_dict(state)
    export_model(model, output)

    provenance_train = [
        {
            **asdict(spec),
            "path": spec.path.as_posix(),
            "sha256": _sha256(spec.path),
            "rows": len(shards[index].records),
        }
        for index, spec in enumerate(config.train_shards)
    ]
    provenance_validation = [
        {
            **asdict(spec),
            "path": spec.path.as_posix(),
            "sha256": _sha256(spec.path),
            "rows": len(validation_records[spec.name]),
        }
        for spec in config.validation_sets
    ]
    init_provenance = (
        None
        if settings.init_model is None
        else {
            "path": settings.init_model.as_posix(),
            "sha256": _sha256(settings.init_model),
        }
    )

    metadata = {
        "schema_version": 2,
        "architecture": "king_bucket_factored_16x768_acc128_head32",
        "config_path": config_path.resolve().as_posix(),
        "config_sha256": _sha256(config_path),
        "trainer_path": Path(__file__).resolve().as_posix(),
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "model": asdict(config.model),
        "training": {
            **asdict(settings),
            "init_model": None if settings.init_model is None else settings.init_model.as_posix(),
            "device_resolved": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": global_step,
            "warmup_steps": warmup_steps,
        },
        "piece_bands": [asdict(band) for band in config.bands],
        "sampling_components": component_manifest,
        "train_shards": provenance_train,
        "validation_sets": provenance_validation,
        "init_model": init_provenance,
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "history": history,
        "output": output.resolve().as_posix(),
        "output_sha256": _sha256(output),
        "checkpoint": checkpoint.resolve().as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    atomic_write_text(manifest, json.dumps(metadata, indent=2, default=str) + "\n")
    print(
        f"exported {output} at epoch {best_epoch}; "
        f"validation objective={best_objective:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="best-state checkpoint; defaults beside the output",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        parser.error(f"config not found: {args.config}")
    checkpoint = (
        args.checkpoint
        if args.checkpoint is not None
        else args.output.with_suffix(".best.pt")
    )
    train(args.config, args.output, args.manifest, checkpoint)


if __name__ == "__main__":
    main()
