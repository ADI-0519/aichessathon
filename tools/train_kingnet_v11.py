"""Train KingNet on a reproducible mixture of packed evaluator shards.

This trainer builds the V11-BIG evaluator while fixing the training-side
limitations of ``tools.train_king_factored``:

* multiple memory-mapped training shards with explicit sampling weights;
* explicit material-band sampling instead of implicit corpus frequencies;
* optional horizontal-mirror augmentation in feature space;
* pairwise accumulator interactions and dual-activation material heads;
* configurable piece-count-to-head mapping stored inside the export;
* separate named validation sets, calibration slopes, and material-aware selection;
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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from tools.backtest_core import atomic_write_text
from tools.king_features import (
    BASE_FEATURE_COUNT,
    FEATURE_COUNT,
    KING_BUCKET_COUNT,
    KING_BUCKETS,
    OWN_KING_SLOT,
)
from tools.king_features import PADDING_INDEX as KING_PADDING_INDEX
from tools.nnue_features import MAX_PIECES
from tools.nnue_features import PADDING_INDEX as BASE_PADDING_INDEX
from tools.pack_nnue_data import PACKED_DTYPE

RUNTIME_INPUT_SCALE = 2_048
MAX_COMPACT_FEATURE_QUANTIZATION_DRIFT = 1


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
    resume_checkpoint: Path | None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    accumulator: int
    hidden: int
    pairwise_width: int
    cp_scale: float
    piece_head_map: tuple[int, ...]

    @property
    def head_count(self) -> int:
        return max(self.piece_head_map) + 1

    @property
    def pairwise_inputs(self) -> int:
        return 2 * self.pairwise_width


@dataclass(frozen=True, slots=True)
class ExportConfig:
    feature_storage: str


@dataclass(frozen=True, slots=True)
class SelectionObjective:
    overall: float
    by_piece_band: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: ModelConfig
    export: ExportConfig
    settings: TrainSettings
    bands: tuple[PieceBand, ...]
    train_shards: tuple[ShardSpec, ...]
    validation_sets: tuple[ValidationSpec, ...]
    selection: SelectionObjective


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


def black_perspective(canonical: Tensor) -> Tensor:
    """Orient canonical piece-square indices from Black's perspective."""
    padding = canonical == BASE_PADDING_INDEX
    piece_slot = torch.div(canonical, 64, rounding_mode="floor")
    square = canonical.remainder(64)
    oriented = (piece_slot + 6).remainder(12) * 64 + torch.bitwise_xor(square, 56)
    return torch.where(padding, BASE_PADDING_INDEX, oriented)


KING_BUCKET_TENSOR = torch.from_numpy(
    np.asarray(KING_BUCKETS, dtype=np.int64)
)


def apply_king_bucket(oriented: Tensor) -> Tensor:
    """Apply the bucket selected by the oriented side's king square."""
    padding = oriented == BASE_PADDING_INDEX
    piece_slot = torch.div(oriented, 64, rounding_mode="floor")
    square = oriented.remainder(64)
    own_king = ((piece_slot == OWN_KING_SLOT) & ~padding).long()
    king_square = (square * own_king).sum(dim=1)
    buckets = KING_BUCKET_TENSOR.to(oriented.device)[king_square]
    shifted = oriented + buckets.unsqueeze(1) * BASE_FEATURE_COUNT
    return torch.where(padding, KING_PADDING_INDEX, shifted)


class V11BigEvaluator(nn.Module):
    """King-conditioned sparse evaluator with pairwise material heads."""

    piece_head_map: Tensor

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.EmbeddingBag(
            FEATURE_COUNT + 1,
            config.accumulator,
            mode="sum",
            padding_idx=KING_PADDING_INDEX,
        )
        self.factor = nn.EmbeddingBag(
            BASE_FEATURE_COUNT + 1,
            config.accumulator,
            mode="sum",
            padding_idx=BASE_PADDING_INDEX,
        )
        self.accumulator_bias = nn.Parameter(torch.zeros(config.accumulator))
        self.hidden_weight = nn.Parameter(
            torch.empty(config.head_count, config.hidden, config.pairwise_inputs)
        )
        self.hidden_bias = nn.Parameter(torch.zeros(config.head_count, config.hidden))
        self.output_relu_weight = nn.Parameter(
            torch.empty(config.head_count, config.hidden)
        )
        self.output_clipped_square_weight = nn.Parameter(
            torch.empty(config.head_count, config.hidden)
        )
        self.output_bias = nn.Parameter(torch.zeros(config.head_count))
        self.register_buffer(
            "piece_head_map",
            torch.tensor(config.piece_head_map, dtype=torch.long),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.embedding.weight)
        nn.init.normal_(self.factor.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.embedding.weight[KING_PADDING_INDEX].zero_()
            self.factor.weight[BASE_PADDING_INDEX].zero_()
        nn.init.zeros_(self.accumulator_bias)
        nn.init.kaiming_uniform_(self.hidden_weight, a=math.sqrt(5))
        nn.init.zeros_(self.hidden_bias)
        nn.init.uniform_(self.output_relu_weight, -0.05, 0.05)
        nn.init.uniform_(self.output_clipped_square_weight, -0.05, 0.05)
        nn.init.zeros_(self.output_bias)

    def _accumulate(self, bucketed: Tensor, base: Tensor) -> Tensor:
        if bucketed.ndim != 2 or bucketed.shape[1] != MAX_PIECES:
            raise ValueError(f"indices must have shape (batch, {MAX_PIECES})")
        offsets = torch.arange(
            0,
            bucketed.numel(),
            MAX_PIECES,
            dtype=torch.long,
            device=bucketed.device,
        )
        bucket_sum = self.embedding(bucketed.reshape(-1), offsets)
        shared_sum = self.factor(base.reshape(-1), offsets)
        return bucket_sum + shared_sum + self.accumulator_bias  # type: ignore[no-any-return]

    def _pairwise(self, accumulator: Tensor) -> Tensor:
        activated = accumulator.clamp(0.0, 1.0)
        width = self.config.pairwise_width
        return activated[:, :width] * activated[:, width : 2 * width]

    def forward(
        self,
        canonical: Tensor,
        white_to_move: Tensor,
        piece_count: Tensor,
    ) -> Tensor:
        black_base = black_perspective(canonical)
        white = self._accumulate(apply_king_bucket(canonical), canonical)
        black = self._accumulate(apply_king_bucket(black_base), black_base)
        selector = white_to_move.bool().unsqueeze(1)
        own = torch.where(selector, white, black)
        opponent = torch.where(selector, black, white)
        pairwise = torch.cat((self._pairwise(own), self._pairwise(opponent)), dim=1)

        head = self.piece_head_map[piece_count.long()]
        hidden_weight = self.hidden_weight[head]
        hidden_bias = self.hidden_bias[head]
        preactivation = torch.bmm(hidden_weight, pairwise.unsqueeze(2)).squeeze(2)
        preactivation = preactivation + hidden_bias
        relu = torch.relu(preactivation)
        clipped_square = preactivation.clamp(0.0, 1.0).square()
        output = self.output_bias[head]
        output = output + torch.sum(self.output_relu_weight[head] * relu, dim=1)
        output = output + torch.sum(
            self.output_clipped_square_weight[head] * clipped_square,
            dim=1,
        )
        return output


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


def _parse_architecture(raw: dict[str, Any]) -> tuple[ModelConfig, ExportConfig]:
    model_raw = cast(dict[str, Any], raw.get("model", {}))
    raw_head_map = cast(list[Any], model_raw.get("piece_head_map", []))
    if len(raw_head_map) != MAX_PIECES + 1:
        raise ValueError(
            f"model.piece_head_map must contain {MAX_PIECES + 1} entries"
        )
    piece_head_map = tuple(int(value) for value in raw_head_map)
    if any(value < 0 for value in piece_head_map):
        raise ValueError("model.piece_head_map cannot contain negative head ids")
    used_heads = sorted(set(piece_head_map[2:]))
    if used_heads != list(range(len(used_heads))):
        raise ValueError("piece head ids used for counts 2..32 must be contiguous from zero")
    required_model_keys = ("accumulator", "hidden", "pairwise_width", "cp_scale")
    missing_model_keys = [key for key in required_model_keys if key not in model_raw]
    if missing_model_keys:
        raise ValueError(f"model configuration is missing: {missing_model_keys}")
    model = ModelConfig(
        accumulator=_positive_int(model_raw["accumulator"], "accumulator"),
        hidden=_positive_int(model_raw["hidden"], "hidden"),
        pairwise_width=_positive_int(model_raw["pairwise_width"], "pairwise_width"),
        cp_scale=_positive_float(model_raw["cp_scale"], "cp_scale"),
        piece_head_map=piece_head_map,
    )
    if 2 * model.pairwise_width > model.accumulator:
        raise ValueError("2 * pairwise_width cannot exceed accumulator width")

    export_raw = cast(dict[str, Any], raw.get("export", {}))
    feature_storage = str(export_raw.get("feature_storage", "float32"))
    if feature_storage not in {"float32", "float16"}:
        raise ValueError("export.feature_storage must be float32 or float16")
    return model, ExportConfig(feature_storage=feature_storage)


def load_architecture_config(path: Path) -> tuple[ModelConfig, ExportConfig]:
    """Load model/export settings without requiring the datasets to exist."""
    raw = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return _parse_architecture(raw)


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment config, resolving paths relative to it."""
    raw = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    base = path.parent.resolve()

    model, export = _parse_architecture(raw)

    training = cast(dict[str, Any], raw["training"])
    init_value = training.get("init_model")
    resume_value = training.get("resume_checkpoint")
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
        resume_checkpoint=(
            None if resume_value is None else _resolve(base, str(resume_value))
        ),
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
    covered = {
        count
        for band in bands
        for count in range(band.min_pieces, band.max_pieces + 1)
    }
    missing_counts = sorted(set(range(2, MAX_PIECES + 1)) - covered)
    if missing_counts:
        raise ValueError(f"piece bands do not cover legal counts: {missing_counts}")

    selection_raw = cast(dict[str, Any], raw["selection_objective"])
    overall_weight = _nonnegative_float(
        selection_raw.get("overall", 0.0), "selection overall weight"
    )
    band_weight_raw = cast(
        dict[str, Any], selection_raw.get("by_piece_band", {})
    )
    known_band_names = {band.name for band in bands}
    unknown_selection_bands = sorted(set(band_weight_raw) - known_band_names)
    if unknown_selection_bands:
        raise ValueError(
            "selection objective references unknown piece bands: "
            f"{unknown_selection_bands}"
        )
    band_weights = tuple(
        (
            band.name,
            _nonnegative_float(
                band_weight_raw.get(band.name, 0.0),
                f"selection weight for {band.name}",
            ),
        )
        for band in bands
    )
    if overall_weight + sum(weight for _, weight in band_weights) <= 0.0:
        raise ValueError("selection objective must assign at least one positive weight")
    selection = SelectionObjective(
        overall=overall_weight,
        by_piece_band=band_weights,
    )

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

    all_specs: tuple[ShardSpec | ValidationSpec, ...] = (
        *train_specs,
        *validation_specs,
    )
    for dataset in all_specs:
        if not dataset.path.is_file():
            raise ValueError(f"dataset not found: {dataset.path}")
    if settings.init_model is not None and not settings.init_model.is_file():
        raise ValueError(f"init_model not found: {settings.init_model}")
    if settings.resume_checkpoint is not None and not settings.resume_checkpoint.is_file():
        raise ValueError(f"resume_checkpoint not found: {settings.resume_checkpoint}")
    if settings.init_model is not None and settings.resume_checkpoint is not None:
        raise ValueError("init_model and resume_checkpoint are mutually exclusive")

    for index, left in enumerate(all_specs):
        for right in all_specs[index + 1 :]:
            if left.path == right.path or left.path.samefile(right.path):
                raise ValueError(
                    f"dataset leakage: {left.name} and {right.name} are the same file"
                )

    return ExperimentConfig(
        model=model,
        export=export,
        settings=settings,
        bands=bands,
        train_shards=tuple(train_specs),
        validation_sets=tuple(validation_specs),
        selection=selection,
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
) -> tuple[
    NDArray[np.int64],
    NDArray[np.bool_],
    NDArray[np.float32],
    NDArray[np.int16],
]:
    probabilities = np.asarray(
        [component.probability for component in components], dtype=np.float64
    )
    allocations = rng.multinomial(batch_size, probabilities)

    index_parts: list[NDArray[np.uint16]] = []
    stm_parts: list[NDArray[np.uint8]] = []
    cp_parts: list[NDArray[np.int16]] = []
    count_parts: list[NDArray[np.uint8]] = []
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
        count_parts.append(np.asarray(selected["count"], dtype=np.uint8))

    indices = np.concatenate(index_parts, axis=0).astype(np.int64, copy=True)
    stm = np.concatenate(stm_parts, axis=0).astype(np.bool_, copy=False)
    cp_white = np.concatenate(cp_parts, axis=0).astype(np.float32, copy=False)
    piece_count = np.concatenate(count_parts, axis=0).astype(np.int16, copy=False)
    if len(indices) != batch_size:
        raise RuntimeError(f"sampled {len(indices)} rows for requested batch of {batch_size}")

    if mirror_probability > 0.0:
        mirror_rows = rng.random(batch_size) < mirror_probability
        _horizontal_mirror(indices, mirror_rows)

    order = rng.permutation(batch_size)
    return indices[order], stm[order], cp_white[order], piece_count[order]


def _batch_to_device(
    indices: NDArray[np.int64],
    stm: NDArray[np.bool_],
    cp_white: NDArray[np.float32],
    piece_count: NDArray[np.int16],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    index_tensor = torch.from_numpy(indices).to(device)
    stm_tensor = torch.from_numpy(stm).to(device)
    cp_tensor = torch.from_numpy(cp_white).to(device)
    count_tensor = torch.from_numpy(piece_count).to(device)
    cp_side_to_move = torch.where(stm_tensor, cp_tensor, -cp_tensor)
    return index_tensor, stm_tensor, cp_side_to_move, count_tensor


def _probability_loss(
    predicted_logit: Tensor, target_cp: Tensor, cp_scale: float
) -> Tensor:
    target_probability = torch.sigmoid(target_cp / cp_scale)
    return torch.mean((torch.sigmoid(predicted_logit) - target_probability) ** 2)


def _cpu_cuda_rng_states(raw_states: object) -> list[Tensor]:
    """Validate saved CUDA RNG states and normalize them for PyTorch restore APIs."""
    if not isinstance(raw_states, (list, tuple)):
        raise ValueError("checkpoint CUDA RNG state must be a sequence")
    states: list[Tensor] = []
    for state in raw_states:
        if not isinstance(state, Tensor) or state.dtype != torch.uint8:
            raise ValueError("checkpoint CUDA RNG states must be torch.ByteTensor values")
        states.append(state.detach().cpu().contiguous())
    return states


def _fold_factor(state: dict[str, Tensor]) -> NDArray[np.float32]:
    bucketed = state["embedding.weight"][:FEATURE_COUNT].cpu().numpy()
    shared = state["factor.weight"][:BASE_FEATURE_COUNT].cpu().numpy()
    accumulator = bucketed.shape[1]
    folded = bucketed.reshape(-1, BASE_FEATURE_COUNT, accumulator) + shared[None, :, :]
    return folded.reshape(FEATURE_COUNT, accumulator)  # type: ignore[no-any-return]


def export_model(
    model: V11BigEvaluator,
    path: Path,
    *,
    feature_storage: str = "float32",
) -> None:
    """Write the complete, self-describing V11 runtime model atomically."""
    if feature_storage not in {"float32", "float16"}:
        raise ValueError("feature_storage must be float32 or float16")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    feature_dtype = np.float16 if feature_storage == "float16" else np.float32
    feature_weights = _fold_factor(state).astype(np.float32)
    stored_feature_weights = feature_weights.astype(feature_dtype)
    reference_quantized = np.rint(feature_weights * RUNTIME_INPUT_SCALE).astype(np.int32)
    stored_quantized = np.rint(
        stored_feature_weights.astype(np.float32) * RUNTIME_INPUT_SCALE
    ).astype(np.int32)
    quantization_delta = np.abs(reference_quantized - stored_quantized)
    maximum_quantization_delta = int(np.max(quantization_delta))
    changed_quantized_values = int(np.count_nonzero(quantization_delta))
    if maximum_quantization_delta > MAX_COMPACT_FEATURE_QUANTIZATION_DRIFT:
        raise ValueError(
            "compact feature storage changes runtime weights by more than "
            f"{MAX_COMPACT_FEATURE_QUANTIZATION_DRIFT} quantum; use float32"
        )
    payload = {
        "format_version": np.asarray(3, dtype=np.int32),
        "architecture": np.asarray("kingnet_v11_big_pairwise_dual_material"),
        "cp_scale": np.asarray(model.config.cp_scale, dtype=np.float32),
        "feature_storage": np.asarray(feature_storage),
        "feature_weights": stored_feature_weights,
        "runtime_input_scale": np.asarray(RUNTIME_INPUT_SCALE, dtype=np.int32),
        "feature_quantization_max_delta": np.asarray(
            maximum_quantization_delta, dtype=np.int32
        ),
        "feature_quantization_changed_values": np.asarray(
            changed_quantized_values, dtype=np.int64
        ),
        "accumulator_bias": state["accumulator_bias"].cpu().numpy().astype(np.float32),
        "pairwise_width": np.asarray(model.config.pairwise_width, dtype=np.int32),
        "piece_head_map": np.asarray(model.config.piece_head_map, dtype=np.int32),
        "hidden_weights": state["hidden_weight"].cpu().numpy().astype(np.float32),
        "hidden_bias": state["hidden_bias"].cpu().numpy().astype(np.float32),
        "output_relu_weights": state["output_relu_weight"].cpu().numpy().astype(np.float32),
        "output_clipped_square_weights": state[
            "output_clipped_square_weight"
        ].cpu().numpy().astype(np.float32),
        "output_bias": state["output_bias"].cpu().numpy().astype(np.float32),
    }
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_exported_model(model: V11BigEvaluator, path: Path) -> None:
    """Warm-start the sparse accumulator from a format-v2 KingNet export."""
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["format_version"]) != 2:
            raise ValueError(f"{path} is not a format-v2 KingNet export")
        feature = np.asarray(archive["feature_weights"], dtype=np.float32)
        accumulator_bias = np.asarray(archive["accumulator_bias"], dtype=np.float32)

    accumulator = model.config.accumulator
    expected = {
        "feature_weights": (FEATURE_COUNT, accumulator),
        "accumulator_bias": (accumulator,),
    }
    actual = {
        "feature_weights": feature.shape,
        "accumulator_bias": accumulator_bias.shape,
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
    model: V11BigEvaluator,
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
        "prediction_teacher": 0.0,
        "teacher_squared": 0.0,
    }
    band_acc = {
        band.name: {
            "count": 0,
            "squared_cp": 0.0,
            "absolute_cp": 0.0,
            "probability": 0.0,
            "prediction_teacher": 0.0,
            "teacher_squared": 0.0,
        }
        for band in bands
    }

    for start in range(0, len(records), batch_size):
        selected = records[start : min(start + batch_size, len(records))]
        indices = np.asarray(selected["indices"], dtype=np.int64)
        stm = np.asarray(selected["stm"], dtype=np.bool_)
        cp_white = np.asarray(selected["cp"], dtype=np.float32)
        piece_count = np.asarray(selected["count"], dtype=np.int16)
        index_tensor, stm_tensor, target_cp, count_tensor = _batch_to_device(
            indices, stm, cp_white, piece_count, device
        )
        prediction = model(index_tensor, stm_tensor, count_tensor)
        predicted_cp = prediction * model.config.cp_scale
        difference = predicted_cp - target_cp
        probability_error = (
            torch.sigmoid(prediction)
            - torch.sigmoid(target_cp / model.config.cp_scale)
        ) ** 2

        squared = (difference * difference).detach().cpu().numpy()
        absolute = torch.abs(difference).detach().cpu().numpy()
        probability = probability_error.detach().cpu().numpy()
        predicted = predicted_cp.detach().cpu().numpy()
        teacher = target_cp.detach().cpu().numpy()
        aggregate["count"] += len(selected)
        aggregate["squared_cp"] += float(np.sum(squared))
        aggregate["absolute_cp"] += float(np.sum(absolute))
        aggregate["probability"] += float(np.sum(probability))
        aggregate["prediction_teacher"] += float(np.sum(predicted * teacher))
        aggregate["teacher_squared"] += float(np.sum(teacher * teacher))

        for band in bands:
            mask = (piece_count >= band.min_pieces) & (piece_count <= band.max_pieces)
            if not np.any(mask):
                continue
            stats = band_acc[band.name]
            stats["count"] += int(np.sum(mask))
            stats["squared_cp"] += float(np.sum(squared[mask]))
            stats["absolute_cp"] += float(np.sum(absolute[mask]))
            stats["probability"] += float(np.sum(probability[mask]))
            stats["prediction_teacher"] += float(
                np.sum(predicted[mask] * teacher[mask])
            )
            stats["teacher_squared"] += float(np.sum(teacher[mask] * teacher[mask]))

    def finish(stats: dict[str, float | int]) -> dict[str, float | int | None]:
        count = int(stats["count"])
        if count == 0:
            return {
                "count": 0,
                "rmse_cp": None,
                "mae_cp": None,
                "probability_mse": None,
                "calibration_slope": None,
            }
        teacher_squared = float(stats["teacher_squared"])
        return {
            "count": count,
            "rmse_cp": math.sqrt(float(stats["squared_cp"]) / count),
            "mae_cp": float(stats["absolute_cp"]) / count,
            "probability_mse": float(stats["probability"]) / count,
            "calibration_slope": (
                None
                if teacher_squared <= 0.0
                else float(stats["prediction_teacher"]) / teacher_squared
            ),
        }

    return {
        "overall": finish(aggregate),
        "by_piece_band": {name: finish(stats) for name, stats in band_acc.items()},
    }


def _selection_score(
    metrics: dict[str, Any], selection: SelectionObjective
) -> float:
    """Calculate the configured material-aware checkpoint objective."""
    numerator = 0.0
    denominator = 0.0
    if selection.overall > 0.0:
        value = metrics["overall"]["probability_mse"]
        if value is None:
            raise RuntimeError("overall validation metric is empty")
        numerator += selection.overall * float(value)
        denominator += selection.overall
    for name, weight in selection.by_piece_band:
        if weight <= 0.0:
            continue
        value = metrics["by_piece_band"][name]["probability_mse"]
        if value is None:
            raise RuntimeError(
                f"selection objective requires non-empty piece band {name!r}"
            )
        numerator += weight * float(value)
        denominator += weight
    if denominator <= 0.0:
        raise RuntimeError("selection objective has no positive components")
    return numerator / denominator


def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def best_partial_path(output: Path) -> Path:
    """Return the deployable checkpoint path written during a training run."""
    return output.with_name(f"{output.stem}.best.partial{output.suffix}")


def train(
    config_path: Path,
    output: Path,
    manifest: Path,
    checkpoint: Path,
) -> None:
    config = load_config(config_path)
    settings = config.settings

    dataset_specs: tuple[ShardSpec | ValidationSpec, ...] = (
        *config.train_shards,
        *config.validation_sets,
    )
    dataset_hashes = {spec.name: _sha256(spec.path) for spec in dataset_specs}
    seen_digests: dict[str, str] = {}
    for spec in dataset_specs:
        digest = dataset_hashes[spec.name]
        previous = seen_digests.get(digest)
        if previous is not None:
            raise ValueError(
                f"dataset leakage: {previous} and {spec.name} have identical content"
            )
        seen_digests[digest] = spec.name

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

    model = V11BigEvaluator(config.model)
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
    warmup_steps = round(settings.warmup_fraction * total_steps)
    training_signature = {
        "epochs": settings.epochs,
        "samples_per_epoch": settings.samples_per_epoch,
        "batch_size": settings.batch_size,
        "learning_rate": settings.learning_rate,
        "min_learning_rate": settings.min_learning_rate,
        "warmup_fraction": settings.warmup_fraction,
        "weight_decay": settings.weight_decay,
        "mirror_probability": settings.mirror_probability,
        "seed": settings.seed,
        "device": settings.device,
        "gradient_clip_norm": settings.gradient_clip_norm,
        "selection_objective": asdict(config.selection),
        "piece_bands": [asdict(band) for band in config.bands],
    }
    global_step = 0

    best_objective = math.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    first_epoch = 1
    partial_output = best_partial_path(output)

    if settings.resume_checkpoint is not None:
        recovery = cast(
            dict[str, Any],
            torch.load(
                settings.resume_checkpoint,
                # RNG state restoration requires CPU ByteTensors. Loading the
                # recovery bundle on CPU also avoids temporarily materializing
                # the model and optimizer twice in accelerator memory; their
                # load_state_dict methods copy state to the live model device.
                map_location="cpu",
                weights_only=False,
            ),
        )
        if recovery.get("schema_version") != 1:
            raise ValueError("unsupported V11 recovery checkpoint")
        if recovery.get("model") != asdict(config.model):
            raise ValueError("resume checkpoint model configuration differs")
        if recovery.get("dataset_hashes") != dataset_hashes:
            raise ValueError("resume checkpoint dataset fingerprints differ")
        if recovery.get("training_signature") != training_signature:
            raise ValueError("resume checkpoint training configuration differs")
        model.load_state_dict(cast(dict[str, Tensor], recovery["model_state"]))
        optimizer.load_state_dict(cast(dict[str, Any], recovery["optimizer_state"]))
        global_step = int(recovery["global_step"])
        best_objective = float(recovery["best_objective"])
        best_epoch = int(recovery["best_epoch"])
        history = cast(list[dict[str, Any]], recovery["history"])
        best_state = cast(dict[str, Tensor], recovery["best_model_state"])
        rng.bit_generator.state = cast(dict[str, Any], recovery["numpy_rng_state"])
        random.setstate(recovery["python_rng_state"])
        torch.set_rng_state(cast(Tensor, recovery["torch_rng_state"]).cpu())
        if device.type == "cuda" and recovery.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(
                _cpu_cuda_rng_states(recovery["cuda_rng_state"])
            )
        first_epoch = int(recovery["completed_epoch"]) + 1
        if first_epoch > settings.epochs:
            raise ValueError("resume checkpoint has already completed all configured epochs")

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
        f"KingNet V11-BIG: {settings.epochs} epochs, "
        f"{settings.samples_per_epoch:,} samples/epoch, {steps_per_epoch:,} steps/epoch",
        flush=True,
    )
    print(f"device={device}; components={len(components)}", flush=True)

    for epoch in range(first_epoch, settings.epochs + 1):
        training_started = time.perf_counter()
        model.train()
        running = 0.0
        seen = 0
        for _ in range(steps_per_epoch):
            current_batch = min(settings.batch_size, settings.samples_per_epoch - seen)
            if current_batch <= 0:
                break
            indices, stm, cp_white, piece_count = _sample_batch(
                shards,
                components,
                current_batch,
                settings.mirror_probability,
                rng,
            )
            index_tensor, stm_tensor, target_cp, count_tensor = _batch_to_device(
                indices, stm, cp_white, piece_count, device
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
            prediction = model(index_tensor, stm_tensor, count_tensor)
            loss = _probability_loss(prediction, target_cp, config.model.cp_scale)
            loss.backward()  # type: ignore[no-untyped-call]
            if settings.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings.gradient_clip_norm
                )
            optimizer.step()

            running += float(loss.detach()) * current_batch
            seen += current_batch
            global_step += 1

        training_seconds = time.perf_counter() - training_started
        validation_started = time.perf_counter()
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
            set_objective = _selection_score(metrics, config.selection)
            validation[spec.name]["selection_objective"] = set_objective
            objective_numerator += spec.weight * set_objective
            objective_denominator += spec.weight

        validation_seconds = time.perf_counter() - validation_started
        objective = objective_numerator / objective_denominator
        epoch_record = {
            "epoch": epoch,
            "train_probability_mse": running / seen,
            "validation_objective": objective,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "training_seconds": training_seconds,
            "training_samples_per_second": seen / training_seconds,
            "validation_seconds": validation_seconds,
            "validation": validation,
        }
        history.append(epoch_record)

        print(
            f"epoch {epoch:02d}: train={running / seen:.6f} "
            f"val={objective:.6f} lr={optimizer.param_groups[0]['lr']:.3g} "
            f"throughput={seen / training_seconds:,.0f} samples/s",
            flush=True,
        )

        if objective < best_objective:
            best_objective = objective
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            # This is a complete runtime model, not merely recovery state. It
            # is exported atomically so an interrupted long run always leaves
            # its best validated epoch ready for engine testing.
            export_model(
                model,
                partial_output,
                feature_storage=config.export.feature_storage,
            )
            print(
                f"updated deployable best: {partial_output} "
                f"(epoch {best_epoch}, objective {best_objective:.6f})",
                flush=True,
            )
        if best_state is None:
            raise RuntimeError("best model state was not initialized")
        recovery_state: dict[str, Any] = {
            "schema_version": 1,
            "completed_epoch": epoch,
            "global_step": global_step,
            "model": asdict(config.model),
            "dataset_hashes": dataset_hashes,
            "training_signature": training_signature,
            "model_state": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "best_model_state": best_state,
            "optimizer_state": optimizer.state_dict(),
            "best_objective": best_objective,
            "best_epoch": best_epoch,
            "history": history,
            "numpy_rng_state": rng.bit_generator.state,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
        }
        _atomic_torch_save(recovery_state, checkpoint)

    if best_epoch == 0 or best_state is None or not checkpoint.is_file():
        raise RuntimeError("training completed without a best checkpoint")

    model.cpu()
    model.load_state_dict(best_state)
    if not partial_output.is_file():
        # Covers recovery from a checkpoint created before partial exports were
        # introduced, or a run directory copied without its partial artifact.
        export_model(
            model,
            partial_output,
            feature_storage=config.export.feature_storage,
        )
    export_model(model, output, feature_storage=config.export.feature_storage)

    provenance_train = [
        {
            **asdict(spec),
            "path": spec.path.as_posix(),
            "sha256": dataset_hashes[spec.name],
            "rows": len(shards[index].records),
        }
        for index, spec in enumerate(config.train_shards)
    ]
    provenance_validation = [
        {
            **asdict(spec),
            "path": spec.path.as_posix(),
            "sha256": dataset_hashes[spec.name],
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
        "schema_version": 3,
        "architecture": (
            "kingnet_v11_big_"
            f"acc{config.model.accumulator}_"
            f"pair{config.model.pairwise_width}_"
            f"hidden{config.model.hidden}_"
            f"heads{config.model.head_count}_dual"
        ),
        "config_path": config_path.resolve().as_posix(),
        "config_sha256": _sha256(config_path),
        "trainer_path": Path(__file__).resolve().as_posix(),
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "model": asdict(config.model),
        "export": asdict(config.export),
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
        "selection_objective": asdict(config.selection),
        "sampling_components": component_manifest,
        "train_shards": provenance_train,
        "validation_sets": provenance_validation,
        "init_model": init_provenance,
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "history": history,
        "output": output.resolve().as_posix(),
        "output_sha256": _sha256(output),
        "best_partial_output": partial_output.resolve().as_posix(),
        "best_partial_output_sha256": _sha256(partial_output),
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
