import argparse
from pathlib import Path

import numpy as np

INPUT_SCALE = 2_048
WEIGHT_SCALE = 2_048
FEATURE_COUNT = 64 * 12 * 64


def quantise(value: np.ndarray, scale: float, dtype: type) -> np.ndarray:
    out = np.rint(value * scale)
    limit = np.iinfo(dtype)
    if out.min() < limit.min or out.max() > limit.max:
        raise SystemExit(
            f"quantisation overflows {dtype.__name__}: "
            f"range {out.min()} to {out.max()}, limit {limit.min} to {limit.max}"
        )
    return out.astype(dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.model, allow_pickle=False) as archive:
        if int(archive["format_version"]) != 2:
            raise SystemExit("unexpected source format version")
        cp_scale = float(archive["cp_scale"])
        feature_weights = archive["feature_weights"]
        accumulator_bias = archive["accumulator_bias"]
        hidden_weights = archive["hidden_weights"]
        hidden_bias = archive["hidden_bias"]
        output_weights = archive["output_weights"]
        output_bias = archive["output_bias"]

    if feature_weights.shape[0] != FEATURE_COUNT:
        raise SystemExit(
            f"{args.model} has {feature_weights.shape[0]} features; "
            f"v6_halfkp requires {FEATURE_COUNT} (64 king squares x 768)"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        format_version=np.asarray(2, dtype=np.int32),
        cp_scale=np.asarray(cp_scale, dtype=np.float32),
        input_scale=np.asarray(INPUT_SCALE, dtype=np.int32),
        weight_scale=np.asarray(WEIGHT_SCALE, dtype=np.int32),
        feature_weights_q=quantise(feature_weights, INPUT_SCALE, np.int16),
        accumulator_bias_q=quantise(accumulator_bias, INPUT_SCALE, np.int32),
        hidden_weights_q=quantise(hidden_weights, WEIGHT_SCALE, np.int16),
        hidden_bias_q=quantise(hidden_bias, INPUT_SCALE * WEIGHT_SCALE, np.int32),
        output_weights_q=quantise(output_weights, WEIGHT_SCALE, np.int16),
        output_bias_q=quantise(
            np.reshape(output_bias, (1,)), INPUT_SCALE * WEIGHT_SCALE, np.int64
        ),
    )
    accumulator = feature_weights.shape[1]
    print(
        f"wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KiB), "
        f"{FEATURE_COUNT} features x {accumulator} accumulator"
    )


if __name__ == "__main__":
    main()
