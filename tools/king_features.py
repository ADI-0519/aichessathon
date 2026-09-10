"""Shared feature constants for the factored king-bucket evaluator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

BASE_FEATURE_COUNT = 12 * 64
BASE_PADDING_INDEX = BASE_FEATURE_COUNT
KING_BUCKET_COUNT = 16
FEATURE_COUNT = KING_BUCKET_COUNT * BASE_FEATURE_COUNT
PADDING_INDEX = FEATURE_COUNT

# After perspective normalisation, the moving side's king is always slot 5.
OWN_KING_SLOT = 5


def king_bucket(square: int) -> int:
    return ((square >> 3) >> 1) * 4 + ((square & 7) >> 1)


KING_BUCKETS: NDArray[np.int16] = np.array(
    [king_bucket(square) for square in range(64)], dtype=np.int16
)


def feature_index(bucket: int, base_index: int) -> int:
    return bucket * BASE_FEATURE_COUNT + base_index
