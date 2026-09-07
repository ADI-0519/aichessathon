from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prune_halfkp import prune_model


def _write_model(path: Path, *, active_removed_connection: bool = False) -> None:
    hidden = np.arange(24, dtype=np.int16).reshape(3, 8)
    hidden[:, 2:4] = 0
    hidden[:, 6:8] = 0
    if active_removed_connection:
        hidden[0, 2] = 1
    np.savez(
        path,
        format_version=np.asarray(2, dtype=np.int32),
        cp_scale=np.asarray(400.0, dtype=np.float32),
        input_scale=np.asarray(2048, dtype=np.int32),
        weight_scale=np.asarray(2048, dtype=np.int32),
        feature_weights_q=np.arange(40, dtype=np.int16).reshape(10, 4),
        accumulator_bias_q=np.arange(4, dtype=np.int32),
        hidden_weights_q=hidden,
        hidden_bias_q=np.arange(3, dtype=np.int32),
        output_weights_q=np.arange(3, dtype=np.int16).reshape(1, 3),
        output_bias_q=np.asarray([7], dtype=np.int64),
    )


class PruneHalfKpTests(unittest.TestCase):
    def test_exactly_removes_disconnected_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.npz"
            output = root / "output.npz"
            _write_model(source)

            details = prune_model(source, output, 2)

            self.assertTrue(details["exact_graph_preserving_transform"])
            with np.load(source, allow_pickle=False) as before, np.load(
                output, allow_pickle=False
            ) as after:
                self.assertEqual(after["cp_scale"].shape, ())
                self.assertEqual(after["input_scale"].shape, ())
                self.assertEqual(after["weight_scale"].shape, ())
                np.testing.assert_array_equal(
                    after["feature_weights_q"], before["feature_weights_q"][:, :2]
                )
                np.testing.assert_array_equal(
                    after["accumulator_bias_q"], before["accumulator_bias_q"][:2]
                )
                expected_hidden = np.concatenate(
                    (before["hidden_weights_q"][:, :2], before["hidden_weights_q"][:, 4:6]),
                    axis=1,
                )
                np.testing.assert_array_equal(after["hidden_weights_q"], expected_hidden)

    def test_rejects_a_nonzero_removed_connection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.npz"
            _write_model(source, active_removed_connection=True)

            with self.assertRaisesRegex(ValueError, "non-zero outgoing connections"):
                prune_model(source, root / "output.npz", 2)


if __name__ == "__main__":
    unittest.main()
