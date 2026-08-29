from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from jhtdb_pipeline.dashboard import (
    _symmetric_color_limit,
    _symlog_transform,
    _global_totals_figure,
    complete_result_paths,
    extract_gradient_slice,
    extract_scalar_slice,
    extract_slice,
    full_index_to_crop,
    spatial_axis_length,
)


class DashboardTests(unittest.TestCase):
    def test_robust_symmetric_color_limit_ignores_sparse_extreme(self) -> None:
        values = np.concatenate((np.linspace(-2.0, 2.0, 10_000), [1_000.0]))
        robust_limit = _symmetric_color_limit(values, 99.0)
        self.assertLess(robust_limit, 2.0)
        self.assertGreater(robust_limit, 1.9)
        self.assertEqual(_symmetric_color_limit(values), 1_000.0)

    def test_symmetric_color_limit_handles_empty_and_constant_fields(self) -> None:
        self.assertEqual(_symmetric_color_limit(np.array([np.nan, np.inf])), 1.0)
        self.assertEqual(_symmetric_color_limit(np.zeros((2, 2)), 99.0), 1.0)
        with self.assertRaisesRegex(ValueError, "percentile"):
            _symmetric_color_limit(np.ones((2, 2)), 0.0)

    def test_symlog_transform_is_symmetric_and_preserves_zero(self) -> None:
        values = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        transformed = _symlog_transform(values, 1.0)
        self.assertEqual(transformed[2], 0.0)
        np.testing.assert_allclose(transformed[:2], -transformed[:2:-1])
        with self.assertRaisesRegex(ValueError, "linear_threshold"):
            _symlog_transform(values, 0.0)

    def test_slice_axis_mapping(self) -> None:
        vector = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
        scalar = np.arange(4 * 5 * 6).reshape(4, 5, 6)
        gradient = np.arange(3 * 3 * 4 * 5 * 6).reshape(3, 3, 4, 5, 6)
        np.testing.assert_array_equal(extract_slice(vector, 1, "x", 2), vector[1, :, :, 2])
        np.testing.assert_array_equal(extract_scalar_slice(scalar, "y", 3), scalar[:, 3, :])
        np.testing.assert_array_equal(
            extract_gradient_slice(gradient, 1, 2, "z", 3), gradient[1, 2, 3, :, :]
        )
        self.assertEqual(spatial_axis_length(vector, "x"), 6)
        self.assertEqual(spatial_axis_length(vector, "y"), 5)
        self.assertEqual(spatial_axis_length(vector, "z"), 4)

    def test_full_index_maps_to_center_crop_only_when_present(self) -> None:
        self.assertEqual(full_index_to_crop(256, "x", (256, 256, 256), (512, 512, 512)), 0)
        self.assertEqual(full_index_to_crop(767, "z", (256, 256, 256), (512, 512, 512)), 511)
        self.assertIsNone(full_index_to_crop(255, "y", (256, 256, 256), (512, 512, 512)))
        self.assertIsNone(full_index_to_crop(768, "x", (256, 256, 256), (512, 512, 512)))

    def test_global_totals_bar_order(self) -> None:
        report = {
            "scope": "full_domain",
            "global_totals": {
                "s_bar": 1.0,
                "pi": 2.0,
                "work_resolved": 3.0,
                "work_full": 4.0,
            },
        }
        figure = _global_totals_figure(report)
        self.assertEqual(list(figure.data[0].y), [1.0, 2.0, 3.0, 4.0])

    def test_only_complete_persistent_results_are_listed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root / "t000001_sigma_1"
            incomplete = root / "t000002_sigma_1"
            hidden = root / ".staging"
            for path in (complete, incomplete, hidden):
                path.mkdir()
            (complete / "COMPLETE").write_text("{}\n", encoding="utf-8")
            self.assertEqual(complete_result_paths(root), [complete])


if __name__ == "__main__":
    unittest.main()
