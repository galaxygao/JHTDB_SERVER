import unittest
from pathlib import Path

import numpy as np

from jhtdb_regimes.config import load_config
from jhtdb_regimes.dashboard import REGIME_LABELS, direct_filter, heatmap, plane
from jhtdb_regimes.physics import gaussian_valid


CONFIG = Path(__file__).parents[1] / "configs" / "task0.yaml"


class DashboardHelperTests(unittest.TestCase):
    def test_plane_axis_mapping(self):
        field = np.arange(8**3).reshape(8, 8, 8)
        np.testing.assert_array_equal(plane(field, "z", 2)[0], field[2, :, :])
        np.testing.assert_array_equal(plane(field, "y", 3)[0], field[:, 3, :])
        np.testing.assert_array_equal(plane(field, "x", 4)[0], field[:, :, 4])

    def test_dashboard_direct_filter_is_independent_match(self):
        cfg = load_config(CONFIG)
        field = np.random.default_rng(52).normal(size=(16, 16, 16))
        production = gaussian_valid(field, cfg.sigma_grid, cfg.support_radius)
        independent = direct_filter(field, cfg)
        np.testing.assert_allclose(production, independent, rtol=2e-13, atol=2e-13)

    def test_regime_legend_has_five_equal_categories(self):
        field = np.arange(5, dtype=np.uint8)[:, None, None] * np.ones((5, 2, 2), dtype=np.uint8)
        figure = heatmap(field, "regime", "z", 0, regime=True)
        trace = figure.data[0]
        self.assertEqual(tuple(trace.colorbar.tickvals), (0, 1, 2, 3, 4))
        self.assertEqual(tuple(trace.colorbar.ticktext), REGIME_LABELS)
        self.assertEqual(trace.zmin, -0.5)
        self.assertEqual(trace.zmax, 4.5)
        self.assertEqual(len(trace.colorscale), 10)


if __name__ == "__main__":
    unittest.main()
