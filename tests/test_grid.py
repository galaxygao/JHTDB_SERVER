import tempfile
import unittest
from pathlib import Path

import numpy as np

from jhtdb_regimes.config import load_config
from jhtdb_regimes.grid import block_indices, point_batches, query_points, rows_to_gradient, rows_to_velocity


CONFIG = Path(__file__).parents[1] / "configs" / "task0.yaml"


class GridTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIG)

    def test_grid_order_and_coordinates(self):
        indices = block_indices(self.cfg)
        points = query_points(self.cfg)
        self.assertEqual(indices.shape, (4096, 3))
        np.testing.assert_array_equal(indices[0], [504, 504, 504])
        np.testing.assert_array_equal(indices[1], [505, 504, 504])
        np.testing.assert_allclose(points[0], indices[0] * 2 * np.pi / 1024)

    def test_testing_token_batches(self):
        batches = point_batches(4096, 4000)
        self.assertEqual([(s.start, s.stop) for s in batches], [(0, 4000), (4000, 4096)])

    def test_row_reshape(self):
        velocity_rows = np.arange(2 * 4096 * 3).reshape(2, 4096, 3)
        velocity = rows_to_velocity(velocity_rows, self.cfg.block_shape)
        self.assertEqual(velocity.shape, (2, 3, 16, 16, 16))
        np.testing.assert_array_equal(velocity[0, :, 0, 0, 1], velocity_rows[0, 1])
        gradient_rows = np.arange(4096 * 9).reshape(1, 4096, 3, 3)
        gradient = rows_to_gradient(gradient_rows, self.cfg.block_shape)
        self.assertEqual(gradient.shape, (1, 3, 3, 16, 16, 16))
        np.testing.assert_array_equal(gradient[0, :, :, 0, 0, 1], gradient_rows[0, 1])


if __name__ == "__main__":
    unittest.main()

