import unittest

import numpy as np

from jhtdb_regimes.jhtdb_client import GRADIENT_NAMES, parse_gradient, parse_velocity


class ColumnTests(unittest.TestCase):
    def test_velocity_shuffled_columns(self):
        columns = ["uz", "ux", "uy"]
        values = np.array([[[30.0, 10.0, 20.0]]])
        parsed = parse_velocity(values, columns)
        np.testing.assert_array_equal(parsed, [[[10.0, 20.0, 30.0]]])

    def test_gradient_shuffled_columns(self):
        columns = list(reversed(GRADIENT_NAMES))
        values = np.arange(9, dtype=float)[None, None, :]
        parsed = parse_gradient(values, columns)
        expected_order = np.arange(8, -1, -1).reshape(1, 1, 3, 3)
        np.testing.assert_array_equal(parsed, expected_order)

    def test_missing_column_fails(self):
        with self.assertRaises(ValueError):
            parse_velocity(np.zeros((1, 1, 2)), ["ux", "uy"])


if __name__ == "__main__":
    unittest.main()

