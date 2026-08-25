import unittest

import numpy as np

from jhtdb_regimes.physics import (
    advective_acceleration,
    divergence,
    gaussian_kernel_1d,
    gaussian_valid,
    regime_codes,
    robust_regime,
    work_and_fields,
)
from jhtdb_regimes.verify import finite_difference_core


class PhysicsTests(unittest.TestCase):
    def test_gaussian_normalization_and_constant(self):
        kernel = gaussian_kernel_1d(1.0, 4)
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=14)
        result = gaussian_valid(np.ones((2, 3, 16, 16, 16)), 1.0, 4)
        self.assertEqual(result.shape, (2, 3, 8, 8, 8))
        np.testing.assert_allclose(result, 1.0, atol=1e-14)

    def test_acceleration_contracts_correct_axis(self):
        velocity = np.zeros((1, 3, 1, 1, 1))
        velocity[:, :, 0, 0, 0] = [2.0, 3.0, 5.0]
        gradient = np.zeros((1, 3, 3, 1, 1, 1))
        gradient[0, :, :, 0, 0, 0] = np.arange(1, 10).reshape(3, 3)
        result = advective_acceleration(velocity, gradient)
        expected = np.arange(1, 10).reshape(3, 3) @ np.array([2.0, 3.0, 5.0])
        np.testing.assert_array_equal(result[0, :, 0, 0, 0], expected)

    def test_divergence_trace(self):
        gradient = np.zeros((1, 3, 3, 2, 2, 2))
        gradient[:, 0, 0] = 1
        gradient[:, 1, 1] = 2
        gradient[:, 2, 2] = 3
        np.testing.assert_array_equal(divergence(gradient), 6 * np.ones((1, 2, 2, 2)))

    def test_four_regimes_boundaries_and_robust(self):
        full = np.array([2.0, 2.0, -2.0, -2.0, 0.0, np.nan])
        resolved = np.array([3.0, -3.0, 3.0, -3.0, 1.0, 1.0])
        codes, _, _ = regime_codes(full, resolved, 1e-12, 0.0)
        np.testing.assert_array_equal(codes, [1, 2, 3, 4, 0, 0])
        audit = np.array([1, 3, 3, 4, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(robust_regime(codes, audit), [1, 0, 3, 4, 0, 0])

    def test_full_derived_pipeline_shapes(self):
        rng = np.random.default_rng(1234)
        velocity = rng.normal(size=(2, 3, 16, 16, 16))
        gradient = rng.normal(size=(2, 3, 3, 16, 16, 16))
        result = work_and_fields(velocity, gradient, 1.0, 4)
        self.assertEqual(result["velocity_bar"].shape, (2, 3, 8, 8, 8))
        self.assertEqual(result["gradient_bar"].shape, (2, 3, 3, 8, 8, 8))
        self.assertEqual(result["acceleration_bar"].shape, (2, 3, 8, 8, 8))
        self.assertEqual(result["acceleration_barbar"].shape, (2, 3, 8, 8, 8))
        self.assertEqual(result["work_full"].shape, (2, 8, 8, 8))
        self.assertEqual(result["divergence_raw"].shape, (2, 8, 8, 8))

    def test_independent_finite_difference_on_linear_field(self):
        spacing = 0.25
        z, y, x = np.meshgrid(
            np.arange(16) * spacing,
            np.arange(16) * spacing,
            np.arange(16) * spacing,
            indexing="ij",
        )
        velocity = np.empty((1, 3, 16, 16, 16))
        velocity[0, 0] = 2 * x + 3 * y + 5 * z
        velocity[0, 1] = -x + 7 * y - 2 * z
        velocity[0, 2] = 11 * x - 13 * y + 17 * z
        expected = np.asarray([[2, 3, 5], [-1, 7, -2], [11, -13, 17]], dtype=float)
        for order in (6, 8):
            gradient = finite_difference_core(velocity, spacing, order, 4)
            expected_grid = np.broadcast_to(expected[None, :, :, None, None, None], gradient.shape)
            np.testing.assert_allclose(gradient, expected_grid, atol=2e-13)


if __name__ == "__main__":
    unittest.main()
