from __future__ import annotations

import unittest

import numpy as np

from jhtdb_pipeline.config import load_config
from jhtdb_pipeline.gradients import (
    ARRAY_AXIS_FOR_DERIVATIVE,
    filtered_space_plan,
    finite_difference_core,
    gradient_space_plan,
)
from jhtdb_pipeline.physics import (
    _acceleration_from_gradient,
    physics_space_plan,
    regime_codes,
    spectral_derivative,
    spectral_gaussian,
)


class PhysicsTests(unittest.TestCase):
    def test_periodic_spectral_derivative(self):
        n = 64
        x = np.arange(n, dtype=np.float32) * (2.0 * np.pi / n)
        field = np.sin(3.0 * x).astype(np.float32)
        derivative = spectral_derivative(field, 0, 2.0 * np.pi)
        np.testing.assert_allclose(derivative, 3.0 * np.cos(3.0 * x), rtol=2e-5, atol=2e-5)

    def test_periodic_gaussian_preserves_constant(self):
        field = np.ones((16, 16, 16), dtype=np.float32)
        filtered = spectral_gaussian(field, 1.0)
        np.testing.assert_allclose(filtered, field, rtol=0, atol=1e-6)

    def test_regime_codes(self):
        full = np.asarray([2.0, 2.0, -2.0, -2.0, 0.0])
        resolved = np.asarray([2.0, -2.0, 2.0, -2.0, 1.0])
        codes, _, _ = regime_codes(full, resolved, 0.1, 0.0)
        np.testing.assert_array_equal(codes, [1, 2, 3, 4, 0])

    def test_gradient_axis_mapping_and_advective_acceleration(self):
        self.assertEqual(ARRAY_AXIS_FOR_DERIVATIVE, (2, 1, 0))
        n = 8
        x = np.arange(n, dtype=np.float32) * (2.0 * np.pi / n)
        velocity = np.zeros((3, n, n, n), dtype=np.float32)
        gradient = np.zeros((3, 3, n, n, n), dtype=np.float32)
        velocity[0] = np.sin(x)[None, None, :]
        gradient[0, 0] = np.cos(x)[None, None, :]
        output = np.empty((n, n, n), dtype=np.float32)
        _acceleration_from_gradient(velocity, gradient, 0, output, slab=2)
        expected = (np.sin(x) * np.cos(x))[None, None, :]
        np.testing.assert_allclose(output, np.broadcast_to(expected, output.shape), atol=1e-6)

    def test_all_nine_spectral_gradient_directions(self):
        n = 32
        coordinates = np.arange(n, dtype=np.float32) * (2.0 * np.pi / n)
        z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
        coefficients = np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        velocity = np.empty((3, n, n, n), dtype=np.float32)
        expected = np.empty((3, 3, n, n, n), dtype=np.float32)
        for component in range(3):
            cx, cy, cz = coefficients[component]
            velocity[component] = (
                cx * np.sin(x) + cy * np.cos(2.0 * y) + cz * np.sin(3.0 * z)
            )
            expected[component, 0] = cx * np.cos(x)
            expected[component, 1] = -2.0 * cy * np.sin(2.0 * y)
            expected[component, 2] = 3.0 * cz * np.cos(3.0 * z)
        for component in range(3):
            for derivative_component in range(3):
                actual = spectral_derivative(
                    velocity[component],
                    ARRAY_AXIS_FOR_DERIVATIVE[derivative_component],
                    2.0 * np.pi,
                )
                np.testing.assert_allclose(
                    actual,
                    expected[component, derivative_component],
                    rtol=2e-5,
                    atol=3e-5,
                )

    def test_fd8_small_block_matches_fft_for_low_periodic_mode(self):
        n = 64
        core_size = 32
        halo = 4
        origin = 12
        coordinates = np.arange(n, dtype=np.float32) * (2.0 * np.pi / n)
        z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
        field = np.sin(2.0 * x) + 0.5 * np.cos(y) + 0.25 * np.sin(3.0 * z)
        haloed = field[
            origin - halo : origin + core_size + halo,
            origin - halo : origin + core_size + halo,
            origin - halo : origin + core_size + halo,
        ]
        fft_derivative = spectral_derivative(field, 2, 2.0 * np.pi)[
            origin : origin + core_size,
            origin : origin + core_size,
            origin : origin + core_size,
        ]
        fd8_derivative = finite_difference_core(
            haloed, 2.0 * np.pi / n, 2, core_size, halo
        )
        difference_rms = np.sqrt(np.mean(np.square(fd8_derivative - fft_derivative)))
        reference_rms = np.sqrt(np.mean(np.square(fft_derivative)))
        self.assertLess(difference_rms / reference_rms, 2e-5)

    def test_resource_plans_are_bounded(self):
        cfg = load_config("configs/pipeline.yaml")
        gradient_plan = gradient_space_plan(cfg)
        filtered_plan = filtered_space_plan(cfg)
        physics_plan = physics_space_plan(cfg)
        self.assertEqual(gradient_plan["gradient_uncompressed_GiB"], 36.0)
        self.assertEqual(gradient_plan["scratch_GiB"], 4.0)
        self.assertEqual(filtered_plan["filtered_total_uncompressed_GiB"], 48.0)
        self.assertEqual(filtered_plan["scratch_GiB"], 12.0)
        self.assertEqual(filtered_plan["full_domain_axis_transform_passes"], 18.0)
        self.assertEqual(physics_plan["scratch_GiB"], 16.0)
        self.assertEqual(physics_plan["derived_uncompressed_GiB"], 9.0)
        self.assertEqual(physics_plan["required_free_GiB"], 65.0)
        self.assertEqual(physics_plan["fft_input_block_MiB"], 16.0)


if __name__ == "__main__":
    unittest.main()
