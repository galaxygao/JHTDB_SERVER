from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from jhtdb_pipeline.physics import (
    ARRAY_AXIS_FOR_DERIVATIVE,
    close_memmap,
    derivative_field,
    filter_field,
    memmap,
    regime_codes,
    spectral_derivative,
    spectral_gaussian,
)


class PhysicsTests(unittest.TestCase):
    def test_periodic_spectral_derivative(self) -> None:
        n = 64
        x = np.arange(n, dtype=np.float32) * (2.0 * np.pi / n)
        derivative = spectral_derivative(np.sin(3.0 * x), 0, 2.0 * np.pi)
        np.testing.assert_allclose(
            derivative, 3.0 * np.cos(3.0 * x), rtol=2e-5, atol=2e-5
        )

    def test_periodic_gaussian_preserves_constant(self) -> None:
        field = np.ones((16, 16, 16), dtype=np.float32)
        np.testing.assert_allclose(
            spectral_gaussian(field, 1.0), field, rtol=0, atol=1e-6
        )

    def test_regime_codes(self) -> None:
        full = np.asarray([2.0, 2.0, -2.0, -2.0, 0.0])
        resolved = np.asarray([2.0, -2.0, 2.0, -2.0, 1.0])
        codes, _, _ = regime_codes(full, resolved, 0.1, 0.0)
        np.testing.assert_array_equal(codes, [1, 2, 3, 4, 0])

    def test_all_gradient_axes(self) -> None:
        self.assertEqual(ARRAY_AXIS_FOR_DERIVATIVE, (2, 1, 0))
        n = 24
        coordinates = np.arange(n, dtype=np.float32) * (2.0 * np.pi / n)
        z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
        field = np.sin(x) + 0.5 * np.cos(2 * y) + 0.25 * np.sin(3 * z)
        expected = (np.cos(x), -np.sin(2 * y), 0.75 * np.cos(3 * z))
        with tempfile.TemporaryDirectory() as temporary:
            output = memmap(Path(temporary) / "derivative.f32", field.shape)
            try:
                for component in range(3):
                    derivative_field(field, output, component, 2.0 * np.pi, slab=3)
                    np.testing.assert_allclose(
                        output, expected[component], rtol=3e-5, atol=3e-5
                    )
            finally:
                close_memmap(output)

    def test_streaming_filter_matches_in_memory_filter(self) -> None:
        n = 16
        rng = np.random.default_rng(7)
        field = rng.normal(size=(n, n, n)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = memmap(root / "output.f32", field.shape)
            temp_a = memmap(root / "a.f32", field.shape)
            temp_b = memmap(root / "b.f32", field.shape)
            try:
                filter_field(field, output, temp_a, temp_b, 1.0, slab=2, workers=2)
                np.testing.assert_allclose(
                    output, spectral_gaussian(field, 1.0), rtol=2e-5, atol=2e-6
                )
            finally:
                close_memmap(output)
                close_memmap(temp_a)
                close_memmap(temp_b)


if __name__ == "__main__":
    unittest.main()
