from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import zarr

from jhtdb_pipeline.catalog import Catalog
from jhtdb_pipeline.config import load_config
from jhtdb_pipeline.gradients import (
    ARRAY_AXIS_FOR_DERIVATIVE,
    _write_verified_field,
    audit_gradients,
    compute_gradients,
    validate_divergence,
)
from jhtdb_pipeline.physics import compute_snapshot, spectral_derivative, spectral_gaussian
from jhtdb_pipeline.planning import Tile
from jhtdb_pipeline.store import VelocityStore
from jhtdb_pipeline.validation import require_divergence_validation


class GradientTests(unittest.TestCase):
    @staticmethod
    def _write_divergence_fixture(root: Path, *, compressible: bool):
        cfg = replace(
            load_config("configs/pipeline.yaml"),
            grid_shape=(16, 16, 16),
            tile_shape=(8, 8, 8),
            storage_root=root,
            safety_free_space_gib=0.0,
        )
        coordinates = np.arange(16, dtype=np.float32) * (2.0 * np.pi / 16)
        z, y, x = np.meshgrid(
            coordinates, coordinates, coordinates, indexing="ij"
        )
        velocity = np.zeros((3, 16, 16, 16), dtype=np.float32)
        if compressible:
            velocity[0] = np.sin(x)
        else:
            velocity[0] = np.sin(x) * np.cos(y)
            velocity[1] = -np.cos(x) * np.sin(y)

        store = VelocityStore(cfg)
        raw_array = store.ensure_snapshot(1, 0.0)
        raw_array[:] = velocity

        with Catalog(cfg.catalog_path) as catalog:
            catalog.plan_snapshot(
                cfg.dataset, 1, 0.0, [Tile(0, 0, 0, 16, 16, 16)]
            )
            catalog.set_snapshot_status(cfg.dataset, 1, "auto_validated", "raw-hash")

        gradient_root = zarr.open_group(str(cfg.gradient_store_path), mode="w")
        gradient_group = gradient_root.require_group("t000001")
        gradient_group.attrs.update(
            {
                "status": "complete",
                "input_manifest_hash": "raw-hash",
                "manifest_hash": "gradient-hash",
            }
        )
        gradient = gradient_group.create_dataset(
            "gradient",
            shape=(3, 3, 16, 16, 16),
            chunks=(1, 1, 8, 8, 8),
            dtype="<f4",
        )
        for velocity_component in range(3):
            for derivative_component in range(3):
                gradient[velocity_component, derivative_component] = spectral_derivative(
                    velocity[velocity_component],
                    ARRAY_AXIS_FOR_DERIVATIVE[derivative_component],
                    2.0 * np.pi,
                )
        return cfg

    def test_field_write_roundtrip_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            array = zarr.open_array(
                str(Path(directory) / "gradient.zarr"),
                mode="w",
                shape=(3, 3, 4, 5, 6),
                chunks=(1, 1, 2, 5, 6),
                dtype="<f4",
            )
            source = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
            digest, byte_count = _write_verified_field(
                source, array, 1, 2, tile_shape_xyz=(3, 5, 2)
            )
            self.assertEqual(len(digest), 64)
            self.assertEqual(byte_count, source.nbytes)
            np.testing.assert_array_equal(array[1, 2], source)

    def test_small_zarr_fd8_audit_covers_all_nine_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = replace(
                load_config("configs/pipeline.yaml"),
                grid_shape=(32, 32, 32),
                tile_shape=(8, 8, 8),
                storage_root=root,
                safety_free_space_gib=0.0,
                fft_slab_width=2,
            )
            coordinates = np.arange(32, dtype=np.float32) * (2.0 * np.pi / 32)
            z, y, x = np.meshgrid(
                coordinates, coordinates, coordinates, indexing="ij"
            )
            velocity = np.empty((3, 32, 32, 32), dtype=np.float32)
            for component in range(3):
                scale = float(component + 1)
                velocity[component] = scale * (
                    np.sin(x) + 0.5 * np.cos(2.0 * y) + 0.25 * np.sin(3.0 * z)
                )

            store = VelocityStore(cfg)
            raw_array = store.ensure_snapshot(1, 0.0)
            raw_array[:] = velocity
            with Catalog(cfg.catalog_path) as catalog:
                catalog.plan_snapshot(
                    cfg.dataset, 1, 0.0, [Tile(0, 0, 0, 32, 32, 32)]
                )
                catalog.set_snapshot_status(cfg.dataset, 1, "auto_validated", "raw-hash")

            gradient_root = zarr.open_group(str(cfg.gradient_store_path), mode="w")
            gradient_group = gradient_root.require_group("t000001")
            gradient_group.attrs.update(
                {
                    "status": "complete",
                    "input_manifest_hash": "raw-hash",
                    "manifest_hash": "gradient-hash",
                }
            )
            gradient = gradient_group.create_dataset(
                "gradient",
                shape=(3, 3, 32, 32, 32),
                chunks=(1, 1, 8, 8, 8),
                dtype="<f4",
            )
            for velocity_component in range(3):
                for derivative_component in range(3):
                    gradient[velocity_component, derivative_component] = spectral_derivative(
                        velocity[velocity_component],
                        ARRAY_AXIS_FOR_DERIVATIVE[derivative_component],
                        2.0 * np.pi,
                    )

            report = audit_gradients(
                cfg, 1, core_size=16, origin_xyz=(8, 8, 8)
            )
            self.assertEqual(len(report["comparisons"]), 9)
            self.assertLess(
                report["aggregate"]["relative_difference_rms"], 2e-3
            )
            self.assertGreater(
                report["aggregate"]["mean_cosine_similarity"], 0.999
            )
            self.assertTrue((cfg.qa_path / "gradient_audit_t000001.json").exists())

    def test_full_domain_divergence_passes_solenoidal_field(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._write_divergence_fixture(Path(directory), compressible=False)
            report = validate_divergence(cfg, 1)
            self.assertTrue(report["passed"])
            self.assertEqual(report["point_count"], 16**3)
            self.assertEqual(report["chunk_count"], 8)
            self.assertLess(report["relative_divergence_rms"], 1.0e-6)
            self.assertLess(report["relative_maximum_divergence"], 1.0e-6)
            self.assertTrue((cfg.qa_path / "divergence_t000001.json").exists())
            accepted = require_divergence_validation(
                cfg, 1, "raw-hash", "gradient-hash"
            )
            self.assertTrue(accepted["passed"])
            with self.assertRaisesRegex(RuntimeError, "different gradient manifest"):
                require_divergence_validation(
                    cfg, 1, "raw-hash", "other-gradient-hash"
                )

    def test_full_domain_divergence_rejects_compressible_field(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._write_divergence_fixture(Path(directory), compressible=True)
            report = validate_divergence(cfg, 1)
            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "failed")
            self.assertGreater(report["relative_divergence_rms"], 0.9)
            self.assertGreater(report["relative_maximum_divergence"], 0.9)

    def test_filtered_preprocessing_and_physics_use_managed_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self._write_divergence_fixture(Path(directory), compressible=False)
            compute_gradients(cfg, 1)
            raw = zarr.open_group(str(cfg.raw_store_path), mode="r")["t000001"][
                "velocity"
            ]

            filtered = zarr.open_group(str(cfg.filtered_store_path), mode="r")[
                "t000001"
            ]
            self.assertEqual(filtered.attrs["status"], "complete")
            self.assertEqual(len(filtered.attrs["field_hashes"]), 12)
            expected_velocity_bar = spectral_gaussian(
                np.asarray(raw[0], dtype=np.float32), cfg.sigma_grid
            )
            np.testing.assert_allclose(
                filtered["velocity_bar"][0],
                expected_velocity_bar,
                rtol=2.0e-5,
                atol=2.0e-6,
            )
            expected_gradient_bar = spectral_derivative(
                expected_velocity_bar,
                ARRAY_AXIS_FOR_DERIVATIVE[0],
                cfg.domain_length,
            )
            np.testing.assert_allclose(
                filtered["gradient_bar"][0, 0],
                expected_gradient_bar,
                rtol=2.0e-5,
                atol=2.0e-6,
            )

            compute_snapshot(cfg, 1)
            physics = zarr.open_group(str(cfg.derived_store_path), mode="r")[
                "t000001"
            ]
            self.assertEqual(physics.attrs["status"], "complete")
            self.assertNotIn("velocity_bar", physics)
            self.assertNotIn("gradient_bar", physics)
            self.assertTrue(np.all(np.isfinite(physics["work_full"][:])))
            self.assertTrue(np.all(np.isfinite(physics["work_resolved"][:])))


if __name__ == "__main__":
    unittest.main()
