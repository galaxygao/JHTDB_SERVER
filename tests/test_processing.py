from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import zarr

from jhtdb_pipeline.catalog import Catalog
from jhtdb_pipeline.config import load_config, result_zarr_name
from jhtdb_pipeline.physics import spectral_derivative, spectral_gaussian
from jhtdb_pipeline.planning import Tile
from jhtdb_pipeline.processing import finalize_result, process_center, resource_plan
from jhtdb_pipeline.store import VelocityStore, open_complete_result


def fixture(root: Path, *, compressible: bool = False):
    cfg = replace(
        load_config("configs/pipeline.yaml"),
        grid_shape=(16, 16, 16),
        request_shape=(16, 16, 16),
        tile_shape=(8, 8, 8),
        crop_start=(4, 4, 4),
        crop_shape=(8, 8, 8),
        state_root=root / "state",
        run_root=root / "runs",
        result_root=root / "results",
        persistent_safety_reserve_gib=0.0,
        scratch_safety_reserve_gib=0.0,
        fft_workers=2,
        fft_slab_width=2,
        cleanup_scratch_on_success=True,
    )
    coordinates = np.arange(16, dtype=np.float32) * (2.0 * np.pi / 16)
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    velocity = np.zeros((3, 16, 16, 16), dtype=np.float32)
    if compressible:
        velocity[0] = np.sin(x)
    else:
        velocity[0] = np.sin(x) * np.cos(y)
        velocity[1] = -np.cos(x) * np.sin(y)

    store = VelocityStore(cfg, 1)
    store.ensure_array()
    store.array[:] = velocity
    store.root.attrs.update({"status": "validated", "manifest_hash": "input-hash"})
    with Catalog(cfg.catalog_path) as catalog:
        catalog.plan_snapshot(
            cfg.dataset, 1, 0.0, [Tile(0, 0, 0, 16, 16, 16)]
        )
        catalog.set_snapshot_status(cfg.dataset, 1, "validated", "input-hash")
    return cfg, velocity


class ProcessingTests(unittest.TestCase):
    def test_center_pipeline_and_persistent_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, velocity = fixture(Path(temporary))
            plan = resource_plan(cfg)
            self.assertEqual(plan["persistent_result_GiB"], 512 * 113 / 1024**3)
            self.assertEqual(plan["configured_sigma_count"], 3)
            self.assertEqual(plan["persistent_batch_GiB"], 3 * 512 * 113 / 1024**3)
            staging = process_center(cfg, 1)
            self.assertTrue((staging / result_zarr_name(cfg.sigma_grid)).is_dir())
            final = finalize_result(cfg, 1)
            self.assertTrue((final / "COMPLETE").is_file())
            self.assertTrue((final / "manifest.json").is_file())
            result = open_complete_result(final)
            self.assertEqual(result["velocity"].shape, (3, 8, 8, 8))
            self.assertEqual(result["gradient"].shape, (3, 3, 8, 8, 8))
            crop = (slice(4, 12), slice(4, 12), slice(4, 12))
            np.testing.assert_allclose(result["velocity"][0], velocity[(0,) + crop])
            expected_filtered = spectral_gaussian(velocity[0], cfg.sigma_grid)[crop]
            np.testing.assert_allclose(
                result["velocity_bar"][0], expected_filtered, rtol=2e-5, atol=2e-6
            )
            expected_gradient = spectral_derivative(
                velocity[0], 2, cfg.domain_length
            )[crop]
            np.testing.assert_allclose(
                result["gradient"][0, 0], expected_gradient, rtol=2e-5, atol=2e-6
            )
            self.assertTrue(np.all(np.isfinite(result["work_full"][:])))
            self.assertTrue(np.all(np.isfinite(result["work_resolved"][:])))
            filtered = np.stack(
                [spectral_gaussian(velocity[i], cfg.sigma_grid) for i in range(3)]
            )
            gradient_bar = np.empty((3, 3, 16, 16, 16), dtype=np.float32)
            tau = np.empty((3, 3, 16, 16, 16), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    gradient_bar[i, j] = spectral_derivative(
                        filtered[i], 2 - j, cfg.domain_length
                    )
                    tau[i, j] = (
                        spectral_gaussian(velocity[i] * velocity[j], cfg.sigma_grid)
                        - filtered[i] * filtered[j]
                    )
            expected_pi = np.einsum("ijzyx,ijzyx->zyx", tau, gradient_bar)
            transport = np.einsum("izyx,ijzyx->jzyx", filtered, tau)
            expected_s_bar = sum(
                spectral_derivative(transport[j], 2 - j, cfg.domain_length)
                for j in range(3)
            )
            np.testing.assert_allclose(
                result["pi"][:], expected_pi[crop], rtol=3e-5, atol=3e-6
            )
            np.testing.assert_allclose(
                result["s_bar"][:], expected_s_bar[crop], rtol=3e-5, atol=3e-6
            )
            np.testing.assert_allclose(
                result["work_full"][:],
                result["work_resolved"][:] - result["pi"][:] + result["s_bar"][:],
                rtol=5e-5,
                atol=5e-6,
            )
            self.assertFalse(cfg.workspace_path(1).exists())
            self.assertFalse(any(final.rglob("*.part-*")))
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(set(manifest["fields"]), {
                "velocity", "gradient", "velocity_bar", "gradient_bar",
                "work_full", "work_resolved", "pi", "s_bar", "regime",
            })

    def test_legacy_complete_result_is_replaced_only_after_new_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, _ = fixture(Path(temporary))
            final = cfg.result_path(1)
            legacy = zarr.open_group(str(final / "center_result.zarr"), mode="w")
            legacy.attrs["status"] = "complete"
            (final / "COMPLETE").write_text("{}\n", encoding="utf-8")

            staging = process_center(cfg, 1)
            self.assertTrue((final / "COMPLETE").is_file())
            self.assertTrue((final / "center_result.zarr").is_dir())
            self.assertTrue((staging / result_zarr_name(cfg.sigma_grid)).is_dir())

            replaced = finalize_result(cfg, 1)
            self.assertEqual(replaced, final)
            self.assertFalse((final / "center_result.zarr").exists())
            self.assertTrue((final / result_zarr_name(cfg.sigma_grid)).is_dir())
            self.assertIn("pi", open_complete_result(final))

    def test_divergence_failure_never_creates_complete_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, _ = fixture(Path(temporary), compressible=True)
            with self.assertRaisesRegex(RuntimeError, "divergence"):
                process_center(cfg, 1)
            self.assertFalse((cfg.result_path(1) / "COMPLETE").exists())
            report = json.loads(
                (cfg.staging_result_path(1) / "divergence.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
