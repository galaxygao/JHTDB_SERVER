from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from jhtdb_pipeline.catalog import Catalog
from jhtdb_pipeline.config import load_config
from jhtdb_pipeline.physics import spectral_derivative, spectral_gaussian
from jhtdb_pipeline.planning import Tile
from jhtdb_pipeline.processing import finalize_result, process_center, resource_plan
from jhtdb_pipeline.store import VelocityStore, open_complete_result


def fixture(root: Path, *, compressible: bool = False):
    cfg = replace(
        load_config("configs/pipeline.yaml"),
        grid_shape=(16, 16, 16),
        tile_shape=(8, 8, 8),
        crop_start=(4, 4, 4),
        crop_shape=(8, 8, 8),
        state_root=root / "state",
        run_root=root / "runs",
        result_root=root / "results",
        persistent_safety_reserve_gib=0.0,
        scratch_safety_reserve_gib=0.0,
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
            self.assertEqual(plan["persistent_result_GiB"], 512 * 105 / 1024**3)
            staging = process_center(cfg, 1)
            self.assertTrue((staging / "center_result.zarr").is_dir())
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
            self.assertFalse(cfg.workspace_path(1).exists())
            self.assertFalse(any(final.rglob("*.part-*")))
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["fields"]), {
                "velocity", "gradient", "velocity_bar", "gradient_bar",
                "work_full", "work_resolved", "regime",
            })

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
