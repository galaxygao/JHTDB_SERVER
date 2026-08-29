from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from jhtdb_pipeline.catalog import Catalog
from jhtdb_pipeline.config import load_config, result_zarr_name
from jhtdb_pipeline.physics import spectral_derivative, spectral_gaussian
from jhtdb_pipeline.planning import Tile
from jhtdb_pipeline.sbar_qa import run_sbar_qa
from jhtdb_pipeline.processing import (
    backfill_full_fields,
    backfill_full_regime,
    filter_field as processing_filter_field,
    finalize_result,
    process_center,
    resource_plan,
)
from jhtdb_pipeline.store import VelocityStore, open_complete_result
from jhtdb_pipeline.validation import atomic_json


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
            expected_result_bytes = 8**3 * 96 + 16**3 * 17
            self.assertEqual(
                plan["persistent_result_GiB"], expected_result_bytes / 1024**3
            )
            self.assertEqual(plan["configured_sigma_count"], 3)
            self.assertEqual(
                plan["persistent_batch_GiB"],
                3 * expected_result_bytes / 1024**3,
            )
            staging = process_center(cfg, 1)
            self.assertTrue((staging / result_zarr_name(cfg.sigma_grid)).is_dir())
            divergence = json.loads(
                (staging / "divergence.json").read_text(encoding="utf-8")
            )
            self.assertTrue(divergence["passed"])
            self.assertTrue(divergence["unfiltered"]["passed"])
            self.assertTrue(divergence["filtered"]["passed"])
            final = finalize_result(cfg, 1)
            self.assertTrue((final / "COMPLETE").is_file())
            self.assertTrue((final / "manifest.json").is_file())
            self.assertTrue((final / "s_bar_qa.json").is_file())
            self.assertTrue((final / "s_bar_global_totals.html").is_file())
            self.assertTrue((final / "cq.json").is_file())
            self.assertTrue((final / "cq.html").is_file())
            self.assertEqual(run_sbar_qa(cfg, 1)["scope"], "full_domain")
            result = open_complete_result(final)
            self.assertEqual(result["velocity"].shape, (3, 8, 8, 8))
            self.assertEqual(result["gradient"].shape, (3, 3, 8, 8, 8))
            self.assertEqual(result["work_full"].shape, (16, 16, 16))
            self.assertEqual(result["regime"].shape, (16, 16, 16))
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
            filtered_divergence = sum(gradient_bar[i, i] for i in range(3))
            self.assertEqual(
                divergence["filtered"]["point_count"], int(np.prod(cfg.grid_shape))
            )
            self.assertAlmostEqual(
                divergence["filtered"]["maximum_abs_divergence"],
                float(np.max(np.abs(filtered_divergence))),
                delta=2.0e-6,
            )
            expected_pi = np.einsum("ijzyx,ijzyx->zyx", tau, gradient_bar)
            transport = np.einsum("izyx,ijzyx->jzyx", filtered, tau)
            expected_s_bar = sum(
                spectral_derivative(transport[j], 2 - j, cfg.domain_length)
                for j in range(3)
            )
            np.testing.assert_allclose(
                result["pi"][:], expected_pi, rtol=3e-5, atol=3e-6
            )
            np.testing.assert_allclose(
                result["s_bar"][:], expected_s_bar, rtol=3e-5, atol=3e-6
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
            self.assertEqual(manifest["schema_version"], 5)
            self.assertEqual(manifest["field_scopes"]["regime"], "full_domain")
            self.assertEqual(manifest["s_bar_qa_report_version"], 2)
            self.assertIn("s_bar_qa_report_hash", manifest)
            self.assertTrue(manifest["cq_passed"])
            self.assertIn("cq_report_hash", manifest)
            self.assertEqual(
                json.loads((final / "COMPLETE").read_text(encoding="utf-8"))[
                    "manifest_hash"
                ],
                result.attrs["manifest_hash"],
            )
            self.assertEqual(set(manifest["fields"]), {
                "velocity", "gradient", "velocity_bar", "gradient_bar",
                "work_full", "work_resolved", "pi", "s_bar", "regime",
            })
            for name in (
                "s_bar_qa.json",
                "s_bar_global_totals.html",
                "cq.json",
                "cq.html",
            ):
                (final / name).unlink()
            shutil.rmtree(cfg.raw_store_path(1))
            self.assertEqual(process_center(cfg, 1), final)
            refreshed_s_bar = json.loads(
                (final / "s_bar_qa.json").read_text(encoding="utf-8")
            )
            self.assertEqual(refreshed_s_bar["report_version"], 2)
            self.assertNotIn("s_bar_rel_self", refreshed_s_bar["metrics"])
            self.assertTrue((final / "s_bar_global_totals.html").is_file())
            self.assertTrue((final / "cq.json").is_file())
            self.assertTrue((final / "cq.html").is_file())

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

    def test_backfill_reuses_validated_filtered_velocity_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, _ = fixture(Path(temporary))
            cfg = replace(cfg, cleanup_scratch_on_success=False)
            process_center(cfg, 1)
            final = finalize_result(cfg, 1)
            root = zarr.open_group(
                str(final / result_zarr_name(cfg.sigma_grid)), mode="a"
            )
            root.attrs["result_schema_version"] = 3

            with patch(
                "jhtdb_pipeline.processing.filter_field",
                wraps=processing_filter_field,
            ) as filtered:
                staging = process_center(cfg, 1)

            qa = json.loads((staging / "qa.json").read_text(encoding="utf-8"))
            self.assertTrue(qa["reuse"]["filtered_velocity"])
            self.assertEqual(filtered.call_count, 9)

    def test_v3_backfill_reuses_raw_cache_and_validates_center_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, _ = fixture(Path(temporary))
            process_center(cfg, 1)
            final = finalize_result(cfg, 1)
            zarr_path = final / result_zarr_name(cfg.sigma_grid)
            current = zarr.open_group(str(zarr_path), mode="r")
            crop = cfg.crop_slices_zyx
            old_fields = {}
            for name in (
                "velocity",
                "gradient",
                "velocity_bar",
                "gradient_bar",
            ):
                old_fields[name] = np.asarray(current[name][:])
            old_fields["regime"] = np.asarray(current["regime"][crop])
            for name in ("work_full", "work_resolved", "pi", "s_bar"):
                old_fields[name] = np.asarray(current[name][crop])
            del current

            old = zarr.open_group(str(zarr_path), mode="w")
            old.attrs.update(
                {
                    "status": "complete",
                    "result_schema_version": 3,
                    "sigma_grid": cfg.sigma_grid,
                }
            )
            for name, values in old_fields.items():
                old.create_dataset(
                    name,
                    data=values,
                    chunks=tuple(min(4, size) for size in values.shape),
                    dtype=values.dtype,
                )
            atomic_json(
                final / "manifest.json",
                {"schema_version": 3, "time_index": 1, "sigma_grid": 1.0},
            )

            upgraded = backfill_full_fields(cfg, 1)

            self.assertEqual(upgraded, final)
            result = open_complete_result(final)
            self.assertEqual(result.attrs["result_schema_version"], 5)
            self.assertEqual(result["work_full"].shape, cfg.full_shape_zyx)
            self.assertEqual(result["regime"].shape, cfg.full_shape_zyx)
            qa = json.loads((final / "qa.json").read_text(encoding="utf-8"))
            overlap = qa["reuse"]["previous_center_overlap"]
            self.assertTrue(overlap["passed"])
            self.assertEqual(overlap["scope"], "stored_center_crop")

    def test_v4_full_regime_backfill_uses_persistent_work_fields_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, _ = fixture(Path(temporary))
            process_center(cfg, 1)
            final = finalize_result(cfg, 1)
            zarr_path = final / result_zarr_name(cfg.sigma_grid)
            root = zarr.open_group(str(zarr_path), mode="a")
            expected = np.asarray(root["regime"][:])
            center = np.asarray(root["regime"][cfg.crop_slices_zyx])
            chunks = tuple(min(4, size) for size in center.shape)
            compressor = root["regime"].compressor
            del root["regime"]
            root.create_dataset(
                "regime", data=center, chunks=chunks, dtype="u1",
                compressor=compressor,
            )
            scopes = dict(root.attrs["field_scopes"])
            scopes["regime"] = "center_crop"
            root.attrs.update(
                {"result_schema_version": 4, "field_scopes": scopes}
            )
            manifest_path = final / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 4
            manifest["field_scopes"]["regime"] = "center_crop"
            manifest["fields"]["regime"]["shape"] = list(cfg.result_shape_zyx)
            for key in ("cq_passed", "cq_report_version", "cq_report_hash"):
                manifest.pop(key, None)
                root.attrs.pop(key, None)
            atomic_json(manifest_path, manifest)
            (final / "cq.json").unlink()
            (final / "cq.html").unlink()
            shutil.rmtree(cfg.raw_store_path(1))

            upgraded = backfill_full_regime(cfg, 1)

            self.assertEqual(upgraded, final)
            current = open_complete_result(final)
            self.assertEqual(current.attrs["result_schema_version"], 5)
            self.assertEqual(current["regime"].shape, cfg.full_shape_zyx)
            np.testing.assert_array_equal(current["regime"][:], expected)
            qa = json.loads((final / "qa.json").read_text(encoding="utf-8"))
            self.assertEqual(qa["regime_scope"], "full_domain")
            self.assertEqual(qa["regime_point_count"], 16**3)
            self.assertEqual(
                qa["reuse"]["regime"],
                "persistent_schema_v4_full_work_fields",
            )
            self.assertTrue((final / "cq.json").is_file())
            self.assertTrue((final / "cq.html").is_file())
            self.assertFalse(any(final.glob(".*regime-v4-backup*")))

    def test_backfill_never_fetches_when_temporary_raw_cache_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, _ = fixture(Path(temporary))
            final = cfg.result_path(1)
            final.mkdir(parents=True)
            atomic_json(final / "COMPLETE", {})
            atomic_json(
                final / "manifest.json",
                {"schema_version": 3, "time_index": 1, "sigma_grid": 1.0},
            )
            raw_store = cfg.raw_store_path(1)
            shutil.rmtree(raw_store)

            with self.assertRaisesRegex(RuntimeError, "never fetches JHTDB"):
                backfill_full_fields(cfg, 1)

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
            self.assertFalse(report["unfiltered"]["passed"])
            self.assertFalse(report["filtered"]["passed"])


if __name__ == "__main__":
    unittest.main()
