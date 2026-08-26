from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from jhtdb_pipeline.catalog import Catalog
from jhtdb_pipeline.config import load_config
from jhtdb_pipeline.planning import Tile
from jhtdb_pipeline.store import VelocityStore, array_sha256
from jhtdb_pipeline.validation import validate_snapshot


class CatalogStoreTests(unittest.TestCase):
    def test_tile_roundtrip_and_unique_catalog(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = replace(
                load_config("configs/pipeline.yaml"),
                grid_shape=(16, 16, 16),
                tile_shape=(8, 8, 8),
                storage_root=Path(temporary),
            )
            tile = Tile(0, 0, 0, 8, 8, 8)
            values = np.arange(3 * 8**3, dtype=np.float32).reshape(3, 8, 8, 8)
            store = VelocityStore(cfg)
            store.ensure_snapshot(1, 0.0)
            digest = store.write_tile(1, tile, values)
            self.assertEqual(digest, array_sha256(values))
            np.testing.assert_array_equal(store.snapshot_array(1)[tile.store_slices], values)
            with Catalog(cfg.catalog_path) as catalog:
                catalog.plan_snapshot(cfg.dataset, 1, 0.0, [tile])
                catalog.plan_snapshot(cfg.dataset, 1, 0.0, [tile])
                self.assertEqual(len(catalog.tiles(cfg.dataset, 1)), 1)
                catalog.mark_verified(cfg.dataset, 1, tile.key, digest, values.nbytes)
                self.assertEqual(catalog.tile(cfg.dataset, 1, tile.key)["status"], "verified")
                self.assertEqual(catalog.tile_progress(cfg.dataset, 1), {"verified": 1, "total": 1})
                catalog.plan_gradient_fields(cfg.dataset, 1, "manifest-a")
                self.assertEqual(catalog.gradient_progress(cfg.dataset, 1), {"planned": 9, "total": 9})
                catalog.mark_gradient_attempt(cfg.dataset, 1, 0, 0)
                catalog.mark_gradient_verified(cfg.dataset, 1, 0, 0, "abc", 4)
                self.assertEqual(catalog.gradient_field(cfg.dataset, 1, 0, 0)["status"], "verified")
                self.assertEqual(
                    catalog.gradient_field(cfg.dataset, 1, 0, 0)["input_manifest_hash"],
                    "manifest-a",
                )
                catalog.plan_gradient_fields(cfg.dataset, 1, "manifest-b")
                changed = catalog.gradient_field(cfg.dataset, 1, 0, 0)
                self.assertEqual(changed["status"], "planned")
                self.assertEqual(changed["input_manifest_hash"], "manifest-b")
                self.assertIsNone(changed["sha256"])

                catalog.mark_gradient_attempt(cfg.dataset, 1, 0, 0)
                catalog.mark_gradient_verified(cfg.dataset, 1, 0, 0, "def", 8)
                catalog.connection.execute(
                    """UPDATE gradient_fields SET input_manifest_hash=NULL
                       WHERE dataset=? AND time_index=?
                         AND velocity_component=0 AND derivative_component=0""",
                    (cfg.dataset, 1),
                )
                catalog.connection.commit()
                catalog.plan_gradient_fields(
                    cfg.dataset,
                    1,
                    "manifest-b",
                    adopt_unbound_verified=True,
                )
                adopted = catalog.gradient_field(cfg.dataset, 1, 0, 0)
                self.assertEqual(adopted["status"], "verified")
                self.assertEqual(adopted["sha256"], "def")
                self.assertEqual(adopted["input_manifest_hash"], "manifest-b")

    def test_complete_small_snapshot_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = replace(
                load_config("configs/pipeline.yaml"),
                grid_shape=(16, 16, 16),
                tile_shape=(8, 8, 8),
                storage_root=Path(temporary),
            )
            tiles = [Tile(x, y, z, 8, 8, 8) for z in (0, 8) for y in (0, 8) for x in (0, 8)]
            store = VelocityStore(cfg)
            store.ensure_snapshot(1, 0.0)
            with Catalog(cfg.catalog_path) as catalog:
                catalog.plan_snapshot(cfg.dataset, 1, 0.0, tiles)
                for tile in tiles:
                    z, y, x = np.mgrid[tile.z0:tile.z0 + 8, tile.y0:tile.y0 + 8, tile.x0:tile.x0 + 8]
                    values = np.stack((x, y, z)).astype(np.float32)
                    digest = store.write_tile(1, tile, values)
                    catalog.mark_verified(cfg.dataset, 1, tile.key, digest, values.nbytes)
            report = validate_snapshot(cfg, 1)
            self.assertEqual(report["status"], "auto_validated")
            self.assertTrue(report["coverage_exactly_once"])
            with Catalog(cfg.catalog_path) as catalog:
                self.assertEqual(catalog.snapshot(cfg.dataset, 1)["status"], "auto_validated")


if __name__ == "__main__":
    unittest.main()
