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


def small_config(root: Path):
    return replace(
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
    )


class CatalogStoreTests(unittest.TestCase):
    def test_tile_roundtrip_and_unique_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg = small_config(Path(temporary))
            tile = Tile(0, 0, 0, 8, 8, 8)
            values = np.arange(3 * 8**3, dtype=np.float32).reshape(3, 8, 8, 8)
            store = VelocityStore(cfg, 1)
            store.ensure_array()
            digest = store.write_tile(tile, values)
            self.assertEqual(digest, array_sha256(values))
            np.testing.assert_array_equal(store.array[tile.store_slices], values)
            with Catalog(cfg.catalog_path) as catalog:
                catalog.plan_snapshot(cfg.dataset, 1, 0.0, [tile])
                catalog.plan_snapshot(cfg.dataset, 1, 0.0, [tile])
                self.assertEqual(len(catalog.tiles(cfg.dataset, 1)), 1)
                catalog.mark_verified(cfg.dataset, 1, tile.key, digest, values.nbytes)
                self.assertEqual(
                    catalog.tile_progress(cfg.dataset, 1), {"verified": 1, "total": 1}
                )

    def test_complete_small_snapshot_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg = small_config(Path(temporary))
            tiles = [
                Tile(x, y, z, 8, 8, 8)
                for z in (0, 8)
                for y in (0, 8)
                for x in (0, 8)
            ]
            store = VelocityStore(cfg, 1)
            store.ensure_array()
            with Catalog(cfg.catalog_path) as catalog:
                catalog.plan_snapshot(cfg.dataset, 1, 0.0, tiles)
                for tile in tiles:
                    z, y, x = np.mgrid[
                        tile.z0 : tile.z0 + 8,
                        tile.y0 : tile.y0 + 8,
                        tile.x0 : tile.x0 + 8,
                    ]
                    values = np.stack((x, y, z)).astype(np.float32)
                    digest = store.write_tile(tile, values)
                    catalog.mark_verified(
                        cfg.dataset, 1, tile.key, digest, values.nbytes
                    )
            report = validate_snapshot(cfg, 1)
            self.assertEqual(report["status"], "validated")
            self.assertTrue(report["coverage_exactly_once"])
            reopened = VelocityStore(cfg, 1)
            self.assertEqual(reopened.root.attrs["status"], "validated")


if __name__ == "__main__":
    unittest.main()
