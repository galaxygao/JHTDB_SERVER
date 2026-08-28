from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from jhtdb_pipeline.catalog import Catalog
from jhtdb_pipeline.config import load_config
from jhtdb_pipeline.jhtdb import fetch_snapshot


class FakeSciServerJHTDB:
    def __init__(self, field: np.ndarray):
        self.field = field
        self.calls = 0

    def fetch_tile(self, request, time_index: int) -> np.ndarray:
        self.calls += 1
        return np.ascontiguousarray(
            self.field[
                :,
                request.z0 : request.z0 + request.nz,
                request.y0 : request.y0 + request.ny,
                request.x0 : request.x0 + request.nx,
            ]
        )


class FetchTests(unittest.TestCase):
    def test_large_request_is_stored_as_verified_checksum_tiles_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            token_file = root / "token"
            token_file.write_text("test-token\n", encoding="utf-8")
            token_file.chmod(0o600)
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
                token_file=token_file,
                scratch_safety_reserve_gib=0.0,
                request_cooldown_seconds=0.0,
            )
            field = np.arange(3 * 16**3, dtype=np.float32).reshape(3, 16, 16, 16)
            client = FakeSciServerJHTDB(field)
            with patch(
                "jhtdb_pipeline.jhtdb.SciServerJHTDB", return_value=client
            ):
                fetch_snapshot(cfg, 1)
                fetch_snapshot(cfg, 1)

            self.assertEqual(client.calls, 1)
            stored = zarr.open_group(str(cfg.raw_store_path(1)), mode="r")["velocity"]
            np.testing.assert_array_equal(stored[:], field)
            with Catalog(cfg.catalog_path) as catalog:
                self.assertEqual(
                    catalog.tile_progress(cfg.dataset, 1),
                    {"verified": 8, "total": 8},
                )


if __name__ == "__main__":
    unittest.main()
