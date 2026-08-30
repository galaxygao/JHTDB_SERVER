from __future__ import annotations

import unittest
from dataclasses import replace

from jhtdb_pipeline.config import load_config
from jhtdb_pipeline.planning import plan, requests_for, tiles_for, tiles_in_request
from jhtdb_pipeline.processing import resource_plan


class PlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("configs/pipeline.yaml")

    def test_full_domain_is_exactly_512_nonoverlapping_tiles(self):
        tiles = tiles_for(self.cfg)
        self.assertEqual(len(tiles), 512)
        self.assertEqual(len({tile.key for tile in tiles}), 512)
        self.assertEqual(sum(tile.nx * tile.ny * tile.nz for tile in tiles), 1024**3)
        self.assertEqual(tiles[0].api_ranges, ((1, 128), (1, 128), (1, 128)))
        self.assertEqual(tiles[-1].api_ranges, ((897, 1024), (897, 1024), (897, 1024)))

    def test_plan_is_single_time_and_serial(self):
        batch_cfg = replace(
            self.cfg,
            sigma_grid=1.0,
            sigma_grids=(1.0, 2.0, 3.0),
        )
        result = plan(batch_cfg, 3)
        self.assertEqual(result["time_index"], 3)
        self.assertEqual(result["physical_time"], 0.004)
        self.assertTrue(result["strictly_serial"])
        self.assertEqual(result["requests_if_no_retry"], 8)
        self.assertEqual(result["checksum_tiles"], 512)
        self.assertEqual(result["sigma_grids"], [1.0, 2.0, 3.0])
        self.assertEqual(
            [
                path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                for path in result["persistent_result_paths"]
            ],
            ["t000003_sigma_1", "t000003_sigma_2", "t000003_sigma_3"],
        )

    def test_eight_requests_each_contain_64_checksum_tiles(self):
        tiles = tiles_for(self.cfg)
        requests = requests_for(self.cfg)
        self.assertEqual(len(requests), 8)
        grouped = [tiles_in_request(request, tiles) for request in requests]
        self.assertTrue(all(len(group) == 64 for group in grouped))
        self.assertEqual(
            {tile.key for group in grouped for tile in group},
            {tile.key for tile in tiles},
        )

    def test_full_domain_energy_fields_are_in_persistent_capacity_plan(self):
        batch_cfg = replace(
            self.cfg,
            sigma_grid=1.0,
            sigma_grids=(1.0, 2.0, 3.0),
        )
        resources = resource_plan(batch_cfg)
        self.assertEqual(resources["persistent_result_GiB"], 29.0)
        self.assertEqual(resources["persistent_batch_GiB"], 87.0)
        self.assertEqual(resources["persistent_v3_to_v5_peak_GiB"], 43.125)
        self.assertEqual(resources["persistent_v4_regime_backfill_peak_GiB"], 29.125)
        self.assertGreater(
            resources["persistent_batch_with_reserve_GiB"],
            resources["observed_account_capacity_GiB"],
        )


if __name__ == "__main__":
    unittest.main()
