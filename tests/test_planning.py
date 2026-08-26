from __future__ import annotations

import unittest

from jhtdb_pipeline.config import load_config
from jhtdb_pipeline.planning import plan, tiles_for


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
        result = plan(self.cfg, 3)
        self.assertEqual(result["time_index"], 3)
        self.assertEqual(result["physical_time"], 0.004)
        self.assertTrue(result["strictly_serial"])
        self.assertEqual(result["requests_if_no_retry"], 512)


if __name__ == "__main__":
    unittest.main()
