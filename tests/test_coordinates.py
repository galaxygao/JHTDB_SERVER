from __future__ import annotations

import unittest

import numpy as np

from jhtdb_pipeline.jhtdb import canonicalize_cutout
from jhtdb_pipeline.planning import Tile, coordinate_for_index


class CoordinateTests(unittest.TestCase):
    def test_cutout_zyxc_to_component_zyx_mapping(self):
        tile = Tile(x0=10, y0=20, z0=30, nx=4, ny=3, nz=2)
        values = np.empty((2, 3, 4, 3), dtype=np.float32)
        for z in range(2):
            for y in range(3):
                for x in range(4):
                    for component in range(3):
                        values[z, y, x, component] = (
                            1000 * component + 100 * z + 10 * y + x
                        )
        canonical = canonicalize_cutout(values, tile)
        self.assertEqual(canonical.shape, (3, 2, 3, 4))
        for component in range(3):
            for z in range(2):
                for y in range(3):
                    for x in range(4):
                        self.assertEqual(
                            canonical[component, z, y, x],
                            1000 * component + 100 * z + 10 * y + x,
                        )

    def test_periodic_coordinates_do_not_duplicate_endpoint(self):
        length = 2.0 * np.pi
        self.assertEqual(coordinate_for_index(0, 1024, length), 0.0)
        self.assertAlmostEqual(
            coordinate_for_index(1023, 1024, length), length - length / 1024
        )
        with self.assertRaises(IndexError):
            coordinate_for_index(1024, 1024, length)

    def test_api_ranges_are_one_based_while_store_is_zero_based(self):
        tile = Tile(x0=128, y0=256, z0=384, nx=128, ny=128, nz=128)
        self.assertEqual(tile.api_ranges, ((129, 256), (257, 384), (385, 512)))
        self.assertEqual(tile.store_slices[3], slice(128, 256))
        self.assertEqual(tile.store_slices[2], slice(256, 384))
        self.assertEqual(tile.store_slices[1], slice(384, 512))


if __name__ == "__main__":
    unittest.main()
