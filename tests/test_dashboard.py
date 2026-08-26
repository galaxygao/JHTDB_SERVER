from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from jhtdb_pipeline.dashboard import (
    extract_scalar_slice,
    extract_gradient_slice,
    extract_slice,
    open_derived_readonly,
    open_filtered_readonly,
    open_gradient_readonly,
    open_snapshot_readonly,
)


class DashboardTests(unittest.TestCase):
    def test_axis_mapping(self):
        array = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
        np.testing.assert_array_equal(extract_slice(array, 1, "x", 2), array[1, :, :, 2])
        np.testing.assert_array_equal(extract_slice(array, 1, "y", 2), array[1, :, 2, :])
        np.testing.assert_array_equal(extract_slice(array, 1, "z", 2), array[1, 2, :, :])

        scalar = array[0]
        np.testing.assert_array_equal(extract_scalar_slice(scalar, "x", 2), scalar[:, :, 2])
        np.testing.assert_array_equal(extract_scalar_slice(scalar, "y", 2), scalar[:, 2, :])
        np.testing.assert_array_equal(extract_scalar_slice(scalar, "z", 2), scalar[2, :, :])

        gradient = np.arange(3 * 3 * 4 * 5 * 6).reshape(3, 3, 4, 5, 6)
        np.testing.assert_array_equal(
            extract_gradient_slice(gradient, 1, 2, "x", 3), gradient[1, 2, :, :, 3]
        )
        np.testing.assert_array_equal(
            extract_gradient_slice(gradient, 1, 2, "y", 3), gradient[1, 2, :, 3, :]
        )
        np.testing.assert_array_equal(
            extract_gradient_slice(gradient, 1, 2, "z", 3), gradient[1, 2, 3, :, :]
        )

    @patch("jhtdb_pipeline.dashboard.zarr.open_group")
    def test_snapshot_is_opened_read_only(self, mock_open_group):
        velocity = MagicMock()
        mock_open_group.return_value.__getitem__.return_value.__getitem__.return_value = velocity
        result = open_snapshot_readonly("velocity.zarr", 7)
        mock_open_group.assert_called_once_with("velocity.zarr", mode="r")
        mock_open_group.return_value.__getitem__.assert_called_once_with("t000007")
        self.assertIs(result, velocity)

    @patch("jhtdb_pipeline.dashboard.zarr.open_group")
    def test_derived_is_opened_read_only(self, mock_open_group):
        group = MagicMock()
        mock_open_group.return_value.__getitem__.return_value = group
        result = open_derived_readonly("physics.zarr", 12)
        mock_open_group.assert_called_once_with("physics.zarr", mode="r")
        mock_open_group.return_value.__getitem__.assert_called_once_with("t000012")
        self.assertIs(result, group)

    @patch("jhtdb_pipeline.dashboard.zarr.open_group")
    def test_gradient_is_opened_read_only(self, mock_open_group):
        group = MagicMock()
        mock_open_group.return_value.__getitem__.return_value = group
        result = open_gradient_readonly("gradients.zarr", 3)
        mock_open_group.assert_called_once_with("gradients.zarr", mode="r")
        mock_open_group.return_value.__getitem__.assert_called_once_with("t000003")
        self.assertIs(result, group)

    @patch("jhtdb_pipeline.dashboard.zarr.open_group")
    def test_filtered_is_opened_read_only(self, mock_open_group):
        group = MagicMock()
        mock_open_group.return_value.__getitem__.return_value = group
        result = open_filtered_readonly("filtered.zarr", 4)
        mock_open_group.assert_called_once_with("filtered.zarr", mode="r")
        mock_open_group.return_value.__getitem__.assert_called_once_with("t000004")
        self.assertIs(result, group)


if __name__ == "__main__":
    unittest.main()
