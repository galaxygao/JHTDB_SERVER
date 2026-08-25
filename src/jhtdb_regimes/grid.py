from __future__ import annotations

import numpy as np

from .config import TaskConfig


def block_indices(cfg: TaskConfig) -> np.ndarray:
    """Return point indices in C-order for arrays shaped (z, y, x)."""
    i0, j0, k0 = cfg.block_start_ijk
    nx, ny, nz = cfg.block_shape
    zz, yy, xx = np.meshgrid(
        np.arange(k0, k0 + nz, dtype=np.int64),
        np.arange(j0, j0 + ny, dtype=np.int64),
        np.arange(i0, i0 + nx, dtype=np.int64),
        indexing="ij",
    )
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


def query_points(cfg: TaskConfig) -> np.ndarray:
    indices = block_indices(cfg)
    spacing = np.float64(cfg.domain_length) / np.float64(cfg.grid_shape[0])
    return indices.astype(np.float64) * spacing


def point_batches(point_count: int, max_points: int) -> list[slice]:
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    return [slice(start, min(start + max_points, point_count)) for start in range(0, point_count, max_points)]


def time_chunks(times: np.ndarray, chunk_size: int) -> list[tuple[int, int, np.ndarray]]:
    return [
        (start, min(start + chunk_size, len(times)), times[start : start + chunk_size])
        for start in range(0, len(times), chunk_size)
    ]


def rows_to_velocity(values: np.ndarray, block_shape: tuple[int, int, int]) -> np.ndarray:
    """Convert (time, point, component) to (time, component, z, y, x)."""
    nx, ny, nz = block_shape
    if values.shape[1:] != (nx * ny * nz, 3):
        raise ValueError(f"unexpected velocity rows shape {values.shape}")
    grid = values.reshape(values.shape[0], nz, ny, nx, 3)
    return np.moveaxis(grid, -1, 1)


def rows_to_gradient(values: np.ndarray, block_shape: tuple[int, int, int]) -> np.ndarray:
    """Convert (time, point, velocity_component, derivative_axis) to named grid axes."""
    nx, ny, nz = block_shape
    if values.shape[1:] != (nx * ny * nz, 3, 3):
        raise ValueError(f"unexpected gradient rows shape {values.shape}")
    grid = values.reshape(values.shape[0], nz, ny, nx, 3, 3)
    return np.moveaxis(grid, (-2, -1), (1, 2))

