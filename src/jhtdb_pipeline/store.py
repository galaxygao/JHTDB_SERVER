from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr
from numcodecs import Blosc

from .config import PipelineConfig
from .planning import Tile


def array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values)
    return hashlib.sha256(canonical.view(np.uint8)).hexdigest()


def compressor(cfg: PipelineConfig) -> Blosc:
    return Blosc(
        cname="zstd",
        clevel=cfg.compression_level,
        shuffle=Blosc.BITSHUFFLE,
    )


class VelocityStore:
    def __init__(self, cfg: PipelineConfig, time_index: int):
        self.cfg = cfg
        self.time_index = time_index
        path = cfg.raw_store_path(time_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(str(path), mode="a")
        self.root.attrs.update(
            {
                "dataset": cfg.dataset,
                "variable": cfg.variable,
                "time_index": time_index,
                "physical_time": cfg.physical_time(time_index),
                "axis_order": ["component", "z", "y", "x"],
                "components": ["ux", "uy", "uz"],
                "domain": "[0,2pi)^3",
                "periodic": [True, True, True],
                "grid_shape_xyz": list(cfg.grid_shape),
                "dtype": "float32",
            }
        )

    def ensure_array(self):
        gx, gy, gz = self.cfg.grid_shape
        tx, ty, tz = self.cfg.tile_shape
        return self.root.require_dataset(
            "velocity",
            shape=(3, gz, gy, gx),
            chunks=(3, tz, ty, tx),
            dtype="<f4",
            fill_value=np.nan,
            compressor=compressor(self.cfg),
            overwrite=False,
        )

    @property
    def array(self):
        return self.root["velocity"]

    def write_tile(self, tile: Tile, values: np.ndarray) -> str:
        expected = (3, tile.nz, tile.ny, tile.nx)
        canonical = np.ascontiguousarray(values, dtype="<f4")
        if canonical.shape != expected:
            raise ValueError(f"tile {tile.key} shape {canonical.shape}, expected {expected}")
        if not np.all(np.isfinite(canonical)):
            raise ValueError(f"tile {tile.key} contains NaN or Inf")
        self.array[tile.store_slices] = canonical
        readback = np.ascontiguousarray(self.array[tile.store_slices], dtype="<f4")
        digest = array_sha256(canonical)
        if array_sha256(readback) != digest:
            raise IOError(f"tile {tile.key} failed write/read checksum")
        return digest

    def mark_validated(self, manifest_hash: str) -> None:
        self.root.attrs.update({"status": "validated", "manifest_hash": manifest_hash})


def create_result_group(
    cfg: PipelineConfig,
    time_index: int,
    sigma_grid: float,
    *,
    overwrite: bool,
):
    staging = cfg.staging_result_path(time_index, sigma_grid)
    staging.mkdir(parents=True, exist_ok=True)
    path = staging / "center_result.zarr"
    root = zarr.open_group(str(path), mode="w" if overwrite else "a")
    nz, ny, nx = cfg.result_shape_zyx
    cz, cy, cx = (min(64, nz), min(64, ny), min(64, nx))
    codec = compressor(cfg)
    root.attrs.update(
        {
            "status": "computing",
            "dataset": cfg.dataset,
            "time_index": time_index,
            "physical_time": cfg.physical_time(time_index),
            "sigma_grid": float(sigma_grid),
            "axis_order": ["component", "z", "y", "x"],
            "gradient_axis_order": [
                "velocity_component", "derivative_component", "z", "y", "x"
            ],
            "components": ["ux", "uy", "uz"],
            "derivative_components": ["x", "y", "z"],
            "crop_start_xyz": list(cfg.crop_start),
            "crop_shape_xyz": list(cfg.crop_shape),
        }
    )
    root.require_dataset(
        "velocity", shape=(3, nz, ny, nx), chunks=(1, cz, cy, cx),
        dtype="<f4", compressor=codec, fill_value=np.nan,
    )
    root.require_dataset(
        "gradient", shape=(3, 3, nz, ny, nx), chunks=(1, 1, cz, cy, cx),
        dtype="<f4", compressor=codec, fill_value=np.nan,
    )
    root.require_dataset(
        "velocity_bar", shape=(3, nz, ny, nx), chunks=(1, cz, cy, cx),
        dtype="<f4", compressor=codec, fill_value=np.nan,
    )
    root.require_dataset(
        "gradient_bar", shape=(3, 3, nz, ny, nx), chunks=(1, 1, cz, cy, cx),
        dtype="<f4", compressor=codec, fill_value=np.nan,
    )
    for name in ("work_full", "work_resolved"):
        root.require_dataset(
            name, shape=(nz, ny, nx), chunks=(cz, cy, cx),
            dtype="<f4", compressor=codec, fill_value=np.nan,
        )
    root.require_dataset(
        "regime", shape=(nz, ny, nx), chunks=(cz, cy, cx),
        dtype="u1", compressor=codec, fill_value=0,
    )
    return root


def spatial_slices(shape: tuple[int, ...], chunk_shape: tuple[int, ...]) -> Iterator[tuple[slice, ...]]:
    if len(shape) != len(chunk_shape):
        raise ValueError("shape and chunk_shape ranks differ")

    def walk(axis: int, prefix: tuple[slice, ...]):
        if axis == len(shape):
            yield prefix
            return
        for start in range(0, shape[axis], chunk_shape[axis]):
            stop = min(start + chunk_shape[axis], shape[axis])
            yield from walk(axis + 1, prefix + (slice(start, stop),))

    yield from walk(0, ())


def hash_zarr_array(array: Any) -> tuple[str, int, float, float]:
    hasher = hashlib.sha256()
    byte_count = 0
    minimum = float("inf")
    maximum = float("-inf")
    for key in spatial_slices(tuple(array.shape), tuple(array.chunks)):
        values = np.ascontiguousarray(array[key])
        if not np.all(np.isfinite(values)):
            raise ValueError("result array contains NaN or Inf")
        hasher.update(values.view(np.uint8))
        byte_count += values.nbytes
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
    return hasher.hexdigest(), byte_count, minimum, maximum


def open_complete_result(path: Path):
    if not (path / "COMPLETE").is_file():
        raise RuntimeError(f"result is not complete: {path}")
    root = zarr.open_group(str(path / "center_result.zarr"), mode="r")
    if root.attrs.get("status") != "complete":
        raise RuntimeError(f"result metadata is incomplete: {path}")
    return root
