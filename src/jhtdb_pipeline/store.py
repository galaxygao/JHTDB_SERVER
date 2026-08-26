from __future__ import annotations

import hashlib

import numpy as np
import zarr
from numcodecs import Blosc

from .config import PipelineConfig
from .planning import Tile


def array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(canonical.view(np.uint8)).hexdigest()


class VelocityStore:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        cfg.raw_store_path.parent.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(str(cfg.raw_store_path), mode="a")
        self.root.attrs.update({
            "dataset": cfg.dataset,
            "variable": cfg.variable,
            "axis_order": ["component", "z", "y", "x"],
            "component": ["ux", "uy", "uz"],
            "domain": "[0,2pi)^3",
            "periodic": [True, True, True],
            "grid_shape_xyz": list(cfg.grid_shape),
            "dtype": "float32",
        })

    @staticmethod
    def group_name(time_index: int) -> str:
        return f"t{time_index:06d}"

    def ensure_snapshot(self, time_index: int, physical_time: float):
        group = self.root.require_group(self.group_name(time_index))
        group.attrs.update({"time_index": time_index, "physical_time": physical_time, "status": "partial"})
        gx, gy, gz = self.cfg.grid_shape
        tx, ty, tz = self.cfg.tile_shape
        compressor = Blosc(cname="zstd", clevel=self.cfg.compression_level, shuffle=Blosc.BITSHUFFLE)
        return group.require_dataset(
            "velocity",
            shape=(3, gz, gy, gx),
            chunks=(3, tz, ty, tx),
            dtype="<f4",
            fill_value=np.nan,
            compressor=compressor,
            overwrite=False,
        )

    def snapshot_array(self, time_index: int):
        return self.root[self.group_name(time_index)]["velocity"]

    def write_tile(self, time_index: int, tile: Tile, values: np.ndarray) -> str:
        expected = (3, tile.nz, tile.ny, tile.nx)
        canonical = np.ascontiguousarray(values, dtype="<f4")
        if canonical.shape != expected:
            raise ValueError(f"tile {tile.key} shape {canonical.shape}, expected {expected}")
        if not np.all(np.isfinite(canonical)):
            raise ValueError(f"tile {tile.key} contains NaN or Inf")
        array = self.snapshot_array(time_index)
        array[tile.store_slices] = canonical
        readback = np.asarray(array[tile.store_slices], dtype="<f4")
        source_hash = array_sha256(canonical)
        if array_sha256(readback) != source_hash:
            raise IOError(f"tile {tile.key} failed write/read checksum")
        return source_hash

    def mark_status(self, time_index: int, status: str, manifest_hash: str | None = None) -> None:
        group = self.root[self.group_name(time_index)]
        group.attrs["status"] = status
        if manifest_hash is not None:
            group.attrs["manifest_hash"] = manifest_hash
