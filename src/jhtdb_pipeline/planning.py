from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .config import PipelineConfig


@dataclass(frozen=True, order=True)
class Tile:
    x0: int
    y0: int
    z0: int
    nx: int
    ny: int
    nz: int

    @property
    def key(self) -> str:
        return f"x{self.x0:04d}_y{self.y0:04d}_z{self.z0:04d}"

    @property
    def api_ranges(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        return (
            (self.x0 + 1, self.x0 + self.nx),
            (self.y0 + 1, self.y0 + self.ny),
            (self.z0 + 1, self.z0 + self.nz),
        )

    @property
    def store_slices(self) -> tuple[slice, slice, slice, slice]:
        return (
            slice(None),
            slice(self.z0, self.z0 + self.nz),
            slice(self.y0, self.y0 + self.ny),
            slice(self.x0, self.x0 + self.nx),
        )


def tiles_for(cfg: PipelineConfig) -> list[Tile]:
    gx, gy, gz = cfg.grid_shape
    tx, ty, tz = cfg.tile_shape
    return [Tile(x, y, z, min(tx, gx - x), min(ty, gy - y), min(tz, gz - z))
            for z, y, x in product(range(0, gz, tz), range(0, gy, ty), range(0, gx, tx))]


def coordinate_for_index(index: int, point_count: int, domain_length: float) -> float:
    if not 0 <= index < point_count:
        raise IndexError(f"grid index {index} is outside [0,{point_count})")
    return index * domain_length / point_count


def plan(cfg: PipelineConfig, time_index: int) -> dict[str, object]:
    if time_index < 1:
        raise ValueError("time_index must be >= 1")
    tiles = tiles_for(cfg)
    tile_bytes = cfg.tile_shape[0] * cfg.tile_shape[1] * cfg.tile_shape[2] * 3 * 4
    return {
        "dataset": cfg.dataset,
        "variable": cfg.variable,
        "backend": cfg.download_backend,
        "time_index": time_index,
        "physical_time": cfg.physical_time(time_index),
        "grid_shape": list(cfg.grid_shape),
        "tile_shape": list(cfg.tile_shape),
        "strictly_serial": True,
        "requests_if_no_retry": len(tiles),
        "tile_uncompressed_MiB": round(tile_bytes / 1024**2, 2),
        "snapshot_uncompressed_GiB": round(cfg.bytes_per_snapshot / 1024**3, 2),
        "local_getcutout_limit_GiB": 3,
        "usage_warning": "JHTDB recommends small targeted subsets and discourages crawling multiple full 3-D fields.",
    }
