from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import Catalog
from .config import PipelineConfig
from .planning import tiles_for
from .store import VelocityStore, array_sha256


def atomic_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest


def _seam_statistics(array: Any) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    axes = {
        "x": (
            (slice(None), slice(None), slice(None), 0),
            (slice(None), slice(None), slice(None), -1),
            (slice(None), slice(None), slice(None), 1),
        ),
        "y": (
            (slice(None), slice(None), 0, slice(None)),
            (slice(None), slice(None), -1, slice(None)),
            (slice(None), slice(None), 1, slice(None)),
        ),
        "z": (
            (slice(None), 0, slice(None), slice(None)),
            (slice(None), -1, slice(None), slice(None)),
            (slice(None), 1, slice(None), slice(None)),
        ),
    }
    for name, (first_key, last_key, next_key) in axes.items():
        first = np.asarray(array[first_key], dtype=np.float64)
        last = np.asarray(array[last_key], dtype=np.float64)
        adjacent = np.asarray(array[next_key], dtype=np.float64)
        wrap_rms = float(np.sqrt(np.mean(np.square(first - last))))
        internal_rms = float(np.sqrt(np.mean(np.square(adjacent - first))))
        result[name] = {
            "wrap_rms": wrap_rms,
            "first_internal_rms": internal_rms,
            "wrap_to_internal_ratio": wrap_rms / max(internal_rms, 1.0e-30),
        }
    return result


def validate_snapshot(cfg: PipelineConfig, time_index: int) -> dict[str, Any]:
    store = VelocityStore(cfg, time_index)
    array = store.array
    gx, gy, gz = cfg.grid_shape
    expected_shape = (3, gz, gy, gx)
    if array.shape != expected_shape or np.dtype(array.dtype) != np.dtype("<f4"):
        raise ValueError(
            f"velocity cache schema mismatch: shape={array.shape}, dtype={array.dtype}"
        )

    expected_tiles = tiles_for(cfg)
    expected_keys = {tile.key for tile in expected_tiles}
    count = 0
    total = 0.0
    sumsq = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    tile_manifest: list[dict[str, Any]] = []
    with Catalog(cfg.catalog_path) as catalog:
        rows = catalog.tiles(cfg.dataset, time_index)
        if {row["tile_key"] for row in rows} != expected_keys:
            raise ValueError("catalog coverage differs from the deterministic tile plan")
        row_map = {row["tile_key"]: row for row in rows}
        for tile in expected_tiles:
            row = row_map[tile.key]
            if row["status"] != "verified" or not row["sha256"]:
                raise ValueError(f"tile {tile.key} is not verified")
            block = np.ascontiguousarray(array[tile.store_slices], dtype="<f4")
            if block.shape != (3, tile.nz, tile.ny, tile.nx):
                raise ValueError(f"tile {tile.key} has an invalid shape")
            if not np.all(np.isfinite(block)):
                raise ValueError(f"tile {tile.key} contains NaN or Inf")
            digest = array_sha256(block)
            if digest != row["sha256"]:
                raise ValueError(f"tile {tile.key} checksum differs from catalog")
            values64 = block.astype(np.float64)
            count += values64.size
            total += float(values64.sum())
            sumsq += float(np.square(values64).sum())
            minimum = min(minimum, float(values64.min()))
            maximum = max(maximum, float(values64.max()))
            tile_manifest.append(
                {
                    "key": tile.key,
                    "origin_xyz_0based": [tile.x0, tile.y0, tile.z0],
                    "shape_xyz": [tile.nx, tile.ny, tile.nz],
                    "sha256": digest,
                }
            )

        qa = {
            "dataset": cfg.dataset,
            "time_index": time_index,
            "physical_time": cfg.physical_time(time_index),
            "status": "validated",
            "shape_component_zyx": list(expected_shape),
            "dtype": "float32",
            "finite": True,
            "coverage_exactly_once": True,
            "periodic_endpoint_duplicated": False,
            "statistics": {
                "minimum": minimum,
                "maximum": maximum,
                "mean": total / count,
                "rms": float(np.sqrt(sumsq / count)),
            },
            "seams": _seam_statistics(array),
        }
        atomic_json(cfg.qa_path / f"input_t{time_index:06d}.json", qa)
        manifest = {
            "schema_version": 2,
            "dataset": cfg.dataset,
            "variable": cfg.variable,
            "time_index": time_index,
            "physical_time": cfg.physical_time(time_index),
            "grid_shape_xyz": list(cfg.grid_shape),
            "axis_order": ["component", "z", "y", "x"],
            "components": ["ux", "uy", "uz"],
            "dtype": "float32",
            "domain": "[0,2pi)^3",
            "periodic": [True, True, True],
            "tile_count": len(expected_tiles),
            "tiles": tile_manifest,
            "qa": qa,
        }
        manifest_hash = atomic_json(
            cfg.manifest_path / f"input_t{time_index:06d}.json", manifest
        )
        catalog.set_snapshot_status(
            cfg.dataset, time_index, "validated", manifest_hash
        )
        store.mark_validated(manifest_hash)
        qa["manifest_hash"] = manifest_hash
        return qa


def input_manifest_hash(cfg: PipelineConfig, time_index: int) -> str:
    with Catalog(cfg.catalog_path) as catalog:
        row = catalog.snapshot(cfg.dataset, time_index)
        if row is None or row["status"] != "validated" or not row["manifest_hash"]:
            raise RuntimeError("the complete velocity cache has not passed validation")
        return str(row["manifest_hash"])
