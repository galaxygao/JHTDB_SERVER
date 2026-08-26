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


def _atomic_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest


def require_divergence_validation(
    cfg: PipelineConfig,
    time_index: int,
    velocity_manifest_hash: str,
    gradient_manifest_hash: str,
) -> dict[str, Any]:
    """Require a passing full-domain divergence report for the exact inputs."""
    path = cfg.qa_path / f"divergence_t{time_index:06d}.json"
    if not path.exists():
        raise RuntimeError(
            "full-domain divergence validation is missing; run: "
            f"python -m jhtdb_pipeline validate-divergence --time-index {time_index}"
        )
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read divergence validation report: {path}") from exc
    if report.get("velocity_manifest_hash") != velocity_manifest_hash:
        raise RuntimeError("divergence validation belongs to a different velocity manifest")
    if report.get("gradient_manifest_hash") != gradient_manifest_hash:
        raise RuntimeError("divergence validation belongs to a different gradient manifest")
    if report.get("status") != "passed" or report.get("passed") is not True:
        raise RuntimeError("full-domain divergence validation did not pass")
    if float(report.get("relative_divergence_rms", float("inf"))) > float(
        cfg.divergence_relative_rms_max
    ):
        raise RuntimeError(
            "divergence validation exceeds the configured relative RMS tolerance"
        )
    if float(report.get("relative_maximum_divergence", float("inf"))) > float(
        cfg.divergence_relative_max_max
    ):
        raise RuntimeError(
            "divergence validation exceeds the configured relative maximum tolerance"
        )
    return report


def _seam_statistics(array: Any) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    axis_slices = {
        "x": ((slice(None), slice(None), slice(None), 0), (slice(None), slice(None), slice(None), -1),
              (slice(None), slice(None), slice(None), 1)),
        "y": ((slice(None), slice(None), 0, slice(None)), (slice(None), slice(None), -1, slice(None)),
              (slice(None), slice(None), 1, slice(None))),
        "z": ((slice(None), 0, slice(None), slice(None)), (slice(None), -1, slice(None), slice(None)),
              (slice(None), 1, slice(None), slice(None))),
    }
    for name, (first_slice, last_slice, next_slice) in axis_slices.items():
        first = np.asarray(array[first_slice], dtype=np.float64)
        last = np.asarray(array[last_slice], dtype=np.float64)
        adjacent = np.asarray(array[next_slice], dtype=np.float64)
        wrap = first - last
        internal = adjacent - first
        result[name] = {
            "wrap_rms": float(np.sqrt(np.mean(wrap * wrap))),
            "first_internal_rms": float(np.sqrt(np.mean(internal * internal))),
            "wrap_to_internal_ratio": float(np.sqrt(np.mean(wrap * wrap)) / max(np.sqrt(np.mean(internal * internal)), 1e-30)),
        }
    return result


def validate_snapshot(cfg: PipelineConfig, time_index: int) -> dict[str, Any]:
    store = VelocityStore(cfg)
    array = store.snapshot_array(time_index)
    gx, gy, gz = cfg.grid_shape
    expected_shape = (3, gz, gy, gx)
    if array.shape != expected_shape or np.dtype(array.dtype) != np.dtype("<f4"):
        raise ValueError(f"raw array schema mismatch: shape={array.shape}, dtype={array.dtype}")
    expected_tiles = tiles_for(cfg)
    expected_keys = {tile.key for tile in expected_tiles}
    total_count = 0
    total_sum = 0.0
    total_sumsq = 0.0
    minimum = np.inf
    maximum = -np.inf
    tile_manifest: list[dict[str, Any]] = []
    with Catalog(cfg.catalog_path) as catalog:
        rows = catalog.tiles(cfg.dataset, time_index)
        if {row["tile_key"] for row in rows} != expected_keys:
            raise ValueError("catalog tile coverage differs from the deterministic full-domain plan")
        row_map = {row["tile_key"]: row for row in rows}
        for tile in expected_tiles:
            row = row_map[tile.key]
            if row["status"] != "verified" or not row["sha256"]:
                raise ValueError(f"tile {tile.key} is not verified")
            block = np.asarray(array[tile.store_slices], dtype="<f4")
            if block.shape != (3, tile.nz, tile.ny, tile.nx) or not np.all(np.isfinite(block)):
                raise ValueError(f"tile {tile.key} failed shape/finite validation")
            digest = array_sha256(block)
            if digest != row["sha256"]:
                raise ValueError(f"tile {tile.key} checksum differs from catalog")
            values64 = block.astype(np.float64)
            total_count += values64.size
            total_sum += float(values64.sum())
            total_sumsq += float(np.square(values64).sum())
            minimum = min(minimum, float(values64.min()))
            maximum = max(maximum, float(values64.max()))
            tile_manifest.append({
                "key": tile.key,
                "origin_xyz_0based": [tile.x0, tile.y0, tile.z0],
                "shape_xyz": [tile.nx, tile.ny, tile.nz],
                "sha256": digest,
            })
        mean = total_sum / total_count
        rms = float(np.sqrt(total_sumsq / total_count))
        qa = {
            "dataset": cfg.dataset,
            "time_index": time_index,
            "physical_time": cfg.physical_time(time_index),
            "status": "auto_validated",
            "shape_component_zyx": list(expected_shape),
            "dtype": "float32",
            "finite": True,
            "coverage_exactly_once": True,
            "periodic_endpoint_duplicated": False,
            "statistics": {"minimum": minimum, "maximum": maximum, "mean": mean, "rms": rms},
            "seams": _seam_statistics(array),
        }
        _atomic_json(cfg.qa_path / f"t{time_index:06d}.json", qa)
        manifest = {
            "schema_version": 1,
            "dataset": cfg.dataset,
            "variable": "velocity",
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
        manifest_hash = _atomic_json(cfg.manifest_path / f"t{time_index:06d}.json", manifest)
        catalog.set_snapshot_status(cfg.dataset, time_index, "auto_validated", manifest_hash)
        store.mark_status(time_index, "auto_validated", manifest_hash)
        qa["manifest_hash"] = manifest_hash
        return qa
