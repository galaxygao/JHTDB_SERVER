from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import TaskConfig
from .grid import point_batches, query_points, rows_to_gradient, rows_to_velocity, time_chunks


VELOCITY_NAMES = ("ux", "uy", "uz")
GRADIENT_NAMES = tuple(f"du{u}d{d}" for u in "xyz" for d in "xyz")


def _normal_name(name: object) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_columns(columns: Iterable[object], required: tuple[str, ...]) -> list[int]:
    normalized = [_normal_name(column) for column in columns]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"duplicate normalized JHTDB columns: {list(columns)}")
    positions = []
    for name in required:
        aliases = {name, name.replace("u", "velocity", 1)} if name.startswith("u") and not name.startswith("du") else {name}
        matches = [index for index, value in enumerate(normalized) if value in aliases]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one column for {name}; got columns {list(columns)}")
        positions.append(matches[0])
    return positions


def parse_velocity(values: np.ndarray, columns: Iterable[object]) -> np.ndarray:
    positions = _find_columns(columns, VELOCITY_NAMES)
    return np.asarray(values[..., positions], dtype=np.float64)


def parse_gradient(values: np.ndarray, columns: Iterable[object]) -> np.ndarray:
    positions = _find_columns(columns, GRADIENT_NAMES)
    ordered = np.asarray(values[..., positions], dtype=np.float64)
    return ordered.reshape(ordered.shape[:-1] + (3, 3))


class JHTDBClient:
    """Strictly serial, checkpointed wrapper around official givernylocal.getData."""

    def __init__(self, cfg: TaskConfig):
        self.cfg = cfg
        cache_identity = {
            "dataset": cfg.dataset,
            "variable": cfg.variable,
            "grid_shape": cfg.grid_shape,
            "domain_length": cfg.domain_length,
            "block_start_ijk": cfg.block_start_ijk,
            "block_shape": cfg.block_shape,
            "times": cfg.times.tolist(),
        }
        self.cache_namespace = hashlib.sha256(
            json.dumps(cache_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        cfg.cache_path.mkdir(parents=True, exist_ok=True)
        api_output = cfg.cache_path / "giverny_output"
        api_output.mkdir(parents=True, exist_ok=True)
        try:
            from givernylocal.turbulence_dataset import turb_dataset
            from givernylocal.turbulence_toolkit import getData
        except ImportError as exc:
            raise RuntimeError("givernylocal is missing; run: python -m pip install givernylocal") from exc

        supplied_token = os.environ.get(cfg.token_env, "").strip()
        # Construct once to load the official metadata. The built-in testing token is
        # then copied only in memory and is never printed or persisted by this project.
        self.cube = turb_dataset(cfg.dataset, str(api_output), supplied_token)
        builtin_token = str(self.cube.metadata["constants"]["pyJHTDB_testing_token"])
        if supplied_token:
            self.token_mode = "testing" if supplied_token == builtin_token else "personal"
        elif cfg.use_builtin_testing_token:
            self.cube.auth_token = builtin_token
            self.token_mode = "testing"
        else:
            raise RuntimeError(f"environment variable {cfg.token_env} is not set")
        if self.token_mode == "testing" and cfg.max_points_per_query > 4000:
            raise ValueError("testing token requires max_points_per_query <= 4000")
        self._get_data = getData

    def _cache_file(self, method: str, operator: str, point_slice: slice, time_indices: tuple[int, int]) -> Path:
        label = (
            f"{self.cache_namespace}_{operator}_{method}_"
            f"p{point_slice.start}-{point_slice.stop}_t{time_indices[0]}-{time_indices[1]}.npz"
        )
        return self.cfg.cache_path / label

    def _query_once(
        self,
        points: np.ndarray,
        times: np.ndarray,
        method: str,
        operator: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kwargs = {"return_times": True, "verbose": False}
        if len(times) > 1:
            deltas = np.diff(times)
            if not np.allclose(deltas, deltas[0], rtol=0.0, atol=1e-12):
                raise ValueError("a getData time chunk must be uniformly spaced")
            kwargs["option"] = [float(times[-1]), float(deltas[0])]
        results, returned_times = self._get_data(
            self.cube,
            self.cfg.variable,
            float(times[0]),
            "none",
            method,
            operator,
            points,
            **kwargs,
        )
        if len(results) != len(times):
            raise RuntimeError(f"JHTDB returned {len(results)} frames for {len(times)} requested times")
        returned_times = np.asarray(returned_times, dtype=np.float64)
        if not np.allclose(returned_times, times, rtol=0.0, atol=2e-10):
            raise RuntimeError(f"returned times differ from requested times: {returned_times} vs {times}")
        columns = np.asarray([str(column) for column in results[0].columns])
        for frame in results:
            if list(map(str, frame.columns)) != columns.tolist():
                raise RuntimeError("JHTDB columns changed within a time-series response")
            if len(frame) != len(points):
                raise RuntimeError("JHTDB point count differs from request")
        values = np.stack([frame.to_numpy(dtype=np.float64, copy=True) for frame in results], axis=0)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("JHTDB response contains NaN or Inf")
        return values, columns, returned_times

    def _cached_query(
        self,
        points: np.ndarray,
        times: np.ndarray,
        method: str,
        operator: str,
        point_slice: slice,
        time_indices: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = self._cache_file(method, operator, point_slice, time_indices)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                values = saved["values"]
                columns = saved["columns"]
                returned_times = saved["times"]
            if values.shape[:2] == (len(times), len(points)) and np.allclose(returned_times, times, atol=2e-10, rtol=0.0):
                return values, columns, returned_times
            raise RuntimeError(f"invalid checkpoint: {path}")

        error: Exception | None = None
        for attempt in range(self.cfg.retries):
            try:
                values, columns, returned_times = self._query_once(points, times, method, operator)
                np.savez_compressed(path, values=values.astype(np.float32), columns=columns, times=returned_times)
                return values, columns, returned_times
            except Exception as exc:  # retain the service message, but never the token
                error = exc
                if attempt + 1 < self.cfg.retries:
                    time.sleep(self.cfg.retry_backoff_seconds * (attempt + 1))
        assert error is not None
        raise error

    def _query_with_split(
        self,
        all_points: np.ndarray,
        times: np.ndarray,
        method: str,
        operator: str,
        point_slice: slice,
        global_start: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            values, columns, _ = self._cached_query(
                all_points[point_slice], times, method, operator, point_slice, (global_start, global_start + len(times))
            )
            return values, columns
        except Exception:
            if len(times) == 1:
                raise
            middle = len(times) // 2
            left, columns_left = self._query_with_split(
                all_points, times[:middle], method, operator, point_slice, global_start
            )
            right, columns_right = self._query_with_split(
                all_points, times[middle:], method, operator, point_slice, global_start + middle
            )
            if columns_left.tolist() != columns_right.tolist():
                raise RuntimeError("JHTDB columns changed between split time queries")
            return np.concatenate((left, right), axis=0), columns_left

    def fetch_rows(self, method: str, operator: str, times: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        requested_times = self.cfg.times if times is None else np.asarray(times, dtype=np.float64)
        points = query_points(self.cfg)
        time_parts = []
        reference_columns: np.ndarray | None = None
        for start, _, chunk_times in time_chunks(requested_times, self.cfg.time_chunk_size):
            spatial_parts = []
            for point_slice in point_batches(len(points), self.cfg.max_points_per_query):
                values, columns = self._query_with_split(
                    points, chunk_times, method, operator, point_slice, start
                )
                if reference_columns is None:
                    reference_columns = columns
                elif reference_columns.tolist() != columns.tolist():
                    raise RuntimeError("JHTDB columns changed between point/time batches")
                spatial_parts.append(values)
            time_parts.append(np.concatenate(spatial_parts, axis=1))
        assert reference_columns is not None
        return np.concatenate(time_parts, axis=0), reference_columns

    def fetch_all(self) -> Path:
        velocity_rows, velocity_columns = self.fetch_rows("none", "field")
        gradient_rows, gradient_columns = self.fetch_rows(self.cfg.gradient_primary, "gradient")
        velocity = rows_to_velocity(parse_velocity(velocity_rows, velocity_columns), self.cfg.block_shape)
        gradient_primary = rows_to_gradient(parse_gradient(gradient_rows, gradient_columns), self.cfg.block_shape)

        arrays: dict[str, np.ndarray] = {
            "times": self.cfg.times,
            "indices_ijk": np.asarray(query_points(self.cfg) * self.cfg.grid_shape[0] / self.cfg.domain_length).round().astype(np.int32),
            "velocity": velocity.astype(np.float32),
            "gradient_primary": gradient_primary.astype(np.float32),
            "velocity_columns": velocity_columns,
            "gradient_columns": gradient_columns,
            "gradient_primary_method": np.asarray(self.cfg.gradient_primary),
        }
        if self.cfg.gradient_audit:
            audit_rows, audit_columns = self.fetch_rows(self.cfg.gradient_audit, "gradient")
            arrays["gradient_audit"] = rows_to_gradient(
                parse_gradient(audit_rows, audit_columns), self.cfg.block_shape
            ).astype(np.float32)
            arrays["gradient_audit_method"] = np.asarray(self.cfg.gradient_audit)
            arrays["gradient_audit_columns"] = audit_columns

        self.cfg.raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.cfg.raw_path, **arrays)
        self._write_manifest(arrays)
        return self.cfg.raw_path

    def smoke(self) -> dict[str, object]:
        points = query_points(self.cfg)[:8]
        times = self.cfg.times[:2]
        velocity, velocity_columns, returned = self._query_once(points, times, "none", "field")
        gradient, gradient_columns, returned_gradient = self._query_once(
            points, times, self.cfg.gradient_primary, "gradient"
        )
        parse_velocity(velocity, velocity_columns)
        parse_gradient(gradient, gradient_columns)
        if not np.array_equal(returned, returned_gradient):
            raise RuntimeError("velocity and gradient smoke times differ")
        result = {
            "status": "ok",
            "token_mode": self.token_mode,
            "dataset": self.cfg.dataset,
            "points": len(points),
            "times": returned.tolist(),
            "velocity_columns": velocity_columns.tolist(),
            "gradient_columns": gradient_columns.tolist(),
        }
        self.cfg.reports_path.mkdir(parents=True, exist_ok=True)
        with (self.cfg.reports_path / "smoke.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        return result

    def _write_manifest(self, arrays: dict[str, np.ndarray]) -> None:
        manifest = {
            "dataset": self.cfg.dataset,
            "token_mode": self.token_mode,
            "token_persisted": False,
            "times": self.cfg.times.tolist(),
            "block_start_ijk": list(self.cfg.block_start_ijk),
            "block_shape": list(self.cfg.block_shape),
            "max_points_per_query": self.cfg.max_points_per_query,
            "strictly_serial": True,
            "gradient_primary": self.cfg.gradient_primary,
            "gradient_audit": self.cfg.gradient_audit,
            "raw_sha256": None,
        }
        with self.cfg.raw_path.open("rb") as handle:
            manifest["raw_sha256"] = hashlib.sha256(handle.read()).hexdigest()
        manifest_path = self.cfg.raw_path.with_suffix(".manifest.json")
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
