from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import zarr
from filelock import FileLock
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from .config import PipelineConfig
from .physics import (
    ComponentView,
    accumulate_product,
    close_memmap,
    derivative_field,
    filter_field,
    memmap,
    zero_field,
)
from .store import create_result_group, hash_zarr_array
from .validation import atomic_json, input_manifest_hash


RESULT_FIELDS = {
    "velocity": ("<f4", 4),
    "gradient": ("<f4", 5),
    "velocity_bar": ("<f4", 4),
    "gradient_bar": ("<f4", 5),
    "work_full": ("<f4", 3),
    "work_resolved": ("<f4", 3),
    "regime": ("u1", 3),
}


def _safe_rmtree(path: Path, required_parent: Path) -> None:
    resolved = path.resolve(strict=False)
    parent = required_parent.resolve(strict=False)
    if resolved.parent != parent or resolved == parent:
        raise RuntimeError(f"refusing to remove unexpected path: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def _spatial_keys(shape: tuple[int, int, int], chunks: tuple[int, int, int]) -> Iterator[tuple[slice, slice, slice]]:
    for z0 in range(0, shape[0], chunks[0]):
        for y0 in range(0, shape[1], chunks[1]):
            for x0 in range(0, shape[2], chunks[2]):
                yield (
                    slice(z0, min(z0 + chunks[0], shape[0])),
                    slice(y0, min(y0 + chunks[1], shape[1])),
                    slice(x0, min(x0 + chunks[2], shape[2])),
                )


def _source_key(cfg: PipelineConfig, relative: tuple[slice, slice, slice]) -> tuple[slice, slice, slice]:
    x0, y0, z0 = cfg.crop_start
    return (
        slice(z0 + relative[0].start, z0 + relative[0].stop),
        slice(y0 + relative[1].start, y0 + relative[1].stop),
        slice(x0 + relative[2].start, x0 + relative[2].stop),
    )


def _copy_crop(
    cfg: PipelineConfig,
    source: Any,
    destination: Any,
    prefix: tuple[int, ...] = (),
) -> None:
    shape = cfg.result_shape_zyx
    chunks = tuple(int(value) for value in destination.chunks[-3:])
    for relative in _spatial_keys(shape, chunks):
        values = np.ascontiguousarray(source[_source_key(cfg, relative)], dtype="<f4")
        if not np.all(np.isfinite(values)):
            raise ValueError("center crop contains NaN or Inf")
        destination[prefix + relative] = values
        readback = np.asarray(destination[prefix + relative], dtype="<f4")
        if not np.array_equal(values, readback):
            raise IOError("persistent center field failed write/read verification")


def _zero_zarr(array: Any) -> None:
    shape = tuple(int(value) for value in array.shape)
    chunks = tuple(int(value) for value in array.chunks)
    for key in _spatial_keys(shape, chunks):
        array[key] = np.zeros(tuple(item.stop - item.start for item in key), dtype=array.dtype)


def _accumulate_center(
    cfg: PipelineConfig,
    destination: Any,
    left_center: Any,
    left_prefix: tuple[int, ...],
    right_full: Any,
) -> None:
    shape = cfg.result_shape_zyx
    chunks = tuple(int(value) for value in destination.chunks)
    for relative in _spatial_keys(shape, chunks):
        current = np.asarray(destination[relative], dtype=np.float32)
        left = np.asarray(left_center[left_prefix + relative], dtype=np.float32)
        right = np.asarray(right_full[_source_key(cfg, relative)], dtype=np.float32)
        destination[relative] = current + left * right


def _field_statistics(field: Any, slab: int) -> tuple[float, float, int]:
    sumsq = 0.0
    maximum = 0.0
    count = 0
    for start in range(0, field.shape[0], slab):
        values = np.asarray(
            field[start : min(start + slab, field.shape[0]), :, :],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("spectral workspace contains NaN or Inf")
        sumsq += float(np.square(values, dtype=np.float64).sum())
        maximum = max(maximum, float(np.max(np.abs(values))))
        count += values.size
    return sumsq, maximum, count


def resource_plan(cfg: PipelineConfig) -> dict[str, float]:
    full_points = int(np.prod(cfg.grid_shape, dtype=np.int64))
    scalar_bytes = full_points * 4
    # filtered velocity (3) + derivative + acceleration + two filter buffers.
    workspace_bytes = 7 * scalar_bytes
    return {
        "velocity_cache_GiB": cfg.bytes_per_snapshot / 1024**3,
        "workspace_GiB": workspace_bytes / 1024**3,
        "scratch_peak_GiB": (cfg.bytes_per_snapshot + workspace_bytes) / 1024**3,
        "persistent_result_GiB": cfg.result_uncompressed_bytes / 1024**3,
        "persistent_reserve_GiB": cfg.persistent_safety_reserve_gib,
        "fft_input_block_MiB": (
            cfg.fft_slab_width
            * cfg.grid_shape[0]
            * cfg.grid_shape[1]
            * 4
            / 1024**2
        ),
    }


def _preflight_result_space(cfg: PipelineConfig) -> dict[str, float]:
    cfg.result_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(cfg.result_root)
    required = cfg.result_uncompressed_bytes + int(
        cfg.persistent_safety_reserve_gib * 1024**3
    )
    if usage.free < required:
        raise RuntimeError(
            f"insufficient persistent space: {usage.free / 1024**3:.2f} GiB free, "
            f"need {required / 1024**3:.2f} GiB including reserve"
        )
    return {
        "filesystem_free_GiB": usage.free / 1024**3,
        "required_GiB": required / 1024**3,
        "observed_account_capacity_GB": cfg.persistent_capacity_gb_observed,
    }


def _preflight_workspace_space(cfg: PipelineConfig, time_index: int) -> None:
    path = cfg.run_path(time_index)
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    full_points = int(np.prod(cfg.grid_shape, dtype=np.int64))
    workspace_bytes = 7 * full_points * 4
    required = workspace_bytes + int(cfg.scratch_safety_reserve_gib * 1024**3)
    if usage.free < required:
        raise RuntimeError(
            f"insufficient scratch workspace: {usage.free / 1024**3:.2f} GiB free, "
            f"need {required / 1024**3:.2f} GiB in addition to the velocity cache"
        )


def process_center(
    cfg: PipelineConfig,
    time_index: int,
    sigma_grid: float | None = None,
) -> Path:
    sigma = cfg.sigma_grid if sigma_grid is None else float(sigma_grid)
    if sigma <= 0:
        raise ValueError("sigma_grid must be positive")
    manifest_hash = input_manifest_hash(cfg, time_index)
    final = cfg.result_path(time_index, sigma)
    if (final / "COMPLETE").is_file():
        return final
    _preflight_result_space(cfg)
    _preflight_workspace_space(cfg, time_index)

    raw_root = zarr.open_group(str(cfg.raw_store_path(time_index)), mode="r")
    raw = raw_root["velocity"]
    expected = (3, cfg.grid_shape[2], cfg.grid_shape[1], cfg.grid_shape[0])
    if raw.shape != expected or np.dtype(raw.dtype) != np.dtype("<f4"):
        raise ValueError("validated velocity cache has an unexpected schema")
    if raw_root.attrs.get("manifest_hash") != manifest_hash:
        raise RuntimeError("velocity cache and persistent input manifest disagree")

    staging = cfg.staging_result_path(time_index, sigma)
    workspace = cfg.workspace_path(time_index, sigma)
    _safe_rmtree(staging, cfg.result_root / ".staging")
    _safe_rmtree(workspace, cfg.run_path(time_index))
    workspace.mkdir(parents=True, exist_ok=True)
    result = create_result_group(cfg, time_index, sigma, overwrite=True)
    result.attrs.update(
        {
            "input_manifest_hash": manifest_hash,
            "algorithm": "center_periodic_spectral_v1",
        }
    )

    full_shape = expected[1:]
    filtered_velocity = memmap(workspace / "filtered_velocity.f32", expected)
    derivative = memmap(workspace / "derivative.f32", full_shape)
    acceleration = memmap(workspace / "acceleration.f32", full_shape)
    temp_a = memmap(workspace / "filter_a.f32", full_shape)
    temp_b = memmap(workspace / "filter_b.f32", full_shape)
    cfg.lock_path.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(cfg.lock_path / "process-center.lock"), timeout=0)
    console = Console()
    gradient_sumsq = 0.0
    gradient_maximum = 0.0
    gradient_count = 0

    with lock, Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        "{task.completed}/{task.total}",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("center spectral pipeline", total=33)
        for component in range(3):
            _copy_crop(
                cfg,
                ComponentView(raw, component),
                result["velocity"],
                (component,),
            )
            progress.advance(task)
            filter_field(
                ComponentView(raw, component),
                ComponentView(filtered_velocity, component),
                temp_a,
                temp_b,
                sigma,
                cfg.fft_slab_width,
            )
            filtered_velocity.flush()
            _copy_crop(
                cfg,
                ComponentView(filtered_velocity, component),
                result["velocity_bar"],
                (component,),
            )
            progress.advance(task)

        _zero_zarr(result["work_full"])
        _zero_zarr(result["work_resolved"])

        for component in range(3):
            zero_field(acceleration, cfg.fft_slab_width)
            for derivative_component in range(3):
                derivative_field(
                    ComponentView(raw, component),
                    derivative,
                    derivative_component,
                    cfg.domain_length,
                    cfg.fft_slab_width,
                )
                derivative.flush()
                _copy_crop(
                    cfg,
                    derivative,
                    result["gradient"],
                    (component, derivative_component),
                )
                sumsq, maximum, count = _field_statistics(
                    derivative, cfg.fft_slab_width
                )
                gradient_sumsq += sumsq
                gradient_maximum = max(gradient_maximum, maximum)
                gradient_count += count
                accumulate_product(
                    acceleration,
                    ComponentView(raw, derivative_component),
                    derivative,
                    cfg.fft_slab_width,
                )
                progress.advance(task)

            filter_field(
                acceleration,
                derivative,
                temp_a,
                temp_b,
                sigma,
                cfg.fft_slab_width,
            )
            derivative.flush()
            _accumulate_center(
                cfg,
                result["work_full"],
                result["velocity_bar"],
                (component,),
                derivative,
            )
            progress.advance(task)

            zero_field(acceleration, cfg.fft_slab_width)
            for derivative_component in range(3):
                derivative_field(
                    ComponentView(filtered_velocity, component),
                    derivative,
                    derivative_component,
                    cfg.domain_length,
                    cfg.fft_slab_width,
                )
                derivative.flush()
                _copy_crop(
                    cfg,
                    derivative,
                    result["gradient_bar"],
                    (component, derivative_component),
                )
                accumulate_product(
                    acceleration,
                    ComponentView(filtered_velocity, derivative_component),
                    derivative,
                    cfg.fft_slab_width,
                )
                progress.advance(task)
            _accumulate_center(
                cfg,
                result["work_resolved"],
                result["velocity_bar"],
                (component,),
                acceleration,
            )
            progress.advance(task)

        zero_field(acceleration, cfg.fft_slab_width)
        for component in range(3):
            derivative_field(
                ComponentView(raw, component),
                derivative,
                component,
                cfg.domain_length,
                cfg.fft_slab_width,
            )
            for start in range(0, acceleration.shape[0], cfg.fft_slab_width):
                key = (
                    slice(start, min(start + cfg.fft_slab_width, acceleration.shape[0])),
                    slice(None),
                    slice(None),
                )
                acceleration[key] = np.asarray(acceleration[key]) + np.asarray(
                    derivative[key]
                )
            progress.advance(task)

    divergence_sumsq, divergence_maximum, point_count = _field_statistics(
        acceleration, cfg.fft_slab_width
    )
    for mapped in (filtered_velocity, derivative, acceleration, temp_a, temp_b):
        close_memmap(mapped)
    divergence_rms = float(np.sqrt(divergence_sumsq / point_count))
    gradient_rms = float(np.sqrt(gradient_sumsq / gradient_count))
    relative_rms = divergence_rms / max(gradient_rms, 1.0e-30)
    relative_maximum = divergence_maximum / max(gradient_maximum, 1.0e-30)
    divergence_report = {
        "passed": (
            relative_rms <= cfg.divergence_relative_rms_max
            and relative_maximum <= cfg.divergence_relative_max_max
        ),
        "divergence_rms": divergence_rms,
        "gradient_rms": gradient_rms,
        "relative_divergence_rms": relative_rms,
        "maximum_abs_divergence": divergence_maximum,
        "maximum_abs_gradient_component": gradient_maximum,
        "relative_maximum_divergence": relative_maximum,
        "point_count": point_count,
    }
    atomic_json(
        staging / "divergence.json", divergence_report
    )
    if not divergence_report["passed"]:
        result.attrs["status"] = "failed_divergence"
        raise RuntimeError(
            "full-domain divergence validation failed: "
            f"relative_rms={relative_rms:.3e}, relative_max={relative_maximum:.3e}"
        )

    full_sumsq = 0.0
    resolved_sumsq = 0.0
    center_count = 0
    work_chunks = tuple(int(value) for value in result["work_full"].chunks)
    for key in _spatial_keys(cfg.result_shape_zyx, work_chunks):
        full = np.asarray(result["work_full"][key], dtype=np.float32)
        resolved = np.asarray(result["work_resolved"][key], dtype=np.float32)
        full_sumsq += float(np.square(full, dtype=np.float64).sum())
        resolved_sumsq += float(np.square(resolved, dtype=np.float64).sum())
        center_count += full.size
    epsilon_full = max(
        cfg.epsilon_abs, cfg.epsilon_rel * np.sqrt(full_sumsq / center_count)
    )
    epsilon_resolved = max(
        cfg.epsilon_abs,
        cfg.epsilon_rel * np.sqrt(resolved_sumsq / center_count),
    )
    occupancy = np.zeros(5, dtype=np.int64)
    for key in _spatial_keys(cfg.result_shape_zyx, work_chunks):
        full = np.asarray(result["work_full"][key], dtype=np.float32)
        resolved = np.asarray(result["work_resolved"][key], dtype=np.float32)
        codes = np.zeros(full.shape, dtype=np.uint8)
        codes[(full > epsilon_full) & (resolved > epsilon_resolved)] = 1
        codes[(full > epsilon_full) & (resolved < -epsilon_resolved)] = 2
        codes[(full < -epsilon_full) & (resolved > epsilon_resolved)] = 3
        codes[(full < -epsilon_full) & (resolved < -epsilon_resolved)] = 4
        result["regime"][key] = codes
        occupancy += np.bincount(codes.ravel(), minlength=5)

    qa = {
        "dataset": cfg.dataset,
        "time_index": time_index,
        "sigma_grid": sigma,
        "input_manifest_hash": manifest_hash,
        "divergence": divergence_report,
        "epsilon_full": float(epsilon_full),
        "epsilon_resolved": float(epsilon_resolved),
        "occupancy": {
            "uncertain" if index == 0 else f"Q{index}": float(value / occupancy.sum())
            for index, value in enumerate(occupancy)
        },
    }
    atomic_json(staging / "qa.json", qa)
    result.attrs.update(
        {
            "status": "processed",
            "epsilon_full": float(epsilon_full),
            "epsilon_resolved": float(epsilon_resolved),
            "occupancy": qa["occupancy"],
        }
    )
    return staging


def finalize_result(
    cfg: PipelineConfig,
    time_index: int,
    sigma_grid: float | None = None,
) -> Path:
    sigma = cfg.sigma_grid if sigma_grid is None else float(sigma_grid)
    staging = cfg.staging_result_path(time_index, sigma)
    final = cfg.result_path(time_index, sigma)
    if (final / "COMPLETE").is_file():
        return final
    if not staging.is_dir():
        raise RuntimeError("persistent result staging directory is missing")
    root = zarr.open_group(str(staging / "center_result.zarr"), mode="a")
    if root.attrs.get("status") != "processed":
        raise RuntimeError("persistent result staging has not completed processing")

    expected_shapes = {
        "velocity": (3, *cfg.result_shape_zyx),
        "gradient": (3, 3, *cfg.result_shape_zyx),
        "velocity_bar": (3, *cfg.result_shape_zyx),
        "gradient_bar": (3, 3, *cfg.result_shape_zyx),
        "work_full": cfg.result_shape_zyx,
        "work_resolved": cfg.result_shape_zyx,
        "regime": cfg.result_shape_zyx,
    }
    fields: dict[str, Any] = {}
    for name, (dtype, rank) in RESULT_FIELDS.items():
        if name not in root:
            raise RuntimeError(f"result field is missing: {name}")
        array = root[name]
        if len(array.shape) != rank or tuple(array.shape) != expected_shapes[name]:
            raise RuntimeError(f"result field has invalid shape: {name}={array.shape}")
        if np.dtype(array.dtype) != np.dtype(dtype):
            raise RuntimeError(f"result field has invalid dtype: {name}={array.dtype}")
        digest, byte_count, minimum, maximum = hash_zarr_array(array)
        fields[name] = {
            "shape": list(array.shape),
            "dtype": str(np.dtype(array.dtype)),
            "chunks": list(array.chunks),
            "sha256": digest,
            "byte_count": byte_count,
            "minimum": minimum,
            "maximum": maximum,
        }

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "dataset": cfg.dataset,
        "time_index": time_index,
        "physical_time": cfg.physical_time(time_index),
        "sigma_grid": sigma,
        "input_manifest_hash": input_manifest_hash(cfg, time_index),
        "algorithm": root.attrs.get("algorithm"),
        "crop_start_xyz": list(cfg.crop_start),
        "crop_shape_xyz": list(cfg.crop_shape),
        "fields": fields,
    }
    manifest_hash = atomic_json(staging / "manifest.json", manifest)
    root.attrs.update(
        {
            "status": "complete",
            "manifest_hash": manifest_hash,
            "output_hashes": {name: item["sha256"] for name, item in fields.items()},
        }
    )

    if final.exists():
        raise FileExistsError(f"refusing to replace existing result: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    complete = final / "COMPLETE"
    with complete.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"manifest_hash": manifest_hash}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    if cfg.cleanup_scratch_on_success:
        workspace = cfg.workspace_path(time_index, sigma)
        _safe_rmtree(workspace, cfg.run_path(time_index))
    return final
