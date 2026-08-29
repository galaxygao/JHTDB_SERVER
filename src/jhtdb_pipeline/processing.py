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
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .config import RESULT_SCHEMA_VERSION, PipelineConfig, result_zarr_name
from .physics import (
    ComponentView,
    ProductView,
    accumulate_product,
    close_memmap,
    derivative_field,
    filter_field,
    memmap,
    subtract_product,
    zero_field,
)
from .store import create_result_group, hash_zarr_array
from .sbar_qa import compute_sbar_qa, write_sbar_artifacts
from .validation import atomic_json, input_manifest_hash


RESULT_FIELDS = {
    "velocity": ("<f4", 4),
    "gradient": ("<f4", 5),
    "velocity_bar": ("<f4", 4),
    "gradient_bar": ("<f4", 5),
    "work_full": ("<f4", 3),
    "work_resolved": ("<f4", 3),
    "pi": ("<f4", 3),
    "s_bar": ("<f4", 3),
    "regime": ("u1", 3),
}


def _complete_result_is_current(path: Path, sigma_grid: float) -> bool:
    if not (path / "COMPLETE").is_file():
        return False
    zarr_path = path / result_zarr_name(sigma_grid)
    if not zarr_path.is_dir():
        return False
    try:
        root = zarr.open_group(str(zarr_path), mode="r")
        return (
            root.attrs.get("status") == "complete"
            and root.attrs.get("result_schema_version") == RESULT_SCHEMA_VERSION
            and all(name in root for name in RESULT_FIELDS)
        )
    except Exception:
        return False


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


def _accumulate_zarr(destination: Any, source: Any) -> None:
    shape = tuple(int(value) for value in destination.shape)
    chunks = tuple(int(value) for value in destination.chunks)
    for key in _spatial_keys(shape, chunks):
        current = np.asarray(destination[key], dtype=np.float32)
        values = np.asarray(source[key], dtype=np.float32)
        destination[key] = current + values


def _accumulate_zarr_product(destination: Any, left: Any, right: Any) -> None:
    shape = tuple(int(value) for value in destination.shape)
    chunks = tuple(int(value) for value in destination.chunks)
    for key in _spatial_keys(shape, chunks):
        current = np.asarray(destination[key], dtype=np.float32)
        left_values = np.asarray(left[key], dtype=np.float32)
        right_values = np.asarray(right[key], dtype=np.float32)
        destination[key] = current + left_values * right_values


def _copy_full(destination: Any, source: Any, slab: int) -> None:
    for start in range(0, destination.shape[0], slab):
        key = (
            slice(start, min(start + slab, destination.shape[0])),
            slice(None),
            slice(None),
        )
        destination[key] = np.asarray(source[key], dtype=np.float32)


def _accumulate_full(destination: Any, source: Any, slab: int) -> None:
    for start in range(0, destination.shape[0], slab):
        key = (
            slice(start, min(start + slab, destination.shape[0])),
            slice(None),
            slice(None),
        )
        destination[key] = np.asarray(
            destination[key], dtype=np.float32
        ) + np.asarray(source[key], dtype=np.float32)


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


def _divergence_metrics(
    cfg: PipelineConfig,
    divergence_sumsq: float,
    divergence_maximum: float,
    point_count: int,
    gradient_sumsq: float,
    gradient_maximum: float,
    gradient_count: int,
) -> dict[str, float | int | bool | str]:
    divergence_rms = float(np.sqrt(divergence_sumsq / point_count))
    gradient_rms = float(np.sqrt(gradient_sumsq / gradient_count))
    relative_rms = divergence_rms / max(gradient_rms, 1.0e-30)
    relative_maximum = divergence_maximum / max(gradient_maximum, 1.0e-30)
    return {
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


def _reusable_filtered_velocity(
    cfg: PipelineConfig,
    time_index: int,
    sigma: float,
    manifest_hash: str,
    expected_shape: tuple[int, ...],
) -> np.memmap | None:
    workspace = cfg.workspace_path(time_index, sigma)
    path = workspace / "filtered_velocity.f32"
    if not path.is_file() or path.stat().st_size != int(np.prod(expected_shape)) * 4:
        return None
    metadata_path = workspace / "filtered_velocity.json"
    trusted = False
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            trusted = (
                metadata.get("status") == "complete"
                and metadata.get("input_manifest_hash") == manifest_hash
                and float(metadata.get("sigma_grid")) == sigma
                and tuple(metadata.get("shape", ())) == expected_shape
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            trusted = False
    mapped = memmap(path, expected_shape, mode="r+")
    if trusted:
        for component in range(3):
            _field_statistics(
                ComponentView(mapped, component), cfg.fft_slab_width
            )
        return mapped

    previous = cfg.result_path(time_index, sigma)
    previous_zarr = previous / result_zarr_name(sigma)
    if not (previous / "COMPLETE").is_file() or not previous_zarr.is_dir():
        close_memmap(mapped)
        return None
    try:
        root = zarr.open_group(str(previous_zarr), mode="r")
        stored = root["velocity_bar"]
        if tuple(stored.shape) != (3, *cfg.result_shape_zyx):
            close_memmap(mapped)
            return None
        chunks = tuple(int(value) for value in stored.chunks[-3:])
        for component in range(3):
            for relative in _spatial_keys(cfg.result_shape_zyx, chunks):
                old = np.asarray(stored[(component,) + relative], dtype=np.float32)
                candidate = np.asarray(
                    mapped[(component,) + _source_key(cfg, relative)],
                    dtype=np.float32,
                )
                if not np.allclose(old, candidate, rtol=2.0e-5, atol=2.0e-6):
                    close_memmap(mapped)
                    return None
    except Exception:
        close_memmap(mapped)
        return None
    return mapped


def _validate_previous_center_overlap(
    cfg: PipelineConfig,
    sigma: float,
    previous: Path,
    current: Any,
) -> dict[str, Any] | None:
    previous_zarr = previous / result_zarr_name(sigma)
    if not (previous / "COMPLETE").is_file() or not previous_zarr.is_dir():
        return None
    old = zarr.open_group(str(previous_zarr), mode="r")
    if any(name not in old for name in ("work_full", "work_resolved", "pi", "s_bar")):
        return None
    if any(tuple(old[name].shape) != cfg.result_shape_zyx for name in ("work_full", "work_resolved", "pi", "s_bar")):
        return None
    maximum = {name: 0.0 for name in ("work_full", "work_resolved", "pi", "s_bar")}
    chunks = tuple(int(value) for value in old["work_full"].chunks)
    for relative in _spatial_keys(cfg.result_shape_zyx, chunks):
        source = _source_key(cfg, relative)
        for name in maximum:
            expected = np.asarray(old[name][relative], dtype=np.float32)
            actual = np.asarray(current[name][source], dtype=np.float32)
            difference = float(np.max(np.abs(expected - actual)))
            maximum[name] = max(maximum[name], difference)
            if not np.allclose(expected, actual, rtol=5.0e-5, atol=5.0e-6):
                raise RuntimeError(
                    f"schema-v4 center overlap differs from existing {name}"
                )
    return {
        "passed": True,
        "scope": "stored_center_crop",
        "maximum_abs_difference": maximum,
    }


def resource_plan(cfg: PipelineConfig) -> dict[str, float]:
    full_points = int(np.prod(cfg.grid_shape, dtype=np.int64))
    center_points = int(np.prod(cfg.crop_shape, dtype=np.int64))
    scalar_bytes = full_points * 4
    # filtered velocity (3) + SGS transport (3) + derivative + acceleration
    # + two filter buffers.
    workspace_bytes = 10 * scalar_bytes
    legacy_result_bytes = center_points * 113
    batch_result_bytes = len(cfg.sigma_grids) * cfg.result_uncompressed_bytes
    return {
        "velocity_cache_GiB": cfg.bytes_per_snapshot / 1024**3,
        "workspace_GiB": workspace_bytes / 1024**3,
        "scratch_peak_GiB": (cfg.bytes_per_snapshot + workspace_bytes) / 1024**3,
        "persistent_result_GiB": cfg.result_uncompressed_bytes / 1024**3,
        "configured_sigma_count": len(cfg.sigma_grids),
        "persistent_batch_GiB": (
            batch_result_bytes / 1024**3
        ),
        "persistent_v3_result_GiB": legacy_result_bytes / 1024**3,
        "persistent_v3_to_v4_peak_GiB": (
            legacy_result_bytes + cfg.result_uncompressed_bytes
        )
        / 1024**3,
        "persistent_batch_with_reserve_GiB": (
            batch_result_bytes / 1024**3 + cfg.persistent_safety_reserve_gib
        ),
        "observed_account_capacity_GiB": (
            cfg.persistent_capacity_gb_observed * 1.0e9 / 1024**3
        ),
        "persistent_reserve_GiB": cfg.persistent_safety_reserve_gib,
        "fft_workers": cfg.fft_workers,
        "compression_threads": cfg.compression_threads,
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
    workspace_bytes = 10 * full_points * 4
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
    if _complete_result_is_current(final, sigma):
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
    filtered_velocity = _reusable_filtered_velocity(
        cfg, time_index, sigma, manifest_hash, expected
    )
    reused_filtered_velocity = filtered_velocity is not None
    if filtered_velocity is None:
        _safe_rmtree(workspace, cfg.run_path(time_index))
    workspace.mkdir(parents=True, exist_ok=True)
    result = create_result_group(cfg, time_index, sigma, overwrite=True)
    result.attrs.update(
        {
            "input_manifest_hash": manifest_hash,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "algorithm": "full_periodic_spectral_pi_sbar_v4",
            "pi_definition": "tau_ij * d_j(velocity_bar_i)",
            "pi_sign_convention": "Equation (2): W_full = W_resolved - pi + s_bar",
            "s_bar_definition": "d_j(velocity_bar_i * tau_ij)",
            "tau_definition": "filter(velocity_i * velocity_j) - velocity_bar_i * velocity_bar_j",
        }
    )

    full_shape = expected[1:]
    if filtered_velocity is None:
        filtered_velocity = memmap(workspace / "filtered_velocity.f32", expected)
    derivative = memmap(workspace / "derivative.f32", full_shape)
    acceleration = memmap(workspace / "acceleration.f32", full_shape)
    temp_a = memmap(workspace / "filter_a.f32", full_shape)
    temp_b = memmap(workspace / "filter_b.f32", full_shape)
    sgs_transport = memmap(workspace / "sgs_transport.f32", expected)
    cfg.lock_path.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(cfg.lock_path / "process-center.lock"), timeout=0)
    console = Console()
    gradient_sumsq = 0.0
    gradient_maximum = 0.0
    gradient_count = 0
    filtered_gradient_sumsq = 0.0
    filtered_gradient_maximum = 0.0
    filtered_gradient_count = 0

    with lock, Progress(
        SpinnerColumn("line"),
        TextColumn("{task.description}"),
        BarColumn(),
        "{task.completed}/{task.total}",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("periodic spectral pipeline", total=42)
        for component in range(3):
            _copy_crop(
                cfg,
                ComponentView(raw, component),
                result["velocity"],
                (component,),
            )
            progress.advance(task)
            if not reused_filtered_velocity:
                filter_field(
                    ComponentView(raw, component),
                    ComponentView(filtered_velocity, component),
                    temp_a,
                    temp_b,
                    sigma,
                    cfg.fft_slab_width,
                    workers=cfg.fft_workers,
                )
            filtered_velocity.flush()
            _copy_crop(
                cfg,
                ComponentView(filtered_velocity, component),
                result["velocity_bar"],
                (component,),
            )
            progress.advance(task)

        atomic_json(
            workspace / "filtered_velocity.json",
            {
                "status": "complete",
                "input_manifest_hash": manifest_hash,
                "sigma_grid": sigma,
                "shape": list(expected),
                "reused": reused_filtered_velocity,
            },
        )

        _zero_zarr(result["work_full"])
        _zero_zarr(result["work_resolved"])
        _zero_zarr(result["pi"])
        _zero_zarr(result["s_bar"])
        filtered_divergence = ComponentView(sgs_transport, 0)
        zero_field(filtered_divergence, cfg.fft_slab_width)

        for component in range(3):
            zero_field(acceleration, cfg.fft_slab_width)
            for derivative_component in range(3):
                derivative_field(
                    ComponentView(raw, component),
                    derivative,
                    derivative_component,
                    cfg.domain_length,
                    cfg.fft_slab_width,
                    workers=cfg.fft_workers,
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
                workers=cfg.fft_workers,
            )
            derivative.flush()
            _accumulate_zarr_product(
                result["work_full"],
                ComponentView(filtered_velocity, component),
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
                    workers=cfg.fft_workers,
                )
                derivative.flush()
                _copy_crop(
                    cfg,
                    derivative,
                    result["gradient_bar"],
                    (component, derivative_component),
                )
                sumsq, maximum, count = _field_statistics(
                    derivative, cfg.fft_slab_width
                )
                filtered_gradient_sumsq += sumsq
                filtered_gradient_maximum = max(
                    filtered_gradient_maximum, maximum
                )
                filtered_gradient_count += count
                if derivative_component == component:
                    _accumulate_full(
                        filtered_divergence,
                        derivative,
                        cfg.fft_slab_width,
                    )
                accumulate_product(
                    acceleration,
                    ComponentView(filtered_velocity, derivative_component),
                    derivative,
                    cfg.fft_slab_width,
                )
                progress.advance(task)
            _accumulate_zarr_product(
                result["work_resolved"],
                ComponentView(filtered_velocity, component),
                acceleration,
            )
            progress.advance(task)

        (
            filtered_divergence_sumsq,
            filtered_divergence_maximum,
            filtered_point_count,
        ) = _field_statistics(filtered_divergence, cfg.fft_slab_width)

        for component in range(3):
            zero_field(ComponentView(sgs_transport, component), cfg.fft_slab_width)

        for left_component in range(3):
            for right_component in range(left_component, 3):
                filter_field(
                    ProductView(
                        ComponentView(raw, left_component),
                        ComponentView(raw, right_component),
                    ),
                    derivative,
                    temp_a,
                    temp_b,
                    sigma,
                    cfg.fft_slab_width,
                    workers=cfg.fft_workers,
                )
                subtract_product(
                    derivative,
                    ComponentView(filtered_velocity, left_component),
                    ComponentView(filtered_velocity, right_component),
                    cfg.fft_slab_width,
                )
                derivative.flush()
                _copy_full(temp_a, derivative, cfg.fft_slab_width)

                derivative_field(
                    ComponentView(filtered_velocity, left_component),
                    derivative,
                    right_component,
                    cfg.domain_length,
                    cfg.fft_slab_width,
                    workers=cfg.fft_workers,
                )
                _accumulate_zarr_product(
                    result["pi"], temp_a, derivative
                )
                if left_component != right_component:
                    derivative_field(
                        ComponentView(filtered_velocity, right_component),
                        derivative,
                        left_component,
                        cfg.domain_length,
                        cfg.fft_slab_width,
                        workers=cfg.fft_workers,
                    )
                    _accumulate_zarr_product(
                        result["pi"], temp_a, derivative
                    )

                accumulate_product(
                    ComponentView(sgs_transport, right_component),
                    ComponentView(filtered_velocity, left_component),
                    temp_a,
                    cfg.fft_slab_width,
                )
                if left_component != right_component:
                    accumulate_product(
                        ComponentView(sgs_transport, left_component),
                        ComponentView(filtered_velocity, right_component),
                        temp_a,
                        cfg.fft_slab_width,
                    )
                progress.advance(task)

        sgs_transport.flush()
        for derivative_component in range(3):
            derivative_field(
                ComponentView(sgs_transport, derivative_component),
                derivative,
                derivative_component,
                cfg.domain_length,
                cfg.fft_slab_width,
                workers=cfg.fft_workers,
            )
            derivative.flush()
            _accumulate_zarr(result["s_bar"], derivative)
            progress.advance(task)

        zero_field(acceleration, cfg.fft_slab_width)
        for component in range(3):
            derivative_field(
                ComponentView(raw, component),
                derivative,
                component,
                cfg.domain_length,
                cfg.fft_slab_width,
                workers=cfg.fft_workers,
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
    for mapped in (
        filtered_velocity,
        derivative,
        acceleration,
        temp_a,
        temp_b,
        sgs_transport,
    ):
        close_memmap(mapped)
    unfiltered_divergence_report = _divergence_metrics(
        cfg,
        divergence_sumsq,
        divergence_maximum,
        point_count,
        gradient_sumsq,
        gradient_maximum,
        gradient_count,
    )
    unfiltered_divergence_report["scope"] = "full_domain"
    filtered_divergence_report = _divergence_metrics(
        cfg,
        filtered_divergence_sumsq,
        filtered_divergence_maximum,
        filtered_point_count,
        filtered_gradient_sumsq,
        filtered_gradient_maximum,
        filtered_gradient_count,
    )
    filtered_divergence_report["scope"] = "full_domain"
    divergence_report = {
        "passed": bool(
            unfiltered_divergence_report["passed"]
            and filtered_divergence_report["passed"]
        ),
        "unfiltered": unfiltered_divergence_report,
        "filtered": filtered_divergence_report,
    }
    atomic_json(
        staging / "divergence.json", divergence_report
    )
    if not divergence_report["passed"]:
        result.attrs["status"] = "failed_divergence"
        raise RuntimeError(
            "full-domain divergence validation failed: "
            "unfiltered_relative_rms="
            f"{unfiltered_divergence_report['relative_divergence_rms']:.3e}, "
            "unfiltered_relative_max="
            f"{unfiltered_divergence_report['relative_maximum_divergence']:.3e}, "
            "filtered_relative_rms="
            f"{filtered_divergence_report['relative_divergence_rms']:.3e}, "
            "filtered_relative_max="
            f"{filtered_divergence_report['relative_maximum_divergence']:.3e}"
        )

    s_bar_report = compute_sbar_qa(result, cfg, scope="full_domain")
    s_bar_report_hash = write_sbar_artifacts(staging, s_bar_report)
    overlap_report = _validate_previous_center_overlap(cfg, sigma, final, result)
    identity_metric = s_bar_report["metrics"]["identity_residual_rms"]
    decomposition_report = {
        "identity": "work_full = work_resolved - pi + s_bar",
        "scope": "full_domain",
        "residual_rms": identity_metric["value"],
        "residual_maximum_abs": identity_metric["maximum_abs"],
    }

    full_sumsq = 0.0
    resolved_sumsq = 0.0
    center_count = 0
    regime_chunks = tuple(int(value) for value in result["regime"].chunks)
    for relative in _spatial_keys(cfg.result_shape_zyx, regime_chunks):
        source = _source_key(cfg, relative)
        full = np.asarray(result["work_full"][source], dtype=np.float32)
        resolved = np.asarray(result["work_resolved"][source], dtype=np.float32)
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
    for relative in _spatial_keys(cfg.result_shape_zyx, regime_chunks):
        source = _source_key(cfg, relative)
        full = np.asarray(result["work_full"][source], dtype=np.float32)
        resolved = np.asarray(result["work_resolved"][source], dtype=np.float32)
        codes = np.zeros(full.shape, dtype=np.uint8)
        codes[(full > epsilon_full) & (resolved > epsilon_resolved)] = 1
        codes[(full > epsilon_full) & (resolved < -epsilon_resolved)] = 2
        codes[(full < -epsilon_full) & (resolved > epsilon_resolved)] = 3
        codes[(full < -epsilon_full) & (resolved < -epsilon_resolved)] = 4
        result["regime"][relative] = codes
        occupancy += np.bincount(codes.ravel(), minlength=5)

    qa = {
        "dataset": cfg.dataset,
        "time_index": time_index,
        "sigma_grid": sigma,
        "input_manifest_hash": manifest_hash,
        "divergence": divergence_report,
        "decomposition": decomposition_report,
        "s_bar_global": {
            "passed": s_bar_report["passed"],
            "scope": s_bar_report["scope"],
            "report_hash": s_bar_report_hash,
            "metrics": s_bar_report["metrics"],
        },
        "reuse": {
            "velocity_cache": True,
            "filtered_velocity": reused_filtered_velocity,
            "previous_center_overlap": overlap_report,
        },
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
            "decomposition": decomposition_report,
            "s_bar_qa_passed": s_bar_report["passed"],
            "s_bar_qa_report_hash": s_bar_report_hash,
            "reused_filtered_velocity": reused_filtered_velocity,
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
    if _complete_result_is_current(final, sigma):
        return final
    if not staging.is_dir():
        raise RuntimeError("persistent result staging directory is missing")
    root = zarr.open_group(str(staging / result_zarr_name(sigma)), mode="a")
    if root.attrs.get("status") != "processed":
        raise RuntimeError("persistent result staging has not completed processing")

    expected_shapes = {
        "velocity": (3, *cfg.result_shape_zyx),
        "gradient": (3, 3, *cfg.result_shape_zyx),
        "velocity_bar": (3, *cfg.result_shape_zyx),
        "gradient_bar": (3, 3, *cfg.result_shape_zyx),
        "work_full": cfg.full_shape_zyx,
        "work_resolved": cfg.full_shape_zyx,
        "pi": cfg.full_shape_zyx,
        "s_bar": cfg.full_shape_zyx,
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
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "dataset": cfg.dataset,
        "time_index": time_index,
        "physical_time": cfg.physical_time(time_index),
        "sigma_grid": sigma,
        "input_manifest_hash": input_manifest_hash(cfg, time_index),
        "algorithm": root.attrs.get("algorithm"),
        "crop_start_xyz": list(cfg.crop_start),
        "crop_shape_xyz": list(cfg.crop_shape),
        "full_shape_xyz": list(cfg.grid_shape),
        "field_scopes": dict(root.attrs.get("field_scopes", {})),
        "s_bar_qa_passed": bool(root.attrs.get("s_bar_qa_passed", False)),
        "s_bar_qa_report_hash": root.attrs.get("s_bar_qa_report_hash"),
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
        if (final / "COMPLETE").is_file():
            _safe_rmtree(final, cfg.result_root)
        else:
            raise FileExistsError(f"refusing to replace incomplete result: {final}")
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


def backfill_full_fields(
    cfg: PipelineConfig,
    time_index: int,
    sigma_grid: float | None = None,
) -> Path:
    """Build schema v4 from an older complete result without fetching JHTDB."""
    sigma = cfg.sigma_grid if sigma_grid is None else float(sigma_grid)
    final = cfg.result_path(time_index, sigma)
    if _complete_result_is_current(final, sigma):
        return final
    if not (final / "COMPLETE").is_file():
        raise RuntimeError("an older complete persistent result is required")
    manifest_path = final / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("older result manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in (2, 3):
        raise RuntimeError("backfill requires a complete schema-v2 or schema-v3 result")
    if not cfg.raw_store_path(time_index).is_dir():
        raise RuntimeError(
            "validated temporary velocity_cache.zarr is required; "
            "backfill-full-fields never fetches JHTDB automatically"
        )
    process_center(cfg, time_index, sigma)
    return finalize_result(cfg, time_index, sigma)


def upgrade_result(
    cfg: PipelineConfig,
    time_index: int,
    sigma_grid: float | None = None,
) -> Path:
    """Backward-compatible alias for full-field backfill."""
    return backfill_full_fields(cfg, time_index, sigma_grid)
