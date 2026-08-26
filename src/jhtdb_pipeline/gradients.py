from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from filelock import FileLock
from numcodecs import Blosc
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from .catalog import Catalog
from .config import PipelineConfig
from .physics import (
    _ComponentView,
    _GradientView,
    FILTER_METHOD,
    _close_memmap,
    _filter_component,
    _memmap,
    _transform_axis,
    _write_verified_scalar,
)
from .validation import _atomic_json, require_divergence_validation


COMPONENT_NAMES = ("x", "y", "z")
# Stored scalar fields are indexed (z,y,x); derivative_component is ordered (x,y,z).
ARRAY_AXIS_FOR_DERIVATIVE = (2, 1, 0)
FD8_WEIGHTS = {
    -4: 1.0 / 280.0,
    -3: -4.0 / 105.0,
    -2: 1.0 / 5.0,
    -1: -4.0 / 5.0,
    1: 4.0 / 5.0,
    2: -1.0 / 5.0,
    3: 4.0 / 105.0,
    4: -1.0 / 280.0,
}


def gradient_space_plan(cfg: PipelineConfig) -> dict[str, float]:
    points = int(np.prod(cfg.grid_shape, dtype=np.int64))
    scalar_bytes = points * np.dtype(np.float32).itemsize
    gradient_bytes = 9 * scalar_bytes
    scratch_bytes = scalar_bytes
    reserve_bytes = int(cfg.safety_free_space_gib * 1024**3)
    return {
        "gradient_uncompressed_GiB": gradient_bytes / 1024**3,
        "scratch_GiB": scratch_bytes / 1024**3,
        "mapped_address_space_GiB": scratch_bytes / 1024**3,
        "safety_reserve_GiB": reserve_bytes / 1024**3,
        "required_free_GiB": (gradient_bytes + scratch_bytes + reserve_bytes) / 1024**3,
        "fft_input_block_MiB": (
            cfg.fft_slab_width
            * cfg.grid_shape[0]
            * cfg.grid_shape[1]
            * np.dtype(np.float32).itemsize
            / 1024**2
        ),
        "estimated_process_peak_RAM_MiB": 256.0,
    }


def filtered_space_plan(cfg: PipelineConfig) -> dict[str, float]:
    points = int(np.prod(cfg.grid_shape, dtype=np.int64))
    scalar_bytes = points * np.dtype(np.float32).itemsize
    filtered_bytes = 12 * scalar_bytes
    scratch_bytes = 3 * scalar_bytes
    reserve_bytes = int(cfg.safety_free_space_gib * 1024**3)
    return {
        "filtered_velocity_uncompressed_GiB": 3 * scalar_bytes / 1024**3,
        "filtered_gradient_uncompressed_GiB": 9 * scalar_bytes / 1024**3,
        "filtered_total_uncompressed_GiB": filtered_bytes / 1024**3,
        "scratch_GiB": scratch_bytes / 1024**3,
        "mapped_address_space_GiB": scratch_bytes / 1024**3,
        "safety_reserve_GiB": reserve_bytes / 1024**3,
        "required_free_GiB": (filtered_bytes + scratch_bytes + reserve_bytes) / 1024**3,
        "full_domain_axis_transform_passes": 18.0,
        "estimated_process_peak_RAM_MiB": 384.0,
    }


def _gradient_group(cfg: PipelineConfig, time_index: int, input_manifest_hash: str):
    cfg.gradient_store_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(cfg.gradient_store_path), mode="a")
    root.attrs.update(
        {
            "dataset": cfg.dataset,
            "variable": "velocity_gradient",
            "axis_order": ["velocity_component", "derivative_component", "z", "y", "x"],
            "velocity_components": ["ux", "uy", "uz"],
            "derivative_components": ["x", "y", "z"],
            "domain": "[0,2pi)^3",
            "periodic": [True, True, True],
            "dtype": "float32",
        }
    )
    group = root.require_group(f"t{time_index:06d}")
    group.attrs.update(
        {
            "time_index": time_index,
            "physical_time": cfg.physical_time(time_index),
        }
    )
    if "input_manifest_hash" not in group.attrs:
        group.attrs["input_manifest_hash"] = input_manifest_hash
    if "status" not in group.attrs:
        group.attrs["status"] = "computing"
    gx, gy, gz = cfg.grid_shape
    tx, ty, tz = cfg.tile_shape
    compressor = Blosc(
        cname="zstd", clevel=cfg.compression_level, shuffle=Blosc.BITSHUFFLE
    )
    array = group.require_dataset(
        "gradient",
        shape=(3, 3, gz, gy, gx),
        chunks=(1, 1, tz, ty, tx),
        dtype="<f4",
        fill_value=np.nan,
        compressor=compressor,
        overwrite=False,
    )
    return group, array


def _filtered_group(cfg: PipelineConfig, time_index: int):
    cfg.filtered_store_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(cfg.filtered_store_path), mode="a")
    root.attrs.update(
        {
            "dataset": cfg.dataset,
            "filter_method": FILTER_METHOD,
            "domain": "[0,2pi)^3",
            "periodic": [True, True, True],
            "dtype": "float32",
        }
    )
    group = root.require_group(f"t{time_index:06d}")
    gx, gy, gz = cfg.grid_shape
    tx, ty, tz = cfg.tile_shape
    compressor = Blosc(
        cname="zstd", clevel=cfg.compression_level, shuffle=Blosc.BITSHUFFLE
    )
    velocity_bar = group.require_dataset(
        "velocity_bar",
        shape=(3, gz, gy, gx),
        chunks=(1, tz, ty, tx),
        dtype="<f4",
        fill_value=np.nan,
        compressor=compressor,
        overwrite=False,
    )
    gradient_bar = group.require_dataset(
        "gradient_bar",
        shape=(3, 3, gz, gy, gx),
        chunks=(1, 1, tz, ty, tx),
        dtype="<f4",
        fill_value=np.nan,
        compressor=compressor,
        overwrite=False,
    )
    return group, velocity_bar, gradient_bar


def _copy_scalar_field(source: Any, destination: Any, tile_shape_xyz: tuple[int, int, int]) -> None:
    tx, ty, tz = tile_shape_xyz
    nz, ny, nx = destination.shape
    for z0 in range(0, nz, tz):
        for y0 in range(0, ny, ty):
            for x0 in range(0, nx, tx):
                key = (
                    slice(z0, min(z0 + tz, nz)),
                    slice(y0, min(y0 + ty, ny)),
                    slice(x0, min(x0 + tx, nx)),
                )
                destination[key] = np.asarray(source[key], dtype=np.float32)


def prepare_filtered_fields(
    cfg: PipelineConfig,
    time_index: int,
    raw: Any,
    velocity_manifest_hash: str,
    gradient_manifest_hash: str,
) -> Path:
    """Persist filtered velocity and its spectral gradient with resumable hashes."""
    group, velocity_bar, gradient_bar = _filtered_group(cfg, time_index)
    identity = {
        "velocity_manifest_hash": velocity_manifest_hash,
        "gradient_manifest_hash": gradient_manifest_hash,
        "filter_method": FILTER_METHOD,
        "sigma_grid": cfg.sigma_grid,
    }
    identity_matches = all(group.attrs.get(key) == value for key, value in identity.items())
    field_hashes = dict(group.attrs.get("field_hashes", {})) if identity_matches else {}
    required_keys = {
        *(f"velocity_bar_{component}" for component in range(3)),
        *(
            f"gradient_bar_{velocity_component}_{derivative_component}"
            for velocity_component in range(3)
            for derivative_component in range(3)
        ),
    }
    if (
        identity_matches
        and group.attrs.get("status") == "complete"
        and group.attrs.get("manifest_hash")
        and required_keys.issubset(field_hashes)
    ):
        Console().print(
            "[green]Filtered velocity and all 9 filtered gradients already match; nothing to do.[/green]"
        )
        return cfg.filtered_store_path

    group.attrs.update(
        {
            **identity,
            "physical_time": cfg.physical_time(time_index),
            "status": "computing",
            "field_hashes": field_hashes,
        }
    )
    if not identity_matches and "manifest_hash" in group.attrs:
        del group.attrs["manifest_hash"]

    points = int(np.prod(cfg.grid_shape, dtype=np.int64))
    scalar_bytes = points * np.dtype(np.float32).itemsize
    missing_count = len(required_keys.difference(field_hashes))
    plan = filtered_space_plan(cfg)
    required_incremental_gib = (
        missing_count * scalar_bytes
        + int(plan["scratch_GiB"] * 1024**3)
        + int(cfg.safety_free_space_gib * 1024**3)
    ) / 1024**3
    free_gib = shutil.disk_usage(cfg.storage_root.anchor).free / 1024**3
    if free_gib < required_incremental_gib:
        raise RuntimeError(
            f"insufficient free space for filtered fields: {free_gib:.2f} GiB free, "
            f"need {required_incremental_gib:.2f} GiB for {missing_count} missing fields "
            "including scratch and safety reserve"
        )

    console = Console()
    console.print(
        "[bold]Managed spectral preprocessing[/bold] "
        f"missing={missing_count}/12 "
        f"output<={plan['filtered_total_uncompressed_GiB']:.2f} GiB "
        f"scratch={plan['scratch_GiB']:.2f} GiB"
    )
    scratch = cfg.scratch_path / f"filtered-t{time_index:06d}"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    scalar_shape = tuple(raw.shape[1:])
    result = _memmap(scratch / "filtered.f32", scalar_shape)
    temp_a = _memmap(scratch / "temp_a.f32", scalar_shape)
    temp_b = _memmap(scratch / "temp_b.f32", scalar_shape)

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        "{task.completed}/{task.total}",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("filtered velocity and gradients", total=12)
        for velocity_component in range(3):
            velocity_key = f"velocity_bar_{velocity_component}"
            gradient_keys = [
                f"gradient_bar_{velocity_component}_{derivative_component}"
                for derivative_component in range(3)
            ]
            missing_gradients = [
                derivative_component
                for derivative_component, key in enumerate(gradient_keys)
                if key not in field_hashes
            ]
            result_ready = False
            if velocity_key not in field_hashes:
                progress.update(task, description=f"filter u{COMPONENT_NAMES[velocity_component]}")
                _filter_component(
                    _ComponentView(raw, velocity_component), result, temp_a, temp_b, cfg
                )
                result.flush()
                temp_a.flush()
                temp_b.flush()
                digest, _, _ = _write_verified_scalar(
                    result,
                    _ComponentView(velocity_bar, velocity_component),
                    cfg.tile_shape,
                )
                field_hashes[velocity_key] = digest
                group.attrs["field_hashes"] = field_hashes
                result_ready = True
            progress.advance(task)

            if missing_gradients and not result_ready:
                progress.update(
                    task,
                    description=f"load filtered u{COMPONENT_NAMES[velocity_component]}",
                )
                _copy_scalar_field(
                    _ComponentView(velocity_bar, velocity_component),
                    result,
                    cfg.tile_shape,
                )
                result.flush()

            for derivative_component, gradient_key in enumerate(gradient_keys):
                label = (
                    f"d(filtered u{COMPONENT_NAMES[velocity_component]})"
                    f"/d{COMPONENT_NAMES[derivative_component]}"
                )
                if gradient_key not in field_hashes:
                    progress.update(task, description=label)
                    _transform_axis(
                        result,
                        temp_a,
                        ARRAY_AXIS_FOR_DERIVATIVE[derivative_component],
                        cfg.fft_slab_width,
                        derivative_domain_length=cfg.domain_length,
                    )
                    temp_a.flush()
                    digest, _, _ = _write_verified_scalar(
                        temp_a,
                        _GradientView(
                            gradient_bar,
                            velocity_component,
                            derivative_component,
                        ),
                        cfg.tile_shape,
                    )
                    field_hashes[gradient_key] = digest
                    group.attrs["field_hashes"] = field_hashes
                progress.advance(task)

    fields = [
        {
            "name": key,
            "sha256": field_hashes[key],
            "byte_count": scalar_bytes,
        }
        for key in sorted(required_keys)
    ]
    manifest = {
        "schema_version": 1,
        "dataset": cfg.dataset,
        "time_index": time_index,
        "physical_time": cfg.physical_time(time_index),
        **identity,
        "identity": "gradient_bar = spectral_gradient(velocity_bar)",
        "velocity_bar_shape": [3, *scalar_shape],
        "gradient_bar_shape": [3, 3, *scalar_shape],
        "axis_order": ["velocity_component", "derivative_component", "z", "y", "x"],
        "dtype": "float32",
        "fields": fields,
    }
    manifest_hash = _atomic_json(
        cfg.manifest_path / f"filtered_t{time_index:06d}.json", manifest
    )
    group.attrs.update(
        {
            **identity,
            "status": "complete",
            "field_hashes": field_hashes,
            "manifest_hash": manifest_hash,
        }
    )
    for mapped in (result, temp_a, temp_b):
        _close_memmap(mapped)
    del result, temp_a, temp_b
    if not cfg.keep_intermediates:
        shutil.rmtree(scratch)
    return cfg.filtered_store_path


def _write_verified_field(
    source: Any,
    destination: Any,
    velocity_component: int,
    derivative_component: int,
    tile_shape_xyz: tuple[int, int, int],
) -> tuple[str, int]:
    source_hasher = hashlib.sha256()
    readback_hasher = hashlib.sha256()
    byte_count = 0
    tx, ty, tz = tile_shape_xyz
    nz, ny, nx = source.shape
    for z0 in range(0, nz, tz):
        for y0 in range(0, ny, ty):
            for x0 in range(0, nx, tx):
                key = (
                    slice(z0, min(z0 + tz, nz)),
                    slice(y0, min(y0 + ty, ny)),
                    slice(x0, min(x0 + tx, nx)),
                )
                values = np.ascontiguousarray(source[key], dtype="<f4")
                if not np.all(np.isfinite(values)):
                    raise ValueError("computed gradient contains NaN or Inf")
                destination[(velocity_component, derivative_component) + key] = values
                readback = np.ascontiguousarray(
                    destination[(velocity_component, derivative_component) + key],
                    dtype="<f4",
                )
                source_hasher.update(values.view(np.uint8))
                readback_hasher.update(readback.view(np.uint8))
                byte_count += values.nbytes
    digest = source_hasher.hexdigest()
    if readback_hasher.hexdigest() != digest:
        raise IOError("gradient failed write/read SHA-256 verification")
    return digest, byte_count


def finite_difference_core(
    block: np.ndarray,
    spacing: float,
    derivative_axis: int,
    core_size: int,
    halo: int = 4,
) -> np.ndarray:
    """Eighth-order central derivative on a haloed scalar (z,y,x) block."""
    expected = (core_size + 2 * halo,) * 3
    if block.shape != expected:
        raise ValueError(f"haloed block shape {block.shape}, expected {expected}")
    if derivative_axis not in (0, 1, 2):
        raise ValueError(f"invalid derivative axis {derivative_axis}")
    if halo < 4:
        raise ValueError("eighth-order finite difference requires halo >= 4")
    result = np.zeros((core_size, core_size, core_size), dtype=np.float64)
    base = [slice(halo, halo + core_size)] * 3
    values = np.asarray(block, dtype=np.float64)
    for offset, weight in FD8_WEIGHTS.items():
        key = list(base)
        key[derivative_axis] = slice(
            halo + offset, halo + offset + core_size
        )
        result += weight * values[tuple(key)]
    return result / spacing


def _rms_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference64 = np.asarray(reference, dtype=np.float64)
    candidate64 = np.asarray(candidate, dtype=np.float64)
    difference = candidate64 - reference64
    reference_rms = float(np.sqrt(np.mean(np.square(reference64))))
    candidate_rms = float(np.sqrt(np.mean(np.square(candidate64))))
    difference_rms = float(np.sqrt(np.mean(np.square(difference))))
    denominator = float(
        np.sqrt(np.sum(np.square(reference64)) * np.sum(np.square(candidate64)))
    )
    correlation = (
        float(np.sum(reference64 * candidate64) / denominator)
        if denominator > 0.0
        else 1.0
    )
    return {
        "fft_rms": reference_rms,
        "fd8_rms": candidate_rms,
        "difference_rms": difference_rms,
        "relative_difference_rms": difference_rms / max(reference_rms, 1e-30),
        "cosine_similarity": correlation,
    }


def audit_gradients(
    cfg: PipelineConfig,
    time_index: int,
    core_size: int = 32,
    origin_xyz: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Compare managed FFT gradients with local FD8 on one small real-data block."""
    halo = 4
    gx, gy, gz = cfg.grid_shape
    if core_size < 1:
        raise ValueError("core_size must be positive")
    if origin_xyz is None:
        origin_xyz = tuple((n - core_size) // 2 for n in (gx, gy, gz))
    x0, y0, z0 = (int(value) for value in origin_xyz)
    for name, origin, size in zip("xyz", (x0, y0, z0), (gx, gy, gz)):
        if origin < halo or origin + core_size + halo > size:
            raise ValueError(
                f"{name} origin must leave a {halo}-point halo inside the domain"
            )

    with Catalog(cfg.catalog_path) as catalog:
        snapshot = catalog.snapshot(cfg.dataset, time_index)
        if snapshot is None or snapshot["status"] != "auto_validated":
            raise RuntimeError("gradient audit requires an auto_validated velocity snapshot")
        velocity_manifest_hash = str(snapshot["manifest_hash"])

    raw_root = zarr.open_group(str(cfg.raw_store_path), mode="r")
    raw = raw_root[f"t{time_index:06d}"]["velocity"]
    gradient_root = zarr.open_group(str(cfg.gradient_store_path), mode="r")
    gradient_group = gradient_root[f"t{time_index:06d}"]
    if gradient_group.attrs.get("status") != "complete":
        raise RuntimeError("gradient audit requires all 9 managed gradients to be complete")
    if gradient_group.attrs.get("input_manifest_hash") != velocity_manifest_hash:
        raise RuntimeError("gradient and velocity manifests do not match")
    gradient = gradient_group["gradient"]

    raw_block = np.asarray(
        raw[
            :,
            z0 - halo : z0 + core_size + halo,
            y0 - halo : y0 + core_size + halo,
            x0 - halo : x0 + core_size + halo,
        ],
        dtype=np.float32,
    )
    fft_block = np.asarray(
        gradient[
            :,
            :,
            z0 : z0 + core_size,
            y0 : y0 + core_size,
            x0 : x0 + core_size,
        ],
        dtype=np.float32,
    )
    spacing = cfg.domain_length / gx
    comparisons: list[dict[str, Any]] = []
    fft_divergence = np.zeros((core_size,) * 3, dtype=np.float64)
    fd8_divergence = np.zeros_like(fft_divergence)
    total_difference_sumsq = 0.0
    total_reference_sumsq = 0.0
    total_count = 0
    for velocity_component in range(3):
        for derivative_component in range(3):
            fd8 = finite_difference_core(
                raw_block[velocity_component],
                spacing,
                ARRAY_AXIS_FOR_DERIVATIVE[derivative_component],
                core_size,
                halo,
            )
            fft_values = np.asarray(
                fft_block[velocity_component, derivative_component], dtype=np.float64
            )
            metrics = _rms_metrics(fft_values, fd8)
            comparisons.append(
                {
                    "velocity_component": COMPONENT_NAMES[velocity_component],
                    "derivative_component": COMPONENT_NAMES[derivative_component],
                    "name": (
                        f"du{COMPONENT_NAMES[velocity_component]}"
                        f"/d{COMPONENT_NAMES[derivative_component]}"
                    ),
                    **metrics,
                }
            )
            difference = fd8 - fft_values
            total_difference_sumsq += float(np.square(difference).sum())
            total_reference_sumsq += float(np.square(fft_values).sum())
            total_count += difference.size
            if velocity_component == derivative_component:
                fft_divergence += fft_values
                fd8_divergence += fd8

    aggregate_difference_rms = float(np.sqrt(total_difference_sumsq / total_count))
    aggregate_fft_rms = float(np.sqrt(total_reference_sumsq / total_count))
    aggregate_correlation = float(
        np.mean([item["cosine_similarity"] for item in comparisons])
    )
    report = {
        "dataset": cfg.dataset,
        "time_index": time_index,
        "physical_time": cfg.physical_time(time_index),
        "velocity_manifest_hash": velocity_manifest_hash,
        "gradient_manifest_hash": gradient_group.attrs.get("manifest_hash"),
        "core_origin_xyz_0based": [x0, y0, z0],
        "core_shape_xyz": [core_size, core_size, core_size],
        "halo": halo,
        "spacing": spacing,
        "finite_difference": "eighth_order_centered",
        "fft_axis_order_zyx_for_derivative_xyz": list(ARRAY_AXIS_FOR_DERIVATIVE),
        "comparisons": comparisons,
        "aggregate": {
            "fft_rms": aggregate_fft_rms,
            "difference_rms": aggregate_difference_rms,
            "relative_difference_rms": aggregate_difference_rms
            / max(aggregate_fft_rms, 1e-30),
            "mean_cosine_similarity": aggregate_correlation,
            "fft_divergence_rms": float(
                np.sqrt(np.mean(np.square(fft_divergence)))
            ),
            "fd8_divergence_rms": float(
                np.sqrt(np.mean(np.square(fd8_divergence)))
            ),
        },
        "estimated_peak_array_MiB": (
            raw_block.nbytes
            + fft_block.nbytes
            + fft_divergence.nbytes
            + fd8_divergence.nbytes
            + 3 * core_size**3 * np.dtype(np.float64).itemsize
        )
        / 1024**2,
        "interpretation": (
            "FD8 and spectral derivatives are different numerical operators; "
            "the RMS difference is an audit metric, not expected to be zero. "
            "High cosine similarity across all nine fields is the primary axis/component check."
        ),
    }
    _atomic_json(cfg.qa_path / f"gradient_audit_t{time_index:06d}.json", report)
    return report


def validate_divergence(
    cfg: PipelineConfig,
    time_index: int,
) -> dict[str, Any]:
    """Validate full-domain incompressibility from the managed gradients."""
    with Catalog(cfg.catalog_path) as catalog:
        snapshot = catalog.snapshot(cfg.dataset, time_index)
        if (
            snapshot is None
            or snapshot["status"] != "auto_validated"
            or not snapshot["manifest_hash"]
        ):
            raise RuntimeError(
                "divergence validation requires an auto_validated velocity snapshot"
            )
        velocity_manifest_hash = str(snapshot["manifest_hash"])

    if not cfg.gradient_store_path.exists():
        raise RuntimeError("divergence validation requires managed gradients")
    gradient_root = zarr.open_group(str(cfg.gradient_store_path), mode="r")
    try:
        gradient_group = gradient_root[f"t{time_index:06d}"]
    except KeyError as exc:
        raise RuntimeError("divergence validation requires managed gradients") from exc
    if gradient_group.attrs.get("status") != "complete":
        raise RuntimeError(
            "divergence validation requires all 9 managed gradients to be complete"
        )
    if gradient_group.attrs.get("input_manifest_hash") != velocity_manifest_hash:
        raise RuntimeError("gradient and velocity manifests do not match")
    gradient_manifest_hash = str(gradient_group.attrs.get("manifest_hash", ""))
    if not gradient_manifest_hash:
        raise RuntimeError("managed gradient manifest hash is missing")

    gradient = gradient_group["gradient"]
    gx, gy, gz = cfg.grid_shape
    expected = (3, 3, gz, gy, gx)
    if gradient.shape != expected or np.dtype(gradient.dtype) != np.dtype("<f4"):
        raise ValueError(
            f"gradient schema mismatch: shape={gradient.shape}, dtype={gradient.dtype}"
        )

    tx, ty, tz = cfg.tile_shape
    chunk_count = (
        ((gx + tx - 1) // tx)
        * ((gy + ty - 1) // ty)
        * ((gz + tz - 1) // tz)
    )
    total_count = 0
    divergence_sum = 0.0
    divergence_abs_sum = 0.0
    divergence_sumsq = 0.0
    diagonal_vector_sumsq = 0.0
    maximum_abs_divergence = 0.0
    maximum_diagonal_gradient_vector = 0.0
    maximum_location_xyz = [0, 0, 0]
    console = Console()
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        "{task.completed}/{task.total}",
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("full-domain divergence", total=chunk_count)
        for z0 in range(0, gz, tz):
            for y0 in range(0, gy, ty):
                for x0 in range(0, gx, tx):
                    key = (
                        slice(z0, min(z0 + tz, gz)),
                        slice(y0, min(y0 + ty, gy)),
                        slice(x0, min(x0 + tx, gx)),
                    )
                    divergence: np.ndarray | None = None
                    diagonal_norm_squared: np.ndarray | None = None
                    for component in range(3):
                        diagonal = np.asarray(
                            gradient[(component, component) + key], dtype=np.float64
                        )
                        if not np.all(np.isfinite(diagonal)):
                            raise ValueError(
                                "managed gradient contains NaN or Inf in a diagonal field"
                            )
                        diagonal_vector_sumsq += float(np.square(diagonal).sum())
                        if diagonal_norm_squared is None:
                            diagonal_norm_squared = np.square(diagonal)
                        else:
                            diagonal_norm_squared += np.square(diagonal)
                        if divergence is None:
                            divergence = diagonal
                        else:
                            divergence += diagonal
                    if divergence is None:  # pragma: no cover - fixed three components
                        raise RuntimeError("no diagonal gradient components were read")
                    if diagonal_norm_squared is None:  # pragma: no cover
                        raise RuntimeError("no diagonal gradient magnitudes were accumulated")
                    maximum_diagonal_gradient_vector = max(
                        maximum_diagonal_gradient_vector,
                        float(np.sqrt(diagonal_norm_squared.max())),
                    )
                    absolute = np.abs(divergence)
                    local_flat_index = int(np.argmax(absolute))
                    local_maximum = float(absolute.ravel()[local_flat_index])
                    if local_maximum > maximum_abs_divergence:
                        local_z, local_y, local_x = np.unravel_index(
                            local_flat_index, divergence.shape
                        )
                        maximum_abs_divergence = local_maximum
                        maximum_location_xyz = [
                            x0 + int(local_x),
                            y0 + int(local_y),
                            z0 + int(local_z),
                        ]
                    total_count += divergence.size
                    divergence_sum += float(divergence.sum())
                    divergence_abs_sum += float(absolute.sum())
                    divergence_sumsq += float(np.square(divergence).sum())
                    progress.advance(task)

    divergence_rms = float(np.sqrt(divergence_sumsq / total_count))
    diagonal_gradient_vector_rms = float(
        np.sqrt(diagonal_vector_sumsq / total_count)
    )
    relative_divergence_rms = divergence_rms / max(
        diagonal_gradient_vector_rms, 1.0e-30
    )
    relative_maximum_divergence = maximum_abs_divergence / max(
        maximum_diagonal_gradient_vector, 1.0e-30
    )
    passed = (
        relative_divergence_rms <= cfg.divergence_relative_rms_max
        and relative_maximum_divergence <= cfg.divergence_relative_max_max
    )
    report = {
        "dataset": cfg.dataset,
        "time_index": time_index,
        "physical_time": cfg.physical_time(time_index),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "velocity_manifest_hash": velocity_manifest_hash,
        "gradient_manifest_hash": gradient_manifest_hash,
        "method": "full_domain_streaming_from_managed_gradients",
        "definition": "divergence = dux/dx + duy/dy + duz/dz",
        "shape_xyz": [gx, gy, gz],
        "chunk_shape_xyz": [tx, ty, tz],
        "chunk_count": chunk_count,
        "point_count": total_count,
        "divergence_mean": divergence_sum / total_count,
        "divergence_mean_absolute": divergence_abs_sum / total_count,
        "divergence_rms": divergence_rms,
        "diagonal_gradient_vector_rms": diagonal_gradient_vector_rms,
        "relative_divergence_rms": relative_divergence_rms,
        "relative_divergence_rms_max": cfg.divergence_relative_rms_max,
        "maximum_absolute_divergence": maximum_abs_divergence,
        "maximum_diagonal_gradient_vector": maximum_diagonal_gradient_vector,
        "relative_maximum_divergence": relative_maximum_divergence,
        "relative_maximum_divergence_max": cfg.divergence_relative_max_max,
        "maximum_location_xyz_0based": maximum_location_xyz,
        "estimated_peak_array_MiB": (
            tx * ty * tz * (3 * np.dtype(np.float64).itemsize + np.dtype(np.float32).itemsize)
            / 1024**2
        ),
        "interpretation": (
            "A passing result means the full-domain spectral-gradient divergence RMS "
            "and maximum divergence are below their configured relative tolerances."
        ),
    }
    _atomic_json(cfg.qa_path / f"divergence_t{time_index:06d}.json", report)
    return report


def _ensure_divergence_validation(
    cfg: PipelineConfig,
    time_index: int,
    velocity_manifest_hash: str,
    gradient_manifest_hash: str,
) -> dict[str, Any]:
    try:
        return require_divergence_validation(
            cfg,
            time_index,
            velocity_manifest_hash,
            gradient_manifest_hash,
        )
    except RuntimeError:
        report = validate_divergence(cfg, time_index)
        if not report["passed"]:
            raise RuntimeError(
                "full-domain divergence validation failed; filtered preprocessing was not started"
            )
        return report


def compute_gradients(cfg: PipelineConfig, time_index: int) -> Path:
    console = Console()
    with Catalog(cfg.catalog_path) as catalog:
        snapshot = catalog.snapshot(cfg.dataset, time_index)
        if snapshot is None or snapshot["status"] != "auto_validated" or not snapshot["manifest_hash"]:
            raise RuntimeError(
                "gradient requires an auto_validated raw snapshot with a manifest hash"
            )
        input_manifest_hash = str(snapshot["manifest_hash"])

    raw_root = zarr.open_group(str(cfg.raw_store_path), mode="r")
    raw = raw_root[f"t{time_index:06d}"]["velocity"]
    expected = (3, cfg.grid_shape[2], cfg.grid_shape[1], cfg.grid_shape[0])
    if raw.shape != expected or np.dtype(raw.dtype) != np.dtype("<f4"):
        raise ValueError(f"raw velocity schema mismatch: shape={raw.shape}, dtype={raw.dtype}")

    lock = FileLock(str(cfg.storage_root / "gradient.lock"), timeout=0)
    scratch = cfg.scratch_path / f"gradient-t{time_index:06d}"
    with lock:
        group, gradient = _gradient_group(cfg, time_index, input_manifest_hash)
        stored_input_manifest_hash = group.attrs.get("input_manifest_hash")
        same_input = stored_input_manifest_hash == input_manifest_hash
        if not same_input:
            group.attrs.update(
                {
                    "input_manifest_hash": input_manifest_hash,
                    "status": "computing",
                }
            )
            if "manifest_hash" in group.attrs:
                del group.attrs["manifest_hash"]

        with Catalog(cfg.catalog_path) as catalog:
            catalog.plan_gradient_fields(
                cfg.dataset,
                time_index,
                input_manifest_hash,
                adopt_unbound_verified=same_input,
            )
            catalog_complete = True
            for velocity_component in range(3):
                for derivative_component in range(3):
                    row = catalog.gradient_field(
                        cfg.dataset,
                        time_index,
                        velocity_component,
                        derivative_component,
                    )
                    catalog_complete = catalog_complete and bool(
                        row is not None
                        and row["input_manifest_hash"] == input_manifest_hash
                        and row["status"] == "verified"
                        and row["sha256"]
                    )
        if (
            same_input
            and catalog_complete
            and group.attrs.get("status") == "complete"
            and group.attrs.get("manifest_hash")
        ):
            console.print(
                "[green]All 9 managed gradients already match the velocity manifest.[/green]"
            )
            gradient_manifest_hash = str(group.attrs["manifest_hash"])
            _ensure_divergence_validation(
                cfg,
                time_index,
                input_manifest_hash,
                gradient_manifest_hash,
            )
            prepare_filtered_fields(
                cfg,
                time_index,
                raw,
                input_manifest_hash,
                gradient_manifest_hash,
            )
            return cfg.gradient_store_path

        group.attrs["status"] = "computing"
        plan = gradient_space_plan(cfg)
        free_gib = shutil.disk_usage(cfg.storage_root.anchor).free / 1024**3
        if free_gib < plan["required_free_GiB"]:
            raise RuntimeError(
                f"insufficient free space for gradients: {free_gib:.2f} GiB free, "
                f"need {plan['required_free_GiB']:.2f} GiB including safety reserve"
            )
        console.print(
            "[bold]Managed periodic gradients[/bold] "
            f"output<={plan['gradient_uncompressed_GiB']:.2f} GiB "
            f"scratch={plan['scratch_GiB']:.2f} GiB "
            f"FFT input block={plan['fft_input_block_MiB']:.2f} MiB"
        )
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        derivative = _memmap(scratch / "derivative.f32", expected[1:])

        with Catalog(cfg.catalog_path) as catalog:
            catalog.plan_gradient_fields(
                cfg.dataset,
                time_index,
                input_manifest_hash,
                adopt_unbound_verified=same_input,
            )
            with Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                "{task.completed}/{task.total}",
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("periodic velocity gradients", total=9)
                for velocity_component in range(3):
                    for derivative_component in range(3):
                        row = catalog.gradient_field(
                            cfg.dataset,
                            time_index,
                            velocity_component,
                            derivative_component,
                        )
                        label = (
                            f"du{COMPONENT_NAMES[velocity_component]}"
                            f"/d{COMPONENT_NAMES[derivative_component]}"
                        )
                        if row is not None and row["status"] == "verified" and row["sha256"]:
                            progress.update(task, advance=1, description=f"skip {label}")
                            continue
                        catalog.mark_gradient_attempt(
                            cfg.dataset,
                            time_index,
                            velocity_component,
                            derivative_component,
                        )
                        progress.update(task, description=label)
                        _transform_axis(
                            _ComponentView(raw, velocity_component),
                            derivative,
                            ARRAY_AXIS_FOR_DERIVATIVE[derivative_component],
                            cfg.fft_slab_width,
                            derivative_domain_length=cfg.domain_length,
                        )
                        derivative.flush()
                        digest, byte_count = _write_verified_field(
                            derivative,
                            gradient,
                            velocity_component,
                            derivative_component,
                            cfg.tile_shape,
                        )
                        catalog.mark_gradient_verified(
                            cfg.dataset,
                            time_index,
                            velocity_component,
                            derivative_component,
                            digest,
                            byte_count,
                        )
                        progress.advance(task)

            fields = []
            for velocity_component in range(3):
                for derivative_component in range(3):
                    row = catalog.gradient_field(
                        cfg.dataset,
                        time_index,
                        velocity_component,
                        derivative_component,
                    )
                    if (
                        row is None
                        or row["input_manifest_hash"] != input_manifest_hash
                        or row["status"] != "verified"
                        or not row["sha256"]
                    ):
                        raise RuntimeError("gradient catalog is incomplete after computation")
                    fields.append(
                        {
                            "velocity_component": velocity_component,
                            "derivative_component": derivative_component,
                            "name": (
                                f"du{COMPONENT_NAMES[velocity_component]}"
                                f"/d{COMPONENT_NAMES[derivative_component]}"
                            ),
                            "sha256": row["sha256"],
                            "byte_count": row["byte_count"],
                        }
                    )

        manifest = {
            "schema_version": 1,
            "dataset": cfg.dataset,
            "time_index": time_index,
            "physical_time": cfg.physical_time(time_index),
            "input_velocity_manifest_hash": input_manifest_hash,
            "method": "periodic_spectral_1d_lines",
            "shape": [3, 3, *expected[1:]],
            "axis_order": ["velocity_component", "derivative_component", "z", "y", "x"],
            "derivative_components": ["x", "y", "z"],
            "dtype": "float32",
            "fields": fields,
        }
        manifest_hash = _atomic_json(
            cfg.manifest_path / f"gradient_t{time_index:06d}.json", manifest
        )
        group.attrs.update(
            {
                "status": "complete",
                "manifest_hash": manifest_hash,
                "input_manifest_hash": input_manifest_hash,
            }
        )
        _close_memmap(derivative)
        del derivative
        if not cfg.keep_intermediates:
            shutil.rmtree(scratch)
        _ensure_divergence_validation(
            cfg,
            time_index,
            input_manifest_hash,
            manifest_hash,
        )
        prepare_filtered_fields(
            cfg,
            time_index,
            raw,
            input_manifest_hash,
            manifest_hash,
        )
        return cfg.gradient_store_path
