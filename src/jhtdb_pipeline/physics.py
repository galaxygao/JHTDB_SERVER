from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from filelock import FileLock
from numcodecs import Blosc
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from scipy import fft

from .catalog import Catalog
from .config import PipelineConfig
from .validation import require_divergence_validation


FILTER_METHOD = "periodic_spectral_gaussian_from_velocity_v1"


def spectral_derivative(values: np.ndarray, axis: int, domain_length: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    n = values.shape[axis]
    wave_number = 2.0 * np.pi * fft.rfftfreq(n, d=domain_length / n)
    spectrum = fft.rfft(values, axis=axis, workers=1)
    shape = [1] * values.ndim
    shape[axis] = len(wave_number)
    spectrum *= (1j * wave_number).reshape(shape)
    return fft.irfft(spectrum, n=n, axis=axis, workers=1).astype(np.float32)


def spectral_gaussian(values: np.ndarray, sigma_grid: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    for axis in range(result.ndim):
        n = result.shape[axis]
        theta = 2.0 * np.pi * fft.rfftfreq(n, d=1.0)
        transfer = np.exp(-0.5 * np.square(sigma_grid * theta)).astype(np.float32)
        spectrum = fft.rfft(result, axis=axis, workers=1)
        shape = [1] * result.ndim
        shape[axis] = len(transfer)
        spectrum *= transfer.reshape(shape)
        result = fft.irfft(spectrum, n=n, axis=axis, workers=1).astype(np.float32)
    return result


def regime_codes(
    work_full: np.ndarray,
    work_resolved: np.ndarray,
    epsilon_abs: float,
    epsilon_rel: float,
) -> tuple[np.ndarray, float, float]:
    full_rms = float(np.sqrt(np.mean(np.square(work_full, dtype=np.float64))))
    resolved_rms = float(np.sqrt(np.mean(np.square(work_resolved, dtype=np.float64))))
    epsilon_full = max(epsilon_abs, epsilon_rel * full_rms)
    epsilon_resolved = max(epsilon_abs, epsilon_rel * resolved_rms)
    codes = np.zeros(work_full.shape, dtype=np.uint8)
    full_pos, full_neg = work_full > epsilon_full, work_full < -epsilon_full
    res_pos, res_neg = work_resolved > epsilon_resolved, work_resolved < -epsilon_resolved
    codes[full_pos & res_pos] = 1
    codes[full_pos & res_neg] = 2
    codes[full_neg & res_pos] = 3
    codes[full_neg & res_neg] = 4
    return codes, epsilon_full, epsilon_resolved


class _ComponentView:
    def __init__(self, parent: Any, component: int):
        self.parent = parent
        self.component = component
        self.shape = tuple(parent.shape[1:])

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        return self.parent[(self.component,) + key]

    def __setitem__(self, key: Any, value: np.ndarray) -> None:
        if not isinstance(key, tuple):
            key = (key,)
        self.parent[(self.component,) + key] = value


class _GradientView:
    def __init__(self, parent: Any, velocity_component: int, derivative_component: int):
        self.parent = parent
        self.velocity_component = velocity_component
        self.derivative_component = derivative_component
        self.shape = tuple(parent.shape[2:])

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key,)
        return self.parent[(self.velocity_component, self.derivative_component) + key]

    def __setitem__(self, key: Any, value: np.ndarray) -> None:
        if not isinstance(key, tuple):
            key = (key,)
        self.parent[(self.velocity_component, self.derivative_component) + key] = value


def _axis_batches(shape: tuple[int, int, int], axis: int, slab: int):
    if axis in (1, 2):
        for start in range(0, shape[0], slab):
            yield (slice(start, min(start + slab, shape[0])), slice(None), slice(None))
    elif axis == 0:
        for start in range(0, shape[1], slab):
            yield (slice(None), slice(start, min(start + slab, shape[1])), slice(None))
    else:
        raise ValueError(f"invalid 3-D axis {axis}")


def _transform_axis(
    source: Any,
    destination: Any,
    axis: int,
    slab: int,
    *,
    derivative_domain_length: float | None = None,
    gaussian_sigma_grid: float | None = None,
) -> None:
    if (derivative_domain_length is None) == (gaussian_sigma_grid is None):
        raise ValueError("select exactly one spectral operation")
    n = source.shape[axis]
    if derivative_domain_length is not None:
        multiplier = 1j * 2.0 * np.pi * fft.rfftfreq(n, d=derivative_domain_length / n)
    else:
        theta = 2.0 * np.pi * fft.rfftfreq(n, d=1.0)
        multiplier = np.exp(-0.5 * np.square(float(gaussian_sigma_grid) * theta)).astype(np.float32)
    shape = [1, 1, 1]
    shape[axis] = len(multiplier)
    shaped_multiplier = multiplier.reshape(shape)
    for key in _axis_batches(source.shape, axis, slab):
        block = np.asarray(source[key], dtype=np.float32)
        spectrum = fft.rfft(block, axis=axis, workers=1)
        spectrum *= shaped_multiplier
        destination[key] = fft.irfft(spectrum, n=n, axis=axis, workers=1).astype(np.float32)


def _filter_component(source: Any, destination: Any, temp_a: Any, temp_b: Any, cfg: PipelineConfig) -> None:
    _transform_axis(source, temp_a, 2, cfg.fft_slab_width, gaussian_sigma_grid=cfg.sigma_grid)
    _transform_axis(temp_a, temp_b, 1, cfg.fft_slab_width, gaussian_sigma_grid=cfg.sigma_grid)
    _transform_axis(temp_b, destination, 0, cfg.fft_slab_width, gaussian_sigma_grid=cfg.sigma_grid)


def _zero_scalar(field: Any, slab: int) -> None:
    for start in range(0, field.shape[0], slab):
        field[start:min(start + slab, field.shape[0]), :, :] = 0.0


def _acceleration_from_gradient(
    velocity: Any,
    gradient: Any,
    velocity_component: int,
    output: Any,
    slab: int,
) -> None:
    """Compute a_i = sum_j u_j * (du_i/dx_j); j is ordered x,y,z."""
    _zero_scalar(output, slab)
    for derivative_component in range(3):
        for start in range(0, output.shape[0], slab):
            zslice = slice(start, min(start + slab, output.shape[0]))
            current = np.asarray(output[zslice, :, :], dtype=np.float32)
            current += (
                np.asarray(velocity[derivative_component, zslice, :, :], dtype=np.float32)
                * np.asarray(
                    gradient[velocity_component, derivative_component, zslice, :, :],
                    dtype=np.float32,
                )
            )
            output[zslice, :, :] = current


def _accumulate_product(destination: Any, left: Any, right: Any, slab: int) -> None:
    for start in range(0, destination.shape[0], slab):
        zslice = slice(start, min(start + slab, destination.shape[0]))
        current = np.asarray(destination[zslice, :, :], dtype=np.float32)
        current += (
            np.asarray(left[zslice, :, :], dtype=np.float32)
            * np.asarray(right[zslice, :, :], dtype=np.float32)
        )
        destination[zslice, :, :] = current


def _write_verified_scalar(
    source: Any, destination: Any, tile_shape_xyz: tuple[int, int, int]
) -> tuple[str, int, float]:
    source_hasher = hashlib.sha256()
    readback_hasher = hashlib.sha256()
    byte_count = 0
    sumsq = 0.0
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
                    raise ValueError("physics field contains NaN or Inf")
                destination[key] = values
                readback = np.ascontiguousarray(destination[key], dtype="<f4")
                source_hasher.update(values.view(np.uint8))
                readback_hasher.update(readback.view(np.uint8))
                byte_count += values.nbytes
                sumsq += float(np.square(values, dtype=np.float64).sum())
    digest = source_hasher.hexdigest()
    if readback_hasher.hexdigest() != digest:
        raise IOError("physics field failed write/read SHA-256 verification")
    return digest, byte_count, sumsq


def _memmap(path: Path, shape: tuple[int, ...], mode: str = "w+") -> np.memmap:
    return np.memmap(path, dtype=np.float32, mode=mode, shape=shape, order="C")


def _close_memmap(mapped: np.memmap) -> None:
    mapped.flush()
    memory_map = getattr(mapped, "_mmap", None)
    if memory_map is not None:
        memory_map.close()


def _derived_group(cfg: PipelineConfig, time_index: int):
    cfg.derived_store_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(cfg.derived_store_path), mode="a")
    group = root.require_group(f"t{time_index:06d}")
    gx, gy, gz = cfg.grid_shape
    tx, ty, tz = cfg.tile_shape
    compressor = Blosc(cname="zstd", clevel=cfg.compression_level, shuffle=Blosc.BITSHUFFLE)
    work_full = group.require_dataset("work_full", shape=(gz, gy, gx), chunks=(tz, ty, tx), dtype="<f4", compressor=compressor)
    work_resolved = group.require_dataset("work_resolved", shape=(gz, gy, gx), chunks=(tz, ty, tx), dtype="<f4", compressor=compressor)
    regime = group.require_dataset("regime", shape=(gz, gy, gx), chunks=(tz, ty, tx), dtype="u1", compressor=compressor)
    return group, work_full, work_resolved, regime


def physics_space_plan(cfg: PipelineConfig) -> dict[str, float]:
    points = int(np.prod(cfg.grid_shape, dtype=np.int64))
    scalar_bytes = points * np.dtype(np.float32).itemsize
    # result + two separable-filter buffers + work accumulator.
    scratch_bytes = 4 * scalar_bytes
    # Two work fields plus one uint8 regime field. Filtered fields are prepared once.
    derived_bytes = 2 * scalar_bytes + points * np.dtype(np.uint8).itemsize
    reserve_bytes = int(cfg.safety_free_space_gib * 1024**3)
    return {
        "scratch_GiB": scratch_bytes / 1024**3,
        "mapped_address_space_GiB": scratch_bytes / 1024**3,
        "derived_uncompressed_GiB": derived_bytes / 1024**3,
        "safety_reserve_GiB": reserve_bytes / 1024**3,
        "required_free_GiB": (scratch_bytes + derived_bytes + reserve_bytes) / 1024**3,
        "fft_input_block_MiB": (
            cfg.fft_slab_width
            * cfg.grid_shape[0]
            * cfg.grid_shape[1]
            * np.dtype(np.float32).itemsize
            / 1024**2
        ),
        "estimated_process_peak_RAM_MiB": 384.0,
    }


def compute_snapshot(cfg: PipelineConfig, time_index: int) -> Path:
    console = Console()
    with Catalog(cfg.catalog_path) as catalog:
        snapshot = catalog.snapshot(cfg.dataset, time_index)
        if snapshot is None or snapshot["status"] != "auto_validated" or not snapshot["manifest_hash"]:
            raise RuntimeError("physics requires an auto_validated raw snapshot with a manifest hash")
        input_manifest_hash = str(snapshot["manifest_hash"])

    raw_root = zarr.open_group(str(cfg.raw_store_path), mode="r")
    raw = raw_root[f"t{time_index:06d}"]["velocity"]
    expected = (3, cfg.grid_shape[2], cfg.grid_shape[1], cfg.grid_shape[0])
    if raw.shape != expected:
        raise ValueError(f"raw shape {raw.shape}, expected {expected}")

    if not cfg.gradient_store_path.exists():
        raise RuntimeError(
            f"managed gradients are missing; run: python -m jhtdb_pipeline gradient --time-index {time_index}"
        )
    gradient_root = zarr.open_group(str(cfg.gradient_store_path), mode="r")
    try:
        gradient_group = gradient_root[f"t{time_index:06d}"]
    except KeyError as exc:
        raise RuntimeError(
            f"managed gradients are missing; run: python -m jhtdb_pipeline gradient --time-index {time_index}"
        ) from exc
    if gradient_group.attrs.get("status") != "complete":
        raise RuntimeError("managed gradients exist but are not complete")
    if gradient_group.attrs.get("input_manifest_hash") != input_manifest_hash:
        raise RuntimeError("managed gradients were built from a different raw manifest")
    gradient_manifest_hash = str(gradient_group.attrs.get("manifest_hash", ""))
    if not gradient_manifest_hash:
        raise RuntimeError("managed gradient manifest hash is missing")
    require_divergence_validation(
        cfg,
        time_index,
        input_manifest_hash,
        gradient_manifest_hash,
    )
    gradient = gradient_group["gradient"]
    expected_gradient = (3, 3, *expected[1:])
    if gradient.shape != expected_gradient or np.dtype(gradient.dtype) != np.dtype("<f4"):
        raise ValueError(
            f"gradient schema mismatch: shape={gradient.shape}, dtype={gradient.dtype}"
        )

    if not cfg.filtered_store_path.exists():
        raise RuntimeError(
            f"filtered fields are missing; run: python -m jhtdb_pipeline gradient --time-index {time_index}"
        )
    filtered_root = zarr.open_group(str(cfg.filtered_store_path), mode="r")
    try:
        filtered_group = filtered_root[f"t{time_index:06d}"]
    except KeyError as exc:
        raise RuntimeError(
            f"filtered fields are missing; run: python -m jhtdb_pipeline gradient --time-index {time_index}"
        ) from exc
    if filtered_group.attrs.get("status") != "complete":
        raise RuntimeError("filtered velocity and gradients are not complete")
    if filtered_group.attrs.get("velocity_manifest_hash") != input_manifest_hash:
        raise RuntimeError("filtered fields belong to a different velocity manifest")
    if filtered_group.attrs.get("gradient_manifest_hash") != gradient_manifest_hash:
        raise RuntimeError("filtered fields belong to a different gradient manifest")
    if filtered_group.attrs.get("filter_method") != FILTER_METHOD:
        raise RuntimeError("filtered fields use an unsupported filter algorithm")
    if float(filtered_group.attrs.get("sigma_grid", -1.0)) != cfg.sigma_grid:
        raise RuntimeError("filtered fields use a different sigma_grid")
    filtered_manifest_hash = str(filtered_group.attrs.get("manifest_hash", ""))
    if not filtered_manifest_hash:
        raise RuntimeError("filtered field manifest hash is missing")
    velocity_bar = filtered_group["velocity_bar"]
    gradient_bar = filtered_group["gradient_bar"]
    if velocity_bar.shape != expected or np.dtype(velocity_bar.dtype) != np.dtype("<f4"):
        raise ValueError("filtered velocity schema mismatch")
    if (
        gradient_bar.shape != expected_gradient
        or np.dtype(gradient_bar.dtype) != np.dtype("<f4")
    ):
        raise ValueError("filtered gradient schema mismatch")

    scratch = cfg.scratch_path / f"t{time_index:06d}"
    lock = FileLock(str(cfg.storage_root / "physics.lock"), timeout=0)
    with lock:
        group, work_full, work_resolved, regime = _derived_group(cfg, time_index)
        cache_identity = {
            "input_manifest_hash": input_manifest_hash,
            "gradient_manifest_hash": gradient_manifest_hash,
            "filtered_manifest_hash": filtered_manifest_hash,
            "physics_algorithm": "managed_filtered_fields_work_v2",
            "sigma_grid": cfg.sigma_grid,
        }
        if (
            all(group.attrs.get(key) == value for key, value in cache_identity.items())
            and group.attrs.get("status") == "complete"
            and group.attrs.get("output_hashes")
        ):
            console.print(
                "[green]Physics outputs already match all input manifests; nothing to do.[/green]"
            )
            return cfg.derived_store_path
        group.attrs.update({**cache_identity, "status": "computing"})
        if "velocity_bar" in group or "gradient_bar" in group:
            group.attrs["legacy_embedded_filtered_fields_ignored"] = True

        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        space = physics_space_plan(cfg)
        free_gib = shutil.disk_usage(cfg.storage_root.anchor).free / 1024**3
        if free_gib < space["required_free_GiB"]:
            raise RuntimeError(
                f"insufficient free space for physics: {free_gib:.2f} GiB free, "
                f"need {space['required_free_GiB']:.2f} GiB including safety reserve"
            )
        console.print(
            "[bold]Managed-gradient periodic physics[/bold] "
            f"scratch={space['scratch_GiB']:.2f} GiB "
            f"derived<={space['derived_uncompressed_GiB']:.2f} GiB "
            f"FFT input block={space['fft_input_block_MiB']:.2f} MiB"
        )

        scalar_shape = expected[1:]
        result = _memmap(scratch / "result.f32", scalar_shape)
        temp_a = _memmap(scratch / "filter_a.f32", scalar_shape)
        temp_b = _memmap(scratch / "filter_b.f32", scalar_shape)
        work_accumulator = _memmap(scratch / "work.f32", scalar_shape)

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            "{task.completed}/{task.total}",
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("managed-gradient physics", total=6)

            _zero_scalar(work_accumulator, cfg.fft_slab_width)
            for component in range(3):
                _acceleration_from_gradient(
                    raw, gradient, component, result, cfg.fft_slab_width
                )
                _filter_component(result, result, temp_a, temp_b, cfg)
                result.flush()
                temp_a.flush()
                temp_b.flush()
                _accumulate_product(
                    work_accumulator,
                    _ComponentView(velocity_bar, component),
                    result,
                    cfg.fft_slab_width,
                )
                progress.advance(task)
            work_accumulator.flush()
            work_full_hash, _, sumsq_full = _write_verified_scalar(
                work_accumulator, work_full, cfg.tile_shape
            )

            _zero_scalar(work_accumulator, cfg.fft_slab_width)
            for component in range(3):
                _acceleration_from_gradient(
                    velocity_bar,
                    gradient_bar,
                    component,
                    result,
                    cfg.fft_slab_width,
                )
                _accumulate_product(
                    work_accumulator,
                    _ComponentView(velocity_bar, component),
                    result,
                    cfg.fft_slab_width,
                )
                progress.advance(task)
            work_accumulator.flush()
            work_resolved_hash, _, sumsq_resolved = _write_verified_scalar(
                work_accumulator, work_resolved, cfg.tile_shape
            )

            count = int(np.prod(scalar_shape, dtype=np.int64))
            epsilon_full = max(cfg.epsilon_abs, cfg.epsilon_rel * np.sqrt(sumsq_full / count))
            epsilon_resolved = max(cfg.epsilon_abs, cfg.epsilon_rel * np.sqrt(sumsq_resolved / count))
            occupancy = np.zeros(5, dtype=np.int64)
            regime_hasher = hashlib.sha256()
            for start in range(0, expected[1], cfg.tile_shape[2]):
                zslice = slice(start, min(start + cfg.tile_shape[2], expected[1]))
                full = np.asarray(work_full[zslice, :, :], dtype=np.float32)
                resolved = np.asarray(work_resolved[zslice, :, :], dtype=np.float32)
                codes = np.zeros(full.shape, dtype=np.uint8)
                codes[(full > epsilon_full) & (resolved > epsilon_resolved)] = 1
                codes[(full > epsilon_full) & (resolved < -epsilon_resolved)] = 2
                codes[(full < -epsilon_full) & (resolved > epsilon_resolved)] = 3
                codes[(full < -epsilon_full) & (resolved < -epsilon_resolved)] = 4
                regime[zslice, :, :] = codes
                readback_codes = np.ascontiguousarray(regime[zslice, :, :], dtype=np.uint8)
                if not np.array_equal(codes, readback_codes):
                    raise IOError("regime failed write/read verification")
                regime_hasher.update(readback_codes)
                occupancy += np.bincount(codes.ravel(), minlength=5)

        report = {
            "dataset": cfg.dataset,
            "time_index": time_index,
            "input_manifest_hash": input_manifest_hash,
            "gradient_manifest_hash": gradient_manifest_hash,
            "filtered_manifest_hash": filtered_manifest_hash,
            "derivative": "managed_periodic_spectral_gradient",
            "filter": filtered_group.attrs.get("filter_method"),
            "sigma_grid": cfg.sigma_grid,
            "epsilon_full": float(epsilon_full),
            "epsilon_resolved": float(epsilon_resolved),
            "occupancy": {"uncertain" if i == 0 else f"Q{i}": float(value / occupancy.sum()) for i, value in enumerate(occupancy)},
            "output_hashes": {
                "work_full": work_full_hash,
                "work_resolved": work_resolved_hash,
                "regime": regime_hasher.hexdigest(),
            },
        }
        report_hash = hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()
        group.attrs.update({**report, "report_hash": report_hash, "status": "complete"})
        cfg.qa_path.mkdir(parents=True, exist_ok=True)
        (cfg.qa_path / f"physics_t{time_index:06d}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        for mapped in (result, temp_a, temp_b, work_accumulator):
            _close_memmap(mapped)
        del result, temp_a, temp_b, work_accumulator
        if not cfg.keep_intermediates:
            shutil.rmtree(scratch)
    return cfg.derived_store_path
