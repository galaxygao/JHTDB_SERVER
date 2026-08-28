from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy import fft


ARRAY_AXIS_FOR_DERIVATIVE = (2, 1, 0)  # derivative labels x,y,z for arrays z,y,x


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
    if sigma_grid <= 0:
        raise ValueError("sigma_grid must be positive")
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
    resolved_rms = float(
        np.sqrt(np.mean(np.square(work_resolved, dtype=np.float64)))
    )
    epsilon_full = max(epsilon_abs, epsilon_rel * full_rms)
    epsilon_resolved = max(epsilon_abs, epsilon_rel * resolved_rms)
    codes = np.zeros(work_full.shape, dtype=np.uint8)
    full_pos, full_neg = work_full > epsilon_full, work_full < -epsilon_full
    res_pos = work_resolved > epsilon_resolved
    res_neg = work_resolved < -epsilon_resolved
    codes[full_pos & res_pos] = 1
    codes[full_pos & res_neg] = 2
    codes[full_neg & res_pos] = 3
    codes[full_neg & res_neg] = 4
    return codes, epsilon_full, epsilon_resolved


class ComponentView:
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


class ProductView:
    """Read-only slab view of the pointwise product of two full-domain fields."""

    def __init__(self, left: Any, right: Any):
        if tuple(left.shape) != tuple(right.shape):
            raise ValueError("product fields must have identical shapes")
        self.left = left
        self.right = right
        self.shape = tuple(left.shape)

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.asarray(self.left[key], dtype=np.float32) * np.asarray(
            self.right[key], dtype=np.float32
        )


def axis_batches(
    shape: tuple[int, int, int], axis: int, slab: int
) -> Iterator[tuple[slice, slice, slice]]:
    if axis in (1, 2):
        for start in range(0, shape[0], slab):
            yield slice(start, min(start + slab, shape[0])), slice(None), slice(None)
    elif axis == 0:
        for start in range(0, shape[1], slab):
            yield slice(None), slice(start, min(start + slab, shape[1])), slice(None)
    else:
        raise ValueError(f"invalid 3-D axis {axis}")


def transform_axis(
    source: Any,
    destination: Any,
    axis: int,
    slab: int,
    *,
    workers: int = 1,
    derivative_domain_length: float | None = None,
    gaussian_sigma_grid: float | None = None,
) -> None:
    if (derivative_domain_length is None) == (gaussian_sigma_grid is None):
        raise ValueError("select exactly one spectral operation")
    n = source.shape[axis]
    if derivative_domain_length is not None:
        multiplier = 1j * 2.0 * np.pi * fft.rfftfreq(
            n, d=derivative_domain_length / n
        )
    else:
        sigma = float(gaussian_sigma_grid)
        if sigma <= 0:
            raise ValueError("gaussian_sigma_grid must be positive")
        theta = 2.0 * np.pi * fft.rfftfreq(n, d=1.0)
        multiplier = np.exp(-0.5 * np.square(sigma * theta)).astype(np.float32)
    multiplier_shape = [1, 1, 1]
    multiplier_shape[axis] = len(multiplier)
    shaped_multiplier = multiplier.reshape(multiplier_shape)
    for key in axis_batches(tuple(source.shape), axis, slab):
        block = np.asarray(source[key], dtype=np.float32)
        spectrum = fft.rfft(block, axis=axis, workers=workers)
        spectrum *= shaped_multiplier
        destination[key] = fft.irfft(
            spectrum, n=n, axis=axis, workers=workers
        ).astype(np.float32)


def derivative_field(
    source: Any,
    destination: Any,
    derivative_component: int,
    domain_length: float,
    slab: int,
    workers: int = 1,
) -> None:
    transform_axis(
        source,
        destination,
        ARRAY_AXIS_FOR_DERIVATIVE[derivative_component],
        slab,
        workers=workers,
        derivative_domain_length=domain_length,
    )


def filter_field(
    source: Any,
    destination: Any,
    temp_a: Any,
    temp_b: Any,
    sigma_grid: float,
    slab: int,
    workers: int = 1,
) -> None:
    transform_axis(
        source, temp_a, 2, slab, workers=workers, gaussian_sigma_grid=sigma_grid
    )
    transform_axis(
        temp_a, temp_b, 1, slab, workers=workers, gaussian_sigma_grid=sigma_grid
    )
    transform_axis(
        temp_b, destination, 0, slab, workers=workers, gaussian_sigma_grid=sigma_grid
    )


def zero_field(field: Any, slab: int) -> None:
    for start in range(0, field.shape[0], slab):
        field[start : min(start + slab, field.shape[0]), :, :] = 0.0


def accumulate_product(
    destination: Any,
    left: Any,
    right: Any,
    slab: int,
) -> None:
    for start in range(0, destination.shape[0], slab):
        key = (slice(start, min(start + slab, destination.shape[0])), slice(None), slice(None))
        values = np.asarray(destination[key], dtype=np.float32)
        values += np.asarray(left[key], dtype=np.float32) * np.asarray(
            right[key], dtype=np.float32
        )
        destination[key] = values


def subtract_product(
    destination: Any,
    left: Any,
    right: Any,
    slab: int,
) -> None:
    for start in range(0, destination.shape[0], slab):
        key = (slice(start, min(start + slab, destination.shape[0])), slice(None), slice(None))
        values = np.asarray(destination[key], dtype=np.float32)
        values -= np.asarray(left[key], dtype=np.float32) * np.asarray(
            right[key], dtype=np.float32
        )
        destination[key] = values


def memmap(path: Path, shape: tuple[int, ...], mode: str = "w+") -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.memmap(path, dtype=np.float32, mode=mode, shape=shape, order="C")


def close_memmap(mapped: np.memmap) -> None:
    mapped.flush()
    memory_map = getattr(mapped, "_mmap", None)
    if memory_map is not None:
        memory_map.close()
