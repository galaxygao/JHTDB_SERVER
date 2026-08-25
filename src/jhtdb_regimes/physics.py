from __future__ import annotations

import numpy as np


def gaussian_kernel_1d(sigma_grid: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma_grid * sigma_grid))
    kernel /= kernel.sum()
    return kernel


def gaussian_valid(field: np.ndarray, sigma_grid: float, radius: int) -> np.ndarray:
    """Separable 3-D Gaussian convolution; last axes are (z,y,x), output is valid/core only."""
    if field.ndim < 3:
        raise ValueError("field needs at least three spatial dimensions")
    width = 2 * radius + 1
    if any(n < width for n in field.shape[-3:]):
        raise ValueError("halo/support is larger than the spatial block")
    kernel = gaussian_kernel_1d(sigma_grid, radius)
    result = np.asarray(field, dtype=np.float64)
    for axis in (-1, -2, -3):
        result = np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"), axis, result)
    return result


def advective_acceleration(velocity: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """a_i = u_j d_j u_i for u=(t,j,z,y,x), G=(t,i,j,z,y,x)."""
    if velocity.ndim != 5 or gradient.ndim != 6:
        raise ValueError("velocity/gradient ranks must be 5/6")
    if velocity.shape[0] != gradient.shape[0] or velocity.shape[1] != 3 or gradient.shape[1:3] != (3, 3):
        raise ValueError("velocity/gradient component dimensions do not match")
    return np.einsum("tjzyx,tijzyx->tizyx", velocity, gradient, optimize=True)


def divergence(gradient: np.ndarray) -> np.ndarray:
    return np.einsum("tiizyx->tzyx", gradient, optimize=True)


def work_and_fields(
    velocity: np.ndarray,
    gradient: np.ndarray,
    sigma_grid: float,
    radius: int,
) -> dict[str, np.ndarray]:
    acceleration = advective_acceleration(velocity, gradient)
    velocity_bar = gaussian_valid(velocity, sigma_grid, radius)
    gradient_bar = gaussian_valid(gradient, sigma_grid, radius)
    acceleration_bar = gaussian_valid(acceleration, sigma_grid, radius)
    acceleration_barbar = advective_acceleration(velocity_bar, gradient_bar)
    work_full = np.einsum("tizyx,tizyx->tzyx", velocity_bar, acceleration_bar, optimize=True)
    work_resolved = np.einsum("tizyx,tizyx->tzyx", velocity_bar, acceleration_barbar, optimize=True)
    core = (slice(None),) + (slice(radius, -radius),) * 3 if radius else (...,)
    divergence_raw = divergence(gradient)
    return {
        "velocity_bar": velocity_bar,
        "gradient_bar": gradient_bar,
        "acceleration_bar": acceleration_bar,
        "acceleration_barbar": acceleration_barbar,
        "work_full": work_full,
        "work_resolved": work_resolved,
        "divergence_raw": divergence_raw[core],
        "divergence_bar": divergence(gradient_bar),
    }


def regime_codes(
    work_full: np.ndarray,
    work_resolved: np.ndarray,
    epsilon_abs: float,
    epsilon_rel: float,
) -> tuple[np.ndarray, float, float]:
    finite = np.isfinite(work_full) & np.isfinite(work_resolved)
    rms_full = float(np.sqrt(np.mean(np.square(work_full[finite])))) if np.any(finite) else np.nan
    rms_res = float(np.sqrt(np.mean(np.square(work_resolved[finite])))) if np.any(finite) else np.nan
    eps_full = max(float(epsilon_abs), float(epsilon_rel) * rms_full)
    eps_res = max(float(epsilon_abs), float(epsilon_rel) * rms_res)
    code = np.zeros(work_full.shape, dtype=np.uint8)
    positive_full = work_full > eps_full
    negative_full = work_full < -eps_full
    positive_res = work_resolved > eps_res
    negative_res = work_resolved < -eps_res
    code[finite & positive_full & positive_res] = 1
    code[finite & positive_full & negative_res] = 2
    code[finite & negative_full & positive_res] = 3
    code[finite & negative_full & negative_res] = 4
    return code, eps_full, eps_res


def robust_regime(primary: np.ndarray, audit: np.ndarray | None) -> np.ndarray:
    if audit is None:
        return primary.copy()
    if audit.shape != primary.shape:
        raise ValueError("primary and audit regimes have different shapes")
    return np.where((primary == audit) & (primary != 0), primary, 0).astype(np.uint8)
