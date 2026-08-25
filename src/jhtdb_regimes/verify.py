from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import convolve

from .config import TaskConfig
from .grid import block_indices
from .physics import gaussian_kernel_1d
from .pipeline import validate_raw


FD_COEFFICIENTS = {
    6: np.asarray([-1 / 60, 3 / 20, -3 / 4, 0.0, 3 / 4, -3 / 20, 1 / 60]),
    8: np.asarray([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0.0, 4 / 5, -1 / 5, 4 / 105, -1 / 280]),
}


def finite_difference_core(
    velocity: np.ndarray,
    spacing: float,
    order: int,
    output_halo: int = 4,
) -> np.ndarray:
    """Independently reconstruct G_ij=d_j u_i on the common trimmed core."""
    if velocity.ndim != 5 or velocity.shape[1] != 3:
        raise ValueError("velocity must have shape (time,3,z,y,x)")
    if order not in FD_COEFFICIENTS:
        raise ValueError("only sixth- and eighth-order centered differences are supported")
    coefficients = FD_COEFFICIENTS[order]
    radius = len(coefficients) // 2
    if output_halo < radius:
        raise ValueError("output_halo is smaller than the derivative stencil radius")
    spatial_shape = velocity.shape[-3:]
    core_shape = tuple(n - 2 * output_halo for n in spatial_shape)
    if any(n <= 0 for n in core_shape):
        raise ValueError("output_halo leaves no core")
    gradient = np.zeros((velocity.shape[0], 3, 3) + core_shape, dtype=np.float64)
    # derivative-axis index j=0,1,2 means physical x,y,z. Array axes are z,y,x.
    array_axis_for_j = {0: 4, 1: 3, 2: 2}
    for j in range(3):
        axis = array_axis_for_j[j]
        for offset, coefficient in zip(range(-radius, radius + 1), coefficients):
            if coefficient == 0:
                continue
            slices = [slice(None), slice(None)]
            for spatial_axis, size in enumerate(spatial_shape, start=2):
                shift = offset if spatial_axis == axis else 0
                slices.append(slice(output_halo + shift, size - output_halo + shift))
            gradient[:, :, j] += coefficient * velocity[tuple(slices)] / spacing
    return gradient


def _error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    reference_rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
    difference_rms = float(np.sqrt(np.mean(np.square(difference))))
    denominator = reference_rms if reference_rms > 0 else 1.0
    return {
        "reference_rms": reference_rms,
        "difference_rms": difference_rms,
        "relative_rms": difference_rms / denominator,
        "max_abs": float(np.max(np.abs(difference))),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _direct_filter(field: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    kernel_1d = gaussian_kernel_1d(sigma, radius)
    kernel_3d = np.einsum("i,j,k->ijk", kernel_1d, kernel_1d, kernel_1d)
    return convolve(np.asarray(field, dtype=np.float64), kernel_3d, mode="valid", method="direct")


def _classification_from_saved_epsilon(
    work_full: np.ndarray, work_resolved: np.ndarray, epsilon: np.ndarray
) -> np.ndarray:
    finite = np.isfinite(work_full) & np.isfinite(work_resolved)
    result = np.zeros(work_full.shape, dtype=np.uint8)
    pf, nf = work_full > epsilon[0], work_full < -epsilon[0]
    pr, nr = work_resolved > epsilon[1], work_resolved < -epsilon[1]
    result[finite & pf & pr] = 1
    result[finite & pf & nr] = 2
    result[finite & nf & pr] = 3
    result[finite & nf & nr] = 4
    return result


def verify_results(cfg: TaskConfig) -> tuple[dict[str, Any], Path, Path]:
    """Audit saved raw/derived data without making any JHTDB request."""
    validate_raw(cfg)
    if not cfg.derived_path.exists():
        raise FileNotFoundError(f"derived file does not exist: {cfg.derived_path}")
    manifest_path = cfg.raw_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with np.load(cfg.raw_path, allow_pickle=False) as saved:
        raw = {name: saved[name] for name in saved.files}
    with np.load(cfg.derived_path, allow_pickle=False) as saved:
        derived = {name: saved[name] for name in saved.files}

    checks: dict[str, bool] = {}
    checks["raw_sha256"] = _sha256(cfg.raw_path) == manifest.get("raw_sha256")
    checks["times_exact"] = np.allclose(raw["times"], cfg.times, rtol=0.0, atol=2e-10) and np.array_equal(
        raw["times"], derived["times"]
    )
    checks["indices_exact"] = np.array_equal(raw["indices_ijk"], block_indices(cfg).astype(np.int32))
    checks["all_raw_finite"] = all(
        np.all(np.isfinite(raw[name])) for name in ("velocity", "gradient_primary", "gradient_audit")
    )
    checks["all_derived_finite"] = all(
        np.all(np.isfinite(value)) for value in derived.values() if np.issubdtype(value.dtype, np.number)
    )

    velocity = raw["velocity"].astype(np.float64)
    primary_gradient = raw["gradient_primary"].astype(np.float64)
    audit_gradient = raw["gradient_audit"].astype(np.float64)
    halo = cfg.support_radius
    core = (slice(None), slice(None), slice(None)) + (slice(halo, -halo),) * 3
    database_fd8_core = primary_gradient[core]
    database_fd6_core = audit_gradient[core]
    spacing = cfg.domain_length / cfg.grid_shape[0]
    local_fd8 = finite_difference_core(velocity, spacing, 8, halo)
    local_fd6 = finite_difference_core(velocity, spacing, 6, halo)
    derivative_fd8 = _error(database_fd8_core, local_fd8)
    derivative_fd6 = _error(database_fd6_core, local_fd6)
    checks["fd8_matches_velocity_stencil"] = derivative_fd8["relative_rms"] < 2e-5
    checks["fd6_matches_velocity_stencil"] = derivative_fd6["relative_rms"] < 2e-5

    fd6_fd8 = _error(database_fd8_core, database_fd6_core)
    divergence = np.trace(database_fd8_core, axis1=1, axis2=2)
    divergence_stats = {
        "mean": float(np.mean(divergence)),
        "rms": float(np.sqrt(np.mean(np.square(divergence)))),
        "max_abs": float(np.max(np.abs(divergence))),
        "rms_over_gradient_rms": float(
            np.sqrt(np.mean(np.square(divergence))) / np.sqrt(np.mean(np.square(database_fd8_core)))
        ),
    }

    sample_times = sorted({0, len(cfg.times) // 2, len(cfg.times) - 1})
    filter_errors: list[dict[str, Any]] = []
    kernel = gaussian_kernel_1d(cfg.sigma_grid, cfg.support_radius)
    checks["kernel_normalized_symmetric"] = bool(
        np.isclose(kernel.sum(), 1.0, rtol=0.0, atol=1e-14)
        and np.allclose(kernel, kernel[::-1], rtol=0.0, atol=1e-14)
        and np.all(kernel >= 0)
    )
    raw_acceleration = np.einsum("tjzyx,tijzyx->tizyx", velocity, primary_gradient, optimize=True)
    for time_index in sample_times:
        for component in range(3):
            filter_errors.append(
                {
                    "field": "velocity_bar",
                    "time_index": time_index,
                    "component": component,
                    **_error(
                        derived["velocity_bar"][time_index, component],
                        _direct_filter(velocity[time_index, component], cfg.sigma_grid, cfg.support_radius),
                    ),
                }
            )
            filter_errors.append(
                {
                    "field": "a_bar",
                    "time_index": time_index,
                    "component": component,
                    **_error(
                        derived["a_bar"][time_index, component],
                        _direct_filter(raw_acceleration[time_index, component], cfg.sigma_grid, cfg.support_radius),
                    ),
                }
            )
        for i in range(3):
            for j in range(3):
                filter_errors.append(
                    {
                        "field": "gradient_bar_primary",
                        "time_index": time_index,
                        "component": 3 * i + j,
                        **_error(
                            derived["gradient_bar_primary"][time_index, i, j],
                            _direct_filter(primary_gradient[time_index, i, j], cfg.sigma_grid, cfg.support_radius),
                        ),
                    }
                )
    maximum_filter_relative_rms = max(item["relative_rms"] for item in filter_errors)
    checks["direct_3d_filter_matches"] = maximum_filter_relative_rms < 2e-6

    acceleration_barbar_independent = np.einsum(
        "tjzyx,tijzyx->tizyx", derived["velocity_bar"], derived["gradient_bar_primary"], optimize=True
    )
    work_full_independent = np.einsum(
        "tizyx,tizyx->tzyx", derived["velocity_bar"], derived["a_bar"], optimize=True
    )
    work_resolved_independent = np.einsum(
        "tizyx,tizyx->tzyx", derived["velocity_bar"], derived["a_barbar"], optimize=True
    )
    algebra_errors = {
        "a_barbar": _error(derived["a_barbar"], acceleration_barbar_independent),
        "work_full": _error(derived["work_full"], work_full_independent),
        "work_resolved": _error(derived["work_resolved"], work_resolved_independent),
    }
    checks["derived_algebra"] = max(value["relative_rms"] for value in algebra_errors.values()) < 2e-6

    primary_regime = _classification_from_saved_epsilon(
        derived["work_full"], derived["work_resolved"], derived["epsilon_primary"]
    )
    audit_regime = _classification_from_saved_epsilon(
        derived["work_full_audit"], derived["work_resolved_audit"], derived["epsilon_audit"]
    )
    robust_regime = np.where((primary_regime == audit_regime) & (primary_regime != 0), primary_regime, 0)
    checks["primary_regime_exact"] = np.array_equal(primary_regime, derived["regime_primary"])
    checks["audit_regime_exact"] = np.array_equal(audit_regime, derived["regime_audit"])
    checks["robust_regime_exact"] = np.array_equal(robust_regime, derived["regime_robust"])
    checks["regime_range"] = bool(np.all((derived["regime_robust"] >= 0) & (derived["regime_robust"] <= 4)))

    report: dict[str, Any] = {
        "overall": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "raw": {
            "snapshots": int(len(raw["times"])),
            "velocity_shape": list(raw["velocity"].shape),
            "gradient_shape": list(raw["gradient_primary"].shape),
            "velocity_rms": float(np.sqrt(np.mean(np.square(velocity)))),
            "gradient_fd8_rms": float(np.sqrt(np.mean(np.square(primary_gradient)))),
        },
        "derivative": {
            "database_fd8_vs_local_fd8": derivative_fd8,
            "database_fd6_vs_local_fd6": derivative_fd6,
            "database_fd6_vs_fd8": fd6_fd8,
            "divergence_fd8_core": divergence_stats,
        },
        "filter": {
            "kernel_sum": float(kernel.sum()),
            "kernel": kernel.tolist(),
            "sample_time_indices": sample_times,
            "maximum_direct_3d_relative_rms": maximum_filter_relative_rms,
            "details": filter_errors,
        },
        "derived_algebra": algebra_errors,
        "regime": {
            "fd6_fd8_disagreement_fraction": float(np.mean(primary_regime != audit_regime)),
            "robust_uncertain_fraction": float(np.mean(robust_regime == 0)),
            "occupancy": {
                str(code): float(np.mean(robust_regime == code)) for code in range(5)
            },
        },
        "limitations": [
            "This verifies the database FD stencils against the downloaded velocity, not against the DNS spectral derivative.",
            "Exact filter-derivative commutation cannot be tested on a 16^3 block when both derivative and filter require radius-4 halos.",
            "A local 8^3 core is not a statistically complete full-domain validation sample.",
        ],
    }
    cfg.reports_path.mkdir(parents=True, exist_ok=True)
    json_path = cfg.reports_path / "task0_verification.json"
    md_path = cfg.reports_path / "task0_verification.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Task 0 独立验证报告",
        "",
        f"总体结果：**{report['overall']}**",
        "",
        "## 检查项",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in checks.items())
    lines.extend(
        [
            "",
            "## 关键误差",
            "",
            f"- database FD8 vs local FD8 relative RMS: `{derivative_fd8['relative_rms']:.6e}`",
            f"- database FD6 vs local FD6 relative RMS: `{derivative_fd6['relative_rms']:.6e}`",
            f"- database FD6 vs FD8 relative RMS: `{fd6_fd8['relative_rms']:.6e}`",
            f"- direct 3-D filter maximum relative RMS: `{maximum_filter_relative_rms:.6e}`",
            f"- divergence RMS / gradient RMS: `{divergence_stats['rms_over_gradient_rms']:.6e}`",
            f"- FD6/FD8 regime disagreement: `{report['regime']['fd6_fd8_disagreement_fraction']:.6e}`",
            f"- robust uncertain fraction: `{report['regime']['robust_uncertain_fraction']:.6e}`",
            "",
            "## 解释边界",
            "",
            "本报告验证了下载完整性、网格轴顺序、数据库有限差分 stencil、局部 Gaussian 实现、派生代数和 regime 分类。它不能把数据库有限差分证明成 DNS 的全局谱导数，也不能用当前 16³ block 验证完整的 filter–derivative commutation。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return report, json_path, md_path

