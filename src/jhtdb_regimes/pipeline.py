from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import TaskConfig
from .physics import regime_codes, robust_regime, work_and_fields


def validate_raw(cfg: TaskConfig) -> dict[str, Any]:
    if not cfg.raw_path.exists():
        raise FileNotFoundError(f"raw file does not exist: {cfg.raw_path}")
    with np.load(cfg.raw_path, allow_pickle=False) as raw:
        times = raw["times"]
        velocity = raw["velocity"]
        primary = raw["gradient_primary"]
        expected_velocity = (len(times), 3, cfg.block_shape[2], cfg.block_shape[1], cfg.block_shape[0])
        expected_gradient = (len(times), 3, 3) + expected_velocity[-3:]
        if velocity.shape != expected_velocity:
            raise ValueError(f"velocity shape {velocity.shape}, expected {expected_velocity}")
        if primary.shape != expected_gradient:
            raise ValueError(f"gradient shape {primary.shape}, expected {expected_gradient}")
        if not np.allclose(times, cfg.times, rtol=0.0, atol=2e-10):
            raise ValueError("saved times differ from config")
        for key in ("velocity", "gradient_primary"):
            if not np.all(np.isfinite(raw[key])):
                raise ValueError(f"{key} contains NaN or Inf")
        if "gradient_audit" in raw and raw["gradient_audit"].shape != expected_gradient:
            raise ValueError("audit gradient shape is invalid")
    return {"status": "ok", "snapshots": len(times), "velocity_shape": list(expected_velocity)}


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "max_abs": float(np.max(np.abs(values))),
    }


def _occupancy(codes: np.ndarray) -> dict[str, float]:
    total = codes.size
    return {f"Q{code}" if code else "uncertain": float(np.count_nonzero(codes == code) / total) for code in range(5)}


def compute(cfg: TaskConfig) -> Path:
    validate_raw(cfg)
    with np.load(cfg.raw_path, allow_pickle=False) as raw:
        times = raw["times"].astype(np.float64)
        velocity = raw["velocity"].astype(np.float64)
        primary_gradient = raw["gradient_primary"].astype(np.float64)
        audit_gradient = raw["gradient_audit"].astype(np.float64) if "gradient_audit" in raw else None

    primary = work_and_fields(velocity, primary_gradient, cfg.sigma_grid, cfg.support_radius)
    regime_primary, eps_full, eps_res = regime_codes(
        primary["work_full"], primary["work_resolved"], cfg.epsilon_abs, cfg.epsilon_rel
    )
    audit = None
    regime_audit = None
    audit_eps = None
    if audit_gradient is not None:
        audit = work_and_fields(velocity, audit_gradient, cfg.sigma_grid, cfg.support_radius)
        regime_audit, audit_eps_full, audit_eps_res = regime_codes(
            audit["work_full"], audit["work_resolved"], cfg.epsilon_abs, cfg.epsilon_rel
        )
        audit_eps = [audit_eps_full, audit_eps_res]
    regime_final = robust_regime(regime_primary, regime_audit)

    output = {
        "times": times,
        "velocity_bar": primary["velocity_bar"].astype(np.float32),
        "gradient_bar_primary": primary["gradient_bar"].astype(np.float32),
        "a_bar": primary["acceleration_bar"].astype(np.float32),
        "a_barbar": primary["acceleration_barbar"].astype(np.float32),
        "work_full": primary["work_full"].astype(np.float32),
        "work_resolved": primary["work_resolved"].astype(np.float32),
        "regime_primary": regime_primary,
        "regime_robust": regime_final,
        "divergence_primary": primary["divergence_raw"].astype(np.float32),
        "divergence_bar_primary": primary["divergence_bar"].astype(np.float32),
        "epsilon_primary": np.asarray([eps_full, eps_res]),
    }
    if audit is not None and regime_audit is not None:
        output.update(
            {
                "work_full_audit": audit["work_full"].astype(np.float32),
                "work_resolved_audit": audit["work_resolved"].astype(np.float32),
                "regime_audit": regime_audit,
                "divergence_audit": audit["divergence_raw"].astype(np.float32),
                "epsilon_audit": np.asarray(audit_eps),
            }
        )
    cfg.derived_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.derived_path, **output)
    write_report(cfg, output)
    return cfg.derived_path


def write_report(cfg: TaskConfig, output: dict[str, np.ndarray] | None = None) -> tuple[Path, Path]:
    if output is None:
        if not cfg.derived_path.exists():
            raise FileNotFoundError(f"derived file does not exist: {cfg.derived_path}")
        with np.load(cfg.derived_path, allow_pickle=False) as saved:
            output = {key: saved[key] for key in saved.files}
    robust = output["regime_robust"]
    report: dict[str, Any] = {
        "dataset": cfg.dataset,
        "snapshots": int(len(output["times"])),
        "block_shape": list(cfg.block_shape),
        "core_shape": list(cfg.core_shape),
        "filter": {"kind": "local_discrete_gaussian", "sigma_grid": cfg.sigma_grid, "radius": cfg.support_radius},
        "gradient": {"primary": cfg.gradient_primary, "audit": cfg.gradient_audit, "documented_rms_error_vs_spectral": "about 7%"},
        "regime_occupancy": _occupancy(robust),
        "per_snapshot_occupancy": [_occupancy(frame) for frame in robust],
        "work_full": _summary_stats(output["work_full"]),
        "work_resolved": _summary_stats(output["work_resolved"]),
        "divergence_primary": _summary_stats(output["divergence_primary"]),
        "divergence_bar_primary": _summary_stats(output["divergence_bar_primary"]),
        "warning": "database local finite-difference gradient; local Gaussian filter; not a full-domain spectral result",
    }
    if "regime_audit" in output:
        report["fd6_fd8_disagreement_fraction"] = float(
            np.mean(output["regime_primary"] != output["regime_audit"])
        )
        report["work_full_fd_difference"] = _summary_stats(output["work_full"] - output["work_full_audit"])
        report["work_resolved_fd_difference"] = _summary_stats(
            output["work_resolved"] - output["work_resolved_audit"]
        )

    cfg.reports_path.mkdir(parents=True, exist_ok=True)
    json_path = cfg.reports_path / "task0_report.json"
    md_path = cfg.reports_path / "task0_report.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    lines = [
        "# Task 0 计算报告",
        "",
        f"- 数据集：`{cfg.dataset}`",
        f"- snapshots：{report['snapshots']}",
        f"- block/core：`{cfg.block_shape}` / `{cfg.core_shape}`",
        f"- 梯度：`{cfg.gradient_primary}`；审计：`{cfg.gradient_audit}`",
        f"- FD6/FD8 disagreement：{report.get('fd6_fd8_disagreement_fraction', '未计算')}",
        "",
        "## Robust regime occupancy",
        "",
    ]
    lines.extend(f"- {name}: {fraction:.8f}" for name, fraction in report["regime_occupancy"].items())
    lines.extend(
        [
            "",
            "## 重要限制",
            "",
            "数据库局部有限差分梯度相对 DNS 全局谱梯度约有 7% RMS 误差；本结果使用局部 Gaussian 滤波，不是全域 spectral result。",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path

