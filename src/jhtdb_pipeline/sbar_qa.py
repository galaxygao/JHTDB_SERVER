from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import zarr
from filelock import FileLock

from .config import RESULT_SCHEMA_VERSION, PipelineConfig, result_zarr_name
from .store import spatial_slices
from .validation import atomic_json


SBAR_REPORT_VERSION = 2
FIELD_ORDER = ("s_bar", "pi", "work_resolved", "work_full")
FIELD_LABELS = {
    "s_bar": "ΣS̄",
    "pi": "ΣΠ",
    "work_resolved": "ΣW_res",
    "work_full": "ΣW_full",
}


def _safe_ratio(numerator: float, denominator: float) -> tuple[float | None, str | None]:
    if denominator != 0.0:
        return abs(numerator) / abs(denominator), None
    if numerator == 0.0:
        return 0.0, None
    return None, "denominator is zero while numerator is nonzero"


def compute_sbar_qa(
    root: Any,
    cfg: PipelineConfig,
    *,
    scope: str,
) -> dict[str, Any]:
    arrays = {name: root[name] for name in FIELD_ORDER}
    shapes = {tuple(array.shape) for array in arrays.values()}
    if len(shapes) != 1:
        raise RuntimeError("the four energy fields do not have identical shapes")
    shape = shapes.pop()
    expected = cfg.full_shape_zyx if scope == "full_domain" else cfg.result_shape_zyx
    if shape != expected:
        raise RuntimeError(
            f"S_bar QA scope {scope} expects {expected}, found {shape}"
        )
    if any(np.dtype(array.dtype) != np.dtype("<f4") for array in arrays.values()):
        raise RuntimeError("the four energy fields must be float32")

    chunks = tuple(int(value) for value in arrays["work_full"].chunks)
    totals = {name: 0.0 for name in FIELD_ORDER}
    residual_sumsq = 0.0
    residual_maximum = 0.0
    point_count = 0
    for key in spatial_slices(shape, chunks):
        values = {
            name: np.asarray(array[key], dtype=np.float32)
            for name, array in arrays.items()
        }
        if any(not np.all(np.isfinite(block)) for block in values.values()):
            raise ValueError("the four energy fields contain NaN or Inf")
        for name, block in values.items():
            values64 = block.astype(np.float64)
            totals[name] += float(values64.sum(dtype=np.float64))
        residual = (
            values["work_full"].astype(np.float64)
            - values["work_resolved"].astype(np.float64)
            + values["pi"].astype(np.float64)
            - values["s_bar"].astype(np.float64)
        )
        residual_sumsq += float(np.square(residual).sum(dtype=np.float64))
        residual_maximum = max(
            residual_maximum, float(np.max(np.abs(residual)))
        )
        point_count += residual.size

    residual_rms = float(np.sqrt(residual_sumsq / point_count))
    vs_pi_net, vs_pi_error = _safe_ratio(totals["s_bar"], totals["pi"])
    identity_passed = residual_rms <= cfg.energy_identity_rms_max
    vs_pi_passed = (
        vs_pi_net is not None and vs_pi_net <= cfg.s_bar_vs_pi_net_max
    )
    return {
        "report_version": SBAR_REPORT_VERSION,
        "scope": scope,
        "point_count": point_count,
        "identity": "work_full = work_resolved - pi + s_bar",
        "global_totals": totals,
        "metrics": {
            "identity_residual_rms": {
                "value": residual_rms,
                "maximum_abs": residual_maximum,
                "threshold": cfg.energy_identity_rms_max,
                "passed": identity_passed,
            },
            "s_bar_vs_pi_net": {
                "value": vs_pi_net,
                "threshold": cfg.s_bar_vs_pi_net_max,
                "passed": vs_pi_passed,
                "error": vs_pi_error,
            },
        },
        "passed": bool(identity_passed and vs_pi_passed),
    }


def write_sbar_artifacts(
    result_dir: Path,
    report: dict[str, Any],
) -> str:
    report_hash = atomic_json(result_dir / "s_bar_qa.json", report)
    totals = report["global_totals"]
    figure = go.Figure(
        go.Bar(
            x=[FIELD_LABELS[name] for name in FIELD_ORDER],
            y=[totals[name] for name in FIELD_ORDER],
            customdata=[totals[name] for name in FIELD_ORDER],
            hovertemplate="%{x}: %{customdata:.8e}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Full-domain net totals ({report['scope']})",
        xaxis_title="field",
        yaxis_title="net sum",
    )
    output = result_dir / "s_bar_global_totals.html"
    temporary = output.with_suffix(output.suffix + ".partial")
    figure.write_html(str(temporary), include_plotlyjs=True, full_html=True)
    os.replace(temporary, output)
    return report_hash


def _report_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sbar_report_is_current(result_dir: Path, root: Any) -> bool:
    path = result_dir / "s_bar_qa.json"
    manifest_path = result_dir / "manifest.json"
    complete_path = result_dir / "COMPLETE"
    if (
        not path.is_file()
        or not (result_dir / "s_bar_global_totals.html").is_file()
        or not manifest_path.is_file()
        or not complete_path.is_file()
    ):
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    report_hash = _report_hash(path)
    manifest_hash = _report_hash(manifest_path)
    return (
        report.get("report_version") == SBAR_REPORT_VERSION
        and report.get("scope") == "full_domain"
        and manifest.get("s_bar_qa_report_version") == SBAR_REPORT_VERSION
        and manifest.get("s_bar_qa_report_hash") == report_hash
        and root.attrs.get("s_bar_qa_report_version") == SBAR_REPORT_VERSION
        and root.attrs.get("s_bar_qa_report_hash") == report_hash
        and root.attrs.get("manifest_hash") == manifest_hash
        and complete.get("manifest_hash") == manifest_hash
    )


def _run_sbar_qa_locked(
    cfg: PipelineConfig,
    time_index: int,
    sigma_grid: float | None = None,
) -> dict[str, Any]:
    sigma = cfg.sigma_grid if sigma_grid is None else float(sigma_grid)
    result_dir = cfg.result_path(time_index, sigma)
    if not (result_dir / "COMPLETE").is_file():
        raise RuntimeError("complete result is missing")
    root = zarr.open_group(
        str(result_dir / result_zarr_name(sigma)), mode="a"
    )
    if root.attrs.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise RuntimeError("current full-domain result schema is required")
    report = compute_sbar_qa(root, cfg, scope="full_domain")
    report_hash = write_sbar_artifacts(result_dir, report)
    qa_path = result_dir / "qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["s_bar_global"] = {
        "passed": report["passed"],
        "scope": report["scope"],
        "report_version": report["report_version"],
        "report_hash": report_hash,
        "metrics": report["metrics"],
    }
    atomic_json(qa_path, qa)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "s_bar_qa_passed": report["passed"],
            "s_bar_qa_report_version": report["report_version"],
            "s_bar_qa_report_hash": report_hash,
        }
    )
    manifest_hash = atomic_json(manifest_path, manifest)
    root.attrs.update(
        {
            "s_bar_qa_passed": report["passed"],
            "s_bar_qa_report_version": report["report_version"],
            "s_bar_qa_report_hash": report_hash,
            "manifest_hash": manifest_hash,
        }
    )
    atomic_json(result_dir / "COMPLETE", {"manifest_hash": manifest_hash})
    return report


def run_sbar_qa(
    cfg: PipelineConfig,
    time_index: int,
    sigma_grid: float | None = None,
) -> dict[str, Any]:
    sigma = cfg.sigma_grid if sigma_grid is None else float(sigma_grid)
    cfg.lock_path.mkdir(parents=True, exist_ok=True)
    lock_name = f"qa-metadata-{cfg.result_id(time_index, sigma)}.lock"
    with FileLock(str(cfg.lock_path / lock_name), timeout=0):
        return _run_sbar_qa_locked(cfg, time_index, sigma)


def ensure_sbar_result(
    cfg: PipelineConfig,
    time_index: int,
    sigma_grid: float | None = None,
) -> Path:
    sigma = cfg.sigma_grid if sigma_grid is None else float(sigma_grid)
    result_dir = cfg.result_path(time_index, sigma)
    if not (result_dir / "COMPLETE").is_file():
        raise RuntimeError("complete result is missing")
    root = zarr.open_group(
        str(result_dir / result_zarr_name(sigma)), mode="r"
    )
    if not sbar_report_is_current(result_dir, root):
        run_sbar_qa(cfg, time_index, sigma)
    return result_dir
