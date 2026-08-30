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
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .config import RESULT_SCHEMA_VERSION, PipelineConfig, result_zarr_name
from .store import spatial_slices
from .validation import atomic_json


SBAR_REPORT_VERSION = 3
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
    field_sumsq = {name: 0.0 for name in FIELD_ORDER}
    residual_sumsq = 0.0
    residual_maximum = 0.0
    point_count = 0
    chunk_count = int(
        np.prod(
            [
                (size + chunk - 1) // chunk
                for size, chunk in zip(shape, chunks)
            ],
            dtype=np.int64,
        )
    )
    with Progress(
        SpinnerColumn("line"),
        TextColumn("{task.description}"),
        BarColumn(),
        "{task.completed}/{task.total}",
        TimeElapsedColumn(),
        console=Console(),
    ) as progress:
        task = progress.add_task("full-domain S_bar QA", total=chunk_count)
        for key in spatial_slices(shape, chunks):
            values = {
                name: np.asarray(array[key], dtype=np.float32)
                for name, array in arrays.items()
            }
            if any(not np.all(np.isfinite(block)) for block in values.values()):
                raise ValueError("the four energy fields contain NaN or Inf")
            values64 = {
                name: block.astype(np.float64)
                for name, block in values.items()
            }
            for name, block in values64.items():
                totals[name] += float(block.sum(dtype=np.float64))
                field_sumsq[name] += float(
                    np.square(block).sum(dtype=np.float64)
                )
            residual = (
                values64["work_full"]
                - values64["work_resolved"]
                + values64["pi"]
                - values64["s_bar"]
            )
            residual_sumsq += float(np.square(residual).sum(dtype=np.float64))
            residual_maximum = max(
                residual_maximum, float(np.max(np.abs(residual)))
            )
            point_count += residual.size
            progress.advance(task)

    residual_rms = float(np.sqrt(residual_sumsq / point_count))
    field_rms = {
        name: float(np.sqrt(value / point_count))
        for name, value in field_sumsq.items()
    }
    joint_energy_rms = float(
        np.sqrt(sum(field_sumsq.values()) / point_count)
    )
    if joint_energy_rms != 0.0:
        relative_residual = residual_rms / joint_energy_rms
        relative_error = None
    elif residual_rms == 0.0:
        relative_residual = 0.0
        relative_error = None
    else:
        relative_residual = None
        relative_error = "joint energy RMS is zero while residual RMS is nonzero"
    vs_pi_net, vs_pi_error = _safe_ratio(totals["s_bar"], totals["pi"])
    identity_passed = (
        relative_residual is not None
        and relative_residual <= cfg.energy_identity_relative_rms_max
    )
    vs_pi_passed = (
        vs_pi_net is not None and vs_pi_net <= cfg.s_bar_vs_pi_net_max
    )
    return {
        "report_version": SBAR_REPORT_VERSION,
        "scope": scope,
        "point_count": point_count,
        "identity": "work_full = work_resolved - pi + s_bar",
        "global_totals": totals,
        "field_sumsq": field_sumsq,
        "field_rms": field_rms,
        "metrics": {
            "identity_relative_residual_rms": {
                "value": relative_residual,
                "residual_rms": residual_rms,
                "joint_energy_rms": joint_energy_rms,
                "maximum_abs": residual_maximum,
                "threshold": cfg.energy_identity_relative_rms_max,
                "passed": identity_passed,
                "error": relative_error,
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


def _load_chained_report(result_dir: Path, root: Any) -> dict[str, Any] | None:
    path = result_dir / "s_bar_qa.json"
    manifest_path = result_dir / "manifest.json"
    complete_path = result_dir / "COMPLETE"
    if (
        not path.is_file()
        or not (result_dir / "s_bar_global_totals.html").is_file()
        or not manifest_path.is_file()
        or not complete_path.is_file()
    ):
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    report_hash = _report_hash(path)
    manifest_hash = _report_hash(manifest_path)
    valid = (
        report.get("report_version") == SBAR_REPORT_VERSION
        and report.get("scope") == "full_domain"
        and manifest.get("s_bar_qa_report_version") == SBAR_REPORT_VERSION
        and manifest.get("s_bar_qa_report_hash") == report_hash
        and root.attrs.get("s_bar_qa_report_version") == SBAR_REPORT_VERSION
        and root.attrs.get("s_bar_qa_report_hash") == report_hash
        and root.attrs.get("manifest_hash") == manifest_hash
        and complete.get("manifest_hash") == manifest_hash
    )
    return report if valid else None


def _reclassify_report(
    report: dict[str, Any], cfg: PipelineConfig
) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    identity = metrics.get("identity_relative_residual_rms", {})
    vs_pi = metrics.get("s_bar_vs_pi_net", {})
    identity_value = identity.get("value")
    vs_pi_value = vs_pi.get("value")
    identity_passed = (
        identity_value is not None
        and float(identity_value) <= cfg.energy_identity_relative_rms_max
    )
    vs_pi_passed = (
        vs_pi_value is not None
        and float(vs_pi_value) <= cfg.s_bar_vs_pi_net_max
    )
    identity.update(
        {
            "threshold": cfg.energy_identity_relative_rms_max,
            "passed": bool(identity_passed),
        }
    )
    vs_pi.update(
        {
            "threshold": cfg.s_bar_vs_pi_net_max,
            "passed": bool(vs_pi_passed),
        }
    )
    report["passed"] = bool(identity_passed and vs_pi_passed)
    return report


def _can_reclassify(report: dict[str, Any]) -> bool:
    metrics = report.get("metrics", {})
    identity = metrics.get("identity_relative_residual_rms", {})
    vs_pi = metrics.get("s_bar_vs_pi_net", {})
    return (
        set(report.get("field_sumsq", {})) == set(FIELD_ORDER)
        and set(report.get("field_rms", {})) == set(FIELD_ORDER)
        and all(
            key in identity
            for key in ("value", "residual_rms", "joint_energy_rms")
        )
        and "value" in vs_pi
    )


def _thresholds_are_current(
    report: dict[str, Any], cfg: PipelineConfig
) -> bool:
    metrics = report.get("metrics", {})
    identity = metrics.get("identity_relative_residual_rms", {})
    vs_pi = metrics.get("s_bar_vs_pi_net", {})
    return _can_reclassify(report) and (
        identity.get("threshold") == cfg.energy_identity_relative_rms_max
        and vs_pi.get("threshold") == cfg.s_bar_vs_pi_net_max
        and _reclassify_report(
            json.loads(json.dumps(report)), cfg
        ).get("passed")
        == report.get("passed")
        and bool(identity.get("passed"))
        == bool(
            identity.get("value") is not None
            and float(identity["value"])
            <= cfg.energy_identity_relative_rms_max
        )
        and bool(vs_pi.get("passed"))
        == bool(
            vs_pi.get("value") is not None
            and float(vs_pi["value"]) <= cfg.s_bar_vs_pi_net_max
        )
    )


def sbar_report_is_current(
    result_dir: Path, root: Any, cfg: PipelineConfig
) -> bool:
    report = _load_chained_report(result_dir, root)
    return report is not None and _thresholds_are_current(report, cfg)


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
    stored_report = _load_chained_report(result_dir, root)
    if (
        stored_report is not None
        and _can_reclassify(stored_report)
        and not _thresholds_are_current(stored_report, cfg)
    ):
        report = _reclassify_report(stored_report, cfg)
    else:
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
    if not sbar_report_is_current(result_dir, root, cfg):
        run_sbar_qa(cfg, time_index, sigma)
    return result_dir
