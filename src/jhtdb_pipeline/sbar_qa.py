from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import zarr

from .config import RESULT_SCHEMA_VERSION, PipelineConfig, result_zarr_name
from .store import spatial_slices
from .validation import atomic_json


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
    absolute_totals = {name: 0.0 for name in FIELD_ORDER}
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
            absolute_totals[name] += float(
                np.abs(values64).sum(dtype=np.float64)
            )
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
    rel_self, rel_self_error = _safe_ratio(
        totals["s_bar"], absolute_totals["s_bar"]
    )
    vs_pi_net, vs_pi_error = _safe_ratio(totals["s_bar"], totals["pi"])
    identity_passed = residual_rms <= cfg.energy_identity_rms_max
    rel_self_passed = (
        rel_self is not None and rel_self <= cfg.s_bar_rel_self_max
    )
    vs_pi_passed = (
        vs_pi_net is not None and vs_pi_net <= cfg.s_bar_vs_pi_net_max
    )
    return {
        "scope": scope,
        "point_count": point_count,
        "identity": "work_full = work_resolved - pi + s_bar",
        "global_totals": totals,
        "global_absolute_totals": absolute_totals,
        "metrics": {
            "identity_residual_rms": {
                "value": residual_rms,
                "maximum_abs": residual_maximum,
                "threshold": cfg.energy_identity_rms_max,
                "passed": identity_passed,
            },
            "s_bar_rel_self": {
                "value": rel_self,
                "threshold": cfg.s_bar_rel_self_max,
                "passed": rel_self_passed,
                "error": rel_self_error,
            },
            "s_bar_vs_pi_net": {
                "value": vs_pi_net,
                "threshold": cfg.s_bar_vs_pi_net_max,
                "passed": vs_pi_passed,
                "error": vs_pi_error,
            },
        },
        "passed": bool(identity_passed and rel_self_passed and vs_pi_passed),
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


def run_sbar_qa(
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
        raise RuntimeError("full-domain schema-v4 result is required")
    report = compute_sbar_qa(root, cfg, scope="full_domain")
    report_hash = write_sbar_artifacts(result_dir, report)
    qa_path = result_dir / "qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["s_bar_global"] = {
        "passed": report["passed"],
        "scope": report["scope"],
        "report_hash": report_hash,
        "metrics": report["metrics"],
    }
    atomic_json(qa_path, qa)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["s_bar_qa_passed"] = report["passed"]
    manifest["s_bar_qa_report_hash"] = report_hash
    manifest_hash = atomic_json(manifest_path, manifest)
    root.attrs.update(
        {
            "s_bar_qa_passed": report["passed"],
            "s_bar_qa_report_hash": report_hash,
            "manifest_hash": manifest_hash,
        }
    )
    atomic_json(result_dir / "COMPLETE", {"manifest_hash": manifest_hash})
    return report
