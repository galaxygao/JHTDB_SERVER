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
from .weak_asymmetry import (
    build_weak_asymmetry_report,
    ensure_weak_asymmetry_result,
    write_weak_asymmetry_artifacts,
)


CQ_REPORT_VERSION = 2
REGIME_CODES = (1, 2, 3, 4)
REGIME_KEYS = ("Q1", "Q2", "Q3", "Q4")


def _chunk_count(shape: tuple[int, ...], chunks: tuple[int, ...]) -> int:
    return int(
        np.prod(
            [(size + chunk - 1) // chunk for size, chunk in zip(shape, chunks)],
            dtype=np.int64,
        )
    )


def compute_cq(
    root: Any,
    cfg: PipelineConfig,
    *,
    scope: str = "full_domain",
) -> dict[str, Any]:
    pi = root["pi"]
    work_full = root["work_full"]
    work_resolved = root["work_resolved"]
    expected = cfg.full_shape_zyx
    if scope != "full_domain":
        raise ValueError("C_q is defined here only for the full periodic domain")
    if any(
        tuple(array.shape) != expected
        for array in (pi, work_full, work_resolved)
    ):
        raise RuntimeError("C_q requires full-domain pi and work fields with identical shapes")
    if any(
        np.dtype(array.dtype) != np.dtype("<f4")
        for array in (pi, work_full, work_resolved)
    ):
        raise RuntimeError("C_q requires float32 pi and work fields")

    chunks = tuple(int(value) for value in pi.chunks)
    counts = np.zeros(4, dtype=np.int64)
    stored_sums = np.zeros(4, dtype=np.float64)
    pi_sum = 0.0
    abs_pi_sum = 0.0
    pi_sumsq = 0.0
    positive_sum = 0.0
    negative_sum = 0.0
    positive_count = 0
    negative_count = 0
    zero_count = 0
    point_count = 0
    console = Console()
    with Progress(
        SpinnerColumn("line"),
        TextColumn("{task.description}"),
        BarColumn(),
        "{task.completed}/{task.total}",
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "full-domain C_q", total=_chunk_count(expected, chunks)
        )
        for key in spatial_slices(expected, chunks):
            values = np.asarray(pi[key], dtype=np.float32)
            full = np.asarray(work_full[key], dtype=np.float32)
            resolved = np.asarray(work_resolved[key], dtype=np.float32)
            if any(
                not np.all(np.isfinite(block))
                for block in (values, full, resolved)
            ):
                raise ValueError("pi or work fields contain NaN or Inf")
            values64 = values.astype(np.float64)
            pi_sum += float(values64.sum(dtype=np.float64))
            abs_pi_sum += float(np.abs(values64).sum(dtype=np.float64))
            pi_sumsq += float(np.square(values64).sum(dtype=np.float64))
            positive = values > 0.0
            negative = values < 0.0
            positive_count += int(np.count_nonzero(positive))
            negative_count += int(np.count_nonzero(negative))
            zero_count += int(values.size - np.count_nonzero(positive | negative))
            if np.any(positive):
                positive_sum += float(values64[positive].sum(dtype=np.float64))
            if np.any(negative):
                negative_sum += float(values64[negative].sum(dtype=np.float64))
            point_count += values.size
            full_negative = full < 0.0
            resolved_negative = resolved < 0.0
            masks = (
                ~full_negative & ~resolved_negative,
                ~full_negative & resolved_negative,
                full_negative & ~resolved_negative,
                full_negative & resolved_negative,
            )
            for index, mask in enumerate(masks):
                count = int(np.count_nonzero(mask))
                counts[index] += count
                if count:
                    stored_sums[index] += float(values64[mask].sum(dtype=np.float64))
            progress.advance(task)

    if point_count != int(np.prod(expected, dtype=np.int64)):
        raise RuntimeError("C_q point coverage is incomplete")
    partition_sum = float(stored_sums.sum(dtype=np.float64))
    residual_sum = partition_sum - pi_sum
    if abs_pi_sum == 0.0:
        relative_residual = 0.0 if residual_sum == 0.0 else None
    else:
        relative_residual = abs(residual_sum) / abs_pi_sum
    partition_count = int(counts.sum(dtype=np.int64))
    coverage_passed = partition_count == point_count
    flux_passed = (
        relative_residual is not None
        and relative_residual <= cfg.cq_partition_relative_max
    )
    passed = coverage_passed and flux_passed

    regimes: dict[str, Any] = {}
    for index, (code, name) in enumerate(zip(REGIME_CODES, REGIME_KEYS)):
        count = int(counts[index])
        stored_sum = float(stored_sums[index])
        contribution = stored_sum / point_count
        conditional = stored_sum / count if count else None
        regimes[name] = {
            "code": code,
            "count": count,
            "volume_fraction": count / point_count,
            "stored_pi_sum": stored_sum,
            "stored_cq": contribution,
            "stored_conditional_mean_pi": conditional,
            "les_forward_cq": -contribution,
            "les_forward_conditional_mean": -conditional if conditional is not None else None,
        }

    pi_mean = pi_sum / point_count
    weak_asymmetry = build_weak_asymmetry_report(
        point_count=point_count,
        pi_sum=pi_sum,
        abs_pi_sum=abs_pi_sum,
        pi_sumsq=pi_sumsq,
        positive_sum=positive_sum,
        negative_sum=negative_sum,
        positive_count=positive_count,
        negative_count=negative_count,
        zero_count=zero_count,
        closure_relative_max=cfg.cq_partition_relative_max,
    )
    return {
        "report_version": CQ_REPORT_VERSION,
        "scope": scope,
        "point_count": point_count,
        "definition": "C_q = mean(pi * I[sign(work_full), sign(work_resolved) in Qq])",
        "quadrant_definition": {
            "Q1": "work_full >= 0 and work_resolved >= 0",
            "Q2": "work_full >= 0 and work_resolved < 0",
            "Q3": "work_full < 0 and work_resolved >= 0",
            "Q4": "work_full < 0 and work_resolved < 0",
            "zero_rule": "zero belongs to the nonnegative side",
            "thresholded_regime_independent": True,
        },
        "partition_identity": "C_1 + C_2 + C_3 + C_4 = mean(pi)",
        "sign_conventions": {
            "stored": "pi = tau_ij * d_j(velocity_bar_i) = tau:S",
            "les_forward": "Pi_LES = -tau:S = -pi",
        },
        "global": {
            "stored_pi_sum": pi_sum,
            "stored_pi_mean": pi_mean,
            "stored_abs_pi_sum": abs_pi_sum,
            "les_forward_flux_mean": -pi_mean,
        },
        "regimes": regimes,
        "partition_check": {
            "sum_quadrant_counts": partition_count,
            "target_point_count": point_count,
            "sum_volume_fraction": partition_count / point_count,
            "coverage_passed": coverage_passed,
            "sum_stored_cq": partition_sum / point_count,
            "target_stored_pi_mean": pi_mean,
            "residual_sum": residual_sum,
            "residual_mean": residual_sum / point_count,
            "relative_to_sum_abs_pi": relative_residual,
            "threshold": cfg.cq_partition_relative_max,
            "flux_passed": flux_passed,
            "passed": bool(passed),
        },
        "weak_asymmetry": weak_asymmetry,
        "passed": bool(passed),
    }


def write_cq_artifacts(result_dir: Path, report: dict[str, Any]) -> str:
    report_hash = atomic_json(result_dir / "cq.json", report)
    names = list(REGIME_KEYS)
    stored = [report["regimes"][name]["stored_cq"] for name in names]
    les = [report["regimes"][name]["les_forward_cq"] for name in names]
    figure = go.Figure()
    figure.add_bar(name="stored Cq (pi=tau:S)", x=names, y=stored)
    figure.add_bar(name="LES-forward Cq (-tau:S)", x=names, y=les)
    figure.update_layout(
        title="Full-domain C_q decomposition",
        barmode="group",
        xaxis_title="regime",
        yaxis_title="contribution to domain mean",
    )
    output = result_dir / "cq.html"
    temporary = output.with_suffix(output.suffix + ".partial")
    figure.write_html(str(temporary), include_plotlyjs=True, full_html=True)
    os.replace(temporary, output)
    return report_hash


def _report_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cq_report_is_current(result_dir: Path, root: Any) -> bool:
    path = result_dir / "cq.json"
    manifest_path = result_dir / "manifest.json"
    complete_path = result_dir / "COMPLETE"
    if (
        not path.is_file()
        or not (result_dir / "cq.html").is_file()
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
        report.get("report_version") == CQ_REPORT_VERSION
        and report.get("scope") == "full_domain"
        and manifest.get("cq_report_version") == CQ_REPORT_VERSION
        and manifest.get("cq_report_hash") == report_hash
        and root.attrs.get("cq_report_version") == CQ_REPORT_VERSION
        and root.attrs.get("cq_report_hash") == report_hash
        and root.attrs.get("manifest_hash") == manifest_hash
        and complete.get("manifest_hash") == manifest_hash
    )


def run_cq(
    cfg: PipelineConfig,
    time_index: int,
    sigma_grid: float | None = None,
) -> dict[str, Any]:
    sigma = cfg.sigma_grid if sigma_grid is None else float(sigma_grid)
    result_dir = cfg.result_path(time_index, sigma)
    if not (result_dir / "COMPLETE").is_file():
        raise RuntimeError("complete result is missing")
    cfg.lock_path.mkdir(parents=True, exist_ok=True)
    lock_name = f"qa-metadata-{cfg.result_id(time_index, sigma)}.lock"
    with FileLock(str(cfg.lock_path / lock_name), timeout=0):
        root = zarr.open_group(
            str(result_dir / result_zarr_name(sigma)), mode="a"
        )
        if root.attrs.get("result_schema_version") != RESULT_SCHEMA_VERSION:
            raise RuntimeError("current full-domain result schema is required")
        report = compute_cq(root, cfg)
        report_hash = write_cq_artifacts(result_dir, report)
        weak_report = report["weak_asymmetry"]
        weak_report_hash = write_weak_asymmetry_artifacts(
            result_dir, weak_report
        )

        qa_path = result_dir / "qa.json"
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        qa["cq"] = {
            "passed": report["passed"],
            "scope": report["scope"],
            "report_version": report["report_version"],
            "report_hash": report_hash,
            "partition_check": report["partition_check"],
        }
        qa["weak_asymmetry"] = {
            "passed": weak_report["passed"],
            "scope": weak_report["scope"],
            "report_version": weak_report["report_version"],
            "report_hash": weak_report_hash,
            "asymmetry_index": weak_report["global"]["asymmetry_index"],
            "closure": weak_report["closure"],
        }
        atomic_json(qa_path, qa)

        manifest_path = result_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "cq_passed": report["passed"],
                "cq_report_version": report["report_version"],
                "cq_report_hash": report_hash,
                "weak_asymmetry_passed": weak_report["passed"],
                "weak_asymmetry_report_version": weak_report["report_version"],
                "weak_asymmetry_report_hash": weak_report_hash,
            }
        )
        manifest_hash = atomic_json(manifest_path, manifest)
        root.attrs.update(
            {
                "cq_passed": report["passed"],
                "cq_report_version": report["report_version"],
                "cq_report_hash": report_hash,
                "weak_asymmetry_passed": weak_report["passed"],
                "weak_asymmetry_report_version": weak_report["report_version"],
                "weak_asymmetry_report_hash": weak_report_hash,
                "manifest_hash": manifest_hash,
            }
        )
        atomic_json(result_dir / "COMPLETE", {"manifest_hash": manifest_hash})
        return report


def ensure_cq_result(
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
    if not cq_report_is_current(result_dir, root):
        run_cq(cfg, time_index, sigma)
    return ensure_weak_asymmetry_result(cfg, time_index, sigma)
