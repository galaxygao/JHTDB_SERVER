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


WEAK_ASYMMETRY_REPORT_VERSION = 1


def _chunk_count(shape: tuple[int, ...], chunks: tuple[int, ...]) -> int:
    return int(
        np.prod(
            [
                (size + chunk - 1) // chunk
                for size, chunk in zip(shape, chunks)
            ],
            dtype=np.int64,
        )
    )


def build_weak_asymmetry_report(
    *,
    point_count: int,
    pi_sum: float,
    abs_pi_sum: float,
    pi_sumsq: float,
    positive_sum: float,
    negative_sum: float,
    positive_count: int,
    negative_count: int,
    zero_count: int,
    closure_relative_max: float,
) -> dict[str, Any]:
    if point_count <= 0:
        raise ValueError("weak-asymmetry point count must be positive")
    covered = positive_count + negative_count + zero_count
    pi_mean = pi_sum / point_count
    pi_rms = float(np.sqrt(pi_sumsq / point_count))
    if pi_rms != 0.0:
        asymmetry_index = pi_mean / pi_rms
        asymmetry_error = None
    elif pi_mean == 0.0:
        asymmetry_index = 0.0
        asymmetry_error = None
    else:
        asymmetry_index = None
        asymmetry_error = "pi RMS is zero while pi mean is nonzero"

    reconstructed_sum = positive_sum + negative_sum
    residual_sum = reconstructed_sum - pi_sum
    if abs_pi_sum != 0.0:
        relative_residual = abs(residual_sum) / abs_pi_sum
    elif residual_sum == 0.0:
        relative_residual = 0.0
    else:
        relative_residual = None
    coverage_passed = covered == point_count
    flux_passed = (
        relative_residual is not None
        and relative_residual <= closure_relative_max
    )
    passed = coverage_passed and flux_passed

    return {
        "report_version": WEAK_ASYMMETRY_REPORT_VERSION,
        "scope": "full_domain",
        "point_count": point_count,
        "sign_convention": {
            "stored": "pi = tau:S",
            "negative": "forward cascade (large to small scales)",
            "positive": "backscatter (small to large scales)",
        },
        "global": {
            "pi_sum": pi_sum,
            "pi_mean": pi_mean,
            "pi_rms": pi_rms,
            "sum_abs_pi": abs_pi_sum,
            "asymmetry_index": asymmetry_index,
            "asymmetry_index_definition": "mean(pi) / rms(pi)",
            "asymmetry_index_error": asymmetry_error,
        },
        "positive_backscatter": {
            "sum": positive_sum,
            "count": positive_count,
            "volume_fraction": positive_count / point_count,
        },
        "negative_forward": {
            "sum": negative_sum,
            "count": negative_count,
            "volume_fraction": negative_count / point_count,
        },
        "zero": {
            "count": zero_count,
            "volume_fraction": zero_count / point_count,
        },
        "closure": {
            "identity": "sum(pi[pi > 0]) + sum(pi[pi < 0]) = sum(pi)",
            "covered_point_count": covered,
            "target_point_count": point_count,
            "coverage_passed": coverage_passed,
            "reconstructed_pi_sum": reconstructed_sum,
            "target_pi_sum": pi_sum,
            "residual_sum": residual_sum,
            "relative_to_sum_abs_pi": relative_residual,
            "threshold": closure_relative_max,
            "flux_passed": flux_passed,
            "passed": passed,
        },
        "passed": passed,
    }


def compute_weak_asymmetry(
    root: Any, cfg: PipelineConfig
) -> dict[str, Any]:
    pi = root["pi"]
    expected = cfg.full_shape_zyx
    if tuple(pi.shape) != expected or np.dtype(pi.dtype) != np.dtype("<f4"):
        raise RuntimeError("weak asymmetry requires full-domain float32 pi")
    chunks = tuple(int(value) for value in pi.chunks)
    pi_sum = 0.0
    abs_pi_sum = 0.0
    pi_sumsq = 0.0
    positive_sum = 0.0
    negative_sum = 0.0
    positive_count = 0
    negative_count = 0
    zero_count = 0
    point_count = 0
    with Progress(
        SpinnerColumn("line"),
        TextColumn("{task.description}"),
        BarColumn(),
        "{task.completed}/{task.total}",
        TimeElapsedColumn(),
        console=Console(),
    ) as progress:
        task = progress.add_task(
            "full-domain weak asymmetry",
            total=_chunk_count(expected, chunks),
        )
        for key in spatial_slices(expected, chunks):
            values = np.asarray(pi[key], dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise ValueError("pi contains NaN or Inf")
            values64 = values.astype(np.float64)
            positive = values > 0.0
            negative = values < 0.0
            pi_sum += float(values64.sum(dtype=np.float64))
            abs_pi_sum += float(np.abs(values64).sum(dtype=np.float64))
            pi_sumsq += float(np.square(values64).sum(dtype=np.float64))
            positive_count += int(np.count_nonzero(positive))
            negative_count += int(np.count_nonzero(negative))
            zero_count += int(values.size - np.count_nonzero(positive | negative))
            if np.any(positive):
                positive_sum += float(values64[positive].sum(dtype=np.float64))
            if np.any(negative):
                negative_sum += float(values64[negative].sum(dtype=np.float64))
            point_count += values.size
            progress.advance(task)
    return build_weak_asymmetry_report(
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


def write_weak_asymmetry_artifacts(
    result_dir: Path, report: dict[str, Any]
) -> str:
    report_hash = atomic_json(result_dir / "weak_asymmetry.json", report)
    global_values = report["global"]
    figure = go.Figure(
        go.Bar(
            x=("positive/backscatter", "negative/forward", "net"),
            y=(
                report["positive_backscatter"]["sum"],
                report["negative_forward"]["sum"],
                global_values["pi_sum"],
            ),
            hovertemplate="%{x}: %{y:.8e}<extra></extra>",
        )
    )
    figure.update_layout(
        title=(
            "Full-domain weak asymmetry: "
            f"mean(pi)/rms(pi)={global_values['asymmetry_index']:.6e}"
        ),
        xaxis_title="stored pi sign",
        yaxis_title="sum(pi)",
    )
    output = result_dir / "weak_asymmetry.html"
    temporary = output.with_suffix(output.suffix + ".partial")
    figure.write_html(str(temporary), include_plotlyjs=True, full_html=True)
    os.replace(temporary, output)
    return report_hash


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weak_asymmetry_report_is_current(result_dir: Path, root: Any) -> bool:
    report_path = result_dir / "weak_asymmetry.json"
    manifest_path = result_dir / "manifest.json"
    complete_path = result_dir / "COMPLETE"
    if not all(
        path.is_file()
        for path in (
            report_path,
            result_dir / "weak_asymmetry.html",
            manifest_path,
            complete_path,
        )
    ):
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    report_hash = _file_hash(report_path)
    manifest_hash = _file_hash(manifest_path)
    return (
        report.get("report_version") == WEAK_ASYMMETRY_REPORT_VERSION
        and report.get("scope") == "full_domain"
        and manifest.get("weak_asymmetry_report_version")
        == WEAK_ASYMMETRY_REPORT_VERSION
        and manifest.get("weak_asymmetry_report_hash") == report_hash
        and root.attrs.get("weak_asymmetry_report_version")
        == WEAK_ASYMMETRY_REPORT_VERSION
        and root.attrs.get("weak_asymmetry_report_hash") == report_hash
        and root.attrs.get("manifest_hash") == manifest_hash
        and complete.get("manifest_hash") == manifest_hash
    )


def _persist_weak_asymmetry(
    result_dir: Path, root: Any, report: dict[str, Any]
) -> None:
    report_hash = write_weak_asymmetry_artifacts(result_dir, report)
    qa_path = result_dir / "qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["weak_asymmetry"] = {
        "passed": report["passed"],
        "scope": report["scope"],
        "report_version": report["report_version"],
        "report_hash": report_hash,
        "asymmetry_index": report["global"]["asymmetry_index"],
        "closure": report["closure"],
    }
    atomic_json(qa_path, qa)
    manifest_path = result_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "weak_asymmetry_passed": report["passed"],
            "weak_asymmetry_report_version": report["report_version"],
            "weak_asymmetry_report_hash": report_hash,
        }
    )
    manifest_hash = atomic_json(manifest_path, manifest)
    root.attrs.update(
        {
            "weak_asymmetry_passed": report["passed"],
            "weak_asymmetry_report_version": report["report_version"],
            "weak_asymmetry_report_hash": report_hash,
            "manifest_hash": manifest_hash,
        }
    )
    atomic_json(result_dir / "COMPLETE", {"manifest_hash": manifest_hash})


def run_weak_asymmetry(
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
        report = compute_weak_asymmetry(root, cfg)
        _persist_weak_asymmetry(result_dir, root, report)
        return report


def ensure_weak_asymmetry_result(
    cfg: PipelineConfig,
    time_index: int,
    sigma_grid: float | None = None,
) -> Path:
    sigma = cfg.sigma_grid if sigma_grid is None else float(sigma_grid)
    result_dir = cfg.result_path(time_index, sigma)
    root = zarr.open_group(
        str(result_dir / result_zarr_name(sigma)), mode="r"
    )
    if not weak_asymmetry_report_is_current(result_dir, root):
        run_weak_asymmetry(cfg, time_index, sigma)
    return result_dir
