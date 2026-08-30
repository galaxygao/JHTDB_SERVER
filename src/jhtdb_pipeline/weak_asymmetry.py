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


WEAK_ASYMMETRY_REPORT_VERSION = 2


class AbsPiPercentileAccumulator:
    """Compute exact linear p99/max from one stream of nonnegative float32 data."""

    def __init__(self, point_count: int) -> None:
        if point_count <= 0:
            raise ValueError("percentile point count must be positive")
        self.point_count = int(point_count)
        self.rank = (self.point_count - 1) * 0.99
        self.lower_index = int(np.floor(self.rank))
        self.upper_index = int(np.ceil(self.rank))
        self.keep_count = self.point_count - self.lower_index
        self.flush_count = max(self.keep_count * 4, self.keep_count)
        self.seen_count = 0
        self.buffered_count = 0
        self.buffers: list[np.ndarray] = []

    def add(self, absolute_values: np.ndarray) -> None:
        values = np.asarray(absolute_values, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("absolute pi values must be finite and nonnegative")
        self.buffers.append(values)
        self.seen_count += int(values.size)
        self.buffered_count += int(values.size)
        if self.buffered_count >= self.flush_count:
            self._consolidate()

    def _consolidate(self) -> None:
        if not self.buffers:
            return
        pieces = self.buffers
        self.buffers = []
        combined = np.concatenate(pieces)
        del pieces
        if combined.size > self.keep_count:
            cutoff = int(combined.size - self.keep_count)
            combined.partition(cutoff)
            combined = combined[cutoff:].copy()
        self.buffers = [combined]
        self.buffered_count = int(combined.size)

    def result(self) -> tuple[float, float]:
        if self.seen_count != self.point_count:
            raise RuntimeError("p99 point coverage is incomplete")
        self._consolidate()
        upper_tail = self.buffers[0]
        upper_offset = self.upper_index - self.lower_index
        upper_tail.partition((0, upper_offset))
        lower_value = float(upper_tail[0])
        upper_value = float(upper_tail[upper_offset])
        fraction = self.rank - self.lower_index
        percentile_99 = lower_value + fraction * (upper_value - lower_value)
        maximum = float(np.max(upper_tail))
        return float(percentile_99), maximum


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
    abs_pi_p99: float,
    abs_pi_max: float,
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
    absolute_mean = abs(pi_mean)
    ratio_p99, ratio_p99_error = _safe_nonnegative_ratio(
        absolute_mean, abs_pi_p99, "percentile(abs(pi), 99)"
    )
    ratio_max, ratio_max_error = _safe_nonnegative_ratio(
        absolute_mean, abs_pi_max, "max(abs(pi))"
    )

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
            "abs_pi_p99": abs_pi_p99,
            "abs_pi_max": abs_pi_max,
            "asymmetry_index": asymmetry_index,
            "asymmetry_index_definition": "mean(pi) / rms(pi)",
            "asymmetry_index_error": asymmetry_error,
            "ratio_p99": ratio_p99,
            "ratio_p99_definition": "abs(mean(pi)) / percentile(abs(pi), 99)",
            "ratio_p99_error": ratio_p99_error,
            "ratio_max": ratio_max,
            "ratio_max_definition": "abs(mean(pi)) / max(abs(pi))",
            "ratio_max_error": ratio_max_error,
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


def _safe_nonnegative_ratio(
    numerator: float, denominator: float, denominator_name: str
) -> tuple[float | None, str | None]:
    if denominator != 0.0:
        return numerator / denominator, None
    if numerator == 0.0:
        return 0.0, None
    return None, f"{denominator_name} is zero while abs(mean(pi)) is nonzero"


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
    expected_count = int(np.prod(expected, dtype=np.int64))
    tail = AbsPiPercentileAccumulator(expected_count)
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
            absolute_values = np.abs(values)
            positive = values > 0.0
            negative = values < 0.0
            pi_sum += float(values64.sum(dtype=np.float64))
            abs_pi_sum += float(absolute_values.sum(dtype=np.float64))
            pi_sumsq += float(np.square(values64).sum(dtype=np.float64))
            tail.add(absolute_values)
            positive_count += int(np.count_nonzero(positive))
            negative_count += int(np.count_nonzero(negative))
            zero_count += int(values.size - np.count_nonzero(positive | negative))
            if np.any(positive):
                positive_sum += float(values64[positive].sum(dtype=np.float64))
            if np.any(negative):
                negative_sum += float(values64[negative].sum(dtype=np.float64))
            point_count += values.size
            progress.advance(task)
    abs_pi_p99, abs_pi_max = tail.result()
    return build_weak_asymmetry_report(
        point_count=point_count,
        pi_sum=pi_sum,
        abs_pi_sum=abs_pi_sum,
        pi_sumsq=pi_sumsq,
        abs_pi_p99=abs_pi_p99,
        abs_pi_max=abs_pi_max,
        positive_sum=positive_sum,
        negative_sum=negative_sum,
        positive_count=positive_count,
        negative_count=negative_count,
        zero_count=zero_count,
        closure_relative_max=cfg.cq_partition_relative_max,
    )


def compute_abs_pi_tail(root: Any, cfg: PipelineConfig) -> tuple[float, float]:
    """Read pi once and compute only the new p99/max metrics for a v1 report."""
    pi = root["pi"]
    expected = cfg.full_shape_zyx
    if tuple(pi.shape) != expected or np.dtype(pi.dtype) != np.dtype("<f4"):
        raise RuntimeError("weak asymmetry requires full-domain float32 pi")
    chunks = tuple(int(value) for value in pi.chunks)
    tail = AbsPiPercentileAccumulator(
        int(np.prod(expected, dtype=np.int64))
    )
    with Progress(
        SpinnerColumn("line"),
        TextColumn("{task.description}"),
        BarColumn(),
        "{task.completed}/{task.total}",
        TimeElapsedColumn(),
        console=Console(),
    ) as progress:
        task = progress.add_task(
            "full-domain weak asymmetry p99/max",
            total=_chunk_count(expected, chunks),
        )
        for key in spatial_slices(expected, chunks):
            values = np.asarray(pi[key], dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise ValueError("pi contains NaN or Inf")
            tail.add(np.abs(values))
            progress.advance(task)
    return tail.result()


def upgrade_v1_report(
    report: dict[str, Any], abs_pi_p99: float, abs_pi_max: float
) -> dict[str, Any]:
    upgraded = json.loads(json.dumps(report))
    global_values = upgraded["global"]
    absolute_mean = abs(float(global_values["pi_mean"]))
    ratio_p99, ratio_p99_error = _safe_nonnegative_ratio(
        absolute_mean, abs_pi_p99, "percentile(abs(pi), 99)"
    )
    ratio_max, ratio_max_error = _safe_nonnegative_ratio(
        absolute_mean, abs_pi_max, "max(abs(pi))"
    )
    global_values.update(
        {
            "abs_pi_p99": abs_pi_p99,
            "abs_pi_max": abs_pi_max,
            "ratio_p99": ratio_p99,
            "ratio_p99_definition": "abs(mean(pi)) / percentile(abs(pi), 99)",
            "ratio_p99_error": ratio_p99_error,
            "ratio_max": ratio_max,
            "ratio_max_definition": "abs(mean(pi)) / max(abs(pi))",
            "ratio_max_error": ratio_max_error,
        }
    )
    upgraded["report_version"] = WEAK_ASYMMETRY_REPORT_VERSION
    return upgraded


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


def _load_chained_report(result_dir: Path, root: Any) -> dict[str, Any] | None:
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
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    report_hash = _file_hash(report_path)
    manifest_hash = _file_hash(manifest_path)
    report_version = report.get("report_version")
    valid = (
        report_version in (1, WEAK_ASYMMETRY_REPORT_VERSION)
        and report.get("scope") == "full_domain"
        and manifest.get("weak_asymmetry_report_version") == report_version
        and manifest.get("weak_asymmetry_report_hash") == report_hash
        and root.attrs.get("weak_asymmetry_report_version") == report_version
        and root.attrs.get("weak_asymmetry_report_hash") == report_hash
        and root.attrs.get("manifest_hash") == manifest_hash
        and complete.get("manifest_hash") == manifest_hash
    )
    return report if valid else None


def weak_asymmetry_report_is_current(result_dir: Path, root: Any) -> bool:
    report = _load_chained_report(result_dir, root)
    global_values = report.get("global", {}) if report is not None else {}
    return bool(
        report is not None
        and report.get("report_version") == WEAK_ASYMMETRY_REPORT_VERSION
        and all(
            key in global_values
            for key in ("abs_pi_p99", "abs_pi_max", "ratio_p99", "ratio_max")
        )
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
        "ratio_p99": report["global"]["ratio_p99"],
        "ratio_max": report["global"]["ratio_max"],
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
        stored_report = _load_chained_report(result_dir, root)
        if stored_report is not None and stored_report.get("report_version") == 1:
            abs_pi_p99, abs_pi_max = compute_abs_pi_tail(root, cfg)
            report = upgrade_v1_report(
                stored_report, abs_pi_p99, abs_pi_max
            )
        else:
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
