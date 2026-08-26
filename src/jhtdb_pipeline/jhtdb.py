from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
from filelock import FileLock
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from .auth import get_token
from .catalog import Catalog
from .config import PipelineConfig
from .planning import Tile, tiles_for
from .store import VelocityStore


class LocalJHTDB:
    """Single-request-at-a-time wrapper around official givernylocal.GetCutout."""

    def __init__(self, cfg: PipelineConfig, token: str):
        try:
            from givernylocal.turbulence_dataset import turb_dataset
            from givernylocal.turbulence_toolkit import getCutout
        except ImportError as exc:
            raise RuntimeError("givernylocal is missing; install the project dependencies") from exc
        output = cfg.storage_root / ".giverny"
        output.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.token = token
        self.cube = turb_dataset(cfg.dataset, str(output), token)
        self._get_cutout = getCutout

    def fetch_tile(self, tile: Tile, time_index: int) -> np.ndarray:
        xyzt_ranges = np.asarray([*tile.api_ranges, (time_index, time_index)], dtype=np.int32)
        strides = np.ones(4, dtype=np.int32)
        result = self._get_cutout(self.cube, self.cfg.variable, xyzt_ranges, strides, verbose=False)
        try:
            names = list(result.data_vars)
            if len(names) != 1:
                raise RuntimeError(f"expected one velocity data variable, got {names}")
            return canonicalize_cutout(np.asarray(result[names[0]].values), tile)
        finally:
            close = getattr(result, "close", None)
            if callable(close):
                close()


def canonicalize_cutout(values: np.ndarray, tile: Tile) -> np.ndarray:
    """Convert Giverny (z,y,x,component) into canonical (component,z,y,x)."""
    expected = (tile.nz, tile.ny, tile.nx, 3)
    if values.shape != expected:
        raise RuntimeError(f"GetCutout shape {values.shape}, expected {expected}")
    canonical = np.ascontiguousarray(np.moveaxis(values, -1, 0), dtype="<f4")
    if not np.all(np.isfinite(canonical)):
        raise RuntimeError("GetCutout returned NaN or Inf")
    return canonical


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"no existing parent for {path}")
        candidate = candidate.parent
    return candidate


def preflight_space(cfg: PipelineConfig) -> dict[str, float]:
    existing = _existing_parent(cfg.storage_root)
    # On Windows, querying the drive root is both sufficient and more reliable than
    # querying a protected parent directory that may not be readable by the process.
    usage_target = Path(existing.anchor) if existing.anchor else existing
    usage = shutil.disk_usage(usage_target)
    required = cfg.bytes_per_snapshot + int(cfg.safety_free_space_gib * 1024**3)
    if usage.free < required:
        raise RuntimeError(
            f"insufficient free space: {usage.free / 1024**3:.2f} GiB free, "
            f"need at least {required / 1024**3:.2f} GiB for one snapshot plus safety reserve"
        )
    return {"free_GiB": usage.free / 1024**3, "required_GiB": required / 1024**3}


def smoke(cfg: PipelineConfig, time_index: int) -> dict[str, object]:
    token = get_token(cfg)
    client = LocalJHTDB(cfg, token)
    tile = Tile(0, 0, 0, 8, 8, 8)
    values = client.fetch_tile(tile, time_index)
    return {
        "status": "ok",
        "dataset": cfg.dataset,
        "time_index": time_index,
        "physical_time": cfg.physical_time(time_index),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite": bool(np.all(np.isfinite(values))),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def download_snapshot(cfg: PipelineConfig, time_index: int) -> Path:
    if time_index < 1:
        raise ValueError("time_index must be >= 1")
    cfg.storage_root.mkdir(parents=True, exist_ok=True)
    space = preflight_space(cfg)
    token = get_token(cfg)
    console = Console()
    console.print(
        f"[bold]JHTDB serial download[/bold] time_index={time_index} "
        f"physical_time={cfg.physical_time(time_index):.6f} free={space['free_GiB']:.2f} GiB"
    )
    console.print("[yellow]Policy:[/yellow] exactly one JHTDB request will be in flight; no concurrency is used.")
    console.print("[yellow]Usage:[/yellow] JHTDB recommends small targeted subsets and discourages crawling full 3-D series.")

    lock = FileLock(str(cfg.storage_root / "jhtdb-request.lock"), timeout=0)
    with lock:
        client = LocalJHTDB(cfg, token)
        tiles = tiles_for(cfg)
        store = VelocityStore(cfg)
        store.ensure_snapshot(time_index, cfg.physical_time(time_index))
        with Catalog(cfg.catalog_path) as catalog:
            catalog.plan_snapshot(cfg.dataset, time_index, cfg.physical_time(time_index), tiles)
            catalog.set_snapshot_status(cfg.dataset, time_index, "downloading")
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("{task.fields[state]}"),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=4,
            ) as progress:
                task = progress.add_task(f"time {time_index}", total=len(tiles), state="starting")
                for tile_number, tile in enumerate(tiles, start=1):
                    row = catalog.tile(cfg.dataset, time_index, tile.key)
                    if row is not None and row["status"] == "verified" and row["sha256"]:
                        progress.update(task, advance=1, state=f"skip {tile.key}")
                        continue
                    last_error: Exception | None = None
                    for attempt in range(1, cfg.retries + 1):
                        catalog.mark_attempt(cfg.dataset, time_index, tile.key)
                        progress.update(task, state=f"{tile.key} attempt {attempt}/{cfg.retries}")
                        try:
                            values = client.fetch_tile(tile, time_index)
                            digest = store.write_tile(time_index, tile, values)
                            catalog.mark_verified(cfg.dataset, time_index, tile.key, digest, values.nbytes)
                            last_error = None
                            break
                        except Exception as exc:
                            last_error = exc
                            safe_message = str(exc).replace(token, "<redacted>")
                            console.print(f"[red]tile {tile.key} attempt {attempt} failed:[/red] {safe_message}")
                            if attempt < cfg.retries:
                                time.sleep(cfg.backoff_seconds * attempt)
                    if last_error is not None:
                        catalog.set_snapshot_status(cfg.dataset, time_index, "partial")
                        raise last_error
                    progress.update(task, advance=1, state=f"verified {tile_number}/{len(tiles)}")
                    if cfg.request_cooldown_seconds:
                        time.sleep(cfg.request_cooldown_seconds)

    from .validation import validate_snapshot

    report = validate_snapshot(cfg, time_index)
    console.print(json.dumps(report, ensure_ascii=False, indent=2))
    return cfg.raw_store_path
