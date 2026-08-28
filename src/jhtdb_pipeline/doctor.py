from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .auth import token_source
from .config import PipelineConfig
from .validation import atomic_json


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def ensure_run_record(cfg: PipelineConfig, time_index: int) -> dict[str, Any]:
    path = cfg.run_path(time_index) / "run.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        expires = datetime.fromisoformat(record["expires_at"])
        if expires <= _utcnow():
            raise RuntimeError(
                f"scratch run t{time_index:06d} has expired; remove that exact "
                "scratch run directory and rebuild it from JHTDB"
            )
        return record
    created = _utcnow()
    payload = {
        "time_index": time_index,
        "created_at": created.isoformat(),
        "expires_at": (
            created + timedelta(hours=cfg.scratch_retention_hours)
        ).isoformat(),
        "retention_hours": cfg.scratch_retention_hours,
    }
    atomic_json(path, payload)
    return payload


def _space(path: Path) -> dict[str, float]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return {
        "total_GiB": usage.total / 1024**3,
        "used_GiB": usage.used / 1024**3,
        "free_GiB": usage.free / 1024**3,
    }


def _writable(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / f".doctor-write-{os.getpid()}"
    try:
        with probe.open("x", encoding="utf-8") as handle:
            handle.write("ok\n")
            handle.flush()
            os.fsync(handle.fileno())
        return probe.read_text(encoding="utf-8") == "ok\n"
    finally:
        if probe.exists():
            probe.unlink()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def doctor(cfg: PipelineConfig, time_index: int | None = None) -> dict[str, Any]:
    storage_text = str(cfg.state_root).replace("\\", "/")
    temporary_text = str(cfg.run_root).replace("\\", "/")
    is_linux = platform.system() == "Linux"
    storage_path_ok = "/home/idies/workspace/Storage/" in storage_text
    temporary_path_ok = "/home/idies/workspace/Temporary/" in temporary_text
    packages = {
        name: _version(name)
        for name in ("giverny", "numpy", "scipy", "zarr", "streamlit")
    }
    checks = {
        "linux": is_linux,
        "storage_path": storage_path_ok,
        "temporary_path": temporary_path_ok,
        "persistent_writable": bool(is_linux and storage_path_ok and _writable(cfg.state_root)),
        "scratch_writable": bool(is_linux and temporary_path_ok and _writable(cfg.run_root)),
        "result_writable": bool(is_linux and storage_path_ok and _writable(cfg.result_root)),
        "token_configured": token_source(cfg) is not None,
        "giverny_available": packages["giverny"] is not None,
    }
    payload: dict[str, Any] = {
        "checks": checks,
        "paths": {
            "state_root": str(cfg.state_root),
            "run_root": str(cfg.run_root),
            "result_root": str(cfg.result_root),
        },
        "persistent": {
            **(_space(cfg.result_root) if is_linux and storage_path_ok else {}),
            "observed_account_capacity_GB": cfg.persistent_capacity_gb_observed,
            "safety_reserve_GiB": cfg.persistent_safety_reserve_gib,
            "quota_note": "account quota is confirmed in the SciServer Quotas UI; filesystem free space is checked here",
        },
        "scratch": {
            **(_space(cfg.run_root) if is_linux and temporary_path_ok else {}),
            "retention_hours": cfg.scratch_retention_hours,
            "safety_reserve_GiB": cfg.scratch_safety_reserve_gib,
        },
        "token_source": token_source(cfg),
        "packages": packages,
    }
    if time_index is not None:
        record_path = cfg.run_path(time_index) / "run.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            expires = datetime.fromisoformat(record["expires_at"])
            remaining_hours = max(
                0.0, (expires - _utcnow()).total_seconds() / 3600.0
            )
            checks["run_not_expired"] = remaining_hours > 0.0
            payload["run"] = {
                **record,
                "remaining_hours": remaining_hours,
            }
        else:
            payload["run"] = {"time_index": time_index, "status": "not_created"}
    payload["status"] = "ok" if all(checks.values()) else "failed"
    return payload
