from __future__ import annotations

import os
import stat

from .config import PipelineConfig


def token_source(cfg: PipelineConfig) -> str | None:
    if os.environ.get("JHTDB_TOKEN", "").strip():
        return "environment"
    if cfg.token_file is not None and cfg.token_file.is_file():
        return "file"
    return None


def has_token(cfg: PipelineConfig) -> bool:
    return token_source(cfg) is not None


def get_token(cfg: PipelineConfig) -> str:
    environment_token = os.environ.get("JHTDB_TOKEN", "").strip()
    if environment_token:
        return environment_token
    if cfg.token_file is None or not cfg.token_file.is_file():
        raise RuntimeError(
            "JHTDB token is not configured; set JHTDB_TOKEN for this process or create "
            "the protected token file configured by auth.token_file"
        )
    if os.name == "posix":
        mode = stat.S_IMODE(cfg.token_file.stat().st_mode)
        if mode & 0o077:
            raise PermissionError(
                f"token file permissions must be 0600 or stricter: {cfg.token_file}"
            )
    token = cfg.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("configured JHTDB token file is empty")
    return token
