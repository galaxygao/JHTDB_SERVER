from __future__ import annotations

import keyring

from .config import PipelineConfig


def set_token(cfg: PipelineConfig) -> None:
    token = input("JHTDB token: ").strip()
    if not token:
        raise ValueError("token cannot be empty")
    keyring.set_password(cfg.auth_service, cfg.auth_username, token)


def get_token(cfg: PipelineConfig) -> str:
    token = keyring.get_password(cfg.auth_service, cfg.auth_username)
    if token is None or not token.strip():
        raise RuntimeError("no JHTDB token in Windows Credential Manager; run: python -m jhtdb_pipeline auth set")
    return token.strip()


def has_token(cfg: PipelineConfig) -> bool:
    token = keyring.get_password(cfg.auth_service, cfg.auth_username)
    return bool(token and token.strip())


def delete_token(cfg: PipelineConfig) -> bool:
    if not has_token(cfg):
        return False
    keyring.delete_password(cfg.auth_service, cfg.auth_username)
    return True
