from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _tuple3(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


@dataclass(frozen=True)
class PipelineConfig:
    dataset: str
    variable: str
    grid_shape: tuple[int, int, int]
    domain_length: float
    stored_time_step: float
    auth_backend: str
    auth_service: str
    auth_username: str
    download_backend: str
    tile_shape: tuple[int, int, int]
    retries: int
    backoff_seconds: float
    request_cooldown_seconds: float
    storage_root: Path
    compression_level: int
    safety_free_space_gib: float
    seam_width: int
    divergence_relative_rms_max: float
    divergence_relative_max_max: float
    sigma_grid: float
    epsilon_abs: float
    epsilon_rel: float
    fft_slab_width: int
    keep_intermediates: bool

    @property
    def raw_store_path(self) -> Path:
        return self.storage_root / "raw" / "velocity.zarr"

    @property
    def catalog_path(self) -> Path:
        return self.storage_root / "catalog.sqlite"

    @property
    def manifest_path(self) -> Path:
        return self.storage_root / "manifests"

    @property
    def qa_path(self) -> Path:
        return self.storage_root / "qa"

    @property
    def derived_store_path(self) -> Path:
        return self.storage_root / "derived" / "physics.zarr"

    @property
    def gradient_store_path(self) -> Path:
        return self.storage_root / "derived" / "gradients.zarr"

    @property
    def filtered_store_path(self) -> Path:
        return self.storage_root / "derived" / "filtered.zarr"

    @property
    def scratch_path(self) -> Path:
        return self.storage_root / "scratch"

    @property
    def bytes_per_snapshot(self) -> int:
        nx, ny, nz = self.grid_shape
        return nx * ny * nz * 3 * 4

    def physical_time(self, time_index: int) -> float:
        if time_index < 1:
            raise ValueError("JHTDB cutout time_index must be >= 1")
        return (time_index - 1) * self.stored_time_step

    def validate(self) -> None:
        if self.dataset != "isotropic1024coarse":
            raise ValueError("this project currently supports only isotropic1024coarse")
        if self.variable != "velocity":
            raise ValueError("only velocity may be downloaded")
        if self.grid_shape != (1024, 1024, 1024):
            raise ValueError("isotropic1024coarse must use the full 1024^3 grid")
        if self.auth_backend != "windows_credential_manager":
            raise ValueError("local mode requires windows_credential_manager")
        if self.download_backend != "local":
            raise ValueError("this build implements the convenient local backend only")
        if any(n % tile != 0 for n, tile in zip(self.grid_shape, self.tile_shape)):
            raise ValueError("tile_shape must divide grid_shape exactly")
        tile_bytes = self.tile_shape[0] * self.tile_shape[1] * self.tile_shape[2] * 3 * 4
        if tile_bytes > 3 * 1024**3:
            raise ValueError("a local GetCutout tile may not exceed 3 GiB")
        if self.retries < 1 or self.backoff_seconds < 0 or self.request_cooldown_seconds < 0:
            raise ValueError("download retry/cooldown settings are invalid")
        if not self.storage_root.is_absolute():
            raise ValueError("storage.root must be an absolute path outside the repository")
        if not (0 <= self.compression_level <= 9):
            raise ValueError("compression_level must be between 0 and 9")
        if self.seam_width < 1 or self.seam_width * 2 >= min(self.grid_shape):
            raise ValueError("seam_width is invalid")
        if self.divergence_relative_rms_max <= 0:
            raise ValueError("divergence_relative_rms_max must be positive")
        if self.divergence_relative_max_max <= 0:
            raise ValueError("divergence_relative_max_max must be positive")
        if self.sigma_grid <= 0 or self.epsilon_abs < 0 or self.epsilon_rel < 0:
            raise ValueError("physics parameters are invalid")
        if self.fft_slab_width < 1:
            raise ValueError("fft_slab_width must be positive")


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    auth = data.get("auth", {})
    download = data.get("download", {})
    storage = data.get("storage", {})
    validation = data.get("validation", {})
    physics = data.get("physics", {})
    cfg = PipelineConfig(
        dataset=str(data.get("dataset", "isotropic1024coarse")),
        variable=str(data.get("variable", "velocity")),
        grid_shape=_tuple3(data.get("grid_shape", [1024, 1024, 1024]), "grid_shape"),
        domain_length=float(data.get("domain_length", 2.0 * 3.141592653589793)),
        stored_time_step=float(data.get("stored_time_step", 0.002)),
        auth_backend=str(auth.get("backend", "windows_credential_manager")),
        auth_service=str(auth.get("service", "jhtdb_pipeline")),
        auth_username=str(auth.get("username", "default")),
        download_backend=str(download.get("backend", "local")),
        tile_shape=_tuple3(download.get("tile_shape", [128, 128, 128]), "download.tile_shape"),
        retries=int(download.get("retries", 5)),
        backoff_seconds=float(download.get("backoff_seconds", 2.0)),
        request_cooldown_seconds=float(download.get("request_cooldown_seconds", 0.25)),
        storage_root=Path(str(storage.get("root", ""))).expanduser(),
        compression_level=int(storage.get("compression_level", 3)),
        safety_free_space_gib=float(storage.get("safety_free_space_gib", 40.0)),
        seam_width=int(validation.get("seam_width", 4)),
        divergence_relative_rms_max=float(
            validation.get("divergence_relative_rms_max", 1.0e-4)
        ),
        divergence_relative_max_max=float(
            validation.get("divergence_relative_max_max", 1.0e-3)
        ),
        sigma_grid=float(physics.get("sigma_grid", 1.0)),
        epsilon_abs=float(physics.get("epsilon_abs", 0.0)),
        epsilon_rel=float(physics.get("epsilon_rel", 0.001)),
        fft_slab_width=int(physics.get("fft_slab_width", 16)),
        keep_intermediates=bool(physics.get("keep_intermediates", False)),
    )
    cfg.validate()
    return cfg
