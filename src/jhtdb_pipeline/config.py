from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


def _tuple3(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _path(value: Any, name: str) -> Path:
    text = os.path.expandvars(str(value)).strip()
    if not text or "<username>" in text:
        raise ValueError(f"{name} must be configured for the SciServer account")
    return Path(text)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(unknown)}")


def sigma_tag(value: float) -> str:
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class PipelineConfig:
    dataset: str
    variable: str
    grid_shape: tuple[int, int, int]
    domain_length: float
    stored_time_step: float
    state_root: Path
    run_root: Path
    result_root: Path
    token_file: Path | None
    tile_shape: tuple[int, int, int]
    retries: int
    backoff_seconds: float
    request_cooldown_seconds: float
    compression_level: int
    persistent_capacity_gb_observed: float
    persistent_safety_reserve_gib: float
    scratch_safety_reserve_gib: float
    scratch_retention_hours: int
    crop_start: tuple[int, int, int]
    crop_shape: tuple[int, int, int]
    divergence_relative_rms_max: float
    divergence_relative_max_max: float
    sigma_grid: float
    epsilon_abs: float
    epsilon_rel: float
    fft_slab_width: int
    cleanup_scratch_on_success: bool

    @property
    def catalog_path(self) -> Path:
        return self.state_root / "catalog.sqlite"

    @property
    def manifest_path(self) -> Path:
        return self.state_root / "manifests"

    @property
    def qa_path(self) -> Path:
        return self.state_root / "qa"

    @property
    def lock_path(self) -> Path:
        return self.state_root / "locks"

    def run_path(self, time_index: int) -> Path:
        self.physical_time(time_index)
        return self.run_root / f"t{time_index:06d}"

    def raw_store_path(self, time_index: int) -> Path:
        return self.run_path(time_index) / "velocity_cache.zarr"

    def workspace_path(self, time_index: int, sigma_grid: float | None = None) -> Path:
        sigma = self.sigma_grid if sigma_grid is None else sigma_grid
        return self.run_path(time_index) / f"work_sigma_{sigma_tag(sigma)}"

    def result_id(self, time_index: int, sigma_grid: float | None = None) -> str:
        sigma = self.sigma_grid if sigma_grid is None else sigma_grid
        return f"t{time_index:06d}_sigma_{sigma_tag(sigma)}"

    def staging_result_path(self, time_index: int, sigma_grid: float | None = None) -> Path:
        return self.result_root / ".staging" / self.result_id(time_index, sigma_grid)

    def result_path(self, time_index: int, sigma_grid: float | None = None) -> Path:
        return self.result_root / self.result_id(time_index, sigma_grid)

    @property
    def crop_slices_zyx(self) -> tuple[slice, slice, slice]:
        x0, y0, z0 = self.crop_start
        nx, ny, nz = self.crop_shape
        return (
            slice(z0, z0 + nz),
            slice(y0, y0 + ny),
            slice(x0, x0 + nx),
        )

    @property
    def result_shape_zyx(self) -> tuple[int, int, int]:
        nx, ny, nz = self.crop_shape
        return nz, ny, nx

    @property
    def bytes_per_snapshot(self) -> int:
        nx, ny, nz = self.grid_shape
        return nx * ny * nz * 3 * 4

    @property
    def result_uncompressed_bytes(self) -> int:
        points = int(self.crop_shape[0] * self.crop_shape[1] * self.crop_shape[2])
        return points * (3 + 9 + 3 + 9 + 1 + 1) * 4 + points

    def physical_time(self, time_index: int) -> float:
        if time_index < 1:
            raise ValueError("JHTDB cutout time_index must be >= 1")
        return (time_index - 1) * self.stored_time_step

    def with_sigma(self, sigma_grid: float) -> "PipelineConfig":
        from dataclasses import replace

        if sigma_grid <= 0:
            raise ValueError("sigma_grid must be positive")
        return replace(self, sigma_grid=float(sigma_grid))

    def validate(self) -> None:
        if self.dataset != "isotropic1024coarse":
            raise ValueError("only isotropic1024coarse is supported")
        if self.variable != "velocity":
            raise ValueError("only velocity may be fetched")
        if self.grid_shape != (1024, 1024, 1024):
            raise ValueError("isotropic1024coarse must use the complete 1024^3 grid")
        if any(n % tile != 0 for n, tile in zip(self.grid_shape, self.tile_shape)):
            raise ValueError("tile_shape must divide grid_shape exactly")
        if self.tile_shape != (128, 128, 128):
            raise ValueError("the first SciServer implementation requires 128^3 tiles")
        if self.retries < 1 or self.backoff_seconds < 0 or self.request_cooldown_seconds < 0:
            raise ValueError("JHTDB retry settings are invalid")
        if not 0 <= self.compression_level <= 9:
            raise ValueError("compression_level must be between 0 and 9")
        if self.persistent_capacity_gb_observed <= 0:
            raise ValueError("persistent_capacity_gb_observed must be positive")
        if self.persistent_safety_reserve_gib < 0 or self.scratch_safety_reserve_gib < 0:
            raise ValueError("storage safety reserves cannot be negative")
        if self.scratch_retention_hours <= 0:
            raise ValueError("scratch_retention_hours must be positive")
        for start, size, full in zip(self.crop_start, self.crop_shape, self.grid_shape):
            if start < 0 or start + size > full:
                raise ValueError("crop lies outside the complete periodic grid")
        if self.crop_start != (256, 256, 256) or self.crop_shape != (512, 512, 512):
            raise ValueError("the production crop must be [256:768)^3")
        if self.divergence_relative_rms_max <= 0 or self.divergence_relative_max_max <= 0:
            raise ValueError("divergence tolerances must be positive")
        if self.sigma_grid <= 0 or self.epsilon_abs < 0 or self.epsilon_rel < 0:
            raise ValueError("physics parameters are invalid")
        if self.fft_slab_width < 1:
            raise ValueError("fft_slab_width must be positive")


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError("configuration root must be a mapping")

    allowed_top = {
        "dataset", "variable", "grid_shape", "domain_length", "stored_time_step",
        "platform", "auth", "jhtdb", "storage", "validation", "physics",
    }
    _reject_unknown(data, allowed_top, "top-level")
    platform = _mapping(data.get("platform"), "platform")
    auth = _mapping(data.get("auth"), "auth")
    jhtdb = _mapping(data.get("jhtdb"), "jhtdb")
    storage = _mapping(data.get("storage"), "storage")
    validation = _mapping(data.get("validation"), "validation")
    physics = _mapping(data.get("physics"), "physics")
    _reject_unknown(platform, {"state_root", "run_root", "result_root", "scratch_retention_hours"}, "platform")
    _reject_unknown(auth, {"token_file"}, "auth")
    _reject_unknown(jhtdb, {"tile_shape", "retries", "backoff_seconds", "request_cooldown_seconds"}, "jhtdb")
    _reject_unknown(storage, {"compression_level", "persistent_capacity_gb_observed", "persistent_safety_reserve_gib", "scratch_safety_reserve_gib"}, "storage")
    _reject_unknown(validation, {"divergence_relative_rms_max", "divergence_relative_max_max"}, "validation")
    _reject_unknown(physics, {"sigma_grid", "crop_start", "crop_shape", "epsilon_abs", "epsilon_rel", "fft_slab_width", "cleanup_scratch_on_success"}, "physics")

    token_value = auth.get("token_file")
    cfg = PipelineConfig(
        dataset=str(data.get("dataset", "isotropic1024coarse")),
        variable=str(data.get("variable", "velocity")),
        grid_shape=_tuple3(data.get("grid_shape", [1024, 1024, 1024]), "grid_shape"),
        domain_length=float(data.get("domain_length", 2.0 * 3.141592653589793)),
        stored_time_step=float(data.get("stored_time_step", 0.002)),
        state_root=_path(platform.get("state_root", ""), "platform.state_root"),
        run_root=_path(platform.get("run_root", ""), "platform.run_root"),
        result_root=_path(platform.get("result_root", ""), "platform.result_root"),
        token_file=_path(token_value, "auth.token_file") if token_value else None,
        tile_shape=_tuple3(jhtdb.get("tile_shape", [128, 128, 128]), "jhtdb.tile_shape"),
        retries=int(jhtdb.get("retries", 5)),
        backoff_seconds=float(jhtdb.get("backoff_seconds", 2.0)),
        request_cooldown_seconds=float(jhtdb.get("request_cooldown_seconds", 0.25)),
        compression_level=int(storage.get("compression_level", 3)),
        persistent_capacity_gb_observed=float(storage.get("persistent_capacity_gb_observed", 100.0)),
        persistent_safety_reserve_gib=float(storage.get("persistent_safety_reserve_gib", 15.0)),
        scratch_safety_reserve_gib=float(storage.get("scratch_safety_reserve_gib", 16.0)),
        scratch_retention_hours=int(platform.get("scratch_retention_hours", 72)),
        crop_start=_tuple3(physics.get("crop_start", [256, 256, 256]), "physics.crop_start"),
        crop_shape=_tuple3(physics.get("crop_shape", [512, 512, 512]), "physics.crop_shape"),
        divergence_relative_rms_max=float(validation.get("divergence_relative_rms_max", 1.0e-4)),
        divergence_relative_max_max=float(validation.get("divergence_relative_max_max", 1.0e-3)),
        sigma_grid=float(physics.get("sigma_grid", 1.0)),
        epsilon_abs=float(physics.get("epsilon_abs", 0.0)),
        epsilon_rel=float(physics.get("epsilon_rel", 0.001)),
        fft_slab_width=int(physics.get("fft_slab_width", 4)),
        cleanup_scratch_on_success=bool(physics.get("cleanup_scratch_on_success", True)),
    )
    cfg.validate()
    return cfg
