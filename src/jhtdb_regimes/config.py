from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class TaskConfig:
    source: Path
    root: Path
    dataset: str
    variable: str
    grid_shape: tuple[int, int, int]
    domain_length: float
    block_start_ijk: tuple[int, int, int]
    block_shape: tuple[int, int, int]
    halo: tuple[int, int, int]
    time_start: float
    time_end: float
    time_step: float
    time_chunk_size: int
    token_env: str
    use_builtin_testing_token: bool
    max_points_per_query: int
    retries: int
    retry_backoff_seconds: float
    gradient_primary: str
    gradient_audit: str | None
    sigma_grid: float
    support_radius: int
    epsilon_abs: float
    epsilon_rel: float
    cache_path: Path
    raw_path: Path
    derived_path: Path
    reports_path: Path

    @property
    def core_shape(self) -> tuple[int, int, int]:
        return tuple(n - 2 * h for n, h in zip(self.block_shape, self.halo))

    @property
    def point_count(self) -> int:
        return int(np.prod(self.block_shape))

    @property
    def times(self) -> np.ndarray:
        count = int(round((self.time_end - self.time_start) / self.time_step)) + 1
        values = self.time_start + np.arange(count, dtype=np.float64) * self.time_step
        values[-1] = self.time_end
        return values

    def validate(self) -> None:
        if any(n <= 0 for n in self.grid_shape + self.block_shape):
            raise ValueError("grid_shape and block_shape must be positive")
        if len(self.block_shape) != 3 or len(self.halo) != 3:
            raise ValueError("block_shape and halo must have three entries")
        if any(c <= 0 for c in self.core_shape):
            raise ValueError("halo leaves an empty core")
        if any(h != self.support_radius for h in self.halo):
            raise ValueError("this implementation requires halo == support_radius on all axes")
        if self.max_points_per_query > 4000:
            raise ValueError("testing-token safety requires max_points_per_query <= 4000")
        if self.time_step <= 0 or self.time_chunk_size <= 0:
            raise ValueError("time step and chunk size must be positive")
        if self.time_end < self.time_start:
            raise ValueError("time.end must be >= time.start")
        for start, size, total in zip(self.block_start_ijk, self.block_shape, self.grid_shape):
            if start < 0 or start + size > total:
                raise ValueError("baseline block must lie inside the represented global grid")
        if self.sigma_grid <= 0:
            raise ValueError("filter sigma must be positive")


def _tuple3(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a YAML list of three integers")
    return tuple(int(v) for v in value)


def load_config(path: str | Path) -> TaskConfig:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    root = source.parent.parent.resolve()
    time = data["time"]
    api = data["api"]
    gradient = data["gradient"]
    filter_cfg = data["filter"]
    regime = data["regime"]
    paths = data["paths"]

    def resolve(value: str) -> Path:
        return (root / value).resolve()

    cfg = TaskConfig(
        source=source,
        root=root,
        dataset=str(data["dataset"]),
        variable=str(data.get("variable", "velocity")),
        grid_shape=_tuple3(data["grid_shape"], "grid_shape"),
        domain_length=float(data["domain_length"]),
        block_start_ijk=_tuple3(data["block_start_ijk"], "block_start_ijk"),
        block_shape=_tuple3(data["block_shape"], "block_shape"),
        halo=_tuple3(data["halo"], "halo"),
        time_start=float(time["start"]),
        time_end=float(time["end"]),
        time_step=float(time["step"]),
        time_chunk_size=int(time["chunk_size"]),
        token_env=str(api.get("token_env", "JHTDB_TOKEN")),
        use_builtin_testing_token=bool(api.get("use_builtin_testing_token", True)),
        max_points_per_query=int(api.get("max_points_per_query", 4000)),
        retries=int(api.get("retries", 3)),
        retry_backoff_seconds=float(api.get("retry_backoff_seconds", 1.0)),
        gradient_primary=str(gradient.get("primary", "fd8noint")),
        gradient_audit=gradient.get("audit"),
        sigma_grid=float(filter_cfg["sigma_grid"]),
        support_radius=int(filter_cfg["support_radius"]),
        epsilon_abs=float(regime.get("epsilon_abs", 0.0)),
        epsilon_rel=float(regime.get("epsilon_rel", 1e-3)),
        cache_path=resolve(paths["cache"]),
        raw_path=resolve(paths["raw"]),
        derived_path=resolve(paths["derived"]),
        reports_path=resolve(paths["reports"]),
    )
    cfg.validate()
    return cfg

