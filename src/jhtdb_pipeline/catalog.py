from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .planning import Tile


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Catalog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                dataset TEXT NOT NULL,
                time_index INTEGER NOT NULL,
                physical_time REAL NOT NULL,
                status TEXT NOT NULL,
                expected_tiles INTEGER NOT NULL,
                manifest_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (dataset, time_index)
            );
            CREATE TABLE IF NOT EXISTS tiles (
                dataset TEXT NOT NULL,
                time_index INTEGER NOT NULL,
                tile_key TEXT NOT NULL,
                x0 INTEGER NOT NULL,
                y0 INTEGER NOT NULL,
                z0 INTEGER NOT NULL,
                nx INTEGER NOT NULL,
                ny INTEGER NOT NULL,
                nz INTEGER NOT NULL,
                status TEXT NOT NULL,
                sha256 TEXT,
                byte_count INTEGER,
                attempts INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (dataset, time_index, tile_key),
                FOREIGN KEY (dataset, time_index) REFERENCES snapshots(dataset, time_index)
            );
            CREATE TABLE IF NOT EXISTS gradient_fields (
                dataset TEXT NOT NULL,
                time_index INTEGER NOT NULL,
                velocity_component INTEGER NOT NULL,
                derivative_component INTEGER NOT NULL,
                input_manifest_hash TEXT,
                status TEXT NOT NULL,
                sha256 TEXT,
                byte_count INTEGER,
                attempts INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (dataset, time_index, velocity_component, derivative_component),
                FOREIGN KEY (dataset, time_index) REFERENCES snapshots(dataset, time_index)
            );
            """
        )
        gradient_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(gradient_fields)").fetchall()
        }
        if "input_manifest_hash" not in gradient_columns:
            self.connection.execute(
                "ALTER TABLE gradient_fields ADD COLUMN input_manifest_hash TEXT"
            )
        self.connection.commit()

    def plan_snapshot(self, dataset: str, time_index: int, physical_time: float, tiles: Iterable[Tile]) -> None:
        tile_list = list(tiles)
        now = _now()
        with self.connection:
            self.connection.execute(
                """INSERT INTO snapshots(dataset,time_index,physical_time,status,expected_tiles,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(dataset,time_index) DO UPDATE SET
                     physical_time=excluded.physical_time,
                     expected_tiles=excluded.expected_tiles,
                     updated_at=excluded.updated_at""",
                (dataset, time_index, physical_time, "planned", len(tile_list), now, now),
            )
            for tile in tile_list:
                self.connection.execute(
                    """INSERT INTO tiles(dataset,time_index,tile_key,x0,y0,z0,nx,ny,nz,status,updated_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(dataset,time_index,tile_key) DO NOTHING""",
                    (dataset, time_index, tile.key, tile.x0, tile.y0, tile.z0,
                     tile.nx, tile.ny, tile.nz, "planned", now),
                )

    def tile(self, dataset: str, time_index: int, key: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM tiles WHERE dataset=? AND time_index=? AND tile_key=?",
            (dataset, time_index, key),
        ).fetchone()

    def mark_attempt(self, dataset: str, time_index: int, key: str) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE tiles SET status='downloading', attempts=attempts+1, updated_at=?
                   WHERE dataset=? AND time_index=? AND tile_key=?""",
                (_now(), dataset, time_index, key),
            )

    def mark_verified(self, dataset: str, time_index: int, key: str, sha256: str, byte_count: int) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE tiles SET status='verified',sha256=?,byte_count=?,updated_at=?
                   WHERE dataset=? AND time_index=? AND tile_key=?""",
                (sha256, byte_count, _now(), dataset, time_index, key),
            )

    def set_snapshot_status(self, dataset: str, time_index: int, status: str, manifest_hash: str | None = None) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE snapshots SET status=?,manifest_hash=COALESCE(?,manifest_hash),updated_at=?
                   WHERE dataset=? AND time_index=?""",
                (status, manifest_hash, _now(), dataset, time_index),
            )

    def snapshot(self, dataset: str, time_index: int) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM snapshots WHERE dataset=? AND time_index=?", (dataset, time_index)
        ).fetchone()

    def snapshots(self, dataset: str) -> list[sqlite3.Row]:
        return list(self.connection.execute(
            "SELECT * FROM snapshots WHERE dataset=? ORDER BY time_index", (dataset,)
        ).fetchall())

    def tiles(self, dataset: str, time_index: int) -> list[sqlite3.Row]:
        return list(self.connection.execute(
            "SELECT * FROM tiles WHERE dataset=? AND time_index=? ORDER BY z0,y0,x0",
            (dataset, time_index),
        ).fetchall())

    def tile_progress(self, dataset: str, time_index: int) -> dict[str, int]:
        rows = self.connection.execute(
            """SELECT status, COUNT(*) AS count FROM tiles
               WHERE dataset=? AND time_index=? GROUP BY status""",
            (dataset, time_index),
        ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        counts["total"] = sum(counts.values())
        return counts

    def plan_gradient_fields(
        self,
        dataset: str,
        time_index: int,
        input_manifest_hash: str,
        *,
        adopt_unbound_verified: bool = False,
    ) -> None:
        if not input_manifest_hash:
            raise ValueError("gradient input manifest hash cannot be empty")
        now = _now()
        with self.connection:
            for velocity_component in range(3):
                for derivative_component in range(3):
                    existing = self.gradient_field(
                        dataset, time_index, velocity_component, derivative_component
                    )
                    if existing is None:
                        self.connection.execute(
                            """INSERT INTO gradient_fields(
                                   dataset,time_index,velocity_component,derivative_component,
                                   input_manifest_hash,status,updated_at
                               ) VALUES(?,?,?,?,?,?,?)""",
                            (
                                dataset,
                                time_index,
                                velocity_component,
                                derivative_component,
                                input_manifest_hash,
                                "planned",
                                now,
                            ),
                        )
                    elif (
                        existing["input_manifest_hash"] is None
                        and adopt_unbound_verified
                        and existing["status"] == "verified"
                        and existing["sha256"]
                    ):
                        self.connection.execute(
                            """UPDATE gradient_fields
                               SET input_manifest_hash=?,updated_at=?
                               WHERE dataset=? AND time_index=?
                                 AND velocity_component=? AND derivative_component=?""",
                            (
                                input_manifest_hash,
                                now,
                                dataset,
                                time_index,
                                velocity_component,
                                derivative_component,
                            ),
                        )
                    elif existing["input_manifest_hash"] != input_manifest_hash:
                        self.connection.execute(
                            """UPDATE gradient_fields
                               SET input_manifest_hash=?,status='planned',sha256=NULL,
                                   byte_count=NULL,attempts=0,updated_at=?
                               WHERE dataset=? AND time_index=?
                                 AND velocity_component=? AND derivative_component=?""",
                            (
                                input_manifest_hash,
                                now,
                                dataset,
                                time_index,
                                velocity_component,
                                derivative_component,
                            ),
                        )

    def gradient_field(
        self, dataset: str, time_index: int, velocity_component: int, derivative_component: int
    ) -> sqlite3.Row | None:
        return self.connection.execute(
            """SELECT * FROM gradient_fields
               WHERE dataset=? AND time_index=? AND velocity_component=? AND derivative_component=?""",
            (dataset, time_index, velocity_component, derivative_component),
        ).fetchone()

    def mark_gradient_attempt(
        self, dataset: str, time_index: int, velocity_component: int, derivative_component: int
    ) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE gradient_fields
                   SET status='computing',attempts=attempts+1,updated_at=?
                   WHERE dataset=? AND time_index=? AND velocity_component=? AND derivative_component=?""",
                (_now(), dataset, time_index, velocity_component, derivative_component),
            )

    def mark_gradient_verified(
        self,
        dataset: str,
        time_index: int,
        velocity_component: int,
        derivative_component: int,
        sha256: str,
        byte_count: int,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE gradient_fields
                   SET status='verified',sha256=?,byte_count=?,updated_at=?
                   WHERE dataset=? AND time_index=? AND velocity_component=? AND derivative_component=?""",
                (
                    sha256,
                    byte_count,
                    _now(),
                    dataset,
                    time_index,
                    velocity_component,
                    derivative_component,
                ),
            )

    def gradient_progress(self, dataset: str, time_index: int) -> dict[str, int]:
        rows = self.connection.execute(
            """SELECT status,COUNT(*) AS count FROM gradient_fields
               WHERE dataset=? AND time_index=? GROUP BY status""",
            (dataset, time_index),
        ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        counts["total"] = sum(counts.values())
        return counts
