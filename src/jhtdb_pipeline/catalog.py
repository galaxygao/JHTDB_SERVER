from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .planning import Tile


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Catalog:
    """Small persistent ledger for resumable JHTDB tile acquisition."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
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
                FOREIGN KEY (dataset, time_index)
                    REFERENCES snapshots(dataset, time_index)
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def plan_snapshot(
        self,
        dataset: str,
        time_index: int,
        physical_time: float,
        tiles: Iterable[Tile],
    ) -> None:
        tile_list = list(tiles)
        now = _now()
        with self.connection:
            self.connection.execute(
                """INSERT INTO snapshots(
                       dataset,time_index,physical_time,status,expected_tiles,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(dataset,time_index) DO UPDATE SET
                       physical_time=excluded.physical_time,
                       expected_tiles=excluded.expected_tiles,
                       updated_at=excluded.updated_at""",
                (dataset, time_index, physical_time, "planned", len(tile_list), now, now),
            )
            for tile in tile_list:
                self.connection.execute(
                    """INSERT INTO tiles(
                           dataset,time_index,tile_key,x0,y0,z0,nx,ny,nz,status,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(dataset,time_index,tile_key) DO NOTHING""",
                    (
                        dataset, time_index, tile.key, tile.x0, tile.y0, tile.z0,
                        tile.nx, tile.ny, tile.nz, "planned", now,
                    ),
                )

    def tile(self, dataset: str, time_index: int, key: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM tiles WHERE dataset=? AND time_index=? AND tile_key=?",
            (dataset, time_index, key),
        ).fetchone()

    def mark_attempt(self, dataset: str, time_index: int, key: str) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE tiles
                   SET status='fetching', attempts=attempts+1, updated_at=?
                   WHERE dataset=? AND time_index=? AND tile_key=?""",
                (_now(), dataset, time_index, key),
            )

    def mark_verified(
        self,
        dataset: str,
        time_index: int,
        key: str,
        sha256: str,
        byte_count: int,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE tiles
                   SET status='verified', sha256=?, byte_count=?, updated_at=?
                   WHERE dataset=? AND time_index=? AND tile_key=?""",
                (sha256, byte_count, _now(), dataset, time_index, key),
            )

    def set_snapshot_status(
        self,
        dataset: str,
        time_index: int,
        status: str,
        manifest_hash: str | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """UPDATE snapshots
                   SET status=?, manifest_hash=COALESCE(?,manifest_hash), updated_at=?
                   WHERE dataset=? AND time_index=?""",
                (status, manifest_hash, _now(), dataset, time_index),
            )

    def snapshot(self, dataset: str, time_index: int) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM snapshots WHERE dataset=? AND time_index=?",
            (dataset, time_index),
        ).fetchone()

    def snapshots(self, dataset: str) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                "SELECT * FROM snapshots WHERE dataset=? ORDER BY time_index", (dataset,)
            ).fetchall()
        )

    def tiles(self, dataset: str, time_index: int) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """SELECT * FROM tiles
                   WHERE dataset=? AND time_index=? ORDER BY z0,y0,x0""",
                (dataset, time_index),
            ).fetchall()
        )

    def tile_progress(self, dataset: str, time_index: int) -> dict[str, int]:
        rows = self.connection.execute(
            """SELECT status, COUNT(*) AS count FROM tiles
               WHERE dataset=? AND time_index=? GROUP BY status""",
            (dataset, time_index),
        ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        counts["total"] = sum(counts.values())
        return counts
