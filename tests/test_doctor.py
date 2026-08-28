from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jhtdb_pipeline.config import load_config
from jhtdb_pipeline.doctor import ensure_run_record
from jhtdb_pipeline.validation import atomic_json


class DoctorTests(unittest.TestCase):
    def test_expired_scratch_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = replace(
                load_config("configs/pipeline.yaml"),
                run_root=root / "runs",
                state_root=root / "state",
                result_root=root / "results",
            )
            record = ensure_run_record(cfg, 1)
            record["expires_at"] = (
                datetime.now(timezone.utc) - timedelta(seconds=1)
            ).isoformat()
            atomic_json(cfg.run_path(1) / "run.json", record)

            with self.assertRaisesRegex(RuntimeError, "has expired"):
                ensure_run_record(cfg, 1)


if __name__ == "__main__":
    unittest.main()
