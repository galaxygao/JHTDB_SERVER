from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import zarr

from jhtdb_pipeline.config import load_config
from jhtdb_pipeline.cq import compute_cq, write_cq_artifacts


class CqTests(unittest.TestCase):
    def _fixture(self, root_path: Path):
        cfg = replace(
            load_config("configs/pipeline.yaml"),
            grid_shape=(2, 2, 2),
            crop_start=(0, 0, 0),
            crop_shape=(2, 2, 2),
            state_root=root_path / "state",
            run_root=root_path / "runs",
            result_root=root_path / "results",
        )
        root = zarr.group()
        pi = np.asarray([1, 2, -3, 4, 5, -6, 7, -8], dtype="<f4").reshape(2, 2, 2)
        work_full = np.asarray([1, 1, 1, 1, -1, -1, -1, -1], dtype="<f4").reshape(2, 2, 2)
        work_resolved = np.asarray([1, 1, -1, -1, 1, 1, -1, -1], dtype="<f4").reshape(2, 2, 2)
        root.create_dataset("pi", data=pi, chunks=(1, 2, 2), dtype="<f4")
        root.create_dataset("work_full", data=work_full, chunks=(1, 2, 2), dtype="<f4")
        root.create_dataset("work_resolved", data=work_resolved, chunks=(1, 2, 2), dtype="<f4")
        return cfg, root, pi, work_full, work_resolved

    def test_four_quadrant_contributions_close_to_mean_pi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, root, pi, _, _ = self._fixture(Path(temporary))
            report = compute_cq(root, cfg)

            self.assertTrue(report["passed"])
            self.assertAlmostEqual(
                sum(item["stored_cq"] for item in report["regimes"].values()),
                float(np.mean(pi, dtype=np.float64)),
            )
            self.assertEqual(set(report["regimes"]), {"Q1", "Q2", "Q3", "Q4"})
            self.assertTrue(all(item["count"] == 2 for item in report["regimes"].values()))
            self.assertAlmostEqual(report["regimes"]["Q1"]["stored_pi_sum"], 3.0)
            self.assertEqual(report["partition_check"]["sum_quadrant_counts"], 8)
            self.assertTrue(report["partition_check"]["coverage_passed"])
            self.assertTrue(report["partition_check"]["flux_passed"])
            self.assertTrue(report["weak_asymmetry"]["passed"])
            self.assertEqual(
                report["weak_asymmetry"]["positive_backscatter"]["sum"],
                19.0,
            )
            for item in report["regimes"].values():
                self.assertEqual(item["les_forward_cq"], -item["stored_cq"])
            self.assertLessEqual(
                report["partition_check"]["relative_to_sum_abs_pi"],
                cfg.cq_partition_relative_max,
            )

    def test_empty_quadrants_are_json_safe_and_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cfg, root, _, _, _ = self._fixture(path)
            root["work_full"][:] = 1.0
            root["work_resolved"][:] = 1.0
            report = compute_cq(root, cfg)
            self.assertIsNone(
                report["regimes"]["Q4"]["stored_conditional_mean_pi"]
            )
            report_hash = write_cq_artifacts(path, report)
            self.assertTrue(report_hash)
            self.assertTrue((path / "cq.json").is_file())
            self.assertTrue((path / "cq.html").is_file())
            json.loads((path / "cq.json").read_text(encoding="utf-8"))

    def test_nonfinite_work_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, root, _, _, _ = self._fixture(Path(temporary))
            root["work_full"][0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                compute_cq(root, cfg)


if __name__ == "__main__":
    unittest.main()
