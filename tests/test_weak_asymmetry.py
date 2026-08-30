from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import zarr

from jhtdb_pipeline.config import load_config
from jhtdb_pipeline.weak_asymmetry import (
    compute_weak_asymmetry,
    write_weak_asymmetry_artifacts,
)


class WeakAsymmetryTests(unittest.TestCase):
    def _fixture(self, path: Path):
        cfg = replace(
            load_config("configs/pipeline.yaml"),
            grid_shape=(2, 2, 2),
            crop_start=(0, 0, 0),
            crop_shape=(2, 2, 2),
        )
        root = zarr.group()
        pi = np.asarray(
            [1, 2, -3, 4, 5, -6, 7, -8], dtype="<f4"
        ).reshape(2, 2, 2)
        root.create_dataset("pi", data=pi, chunks=(1, 2, 2), dtype="<f4")
        return cfg, root, pi

    def test_positive_negative_split_and_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, root, pi = self._fixture(Path(temporary))
            report = compute_weak_asymmetry(root, cfg)
            self.assertTrue(report["passed"])
            self.assertEqual(report["positive_backscatter"]["sum"], 19.0)
            self.assertEqual(report["negative_forward"]["sum"], -17.0)
            self.assertEqual(report["positive_backscatter"]["count"], 5)
            self.assertEqual(report["negative_forward"]["count"], 3)
            self.assertAlmostEqual(
                report["global"]["pi_mean"],
                float(np.mean(pi, dtype=np.float64)),
            )
            self.assertAlmostEqual(
                report["global"]["pi_rms"],
                float(np.sqrt(np.mean(pi.astype(np.float64) ** 2))),
            )
            self.assertEqual(report["closure"]["residual_sum"], 0.0)

    def test_artifacts_are_json_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cfg, root, _ = self._fixture(path)
            report = compute_weak_asymmetry(root, cfg)
            report_hash = write_weak_asymmetry_artifacts(path, report)
            self.assertEqual(len(report_hash), 64)
            self.assertTrue((path / "weak_asymmetry.json").is_file())
            self.assertTrue((path / "weak_asymmetry.html").is_file())
            json.loads(
                (path / "weak_asymmetry.json").read_text(encoding="utf-8")
            )

    def test_nonfinite_pi_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, root, _ = self._fixture(Path(temporary))
            root["pi"][0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                compute_weak_asymmetry(root, cfg)


if __name__ == "__main__":
    unittest.main()
