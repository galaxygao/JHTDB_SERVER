from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import zarr

from jhtdb_pipeline.config import load_config
from jhtdb_pipeline.sbar_qa import compute_sbar_qa, write_sbar_artifacts


def fixture(root_path: Path, *, contaminated: bool = False, zero_pi: bool = False):
    cfg = replace(
        load_config("configs/pipeline.yaml"),
        grid_shape=(4, 4, 4),
        crop_start=(0, 0, 0),
        crop_shape=(4, 4, 4),
    )
    root = zarr.open_group(str(root_path / "result.zarr"), mode="w")
    coordinates = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    work_resolved = 0.01 * coordinates
    pi = np.zeros_like(coordinates) if zero_pi else np.full_like(coordinates, 0.25)
    if contaminated:
        s_bar = np.ones_like(coordinates)
    else:
        s_bar = np.ones_like(coordinates)
        s_bar.reshape(-1)[::2] = -1.0
    work_full = work_resolved - pi + s_bar
    for name, values in {
        "work_full": work_full,
        "work_resolved": work_resolved,
        "pi": pi,
        "s_bar": s_bar,
    }.items():
        root.create_dataset(name, data=values, chunks=(2, 2, 2), dtype="<f4")
    return cfg, root


class SBarQATests(unittest.TestCase):
    def test_two_global_checks_pass_and_artifacts_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cfg, root = fixture(path)
            report = compute_sbar_qa(root, cfg, scope="full_domain")
            self.assertTrue(report["passed"])
            self.assertEqual(report["global_totals"]["s_bar"], 0.0)
            self.assertLess(
                report["metrics"]["identity_relative_residual_rms"]["value"],
                1.0e-7,
            )
            self.assertEqual(set(report["field_rms"]), {
                "s_bar", "pi", "work_resolved", "work_full",
            })
            self.assertNotIn("s_bar_rel_self", report["metrics"])
            self.assertNotIn("global_absolute_totals", report)
            json.dumps(report, allow_nan=False)
            digest = write_sbar_artifacts(path, report)
            self.assertEqual(len(digest), 64)
            self.assertTrue((path / "s_bar_qa.json").is_file())
            self.assertTrue((path / "s_bar_global_totals.html").is_file())

    def test_net_s_bar_contamination_fails_vs_pi_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, root = fixture(Path(temporary), contaminated=True)
            report = compute_sbar_qa(root, cfg, scope="full_domain")
            self.assertFalse(report["passed"])
            self.assertFalse(report["metrics"]["s_bar_vs_pi_net"]["passed"])

    def test_zero_pi_denominator_is_json_safe_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cfg, root = fixture(
                Path(temporary), contaminated=True, zero_pi=True
            )
            report = compute_sbar_qa(root, cfg, scope="full_domain")
            metric = report["metrics"]["s_bar_vs_pi_net"]
            self.assertIsNone(metric["value"])
            self.assertFalse(metric["passed"])
            self.assertIn("denominator is zero", metric["error"])
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
