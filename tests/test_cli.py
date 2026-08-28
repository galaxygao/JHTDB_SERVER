from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from jhtdb_pipeline.cli import _run_single_frame, _selected_sigmas


class CliBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SimpleNamespace(sigma_grids=(1.0, 2.0, 3.0))

    def test_configured_sigmas_are_selected_without_override(self) -> None:
        self.assertEqual(_selected_sigmas(self.cfg, None), (1.0, 2.0, 3.0))
        self.assertEqual(_selected_sigmas(self.cfg, 2.5), (2.5,))

    @patch("jhtdb_pipeline.cli.finalize_result")
    @patch("jhtdb_pipeline.cli.process_center")
    @patch("jhtdb_pipeline.cli.validate_snapshot")
    @patch("jhtdb_pipeline.cli.fetch_snapshot")
    @patch("jhtdb_pipeline.cli.doctor")
    def test_single_frame_fetches_once_and_processes_each_sigma(
        self, doctor, fetch, validate, process, finalize
    ) -> None:
        doctor.return_value = {"status": "ok"}
        finalize.side_effect = lambda cfg, frame, sigma: Path(
            f"result_sigma_{sigma:g}"
        )

        results = _run_single_frame(self.cfg, 1, None)

        doctor.assert_called_once_with(self.cfg, 1)
        fetch.assert_called_once_with(self.cfg, 1)
        validate.assert_called_once_with(self.cfg, 1)
        self.assertEqual(
            process.call_args_list,
            [call(self.cfg, 1, 1.0), call(self.cfg, 1, 2.0), call(self.cfg, 1, 3.0)],
        )
        self.assertEqual(finalize.call_args_list, process.call_args_list)
        self.assertEqual(
            results,
            [Path("result_sigma_1"), Path("result_sigma_2"), Path("result_sigma_3")],
        )


if __name__ == "__main__":
    unittest.main()
