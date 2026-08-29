from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from jhtdb_pipeline.cli import _run_single_frame, _selected_sigmas, build_parser


class CliBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SimpleNamespace(sigma_grids=(1.0, 2.0, 3.0))

    def test_configured_sigmas_are_selected_without_override(self) -> None:
        self.assertEqual(_selected_sigmas(self.cfg, None), (1.0, 2.0, 3.0))
        self.assertEqual(_selected_sigmas(self.cfg, 2.5), (2.5,))

    def test_upgrade_result_accepts_frame_and_sigma(self) -> None:
        args = build_parser().parse_args(
            ["upgrade-result", "--time-index", "7", "--sigma-grid", "2.0"]
        )
        self.assertEqual(args.command, "upgrade-result")
        self.assertEqual(args.time_index, 7)
        self.assertEqual(args.sigma_grid, 2.0)

    def test_backfill_and_sbar_qa_commands_accept_frame_and_sigma(self) -> None:
        for command in (
            "backfill-full-fields",
            "backfill-full-regime",
            "compute-cq",
            "qa-sbar",
        ):
            args = build_parser().parse_args(
                [command, "--time-index", "7", "--sigma-grid", "2.0"]
            )
            self.assertEqual(args.command, command)
            self.assertEqual(args.time_index, 7)
            self.assertEqual(args.sigma_grid, 2.0)

    @patch("jhtdb_pipeline.cli.finalize_result")
    @patch("jhtdb_pipeline.cli.process_center")
    @patch("jhtdb_pipeline.cli.validate_snapshot")
    @patch("jhtdb_pipeline.cli.fetch_snapshot")
    @patch("jhtdb_pipeline.cli.doctor")
    @patch("jhtdb_pipeline.cli.reuse_or_backfill_result")
    def test_single_frame_fetches_once_and_processes_each_sigma(
        self, reuse, doctor, fetch, validate, process, finalize
    ) -> None:
        reuse.return_value = None
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

    @patch("jhtdb_pipeline.cli.validate_snapshot")
    @patch("jhtdb_pipeline.cli.fetch_snapshot")
    @patch("jhtdb_pipeline.cli.doctor")
    @patch("jhtdb_pipeline.cli.reuse_or_backfill_result")
    def test_single_frame_fast_upgrades_existing_results_without_fetch(
        self, reuse, doctor, fetch, validate
    ) -> None:
        reuse.side_effect = lambda cfg, frame, sigma: Path(
            f"upgraded_sigma_{sigma:g}"
        )

        results = _run_single_frame(self.cfg, 1, None)

        doctor.assert_not_called()
        fetch.assert_not_called()
        validate.assert_not_called()
        self.assertEqual(
            results,
            [
                Path("upgraded_sigma_1"),
                Path("upgraded_sigma_2"),
                Path("upgraded_sigma_3"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
