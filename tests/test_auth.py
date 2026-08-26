from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from jhtdb_pipeline.auth import set_token
from jhtdb_pipeline.cli import main
from jhtdb_pipeline.config import load_config


class AuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config("configs/pipeline.yaml")

    @patch("jhtdb_pipeline.auth.keyring.set_password")
    @patch("builtins.input", return_value="plain-token")
    def test_set_token_uses_visible_input(self, mock_input, mock_set_password) -> None:
        set_token(self.cfg)
        mock_input.assert_called_once_with("JHTDB token: ")
        mock_set_password.assert_called_once_with(
            self.cfg.auth_service, self.cfg.auth_username, "plain-token"
        )

    @patch("jhtdb_pipeline.cli.get_token", return_value="plain-token")
    @patch("jhtdb_pipeline.cli.has_token", return_value=True)
    def test_auth_status_can_show_token(self, _mock_has_token, _mock_get_token) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["auth", "status", "--show-token"])
        self.assertEqual(exit_code, 0)
        self.assertIn('"token": "plain-token"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
