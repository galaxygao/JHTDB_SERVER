from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from jhtdb_pipeline.auth import get_token, has_token, token_source
from jhtdb_pipeline.cli import main
from jhtdb_pipeline.config import load_config


class AuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = replace(load_config("configs/pipeline.yaml"), token_file=None)

    def test_environment_token_is_used_without_being_reported(self) -> None:
        with patch.dict(os.environ, {"JHTDB_TOKEN": "secret-value"}, clear=False):
            self.assertTrue(has_token(self.cfg))
            self.assertEqual(token_source(self.cfg), "environment")
            self.assertEqual(get_token(self.cfg), "secret-value")
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["auth", "status"])
            self.assertEqual(code, 0)
            self.assertNotIn("secret-value", output.getvalue())

    def test_token_file_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "token"
            path.write_text("file-token\n", encoding="utf-8")
            path.chmod(0o600)
            cfg = replace(self.cfg, token_file=path)
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(token_source(cfg), "file")
                self.assertEqual(get_token(cfg), "file-token")

    def test_missing_token_fails_closed(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(has_token(self.cfg))
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                get_token(self.cfg)

    def test_empty_token_file_is_not_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "token"
            path.touch(mode=0o600)
            cfg = replace(self.cfg, token_file=path)
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(has_token(cfg))
                self.assertIsNone(token_source(cfg))


if __name__ == "__main__":
    unittest.main()
