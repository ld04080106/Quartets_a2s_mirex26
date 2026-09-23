from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import ensure_model_assets as assets  # noqa: E402


class _InteractiveInput:
    def isatty(self) -> bool:
        return True


class ModelAssetDownloadTest(unittest.TestCase):
    def test_interactive_confirmation_accepts_enter_and_y(self) -> None:
        specs = [{"name": "model", "path": Path("model.pt"), "bytes": 1024}]
        with patch.object(assets.sys, "stdin", _InteractiveInput()):
            with patch("builtins.input", return_value=""):
                self.assertTrue(assets._confirm_download(specs, assume_yes=False))
            with patch("builtins.input", return_value="y"):
                self.assertTrue(assets._confirm_download(specs, assume_yes=False))
            with patch("builtins.input", return_value="n"):
                self.assertFalse(assets._confirm_download(specs, assume_yes=False))

    def test_download_reports_progress_and_verifies_file(self) -> None:
        payload = b"quartets-a2s-test" * 1024
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.bin"
            target = root / "installed" / "model.bin"
            source.write_bytes(payload)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                assets._download(
                    "test_model",
                    source.resolve().as_uri(),
                    target,
                    expected,
                    len(payload),
                    timeout=10,
                    retries=1,
                )
            self.assertEqual(target.read_bytes(), payload)
            self.assertIn("100%", output.getvalue())
            self.assertIn("Verifying SHA-256", output.getvalue())


if __name__ == "__main__":
    unittest.main()
