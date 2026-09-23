from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_submission_pipeline import _find_metadata  # noqa: E402


FOUR_SPINE_HEADER = "\n".join(
    [
        "**kern\t**kern\t**kern\t**kern",
        "*Icello\t*Iviola\t*Ivioln\t*Ivioln",
        "*M4/4\t*M4/4\t*M4/4\t*M4/4",
        "=1\t=1\t=1\t=1",
    ]
) + "\n"


class RequiredStavesMetadataTest(unittest.TestCase):
    def test_metadata_source_is_required(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _find_metadata(None, "piece")

    def test_matching_four_spine_header_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "piece.krn"
            path.write_text(FOUR_SPINE_HEADER, encoding="utf-8")
            metadata = _find_metadata(Path(temp_dir), "piece")
            self.assertEqual(metadata.kern_indices, [0, 1, 2, 3])

    def test_missing_sample_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                _find_metadata(Path(temp_dir), "piece")

    def test_json_without_staves_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "piece.json"
            path.write_text('{"meter": "4/4"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                _find_metadata(Path(temp_dir), "piece")


if __name__ == "__main__":
    unittest.main()
