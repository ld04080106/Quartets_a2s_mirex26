from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
from a2s.evaluation.kern_validity import validate_kern_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the MIREX pre-computed-kern fallback submission.")
    parser.add_argument("--input_audio", required=True)
    parser.add_argument("--kern_dir", required=True)
    parser.add_argument("--out", required=True, help="Archive base path without extension.")
    args = parser.parse_args()

    audio_source = Path(args.input_audio)
    audio_paths = [audio_source] if audio_source.is_file() else [
        path for path in audio_source.rglob("*") if path.is_file() and path.suffix.lower() == ".flac"
    ]
    expected = {path.stem for path in audio_paths}
    kern_dir = Path(args.kern_dir)
    kern_paths = sorted(kern_dir.glob("*.krn"))
    actual = {path.stem for path in kern_paths}
    if expected != actual:
        raise ValueError(
            f"audio/kern basename mismatch: missing={sorted(expected - actual)[:20]} "
            f"extra={sorted(actual - expected)[:20]}"
        )
    invalid = {}
    for path in kern_paths:
        result = validate_kern_text(path.read_text(encoding="utf-8-sig", errors="replace"))
        if not result.valid:
            invalid[path.name] = result.issues
    if invalid:
        raise ValueError(f"invalid kern files: {dict(list(invalid.items())[:20])}")

    archive_base = Path(args.out).resolve()
    archive_base.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mirex_kern_submission_") as temporary:
        staging = Path(temporary) / "kern_predictions"
        staging.mkdir()
        for path in kern_paths:
            shutil.copy2(path, staging / path.name)
        archive = Path(shutil.make_archive(str(archive_base), "zip", staging))
    hasher = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    checksum = archive.with_name(archive.name + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(f"files={len(kern_paths)}")
    print(archive)
    print(checksum)


if __name__ == "__main__":
    main()
