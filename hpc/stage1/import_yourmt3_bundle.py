#!/usr/bin/env python3
"""Safely extract and verify a YourMT3 transfer bundle on the cluster."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path


FORBIDDEN_PARTS = {".cache", "__pycache__", ".git", "extras", "tests"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", nargs="?")
    parser.add_argument("--bundle", dest="bundle_option")
    parser.add_argument("--source_dir", default="third_party/YourMT3")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    bundle_value = args.bundle_option or args.bundle
    if not bundle_value:
        parser.error("provide BUNDLE or --bundle BUNDLE")
    bundle = Path(bundle_value).resolve()
    source_dir = Path(args.source_dir).resolve()
    if source_dir.name != "YourMT3":
        parser.error("--source_dir must end in YourMT3 because the bundle has a YourMT3/ root")
    out_dir = source_dir.parent
    if source_dir.exists() and not args.force:
        raise SystemExit(f"Target exists: {source_dir}; use --force to replace it")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            parts = Path(member.name).parts
            if any(part in FORBIDDEN_PARTS for part in parts):
                raise SystemExit(f"Bundle contains excluded or sensitive content: {member.name}")
            if member.name.endswith("/utils/preprocess/preprocess_rnsynth.py"):
                raise SystemExit(f"Bundle contains excluded broken utility: {member.name}")
            target = (out_dir / member.name).resolve()
            if out_dir != target and out_dir not in target.parents:
                raise SystemExit(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise SystemExit(f"Links are not allowed in bundle: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise SystemExit(f"Unsupported archive member: {member.name}")
        with tempfile.TemporaryDirectory(prefix=".yourmt3-import-", dir=out_dir) as temp_name:
            temp_root = Path(temp_name)
            archive.extractall(temp_root)
            checksum_file = temp_root / "SHA256SUMS"
            failures = []
            for line in checksum_file.read_text(encoding="utf-8").splitlines():
                expected, relative = line.split(maxsplit=1)
                path = temp_root / relative.strip()
                actual = sha256(path) if path.is_file() else "missing"
                if actual != expected:
                    failures.append({"path": relative, "expected": expected, "actual": actual})
            if failures:
                raise SystemExit(f"Checksum verification failed: {failures[:5]}")
            candidate = temp_root / "YourMT3"
            if not (candidate / "model_helper.py").exists() or not (candidate / "amt" / "src").is_dir():
                raise SystemExit("Bundle verification passed but required YourMT3 source files are missing")
            if source_dir.exists():
                shutil.rmtree(source_dir)
            shutil.move(str(candidate), str(source_dir))
    print(f"Verified YourMT3 source: {source_dir}")
    print(f"export YOURMT3_SOURCE_DIR={source_dir}")


if __name__ == "__main__":
    main()
