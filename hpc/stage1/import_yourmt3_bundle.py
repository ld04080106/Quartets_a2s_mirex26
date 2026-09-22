#!/usr/bin/env python3
"""Safely extract and verify a YourMT3 transfer bundle on the cluster."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("--out_dir", required=True, help="Parent directory that will contain YourMT3/.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    out_dir = Path(args.out_dir).resolve()
    source_dir = out_dir / "YourMT3"
    if source_dir.exists():
        if not args.force:
            raise SystemExit(f"Target exists: {source_dir}; use --force to replace it")
        shutil.rmtree(source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (out_dir / member.name).resolve()
            if out_dir != target and out_dir not in target.parents:
                raise SystemExit(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise SystemExit(f"Links are not allowed in bundle: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise SystemExit(f"Unsupported archive member: {member.name}")
        archive.extractall(out_dir)
    checksum_file = out_dir / "SHA256SUMS"
    failures = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = out_dir / relative.strip()
        actual = sha256(path) if path.is_file() else "missing"
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    if failures:
        raise SystemExit(f"Checksum verification failed: {failures[:5]}")
    if not (source_dir / "model_helper.py").exists() or not (source_dir / "amt" / "src").is_dir():
        raise SystemExit("Bundle verification passed but required YourMT3 source files are missing")
    print(f"Verified YourMT3 source: {source_dir}")
    print(f"export YOURMT3_SOURCE_DIR={source_dir}")


if __name__ == "__main__":
    main()
