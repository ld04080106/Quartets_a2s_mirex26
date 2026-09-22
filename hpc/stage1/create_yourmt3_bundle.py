#!/usr/bin/env python3
"""Download (or reuse) YourMT3 and create a self-verifying transfer bundle."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path


EXCLUDED_PARTS = {".cache", "__pycache__", ".git", "model_output"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def add_regular_file(archive: tarfile.TarFile, source: Path, name: str) -> None:
    """Dereference Hub-cache symlinks and always store a regular file."""
    info = tarfile.TarInfo(name)
    info.size = source.stat().st_size
    info.mtime = 0
    info.mode = 0o644
    with source.open("rb") as handle:
        archive.addfile(info, handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="YourMT3_bundle.tar.gz")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--download_dir", default="downloads/YourMT3",
        help="Hub cache root in cache mode, or snapshot destination in local_dir mode.",
    )
    parser.add_argument(
        "--download_mode", choices=("auto", "cache", "local_dir"), default="auto",
        help="auto uses cache mode on Windows to avoid long .incomplete paths.",
    )
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--disable_xet", action="store_true")
    parser.add_argument("--source_dir", help="Package an existing complete snapshot instead of downloading.")
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()

    if args.source_dir:
        source = Path(args.source_dir).resolve()
        selected_mode = "existing_source"
    else:
        if args.disable_xet:
            # huggingface_hub reads environment variables during import.
            os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SystemExit("Install the downloader locally: python -m pip install -U huggingface_hub") from exc
        download_root = Path(args.download_dir).resolve()
        download_root.mkdir(parents=True, exist_ok=True)
        selected_mode = "cache" if args.download_mode == "auto" and os.name == "nt" else args.download_mode
        if selected_mode == "auto":
            selected_mode = "local_dir"
        try:
            if selected_mode == "cache":
                source = Path(snapshot_download(
                    repo_id="mimbres/YourMT3",
                    repo_type="space",
                    revision=args.revision,
                    cache_dir=str(download_root),
                    force_download=args.force_download,
                    max_workers=args.max_workers,
                )).resolve()
            else:
                source = download_root
                snapshot_download(
                    repo_id="mimbres/YourMT3",
                    repo_type="space",
                    revision=args.revision,
                    local_dir=str(source),
                    force_download=args.force_download,
                    max_workers=args.max_workers,
                )
        except FileNotFoundError as exc:
            if os.name == "nt":
                raise SystemExit(
                    "Hugging Face could not create a temporary file. This is usually a Windows path-length issue. "
                    "Retry with a very short cache root, for example --download_dir ./y --download_mode cache. "
                    "The cache downloader will resume completed files."
                ) from exc
            raise

    required = [source / "model_helper.py", source / "amt" / "src"]
    if not all(path.exists() for path in required):
        raise SystemExit(f"Incomplete YourMT3 snapshot: {source}")
    files = sorted(
        path for path in source.rglob("*")
        if path.is_file() and not any(part in EXCLUDED_PARTS for part in path.relative_to(source).parts)
    )
    if not files:
        raise SystemExit("No files selected for bundle")
    checksums = []
    total_bytes = 0
    for path in files:
        relative = Path("YourMT3") / path.relative_to(source)
        checksums.append(f"{sha256(path)}  {relative.as_posix()}")
        total_bytes += path.stat().st_size
    info = {
        "repo_id": "mimbres/YourMT3",
        "repo_type": "space",
        "upstream_space_url": "https://huggingface.co/spaces/mimbres/YourMT3",
        "upstream_code_url": "https://github.com/mimbres/YourMT3",
        "space_declared_license": "apache-2.0",
        "github_declared_license": "GPL-3.0",
        "requested_revision": args.revision,
        "download_mode": selected_mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "num_files": len(files),
        "uncompressed_bytes": total_bytes,
        "source_dir": str(source),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            add_regular_file(
                archive, path, (Path("YourMT3") / path.relative_to(source)).as_posix()
            )
        add_bytes(archive, "BUNDLE_INFO.json", json.dumps(info, indent=2).encode("utf-8"))
        add_bytes(archive, "SHA256SUMS", ("\n".join(checksums) + "\n").encode("utf-8"))
    bundle_hash = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{bundle_hash}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({**info, "bundle": str(output), "bundle_sha256": bundle_hash}, indent=2))


if __name__ == "__main__":
    main()
