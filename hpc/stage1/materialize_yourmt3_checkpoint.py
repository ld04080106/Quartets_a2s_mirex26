#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _source_dir(value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path.resolve()


def _latest_checkpoint(source_dir: Path, experiment_id: str | None) -> Path:
    roots = [source_dir / "amt" / "logs", source_dir / "lightning_logs"]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(path for path in root.rglob("*.ckpt") if path.is_file())
    if experiment_id:
        matched = [path for path in candidates if experiment_id in path.as_posix()]
        if matched:
            candidates = matched
    if not candidates:
        raise SystemExit(f"no checkpoint found under {source_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy latest YourMT3 Lightning checkpoint into amt/logs/<project>/<experiment>/checkpoints/last.ckpt.")
    parser.add_argument("--source_dir", default=os.environ.get("YOURMT3_SOURCE_DIR", "hpc_assets/stage1/sources/YourMT3"))
    parser.add_argument("--project", default="2026_quartets")
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--checkpoint_name", default="last.ckpt")
    parser.add_argument("--from_checkpoint", help="Optional explicit checkpoint to copy instead of auto-selecting latest.")
    args = parser.parse_args()

    source_dir = _source_dir(args.source_dir)
    source = Path(args.from_checkpoint).resolve() if args.from_checkpoint else _latest_checkpoint(source_dir, args.experiment_id)
    if not source.is_file():
        raise SystemExit(f"checkpoint does not exist: {source}")
    target = source_dir / "amt" / "logs" / args.project / args.experiment_id / "checkpoints" / args.checkpoint_name
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    print(f"materialized checkpoint:\n  source: {source}\n  target: {target}")


if __name__ == "__main__":
    main()
