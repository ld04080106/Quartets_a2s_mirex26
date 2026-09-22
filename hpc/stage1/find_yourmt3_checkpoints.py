#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="List YourMT3 checkpoints under a source snapshot.")
    parser.add_argument("--source_dir", default=os.environ.get("YOURMT3_SOURCE_DIR", "hpc_assets/stage1/sources/YourMT3"))
    parser.add_argument("--contains", default="quartets")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    source_dir = Path(os.path.expandvars(os.path.expanduser(args.source_dir))).resolve()
    roots = [
        source_dir / "amt" / "logs",
        source_dir / "lightning_logs",
    ]
    items = []
    for root in roots:
        if not root.exists():
            continue
        for ckpt in root.rglob("*.ckpt"):
            text = ckpt.as_posix()
            if args.contains and args.contains not in text:
                continue
            stat = ckpt.stat()
            items.append({
                "path": str(ckpt),
                "relative_to_source": ckpt.relative_to(source_dir).as_posix(),
                "bytes": stat.st_size,
                "mtime": stat.st_mtime,
            })
    items.sort(key=lambda item: item["mtime"], reverse=True)
    print(json.dumps({
        "source_dir": str(source_dir),
        "num_checkpoints": len(items),
        "checkpoints": items[: args.limit],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
