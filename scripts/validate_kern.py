from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from a2s.evaluation.kern_validity import validate_kern_text
from a2s.utils.config import load_config
from a2s.utils.json_io import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_midi_pred_events_long_v2_infer.yaml")
    parser.add_argument("path", nargs="?")
    parser.add_argument("--input_dir", help="Compatibility alias for a directory path.")
    parser.add_argument("--out_json")
    args = parser.parse_args()
    load_config(args.config)
    input_path = Path(args.input_dir or args.path or "")
    if not str(input_path):
        parser.error("path or --input_dir is required")
    paths = sorted(input_path.glob("*.krn")) if input_path.is_dir() else [input_path]
    report = {}
    for path in paths:
        result = validate_kern_text(path.read_text(encoding="utf-8-sig", errors="replace"))
        report[path.name] = {"valid": result.valid, "issues": result.issues, "spine_count": result.spine_count}
    if args.out_json:
        save_json(args.out_json, report)
    else:
        print(json.dumps(report, indent=2))
    raise SystemExit(0 if all(item["valid"] for item in report.values()) else 1)


if __name__ == "__main__":
    main()
