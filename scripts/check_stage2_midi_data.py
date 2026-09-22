from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from a2s.utils.config import deep_get, load_config
from a2s.utils.json_io import save_json


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _inspect_jsonl(path: Path, min_avg_source_tokens: float) -> dict:
    counts: Counter[str] = Counter()
    source_lengths: list[int] = []
    target_lengths: list[int] = []
    missing = []
    if not path.exists():
        return {"path": str(path), "exists": False, "ok": False, "error": "missing_jsonl"}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            counts[str(item.get("source", "unknown"))] += 1
            source_lengths.append(len(item.get("source_tokens", [])))
            target_lengths.append(len(item.get("target_tokens", [])))
            if not item.get("sample_id"):
                missing.append({"line_no": line_no, "field": "sample_id"})
    avg_source = sum(source_lengths) / max(1, len(source_lengths))
    avg_target = sum(target_lengths) / max(1, len(target_lengths))
    result = {
        "path": str(path),
        "exists": True,
        "num_examples": len(source_lengths),
        "source_counts": dict(counts),
        "avg_source_tokens": avg_source,
        "avg_target_tokens": avg_target,
        "min_source_tokens": min(source_lengths) if source_lengths else 0,
        "max_source_tokens": max(source_lengths) if source_lengths else 0,
        "min_target_tokens": min(target_lengths) if target_lengths else 0,
        "max_target_tokens": max(target_lengths) if target_lengths else 0,
        "missing_required_fields": missing[:50],
    }
    result["ok"] = (
        result["num_examples"] > 0
        and counts.get("noisy_oracle", 0) == 0
        and counts.get("pred_events", 0) == result["num_examples"]
        and avg_source >= min_avg_source_tokens
        and not missing
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check prepared Stage2 MIDI pred-events data.")
    parser.add_argument("--config", default="configs/stage2_midi_train_pred_events_long_v2.yaml")
    parser.add_argument("--out_json")
    parser.add_argument("--min_avg_source_tokens", type=float, default=100.0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    train_path = _project_path(project_root, deep_get(config, "data.train_jsonl", "outputs/stage2_midi_pred_events_data/train.jsonl"))
    valid_path = _project_path(project_root, deep_get(config, "data.valid_jsonl", "outputs/stage2_midi_pred_events_data/valid.jsonl"))
    report = {
        "train": _inspect_jsonl(train_path, args.min_avg_source_tokens),
        "valid": _inspect_jsonl(valid_path, args.min_avg_source_tokens),
    }
    report["ok"] = bool(report["train"]["ok"] and report["valid"]["ok"])
    out_json = args.out_json or deep_get(
        config,
        "data.check_report",
        "outputs/stage2_midi_pred_events_data/reports/stage2_midi_data_check.json",
    )
    save_json(_project_path(project_root, out_json), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
