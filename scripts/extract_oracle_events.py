from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from _common import read_manifest, write_failures
from a2s.data.kern_parser import parse_kern_file
from a2s.data.note_event import VOICES
from a2s.data.oracle_events import kern_score_to_oracle
from a2s.utils.config import load_config
from a2s.utils.json_io import save_json
from a2s.utils.logging import configure_logging


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract four-voice oracle events from Quartets **kern files.")
    parser.add_argument("--config", default="configs/data_quartets.yaml")
    parser.add_argument("--limit", type=int, help="Optional per-split smoke-test limit.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    output = config.get("output", {})
    manifest_dir = _project_path(project_root, output.get("manifest_dir", "data/manifests"))
    oracle_dir = _project_path(project_root, output.get("oracle_events_dir", "data/oracle_events"))
    reports_dir = _project_path(project_root, output.get("reports_dir", "data/reports"))
    logger = configure_logging(reports_dir / "extract_oracle_events.log", args.verbose)
    failures: list[dict[str, str]] = []
    voice_counts = Counter({voice: 0 for voice in VOICES})
    meter_counts: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    total_files = success = total_notes = unknown_count = tie_count = chord_count = 0

    for split in ("train", "valid", "test"):
        manifest_path = manifest_dir / f"{split}.csv"
        if not manifest_path.exists():
            failures.append({"sample_id": split, "stage": "oracle_extraction", "error": f"missing manifest: {manifest_path}"})
            continue
        rows = read_manifest(manifest_path)
        if args.limit:
            rows = rows[:args.limit]
        target_dir = oracle_dir / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            total_files += 1
            if total_files == 1 or total_files % 1000 == 0:
                logger.info("extracting oracle file %d", total_files)
            sample_id = row.get("sample_id") or "unknown"
            try:
                if row.get("has_kern", "true").lower() != "true" or not row.get("kern_path"):
                    raise FileNotFoundError("manifest marks kern as missing")
                kern_path = _project_path(project_root, row["kern_path"])
                score = parse_kern_file(kern_path)
                score.sample_id = sample_id
                payload = kern_score_to_oracle(score, row.get("composer") or "unknown", sample_id)
                save_json(target_dir / f"{sample_id}.json", payload)
                success += 1
                total_notes += len(score.notes)
                voice_counts.update(note.voice for note in score.notes)
                meter_counts[score.meter or "unknown"] += 1
                key_counts[score.key or "unknown"] += 1
                unknown_count += score.unknown_token_count
                tie_count += score.tie_token_count
                chord_count += score.chord_token_count
            except Exception as exc:
                logger.warning("oracle extraction failed for %s: %s", sample_id, exc)
                failures.append({"sample_id": sample_id, "stage": "oracle_extraction", "error": str(exc)})

    report = {
        "num_files": total_files, "num_success": success, "num_failed": len(failures),
        "total_notes": total_notes, "avg_notes_per_file": total_notes / success if success else 0.0,
        "voice_note_counts": dict(voice_counts), "meter_counts": dict(sorted(meter_counts.items())),
        "key_counts": dict(sorted(key_counts.items())), "unknown_token_count": unknown_count,
        "tie_token_count": tie_count, "chord_token_count": chord_count, "failed_files": failures,
    }
    write_failures(oracle_dir / "failed_samples.csv", failures)
    save_json(reports_dir / "oracle_events_report.json", report)
    logger.info("oracle extraction: files=%d success=%d failed=%d notes=%d", total_files, success, len(failures), total_notes)


if __name__ == "__main__":
    main()
