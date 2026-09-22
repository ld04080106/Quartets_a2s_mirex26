from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from a2s.data.note_event import VOICES
from a2s.data.oracle_events import validate_oracle_payload
from a2s.utils.config import load_config
from a2s.utils.json_io import load_json, save_json
from a2s.utils.logging import configure_logging

PITCH_RANGES = {
    "cello": (36, 76), "viola": (48, 91),
    "violin_1": (55, 103), "violin_2": (55, 103),
}


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check extracted oracle event JSON files.")
    parser.add_argument("--config", default="configs/data_quartets.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    output = config.get("output", {})
    oracle_dir = _project_path(project_root, output.get("oracle_events_dir", "data/oracle_events"))
    reports_dir = _project_path(project_root, output.get("reports_dir", "data/reports"))
    logger = configure_logging(reports_dir / "check_oracle_events.log", args.verbose)
    checked = empty_files = range_warnings = 0
    invalid_files: list[dict[str, object]] = []
    warning_files: list[dict[str, object]] = []
    split_counts = {split: 0 for split in ("train", "valid", "test")}

    for split in split_counts:
        for path in sorted((oracle_dir / split).glob("*.json")):
            checked += 1
            if checked == 1 or checked % 1000 == 0:
                logger.info("checking oracle file %d", checked)
            split_counts[split] += 1
            try:
                payload = load_json(path)
                issues = validate_oracle_payload(payload)
                notes = payload.get("notes", []) if isinstance(payload, dict) else []
                if not notes:
                    empty_files += 1
                    warning_files.append({"file": path.name, "warnings": ["empty notes"]})
                pitch_messages = []
                for index, note in enumerate(notes):
                    voice = note.get("voice")
                    if voice not in VOICES:
                        continue
                    low, high = PITCH_RANGES[voice]
                    pitch = int(note.get("pitch", -1))
                    if not low <= pitch <= high:
                        pitch_messages.append(f"note {index}: {voice} pitch {pitch} outside {low}-{high}")
                range_warnings += len(pitch_messages)
                if pitch_messages:
                    warning_files.append({"file": path.name, "warnings": pitch_messages})
                if issues:
                    invalid_files.append({"file": path.name, "issues": issues})
            except Exception as exc:
                invalid_files.append({"file": path.name, "issues": [str(exc)]})

    report = {
        "num_files_checked": checked, "split_counts": split_counts,
        "num_invalid_files": len(invalid_files), "num_empty_files": empty_files,
        "pitch_range_warning_count": range_warnings,
        "invalid_files": invalid_files, "warning_files": warning_files,
    }
    save_json(reports_dir / "oracle_events_check_report.json", report)
    logger.info(
        "oracle sanity check: checked=%d invalid=%d empty=%d pitch_warnings=%d",
        checked, len(invalid_files), empty_files, range_warnings,
    )
    if checked == 0 or invalid_files:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
