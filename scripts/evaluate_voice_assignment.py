#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from a2s.data.note_event import NoteEventSequence
from a2s.stage1_amt.voice_assignment_dataset import match_predicted_to_reference, reference_seconds
from a2s.utils.config import deep_get, load_config
from a2s.utils.json_io import load_json, save_json
from a2s.utils.logging import configure_logging


def _path(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_voice_assignment.yaml")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--oracle_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    pred_dir, oracle_dir = _path(project, args.pred_dir), _path(project, args.oracle_dir)
    logger = configure_logging(Path(args.out_json).with_suffix(".log"))
    onset_tol = float(deep_get(config, "matching.onset_tolerance_sec", 0.05))
    offset_tol = float(deep_get(config, "matching.offset_tolerance_sec", 0.10))
    paths = sorted(path for path in pred_dir.glob("*.json") if path.name != "stage1_infer_report.json")
    if args.limit:
        paths = paths[: args.limit]
    total = correct = 0
    sample_count = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    missing = []
    for index, pred_path in enumerate(paths, start=1):
        oracle_path = oracle_dir / pred_path.name
        if not oracle_path.exists():
            missing.append(pred_path.stem)
            continue
        pred = NoteEventSequence.from_dict(load_json(pred_path))
        ref = NoteEventSequence.from_dict(load_json(oracle_path))
        pred_notes, ref_notes = reference_seconds(pred), reference_seconds(ref)
        matched = match_predicted_to_reference(pred_notes, ref_notes, onset_tol, offset_tol)
        for pred_index, ref_note in matched:
            pred_voice = pred_notes[pred_index].voice
            confusion[ref_note.voice][pred_voice] += 1
            correct += int(pred_voice == ref_note.voice)
            total += 1
        sample_count += 1
        if index == 1 or index % 500 == 0:
            logger.info("evaluating voice assignment %d/%d", index, len(paths))
    swaps = confusion["violin_1"]["violin_2"] + confusion["violin_2"]["violin_1"]
    violin_total = sum(sum(confusion[voice].values()) for voice in ("violin_1", "violin_2"))
    save_json(args.out_json, {
        "evaluated_samples": sample_count,
        "matched_notes": total,
        "voice_accuracy": correct / total if total else None,
        "voice_swap_rate": swaps / max(1, violin_total),
        "instrument_confusion_matrix": {voice: dict(row) for voice, row in confusion.items()},
        "missing_oracle_count": len(missing),
        "missing_oracles": missing,
    })


if __name__ == "__main__":
    main()
