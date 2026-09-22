#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from a2s.data.note_event import NoteEventSequence
from a2s.stage1_amt.voice_assignment_model import VoiceAssignmentModel
from a2s.utils.config import deep_get, load_config
from a2s.utils.json_io import load_json, save_json
from a2s.utils.logging import configure_logging
from a2s.utils.midi_io import events_to_midi


def _path(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_voice_assignment.yaml")
    parser.add_argument("--model")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--midi_dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    model_path = _path(project, args.model or deep_get(config, "model.out_path", "outputs/voice_assignment/model.pkl"))
    model = VoiceAssignmentModel.load(model_path)
    pred_dir, out_dir = _path(project, args.pred_dir), _path(project, args.out_dir)
    midi_dir = _path(project, args.midi_dir) if args.midi_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)
    if midi_dir:
        midi_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(out_dir / "infer_voice_assignment.log")
    failures: list[dict[str, str]] = []
    paths = sorted(path for path in pred_dir.glob("*.json") if path.name != "stage1_infer_report.json")
    if args.limit:
        paths = paths[: args.limit]
    for index, path in enumerate(paths, start=1):
        target = out_dir / path.name
        if args.skip_existing and target.exists():
            continue
        try:
            sequence = NoteEventSequence.from_dict(load_json(path))
            sequence = model.predict_sequence(sequence)
            save_json(target, sequence.to_dict())
            if midi_dir:
                events_to_midi(sequence, midi_dir / f"{path.stem}.voice.mid")
            if index == 1 or index % 500 == 0:
                logger.info("voice assignment %d/%d (%s)", index, len(paths), path.stem)
        except Exception as exc:
            logger.exception("voice assignment failed: %s", path.stem)
            failures.append({"sample_id": path.stem, "stage": "voice_assignment", "error": str(exc)})
    with (out_dir / "failed_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "stage", "error"))
        writer.writeheader()
        writer.writerows(failures)
    save_json(out_dir / "voice_assignment_infer_report.json", {
        "model_path": str(model_path),
        "num_files": len(paths),
        "num_failed": len(failures),
        "midi_dir": str(midi_dir) if midi_dir else None,
    })


if __name__ == "__main__":
    main()
