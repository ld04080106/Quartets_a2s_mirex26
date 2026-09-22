from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from _common import read_manifest, write_failures
from a2s.data.note_event import NoteEventSequence
from a2s.stage1_amt.voice_assignment import assign_voices
from a2s.stage2_score.midi_tokenizer import beats_to_seconds, sequence_to_midi_tokens
from a2s.stage2_score.tie_merger import merge_tied_notes
from a2s.utils.config import deep_get, load_config
from a2s.utils.json_io import load_json, save_json
from a2s.utils.logging import configure_logging
from a2s.utils.midi_io import events_to_midi


def _project_path(project_root: Path, value: str | None, default: str | None = None) -> Path:
    raw = value if value not in (None, "") else default
    if raw is None:
        raise ValueError("path value is required")
    path = Path(str(raw))
    return path if path.is_absolute() else project_root / path


def _template_path(project_root: Path, value: str | None, split: str) -> Path | None:
    if not value:
        return None
    return _project_path(project_root, value.format(split=split))


def _oracle_path(oracle_root: Path, split: str, sample_id: str) -> Path:
    split_path = oracle_root / split / f"{sample_id}.json"
    if split_path.exists():
        return split_path
    return oracle_root / f"{sample_id}.json"


def _load_sequence(path: Path) -> NoteEventSequence:
    sequence = NoteEventSequence.from_dict(load_json(path))
    if any(note.voice == "unknown" for note in sequence.notes):
        sequence.notes = assign_voices(sequence.notes)
    sequence.notes = merge_tied_notes(sequence.notes)
    return sequence


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _copy_metadata(source: NoteEventSequence, target: NoteEventSequence) -> NoteEventSequence:
    source.meter = source.meter or target.meter
    source.key = source.key or target.key
    source.tempo_bpm = source.tempo_bpm or target.tempo_bpm or 120.0
    source.num_measures = source.num_measures or target.num_measures
    source.staves = target.staves
    return source


def _for_midi_sidecar(sequence: NoteEventSequence) -> NoteEventSequence:
    if any(note.onset_sec is not None and note.offset_sec is not None for note in sequence.notes):
        return sequence
    return beats_to_seconds(sequence, sequence.tempo_bpm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage2 MIDI-to-MIDI correction data.")
    parser.add_argument("--config", default="configs/stage2_midi_train_pred_events_long_v2.yaml")
    parser.add_argument("--splits")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    out_dir = _project_path(project_root, deep_get(config, "data.prepared_dir", "outputs/stage2_midi_data"))
    reports_dir = _project_path(project_root, deep_get(config, "data.reports_dir", str(out_dir / "reports")))
    source_midi_root = _project_path(project_root, deep_get(config, "data.source_midi_dir", str(out_dir / "source_midi")))
    target_midi_root = _project_path(project_root, deep_get(config, "data.target_midi_dir", str(out_dir / "target_midi")))
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(reports_dir / "prepare_stage2_midi_data.log")

    manifest_dir = _project_path(project_root, deep_get(config, "data.manifest_dir", "data/manifests_dedup"))
    if not manifest_dir.exists():
        manifest_dir = _project_path(project_root, None, "data/manifests")
    oracle_root = _project_path(project_root, deep_get(config, "data.oracle_dir", "data/oracle_events"))
    pred_template = deep_get(config, "data.pred_events_dir_template", None)
    split_names = [s.strip() for s in (args.splits or ",".join(deep_get(config, "data.splits", ["train", "valid"]))).split(",") if s.strip()]
    subdivisions = int(deep_get(config, "data.subdivisions_per_beat", 24))
    report: dict[str, Any] = {"splits": {}, "failed_samples": []}
    all_failures: list[dict[str, str]] = []
    for split in split_names:
        manifest = manifest_dir / f"{split}.csv"
        rows = read_manifest(manifest)
        if args.limit:
            rows = rows[: args.limit]
        pred_dir = _template_path(project_root, pred_template, split)
        source_midi_dir = source_midi_root / split
        target_midi_dir = target_midi_root / split
        source_midi_dir.mkdir(parents=True, exist_ok=True)
        target_midi_dir.mkdir(parents=True, exist_ok=True)
        examples: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        counts = {"pred_events": 0}
        logger.info("preparing Stage2 MIDI split=%s rows=%d pred_dir=%s", split, len(rows), pred_dir)

        for index, row in enumerate(rows, start=1):
            if index == 1 or index % 500 == 0:
                logger.info("split=%s sample %d/%d", split, index, len(rows))
            sample_id = row["sample_id"]
            try:
                target = _load_sequence(_oracle_path(oracle_root, split, sample_id))
                target.source_model = "kern_oracle_midi"
                target.tempo_bpm = target.tempo_bpm or float(deep_get(config, "data.default_tempo_bpm", 120.0))

                pred_path = pred_dir / f"{sample_id}.json" if pred_dir else None
                if pred_path and pred_path.exists():
                    source = _load_sequence(pred_path)
                    counts["pred_events"] += 1
                else:
                    raise FileNotFoundError(f"missing predicted events: {pred_path}")
                source = _copy_metadata(source, target)

                source_tokens = sequence_to_midi_tokens(source, subdivisions)
                target_tokens = sequence_to_midi_tokens(target, subdivisions, include_confidence=False)
                examples.append({
                    "sample_id": sample_id,
                    "split": split,
                    "source": "pred_events",
                    "source_tokens": source_tokens,
                    "target_tokens": target_tokens,
                    "source_midi": str(source_midi_dir / f"{sample_id}.mid"),
                    "target_midi": str(target_midi_dir / f"{sample_id}.mid"),
                })
                events_to_midi(_for_midi_sidecar(source), source_midi_dir / f"{sample_id}.mid")
                events_to_midi(_for_midi_sidecar(target), target_midi_dir / f"{sample_id}.mid")
            except Exception as exc:
                logger.warning("failed Stage2 MIDI data sample %s/%s: %s", split, sample_id, exc)
                failure = {"sample_id": sample_id, "stage": f"stage2_midi_prepare_{split}", "error": str(exc)}
                failures.append(failure)
                all_failures.append(failure)

        _write_jsonl(out_dir / f"{split}.jsonl", examples)
        write_failures(out_dir / f"failed_samples_{split}.csv", failures)
        report["splits"][split] = {
            "manifest": str(manifest),
            "num_manifest_rows": len(rows),
            "num_examples": len(examples),
            "num_failed": len(failures),
            "source_counts": counts,
            "jsonl": str(out_dir / f"{split}.jsonl"),
            "source_midi_dir": str(source_midi_dir),
            "target_midi_dir": str(target_midi_dir),
            "avg_source_tokens": sum(len(x["source_tokens"]) for x in examples) / max(1, len(examples)),
            "avg_target_tokens": sum(len(x["target_tokens"]) for x in examples) / max(1, len(examples)),
        }
    report["failed_samples"] = all_failures
    report["num_failed_total"] = len(all_failures)
    save_json(reports_dir / "stage2_midi_data_report.json", report)


if __name__ == "__main__":
    main()
