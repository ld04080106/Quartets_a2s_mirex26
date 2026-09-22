from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
from _common import read_manifest
from a2s.data.metadata_parser import parse_kern_metadata
from a2s.data.note_event import NoteEventSequence
from a2s.evaluation.note_f1 import note_match_counts
from a2s.utils.config import deep_get, load_config
from a2s.utils.json_io import load_json, save_json
from a2s.utils.logging import configure_logging
from a2s.stage2_score.tie_merger import merge_tied_notes


def _reference_seconds(sequence: NoteEventSequence) -> list:
    bpm = sequence.tempo_bpm or 120.0
    return [replace(
        note,
        onset_sec=note.onset_sec if note.onset_sec is not None else (note.onset_beat or 0.0) * 60.0 / bpm,
        offset_sec=note.offset_sec if note.offset_sec is not None else (note.offset_beat or 0.0) * 60.0 / bpm,
    ) for note in merge_tied_notes(sequence.notes)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_yourmt3_synth_finetuned_pad05_nops_hpc.yaml")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--oracle_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--manifest", help="Optional manifest for ground-truth tempo metadata.")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--pred_only",
        action="store_true",
        help="Evaluate only oracle files whose JSON exists in --pred_dir. Useful for limited/smoke inference outputs.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    pred_dir, oracle_dir = Path(args.pred_dir), Path(args.oracle_dir)
    logger = configure_logging(Path(args.out_json).with_suffix(".log"))
    project_root = Path(__file__).resolve().parents[1]
    manifest_list = read_manifest(args.manifest) if args.manifest else []
    manifest_rows = {row["sample_id"]: row for row in manifest_list}
    onset_tolerance = float(deep_get(config, "evaluation.onset_tolerance_sec", 0.05))
    offset_tolerance = float(deep_get(config, "evaluation.offset_tolerance_sec", 0.1))
    sample_count = empty_count = note_count = 0
    totals = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    missing_predictions: list[str] = []
    if args.pred_only:
        oracle_paths = [
            oracle_dir / path.name
            for path in sorted(pred_dir.glob("*.json"))
            if path.name != "stage1_infer_report.json" and (oracle_dir / path.name).exists()
        ]
    elif manifest_list:
        oracle_paths = [
            oracle_dir / f"{row['sample_id']}.json" for row in manifest_list
            if (oracle_dir / f"{row['sample_id']}.json").exists()
        ]
    else:
        oracle_paths = sorted(oracle_dir.glob("*.json"))
    if args.limit:
        oracle_paths = oracle_paths[:args.limit]
    for index, oracle_path in enumerate(oracle_paths, start=1):
        if index == 1 or index % 500 == 0:
            logger.info("evaluating Stage 1 file %d/%d", index, len(oracle_paths))
        pred_path = pred_dir / oracle_path.name
        ref = NoteEventSequence.from_dict(load_json(oracle_path))
        if not pred_path.exists():
            pred_notes = []
            missing_predictions.append(oracle_path.stem)
        else:
            pred = NoteEventSequence.from_dict(load_json(pred_path))
            pred_notes = _reference_seconds(pred)
            ref.tempo_bpm = pred.tempo_bpm or ref.tempo_bpm
        if ref.tempo_bpm is None and oracle_path.stem in manifest_rows:
            metadata_value = manifest_rows[oracle_path.stem].get("metadata_path")
            if metadata_value:
                metadata_path = Path(metadata_value)
                if not metadata_path.is_absolute():
                    metadata_path = project_root / metadata_path
                if metadata_path.exists():
                    ref.tempo_bpm = parse_kern_metadata(metadata_path).tempo_bpm
        ref_notes = _reference_seconds(ref)
        counts = note_match_counts(pred_notes, ref_notes, onset_tolerance, offset_tolerance)
        totals.update({key: value for key, value in counts.items() if isinstance(value, int)})
        for ref_voice, row in counts["instrument_confusion_matrix"].items():
            confusion[ref_voice].update(row)
        sample_count += 1
        note_count += len(pred_notes)
        empty_count += not pred_notes

    def prf(matches: int) -> dict[str, float]:
        precision = matches / totals["predicted"] if totals["predicted"] else 0.0
        recall = matches / totals["reference"] if totals["reference"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    confusion_dict = {voice: dict(row) for voice, row in confusion.items()}
    swaps = confusion["violin_1"]["violin_2"] + confusion["violin_2"]["violin_1"]
    violin_total = sum(sum(confusion[voice].values()) for voice in ("violin_1", "violin_2"))
    metrics = {
        "pitch": prf(totals["pitch_matches"]), "onset": prf(totals["onset_matches"]),
        "offset": prf(totals["offset_matches"]), "note": prf(totals["note_matches"]),
        "voice_accuracy": (
            totals["correct_voice_matches"] / totals["known_voice_matches"]
            if totals["known_voice_matches"] else None
        ),
        "instrument_confusion_matrix": confusion_dict,
        "evaluated_samples": sample_count, "empty_output_rate": empty_count / max(1, sample_count),
        "avg_notes_per_sample": note_count / max(1, sample_count),
        "voice_swap_rate": swaps / max(1, violin_total),
        "missing_prediction_count": len(missing_predictions),
        "missing_predictions": missing_predictions,
    }
    save_json(args.out_json, metrics)
    logger.info(
        "Stage 1 evaluation: samples=%d note_F1=%.4f voice_accuracy=%s empty_rate=%.4f",
        sample_count, metrics["note"]["f1"], metrics["voice_accuracy"],
        metrics["empty_output_rate"],
    )


if __name__ == "__main__":
    main()
