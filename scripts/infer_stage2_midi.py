from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _common import read_manifest, write_failures
from a2s.data.metadata_parser import parse_kern_metadata
from a2s.data.note_event import NoteEvent, NoteEventSequence, VOICES
from a2s.evaluation.kern_validity import validate_kern_text
from a2s.stage1_amt.voice_assignment import assign_voices
from a2s.stage2_score.kern_repair import repair_kern, safe_fallback_kern
from a2s.stage2_score.midi_tokenizer import beats_to_seconds, midi_tokens_to_sequence, sequence_to_midi_tokens
from a2s.stage2_score.rule_based_converter import RuleBasedConverter
from a2s.stage2_score.tie_merger import merge_tied_notes
from a2s.utils.config import deep_get, load_config
from a2s.utils.json_io import load_json, save_json
from a2s.utils.logging import configure_logging
from a2s.utils.midi_io import events_to_midi


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _apply_manifest_metadata(project_root: Path, sequence: NoteEventSequence, row: dict[str, str]) -> None:
    for key in ("metadata_path", "kern_path"):
        value = row.get(key)
        if not value:
            continue
        path = _project_path(project_root, value)
        if not path.exists():
            continue
        metadata = parse_kern_metadata(path)
        sequence.meter = sequence.meter or metadata.meter
        sequence.key = sequence.key or metadata.key
        sequence.tempo_bpm = sequence.tempo_bpm or metadata.tempo_bpm
        return


def _note_onset(note: NoteEvent, tempo_bpm: float | None) -> float | None:
    if note.onset_beat is not None:
        return float(note.onset_beat)
    if note.onset_sec is not None:
        return float(note.onset_sec) * float(tempo_bpm or 120.0) / 60.0
    return None


def _note_offset(note: NoteEvent, tempo_bpm: float | None) -> float | None:
    if note.offset_beat is not None:
        return float(note.offset_beat)
    if note.duration_beat is not None and note.onset_beat is not None:
        return float(note.onset_beat) + float(note.duration_beat)
    if note.offset_sec is not None:
        return float(note.offset_sec) * float(tempo_bpm or 120.0) / 60.0
    return None


def _span_beats(sequence: NoteEventSequence) -> float:
    tempo = sequence.tempo_bpm or 120.0
    starts = [_note_onset(note, tempo) for note in sequence.notes]
    ends = [_note_offset(note, tempo) for note in sequence.notes]
    starts = [value for value in starts if value is not None]
    ends = [value for value in ends if value is not None]
    if not starts or not ends:
        return 0.0
    return max(0.0, max(ends) - min(starts))


def _voice_counts(sequence: NoteEventSequence) -> dict[str, int]:
    counts = {voice: 0 for voice in VOICES}
    counts["unknown"] = 0
    for note in sequence.notes:
        counts[note.voice if note.voice in counts else "unknown"] += 1
    return counts


def _quality_gate(
    source: NoteEventSequence,
    corrected: NoteEventSequence,
    config: dict,
) -> tuple[bool, list[str], dict[str, float | int | dict[str, int]]]:
    """Reject obviously broken neural MIDI corrections.

    The gate is intentionally conservative and GT-free.  It only compares the
    corrected event stream against the Stage1 source stream, then lets the
    rule-based source->kern path handle rejected cases.
    """

    gate = deep_get(config, "stage2_midi.quality_gate", {}) or {}
    if not bool(gate.get("enabled", False)):
        return True, [], {}

    source_notes = len(source.notes)
    corrected_notes = len(corrected.notes)
    source_span = _span_beats(source)
    corrected_span = _span_beats(corrected)
    note_ratio = corrected_notes / max(1, source_notes)
    span_ratio = corrected_span / max(1e-6, source_span)
    unknown_voice_ratio = sum(1 for note in corrected.notes if note.voice not in VOICES) / max(1, corrected_notes)
    pitches = [int(note.pitch) for note in corrected.notes]
    stats: dict[str, float | int | dict[str, int]] = {
        "source_notes": source_notes,
        "corrected_notes": corrected_notes,
        "note_ratio": note_ratio,
        "source_span_beat": source_span,
        "corrected_span_beat": corrected_span,
        "span_ratio": span_ratio,
        "unknown_voice_ratio": unknown_voice_ratio,
        "min_pitch": min(pitches) if pitches else -1,
        "max_pitch": max(pitches) if pitches else -1,
        "voice_counts": _voice_counts(corrected),
    }

    reasons: list[str] = []
    if corrected_notes == 0:
        reasons.append("empty_corrected_notes")
    if note_ratio < float(gate.get("min_note_ratio_vs_source", 0.45)):
        reasons.append("too_few_notes")
    if (
        note_ratio > float(gate.get("max_note_ratio_vs_source", 2.20))
        and span_ratio >= float(gate.get("min_span_ratio_for_too_many_notes", 0.0))
    ):
        reasons.append("too_many_notes")
    if source_span > 0 and span_ratio < float(gate.get("min_span_ratio_vs_source", 0.45)):
        reasons.append("span_too_short")
    if source_span > 0 and span_ratio > float(gate.get("max_span_ratio_vs_source", 2.40)):
        reasons.append("span_too_long")
    if unknown_voice_ratio > float(gate.get("max_unknown_voice_ratio", 0.05)):
        reasons.append("too_many_unknown_voices")
    if pitches and min(pitches) < int(gate.get("min_pitch", 32)):
        reasons.append("pitch_too_low")
    if pitches and max(pitches) > int(gate.get("max_pitch", 108)):
        reasons.append("pitch_too_high")
    return not reasons, reasons, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer Stage2 MIDI correction, then convert corrected MIDI/events to **kern.")
    parser.add_argument("--config", default="configs/stage2_midi_pred_events_long_v2_infer.yaml")
    parser.add_argument("--events_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    project_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_dir)
    events_out = out_root / "events"
    midi_out = out_root / "midi"
    kern_out = out_root / "kern"
    for path in (events_out, midi_out, kern_out):
        path.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(out_root / "infer_stage2_midi.log")
    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    wanted = {row["sample_id"]: row for row in rows}
    from a2s.stage2_score.midi_correction_model import MidiCorrectionModel

    model = MidiCorrectionModel(
        deep_get(config, "stage2_midi.checkpoint"),
        deep_get(config, "stage2_midi.device", "cpu"),
    )
    subdivisions = int(deep_get(config, "stage2_midi.subdivisions_per_beat", 24))
    converter = RuleBasedConverter(
        subdivisions_per_beat=int(deep_get(config, "stage2.quantization.subdivisions_per_beat", 4)),
        default_tempo_bpm=float(deep_get(config, "stage2.default_tempo_bpm", 120.0)),
        alignment_tolerance_beat=float(deep_get(config, "stage2.alignment.tolerance_beat", 0.12)),
        alignment_max_span_beat=float(deep_get(config, "stage2.alignment.max_cluster_span_beat", 0.20)),
    )
    failures: list[dict[str, str]] = []
    gate_rejections: list[dict] = []
    accepted = 0
    neural_accepted = 0
    rule_fallback = 0
    for index, sample_id in enumerate(wanted, start=1):
        if index == 1 or index % 500 == 0:
            logger.info("Stage2 MIDI inference %d/%d", index, len(wanted))
        try:
            row = wanted[sample_id]
            source = NoteEventSequence.from_dict(load_json(Path(args.events_dir) / f"{sample_id}.json"))
            _apply_manifest_metadata(project_root, source, row)
            source.tempo_bpm = source.tempo_bpm or float(deep_get(config, "stage2.default_tempo_bpm", 120.0))
            if any(note.voice == "unknown" for note in source.notes):
                source.notes = assign_voices(source.notes)
            source.notes = merge_tied_notes(source.notes)
            input_tokens = [
                token for token in sequence_to_midi_tokens(source, subdivisions)
                if token not in {"<BOS_MIDI>", "<EOS_MIDI>"}
            ]
            output_tokens = model.generate(input_tokens, int(deep_get(config, "stage2_midi.max_decode_tokens", 2048)))
            corrected = midi_tokens_to_sequence(
                output_tokens, sample_id,
                meter=source.meter, key=source.key,
                tempo_bpm=source.tempo_bpm or float(deep_get(config, "stage2.default_tempo_bpm", 120.0)),
            )
            if not corrected.notes:
                raise ValueError("empty corrected MIDI events")
            gate_ok, gate_reasons, gate_stats = _quality_gate(source, corrected, config)
            selected = corrected if gate_ok else source
            selected.source_model = "stage2_midi_neural" if gate_ok else "stage2_midi_rule_fallback"
            if gate_ok:
                neural_accepted += 1
            else:
                rule_fallback += 1
                gate_rejections.append({
                    "sample_id": sample_id,
                    "reasons": gate_reasons,
                    "stats": gate_stats,
                })
                logger.warning("Stage2 MIDI gate rejected %s: %s", sample_id, ",".join(gate_reasons))
            save_json(events_out / f"{sample_id}.json", selected.to_dict())
            midi_sequence = beats_to_seconds(selected, selected.tempo_bpm) if gate_ok else selected
            events_to_midi(midi_sequence, midi_out / f"{sample_id}.mid")
            text = converter.convert(selected)
            validation = validate_kern_text(text)
            if not validation.valid:
                text = repair_kern(text)
                validation = validate_kern_text(text)
            if not validation.valid:
                raise ValueError("invalid kern after MIDI correction: " + ",".join(validation.issues))
            kern_out.joinpath(f"{sample_id}.krn").write_text(text, encoding="utf-8")
            accepted += 1
        except Exception as exc:
            logger.exception("Stage2 MIDI failed for %s", sample_id)
            failures.append({"sample_id": sample_id, "stage": "stage2_midi", "error": str(exc)})
            kern_out.joinpath(f"{sample_id}.krn").write_text(safe_fallback_kern(), encoding="utf-8")
    write_failures(out_root / "failed_samples.csv", failures)
    save_json(out_root / "stage2_midi_gate_rejections.json", gate_rejections)
    save_json(out_root / "stage2_midi_infer_report.json", {
        "num_files": len(wanted),
        "num_success": accepted,
        "num_failed": len(failures),
        "num_neural_accepted": neural_accepted,
        "num_rule_fallback": rule_fallback,
        "quality_gate_enabled": bool(deep_get(config, "stage2_midi.quality_gate.enabled", False)),
        "failed_samples": failures,
        "gate_rejections_json": "stage2_midi_gate_rejections.json",
    })


if __name__ == "__main__":
    main()
