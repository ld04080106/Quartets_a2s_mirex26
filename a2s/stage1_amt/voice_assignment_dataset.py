from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from a2s.data.note_event import NoteEvent, NoteEventSequence, VOICES
from a2s.stage1_amt.voice_assignment_features import feature_rows, onset, offset
from a2s.stage1_amt.voice_assignment_model import VOICE_TO_ID
from a2s.stage2_score.tie_merger import merge_tied_notes
from a2s.utils.json_io import load_json


def reference_seconds(sequence: NoteEventSequence) -> list[NoteEvent]:
    bpm = sequence.tempo_bpm or 120.0
    return [
        replace(
            note,
            onset_sec=note.onset_sec if note.onset_sec is not None else (note.onset_beat or 0.0) * 60.0 / bpm,
            offset_sec=note.offset_sec if note.offset_sec is not None else (note.offset_beat or 0.0) * 60.0 / bpm,
        )
        for note in merge_tied_notes(sequence.notes)
    ]


def match_predicted_to_reference(
    predicted: list[NoteEvent],
    reference: list[NoteEvent],
    onset_tolerance_sec: float = 0.05,
    offset_tolerance_sec: float = 0.10,
) -> list[tuple[int, NoteEvent]]:
    candidates = sorted(
        (
            (abs(onset(pred) - onset(ref)) + 0.25 * abs(offset(pred) - offset(ref)), pred_index, ref)
            for pred_index, pred in enumerate(predicted)
            for ref in reference
            if pred.pitch == ref.pitch
            and abs(onset(pred) - onset(ref)) <= onset_tolerance_sec
            and abs(offset(pred) - offset(ref)) <= offset_tolerance_sec
            and ref.voice in VOICES
        ),
        key=lambda item: item[0],
    )
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    result: list[tuple[int, NoteEvent]] = []
    for _, pred_index, ref in candidates:
        ref_id = id(ref)
        if pred_index in used_pred or ref_id in used_ref:
            continue
        used_pred.add(pred_index)
        used_ref.add(ref_id)
        result.append((pred_index, ref))
    return result


def load_labeled_voice_examples(
    pred_path: str | Path,
    oracle_path: str | Path,
    onset_tolerance_sec: float = 0.05,
    offset_tolerance_sec: float = 0.10,
    onset_group_sec: float = 0.04,
) -> tuple[list[list[float]], list[int], dict[str, Any]]:
    pred = NoteEventSequence.from_dict(load_json(pred_path))
    ref = NoteEventSequence.from_dict(load_json(oracle_path))
    pred_notes = reference_seconds(pred)
    ref_notes = reference_seconds(ref)
    rows = feature_rows(pred_notes, onset_group_sec)
    matched = match_predicted_to_reference(pred_notes, ref_notes, onset_tolerance_sec, offset_tolerance_sec)
    x, y = [], []
    for pred_index, ref_note in matched:
        x.append(rows[pred_index])
        y.append(VOICE_TO_ID[ref_note.voice])
    stats = {
        "sample_id": pred.sample_id,
        "predicted_notes": len(pred_notes),
        "reference_notes": len(ref_notes),
        "matched_notes": len(matched),
    }
    return x, y, stats
