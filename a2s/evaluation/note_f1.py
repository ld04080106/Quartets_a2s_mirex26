from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

from a2s.data.note_event import NoteEvent


def _prf(matches: int, predicted: int, reference: int) -> dict[str, float]:
    precision = matches / predicted if predicted else 0.0
    recall = matches / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def note_match_counts(
    predicted: list[NoteEvent], reference: list[NoteEvent],
    onset_tolerance: float = 0.05, offset_tolerance: float = 0.1,
) -> dict:
    pred_pitch, ref_pitch = Counter(note.pitch for note in predicted), Counter(note.pitch for note in reference)
    pitch_matches = sum((pred_pitch & ref_pitch).values())
    onset_pairs = _greedy_match(predicted, reference, lambda p, r: p.pitch == r.pitch and abs(_onset(p) - _onset(r)) <= onset_tolerance)
    offset_pairs = _greedy_match(predicted, reference, lambda p, r: p.pitch == r.pitch and abs(_offset(p) - _offset(r)) <= offset_tolerance)
    note_pairs = _greedy_match(predicted, reference, lambda p, r: p.pitch == r.pitch and abs(_onset(p) - _onset(r)) <= onset_tolerance and abs(_offset(p) - _offset(r)) <= offset_tolerance)
    known_pairs = [(p, r) for p, r in note_pairs if p.voice != "unknown" and r.voice != "unknown"]
    confusion: dict[str, Counter] = defaultdict(Counter)
    for pred, ref in known_pairs:
        confusion[ref.voice][pred.voice] += 1
    return {
        "predicted": len(predicted), "reference": len(reference),
        "pitch_matches": pitch_matches, "onset_matches": len(onset_pairs),
        "offset_matches": len(offset_pairs), "note_matches": len(note_pairs),
        "known_voice_matches": len(known_pairs),
        "correct_voice_matches": sum(p.voice == r.voice for p, r in known_pairs),
        "instrument_confusion_matrix": {ref: dict(counts) for ref, counts in confusion.items()},
    }


def evaluate_notes(predicted: list[NoteEvent], reference: list[NoteEvent], onset_tolerance: float = 0.05, offset_tolerance: float = 0.1) -> dict:
    counts = note_match_counts(predicted, reference, onset_tolerance, offset_tolerance)
    return {
        "pitch": _prf(counts["pitch_matches"], counts["predicted"], counts["reference"]),
        "onset": _prf(counts["onset_matches"], counts["predicted"], counts["reference"]),
        "offset": _prf(counts["offset_matches"], counts["predicted"], counts["reference"]),
        "note": _prf(counts["note_matches"], counts["predicted"], counts["reference"]),
        "voice_accuracy": (
            counts["correct_voice_matches"] / counts["known_voice_matches"]
            if counts["known_voice_matches"] else None
        ),
        "instrument_confusion_matrix": counts["instrument_confusion_matrix"],
    }


def _onset(note: NoteEvent) -> float:
    return note.onset_sec if note.onset_sec is not None else note.onset_beat or 0.0


def _offset(note: NoteEvent) -> float:
    return note.offset_sec if note.offset_sec is not None else note.offset_beat or 0.0


def _greedy_match(predicted: list[NoteEvent], reference: list[NoteEvent], compatible: Callable[[NoteEvent, NoteEvent], bool]) -> list[tuple[NoteEvent, NoteEvent]]:
    candidates = sorted(
        ((abs(_onset(p) - _onset(r)), pi, ri) for pi, p in enumerate(predicted) for ri, r in enumerate(reference) if compatible(p, r)),
        key=lambda item: item[0],
    )
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    result = []
    for _, pi, ri in candidates:
        if pi not in used_pred and ri not in used_ref:
            used_pred.add(pi); used_ref.add(ri)
            result.append((predicted[pi], reference[ri]))
    return result
