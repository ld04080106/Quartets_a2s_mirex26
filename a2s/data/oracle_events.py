from __future__ import annotations

from typing import Any

from .kern_parser import KernScore
from .note_event import VOICES

REQUIRED_NOTE_FIELDS = {
    "voice", "pitch", "pitch_name", "onset_beat", "offset_beat", "duration_beat",
    "measure_index", "beat_in_measure", "confidence", "source", "original_token",
}


def kern_score_to_oracle(score: KernScore, composer: str = "unknown", sample_id: str | None = None) -> dict[str, Any]:
    return {
        "sample_id": sample_id or score.sample_id, "composer": composer or "unknown",
        "meter": score.meter, "key": score.key, "num_measures": score.num_measures,
        "staves": list(VOICES),
        "notes": [{
            "voice": note.voice, "pitch": note.pitch, "pitch_name": note.pitch_name,
            "onset_beat": note.onset_beat, "offset_beat": note.offset_beat,
            "duration_beat": note.duration_beat, "measure_index": note.measure_index,
            "beat_in_measure": note.beat_in_measure, "confidence": 1.0,
            "source": "gt_kern", "original_token": note.token,
            "is_tied_start": note.is_tied_start, "is_tied_stop": note.is_tied_stop,
        } for note in score.notes],
        "warnings": list(score.warnings),
    }


def validate_oracle_payload(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if not payload.get("sample_id"):
        issues.append("missing sample_id")
    notes = payload.get("notes")
    if not isinstance(notes, list):
        return issues + ["notes is not a list"]
    previous_onset = float("-inf")
    for index, note in enumerate(notes):
        missing = REQUIRED_NOTE_FIELDS - set(note)
        if missing:
            issues.append(f"note {index}: missing fields {sorted(missing)}")
            continue
        onset = note["onset_beat"]
        offset = note["offset_beat"]
        duration = note["duration_beat"]
        if onset < previous_onset:
            issues.append(f"note {index}: onset order violation")
        previous_onset = onset
        if onset < 0 or duration <= 0 or offset <= onset:
            issues.append(f"note {index}: invalid timing")
        if note["voice"] not in VOICES:
            issues.append(f"note {index}: invalid voice {note['voice']!r}")
        if not 0 <= int(note["pitch"]) <= 127:
            issues.append(f"note {index}: MIDI pitch out of range")
    return issues
