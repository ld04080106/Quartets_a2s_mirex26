from __future__ import annotations

from dataclasses import replace

from a2s.data.note_event import NoteEvent


def merge_tied_notes(notes: list[NoteEvent], tolerance: float = 1e-5) -> list[NoteEvent]:
    """Merge simple **kern tie chains before score reconstruction.

    The oracle extractor intentionally preserves tie fragments.  Stage 2 should
    see the performed duration as one event, otherwise every tie continuation is
    reconstructed as an attack.  Malformed or non-contiguous chains are left as
    separate notes instead of being guessed.
    """
    ordered = sorted(
        notes,
        key=lambda note: (
            note.voice,
            note.pitch,
            float(note.onset_beat if note.onset_beat is not None else float("inf")),
        ),
    )
    pending: dict[tuple[str, int], NoteEvent] = {}
    merged: list[NoteEvent] = []

    for note in ordered:
        key = (note.voice, note.pitch)
        current = pending.get(key)
        continuation = "_" in (note.original_token or "")
        contiguous = (
            current is not None
            and current.offset_beat is not None
            and note.onset_beat is not None
            and abs(current.offset_beat - note.onset_beat) <= tolerance
        )
        if current is not None and contiguous and (continuation or note.is_tied_stop):
            current.offset_beat = note.offset_beat
            if current.onset_beat is not None and current.offset_beat is not None:
                current.duration_beat = current.offset_beat - current.onset_beat
            current.is_tied_stop = note.is_tied_stop
            current.original_token = f"{current.original_token or ''} {note.original_token or ''}".strip()
            if note.is_tied_stop:
                pending.pop(key)
                current.is_tied_start = False
                current.is_tied_stop = False
                merged.append(current)
            continue

        if current is not None:
            pending.pop(key)
            merged.append(current)
        if note.is_tied_start and not note.is_tied_stop:
            pending[key] = replace(note)
        else:
            merged.append(replace(note))

    merged.extend(pending.values())
    return sorted(
        merged,
        key=lambda note: (
            float(note.onset_beat if note.onset_beat is not None else float("inf")),
            note.voice,
            note.pitch,
        ),
    )
