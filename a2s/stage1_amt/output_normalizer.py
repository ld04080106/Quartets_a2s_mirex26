from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from a2s.data.note_event import ALLOWED_VOICES, NoteEvent, NoteEventSequence


def instrument_to_voice(
    name: str | None = None, program: int | None = None,
    instrument: int | str | None = None,
) -> str:
    if program is not None:
        try:
            program = int(program)
        except (TypeError, ValueError):
            program = None
    value = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if any(token in value for token in ("cello", "violoncello", "vc")):
        return "cello"
    if "viola" in value or value in {"va", "vla"}:
        return "viola"
    if any(token in value for token in ("violin_1", "violin_i", "vln_1", "vn1", "vl1")):
        return "violin_1"
    if any(token in value for token in ("violin_2", "violin_ii", "vln_2", "vn2", "vl2")):
        return "violin_2"
    if program == 41:
        return "viola"
    if program == 42:
        return "cello"
    # A generic violin program cannot distinguish violin I from violin II.
    if program == 40 or "violin" in value:
        return "unknown"
    if isinstance(instrument, str):
        return instrument_to_voice(instrument)
    return "unknown"


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def _json_note(item: dict[str, Any]) -> NoteEvent:
    pitch = _first(item, "pitch", "midi_pitch", "midiPitch")
    onset = _first(item, "onset_sec", "onset", "start_time", "startTime", "start")
    offset = _first(item, "offset_sec", "offset", "end_time", "endTime", "end")
    if pitch is None or onset is None or offset is None:
        raise ValueError(f"AMT note lacks pitch/start/end: {item}")
    voice = item.get("voice")
    if voice not in ALLOWED_VOICES:
        voice = instrument_to_voice(
            str(voice) if voice is not None else _first(
                item, "instrument_name", "instrumentName", "track_name", "trackName", "name"
            ),
            _first(item, "program", "midi_program", "midiProgram"), item.get("instrument"),
        )
    velocity = _first(item, "velocity", "midi_velocity", "midiVelocity")
    confidence = _first(item, "confidence", "probability", "score")
    return NoteEvent(
        pitch=int(pitch), onset_sec=float(onset), offset_sec=float(offset),
        velocity=int(80 if velocity is None else velocity),
        voice=str(voice), confidence=float(1.0 if confidence is None else confidence),
    ).normalized()


def sequence_from_json(payload: Any, sample_id: str, source_model: str = "external_amt") -> NoteEventSequence:
    if isinstance(payload, list):
        note_items, metadata = payload, {}
    elif isinstance(payload, dict):
        note_items = payload.get("notes") or payload.get("note_events") or payload.get("events") or []
        metadata = payload
    else:
        raise ValueError("AMT JSON must be an object or a list of notes")
    notes: list[NoteEvent] = []
    rejected = 0
    for item in note_items:
        try:
            notes.append(_json_note(item))
        except (TypeError, ValueError):
            rejected += 1
    notes.sort(key=lambda note: (note.onset_sec or 0.0, note.pitch, note.voice))
    sequence = NoteEventSequence(
        sample_id=sample_id, notes=notes, meter=metadata.get("meter"), key=metadata.get("key"),
        tempo_bpm=metadata.get("tempo_bpm") or metadata.get("tempo"),
        source_model=metadata.get("source_model", source_model),
        metadata={"rejected_output_notes": rejected},
    )
    sequence.validate()
    return sequence


def sequence_from_midi(path: str | Path, sample_id: str, source_model: str = "midi_adapter") -> NoteEventSequence:
    try:
        import pretty_midi
    except ImportError as exc:
        raise RuntimeError("pretty_midi is required to normalize MIDI AMT output") from exc
    midi = pretty_midi.PrettyMIDI(str(path))
    notes: list[NoteEvent] = []
    track_metadata = []
    for instrument_index, track in enumerate(midi.instruments):
        voice = instrument_to_voice(track.name, track.program, instrument_index)
        track_metadata.append({
            "index": int(instrument_index), "name": str(track.name or ""),
            "program": int(track.program), "voice": voice, "is_drum": bool(track.is_drum),
        })
        if track.is_drum:
            continue
        for note in track.notes:
            if note.end <= note.start:
                continue
            notes.append(NoteEvent(
                pitch=note.pitch, onset_sec=note.start, offset_sec=note.end,
                velocity=note.velocity, voice=voice, confidence=1.0,
            ))
    notes.sort(key=lambda note: (note.onset_sec or 0.0, note.pitch, note.voice))
    return NoteEventSequence(
        sample_id=sample_id, notes=notes, source_model=source_model,
        metadata={"midi_path": str(path), "midi_tracks": track_metadata},
    )


def load_amt_output(
    path: str | Path, sample_id: str, output_format: str = "auto",
    source_model: str = "external_amt",
) -> NoteEventSequence:
    source = Path(path)
    selected = output_format.lower()
    if selected == "auto":
        selected = "midi" if source.suffix.lower() in {".mid", ".midi"} else "json"
    if selected == "midi":
        return sequence_from_midi(source, sample_id, source_model)
    if selected == "json":
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        # Preserve already-unified beat-level sidecars as well as sec-level AMT JSON.
        if isinstance(payload, dict) and payload.get("notes") and any(
            "onset_beat" in note for note in payload["notes"]
        ):
            sequence = NoteEventSequence.from_dict({**payload, "sample_id": sample_id})
            sequence.source_model = payload.get("source_model", source_model)
            return sequence
        return sequence_from_json(payload, sample_id, source_model)
    raise ValueError(f"unsupported AMT output format: {output_format}")
