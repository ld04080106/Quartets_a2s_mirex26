from __future__ import annotations

from pathlib import Path

from a2s.data.kern_parser import parse_kern_file
from a2s.data.metadata_parser import parse_kern_metadata
from a2s.data.note_event import NoteEventSequence
from a2s.stage2_score.tie_merger import merge_tied_notes


GM_PROGRAMS = {
    "violn": 40, "violin": 40, "viola": 41, "cello": 42, "violoncello": 42,
    "cbass": 43, "contrabass": 43, "flt": 73, "flute": 73, "piccolo": 72,
    "oboe": 68, "englhorn": 69, "bassoon": 70, "clarinet": 71,
    "sax": 65, "trumpet": 56, "trombone": 57, "tuba": 58, "horn": 60,
}
VOICE_FALLBACK_PROGRAMS = {"violin_1": 40, "violin_2": 40, "viola": 41, "cello": 42}


def instrument_to_gm_program(instrument: str) -> int | None:
    value = instrument.lower().replace("*i", "").replace("-", "").replace("_", "").strip()
    if value in GM_PROGRAMS:
        return GM_PROGRAMS[value]
    for name, program in GM_PROGRAMS.items():
        if name in value:
            return program
    return None


def voice_programs_from_kern(path: str | Path) -> tuple[dict[str, int], list[str]]:
    score = parse_kern_file(path)
    metadata = parse_kern_metadata(path)
    mapping: dict[str, int] = {}
    warnings: list[str] = []
    for voice, instrument in zip(score.voices, metadata.instruments):
        program = instrument_to_gm_program(instrument)
        if program is None:
            program = VOICE_FALLBACK_PROGRAMS[voice]
            warnings.append(f"unknown instrument {instrument!r} for {voice}; using GM {program}")
        mapping[voice] = program
    for voice, program in VOICE_FALLBACK_PROGRAMS.items():
        mapping.setdefault(voice, program)
    return mapping, warnings


def oracle_notes_in_seconds(
    sequence: NoteEventSequence,
    tempo_bpm: float,
    voice_programs: dict[str, int],
) -> list[dict]:
    if tempo_bpm <= 0:
        raise ValueError(f"tempo must be positive, got {tempo_bpm}")
    seconds_per_beat = 60.0 / tempo_bpm
    result = []
    for note in merge_tied_notes(sequence.notes):
        if note.onset_beat is None or note.offset_beat is None:
            continue
        onset = float(note.onset_beat) * seconds_per_beat
        offset = float(note.offset_beat) * seconds_per_beat
        if offset <= onset:
            continue
        result.append({
            "is_drum": False,
            "program": int(voice_programs.get(note.voice, VOICE_FALLBACK_PROGRAMS.get(note.voice, 40))),
            "onset": onset,
            "offset": offset,
            "pitch": int(note.pitch),
            "velocity": 1,
            "voice": note.voice,
        })
    return sorted(result, key=lambda n: (n["onset"], n["program"], n["pitch"], n["offset"]))
