from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from .metadata_parser import parse_kern_metadata
from .note_event import NoteEvent, NoteEventSequence, VOICES

_DURATION_RE = re.compile(r"(\d+)(\.*)")
_PITCH_RE = re.compile(r"([A-Ga-g]+)([#n-]*)")
_PC = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


@dataclass
class KernNote:
    voice: str
    pitch: int
    pitch_name: str
    onset_beat: float
    duration_beat: float
    offset_beat: float
    measure_index: int
    beat_in_measure: float
    token: str
    is_tied_start: bool = False
    is_tied_stop: bool = False


@dataclass
class KernScore:
    sample_id: str
    spines: list[str]
    voices: list[str]
    meter: str | None
    key: str | None
    num_measures: int
    notes: list[KernNote] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unknown_token_count: int = 0
    tie_token_count: int = 0
    chord_token_count: int = 0


def kern_duration_to_beats(token: str, meter: str = "4/4") -> float:
    """Parse a common **kern reciprocal duration into quarter-note beats."""
    del meter  # Reserved for future meter-relative/tuplet handling.
    match = _DURATION_RE.search(token)
    if not match:
        raise ValueError(f"missing reciprocal duration: {token}")
    denominator = int(match.group(1))
    if denominator <= 0:
        raise ValueError(f"unsupported reciprocal duration: {token}")
    base = Fraction(4, denominator)
    total, addition = base, base
    for _ in match.group(2):
        addition /= 2
        total += addition
    return float(total)


def reciprocal_to_beats(token: str) -> Fraction:
    """Backward-compatible exact-duration helper."""
    return Fraction(kern_duration_to_beats(token)).limit_denominator(192)


def kern_pitch_to_midi(token: str) -> tuple[int, str]:
    """Extract the core **kern pitch, ignoring common ornaments/articulations."""
    match = _PITCH_RE.search(token)
    if not match:
        raise ValueError(f"missing kern pitch: {token}")
    letters, accidental = match.groups()
    # Repeated lower-case letters move upward; repeated upper-case letters move downward.
    octave = 4 + len(letters) - 1 if letters[0].islower() else 3 - (len(letters) - 1)
    alteration = 0 if "n" in accidental else accidental.count("#") - accidental.count("-")
    pitch = 12 * (octave + 1) + _PC[letters[0].lower()] + alteration
    if not 0 <= pitch <= 127:
        raise ValueError(f"pitch out of MIDI range: {token}")
    accidental_name = "" if "n" in accidental else "#" * accidental.count("#") + "b" * accidental.count("-")
    return pitch, f"{letters[0].upper()}{accidental_name}{octave}"


def _read_source(source: str | Path) -> tuple[str, str]:
    raw = str(source)
    if "\n" in raw or "\t" in raw:
        return raw, "unknown"
    path = Path(source)
    return path.read_text(encoding="utf-8-sig", errors="replace"), path.stem


def _resolve_voices(instruments: list[str], clefs: list[str], spine_count: int) -> list[str]:
    defaults = list(VOICES[:spine_count])
    if spine_count != 4 or not instruments or all(value == "unknown" for value in instruments):
        return defaults
    if clefs and clefs[0].startswith("G") and clefs[-1].startswith("F"):
        return defaults
    resolved: list[str | None] = [None] * spine_count
    violin_positions: list[int] = []
    for index, instrument in enumerate(instruments[:spine_count]):
        value = instrument.lower()
        if "cello" in value or "violonc" in value:
            resolved[index] = "cello"
        elif "viola" in value:
            resolved[index] = "viola"
        elif "viol" in value or "flt" in value or "flute" in value:
            violin_positions.append(index)
    # Quartets files in the local corpus are commonly stored bottom staff first.
    bottom_first = resolved[:2] == ["cello", "viola"]
    ordered = list(reversed(violin_positions)) if bottom_first else violin_positions
    for voice, index in zip(("violin_1", "violin_2"), ordered):
        resolved[index] = voice
    used = {voice for voice in resolved if voice}
    remaining = [voice for voice in VOICES if voice not in used]
    for index, voice in enumerate(resolved):
        if voice is None:
            resolved[index] = remaining.pop(0) if remaining else defaults[index]
    return [str(voice) for voice in resolved]


def _initial_clefs(text: str, indices: list[int]) -> list[str]:
    clefs = ["" for _ in indices]
    for line in text.replace("\r", "").splitlines():
        fields = line.split("\t")
        for local_index, field_index in enumerate(indices):
            if field_index < len(fields) and fields[field_index].startswith("*clef"):
                clefs[local_index] = fields[field_index][5:]
        if line.startswith("="):
            break
    return clefs


def _raw_key_token(text: str, indices: list[int]) -> str | None:
    for line in text.replace("\r", "").splitlines():
        fields = line.split("\t")
        for index in indices:
            if index >= len(fields):
                continue
            token = fields[index]
            if re.fullmatch(r"\*k\[[^]]*\]", token):
                return token
        if line.startswith("="):
            break
    return None


def parse_kern_file(path: str | Path) -> KernScore:
    text, inferred_id = _read_source(path)
    meta = parse_kern_metadata(text)
    indices = meta.kern_indices
    if len(indices) != 4:
        raise ValueError(f"expected 4 **kern spines, found {len(indices)}")
    voices = _resolve_voices(meta.instruments, _initial_clefs(text, indices), len(indices))
    meter: str | None = None
    key = _raw_key_token(text, indices)
    times = [Fraction(0) for _ in indices]
    measure_starts = [Fraction(0) for _ in indices]
    measure_index = 1
    barline_count = 0
    notes: list[KernNote] = []
    warnings: list[str] = []
    unknown_count = tie_count = chord_count = 0

    for line_number, line in enumerate(text.replace("\r", "").splitlines(), start=1):
        if not line.strip() or line.startswith("!!!"):
            continue
        fields = line.split("\t")
        if line.startswith("*"):
            for index in indices:
                if index < len(fields) and re.fullmatch(r"\*M\d+/\d+", fields[index]):
                    meter = fields[index][2:]
                    break
            continue
        if line.startswith("="):
            if not line.startswith("=="):
                barline_count += 1
                if any(time > 0 for time in times):
                    measure_index += 1
                measure_starts = times[:]
            continue
        if line.startswith("!"):
            continue

        for local_index, field_index in enumerate(indices):
            if field_index >= len(fields):
                warnings.append(f"line {line_number}: missing spine field {field_index}")
                unknown_count += 1
                continue
            field_token = fields[field_index].strip()
            if field_token in {"", "."} or field_token.startswith(("*", "!", "=")):
                continue
            subtokens = field_token.split()
            if len(subtokens) > 1:
                chord_count += 1
            inherited_duration: float | None = None
            parsed_durations: list[Fraction] = []
            for subtoken in subtokens:
                try:
                    duration = kern_duration_to_beats(subtoken, meter or "4/4")
                    inherited_duration = inherited_duration or duration
                except ValueError:
                    if inherited_duration is None:
                        warnings.append(f"line {line_number}: unknown duration token {subtoken!r}")
                        unknown_count += 1
                        continue
                    duration = inherited_duration
                parsed_durations.append(Fraction(duration).limit_denominator(192))
                if "[" in subtoken or "]" in subtoken or "_" in subtoken:
                    tie_count += 1
                if "r" in subtoken:
                    continue
                try:
                    pitch, pitch_name = kern_pitch_to_midi(subtoken)
                except ValueError:
                    warnings.append(f"line {line_number}: unknown pitch token {subtoken!r}")
                    unknown_count += 1
                    continue
                onset = times[local_index]
                duration_fraction = Fraction(duration).limit_denominator(192)
                notes.append(KernNote(
                    voice=voices[local_index], pitch=pitch, pitch_name=pitch_name,
                    onset_beat=float(onset), duration_beat=float(duration_fraction),
                    offset_beat=float(onset + duration_fraction), measure_index=measure_index,
                    beat_in_measure=float(onset - measure_starts[local_index]), token=subtoken,
                    is_tied_start="[" in subtoken, is_tied_stop="]" in subtoken,
                ))
            if parsed_durations:
                times[local_index] += max(parsed_durations)

    if meter is None:
        warnings.append("missing meter interpretation")
    if key is None:
        warnings.append("missing key signature interpretation")
    notes.sort(key=lambda note: (note.onset_beat, VOICES.index(note.voice), note.pitch))
    return KernScore(
        sample_id=inferred_id, spines=["**kern"] * len(indices), voices=voices,
        meter=meter, key=key, num_measures=max(barline_count, 1 if notes else 0), notes=notes,
        warnings=warnings, unknown_token_count=unknown_count,
        tie_token_count=tie_count, chord_token_count=chord_count,
    )


def parse_kern_notes(source: str | Path, sample_id: str | None = None) -> NoteEventSequence:
    """Compatibility wrapper exposing KernScore through the unified event schema."""
    score = parse_kern_file(source)
    compatibility_metadata = parse_kern_metadata(source)
    return NoteEventSequence(
        sample_id=sample_id or score.sample_id,
        notes=[NoteEvent(
            pitch=note.pitch, voice=note.voice, onset_beat=note.onset_beat,
            offset_beat=note.offset_beat, duration_beat=note.duration_beat, confidence=1.0,
            pitch_name=note.pitch_name, measure_index=note.measure_index,
            beat_in_measure=note.beat_in_measure, original_token=note.token,
            is_tied_start=note.is_tied_start, is_tied_stop=note.is_tied_stop,
        ) for note in score.notes],
        meter=score.meter, key=compatibility_metadata.key,
        tempo_bpm=compatibility_metadata.tempo_bpm, source_model="oracle_kern",
        num_measures=score.num_measures,
        metadata={"warnings": score.warnings, "num_measures": score.num_measures},
    )


def count_measures(text: str) -> int:
    return sum(1 for line in text.replace("\r", "").splitlines() if line.startswith("=") and not line.startswith("=="))
