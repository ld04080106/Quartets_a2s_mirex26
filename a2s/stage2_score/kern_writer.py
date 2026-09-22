from __future__ import annotations

from fractions import Fraction

from .duration_spelling import spell_duration
from .score_grid import Segment
from .spine_aligner import align_spines

VOICE_ORDER = ("violin_1", "violin_2", "viola", "cello")
_PC_NAMES = ("c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b")


def midi_to_kern_pitch(pitch: int) -> str:
    octave = pitch // 12 - 1
    name = _PC_NAMES[pitch % 12]
    letter, accidental = name[0], name[1:]
    if octave >= 4:
        letters = letter * (octave - 3)
    else:
        letters = letter.upper() * (4 - octave)
    return letters + accidental


def _segment_token(segment: Segment) -> str:
    duration = spell_duration(segment.duration)[0]
    if segment.is_rest:
        return duration + "r"
    pitches: list[str] = []
    for pitch in segment.pitches:
        value = duration + midi_to_kern_pitch(pitch)
        if segment.tie_from and segment.tie_to:
            value += "_"
        elif segment.tie_from:
            value += "]"
        elif segment.tie_to:
            value = "[" + value
        pitches.append(value)
    return " ".join(pitches)


def write_kern(
    spines: list[list[Segment]], meter: str = "4/4", key: str = "C major",
    tempo_bpm: float = 120.0, tonal_key: str | None = None,
) -> str:
    count = len(spines)
    if count != 4:
        raise ValueError(f"quartet writer requires four spines, got {count}")
    repeated = lambda value: "\t".join([value] * count)
    raw_key_signature = key if key and key.startswith("*k[") else "*k[]"
    tonal_key = tonal_key or (key if key and not key.startswith("*") else None)
    tonal_token = None
    if tonal_key:
        key_token = tonal_key.split()[0].replace("b", "-")
        major = "minor" not in tonal_key.lower()
        tonal_token = f"*{key_token.upper() if major else key_token.lower()}:"
    lines = [
        repeated("**kern"),
        "\t".join(("*Iviolin", "*Iviolin", "*Iviola", "*Icello")),
        "\t".join(("*clefG2", "*clefG2", "*clefC3", "*clefF4")),
        repeated(raw_key_signature),
    ]
    if tonal_token:
        lines.append(repeated(tonal_token))
    lines.extend((repeated(f"*M{meter}"), repeated(f"*MM{tempo_bpm:g}")))
    numerator, denominator = (int(item) for item in meter.split("/", 1))
    measure_beats = Fraction(numerator * 4, denominator)
    emitted_bars: set[Fraction] = set()
    for time, row in align_spines(spines):
        if time > 0 and time % measure_beats == 0 and time not in emitted_bars:
            number = int(time / measure_beats) + 1
            lines.append(repeated(f"={number}"))
            emitted_bars.add(time)
        lines.append("\t".join(_segment_token(segment) if segment else "." for segment in row))
    lines.extend((repeated("=="), repeated("*-")))
    return "\n".join(lines) + "\n"
