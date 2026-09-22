"""Deterministic note-event to valid four-spine **kern conversion."""

from __future__ import annotations

import math
from fractions import Fraction

from a2s.data.note_event import NoteEventSequence, VOICES

from .barline_repair import split_at_barlines
from .event_quantizer import quantize_events
from .kern_writer import write_kern
from .rest_filler import make_voice_segments
from .score_grid import meter_beats
from .tie_merger import merge_tied_notes


class RuleBasedConverter:
    """Quantize corrected events, fill rests, split bars, and write **kern.

    This is the notation backend for neural Stage 2 and the safe fallback when
    model inference fails; the neural model itself only corrects MIDI tokens.
    """
    def __init__(
        self, subdivisions_per_beat: int = 4, default_tempo_bpm: float = 120.0,
        alignment_tolerance_beat: float = 0.0,
        alignment_max_span_beat: float | None = None,
    ):
        self.subdivisions_per_beat = subdivisions_per_beat
        self.default_tempo_bpm = default_tempo_bpm
        self.alignment_tolerance_beat = alignment_tolerance_beat
        self.alignment_max_span_beat = alignment_max_span_beat

    def convert(self, sequence: NoteEventSequence) -> str:
        meter = sequence.meter or "4/4"
        tempo = sequence.tempo_bpm or self.default_tempo_bpm
        notes = quantize_events(
            merge_tied_notes(sequence.notes), tempo, self.subdivisions_per_beat,
            alignment_tolerance_beat=self.alignment_tolerance_beat,
            alignment_max_span_beat=self.alignment_max_span_beat,
        )
        bar = meter_beats(meter)
        last_offset = max((note.offset_beat or 0.0 for note in notes), default=float(bar))
        inferred_measures = max(1, math.ceil(Fraction(last_offset).limit_denominator(192) / bar))
        total = bar * max(inferred_measures, sequence.num_measures or 0)
        spines = []
        for voice in VOICES:
            voice_notes = [note for note in notes if note.voice == voice]
            segments = make_voice_segments(voice_notes, total)
            spines.append(split_at_barlines(segments, bar))
        return write_kern(
            spines, meter=meter, key=sequence.key or "C major", tempo_bpm=tempo,
            tonal_key=sequence.metadata.get("tonal_key"),
        )
