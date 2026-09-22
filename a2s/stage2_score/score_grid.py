from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Segment:
    onset: Fraction
    duration: Fraction
    pitches: tuple[int, ...] = ()
    tie_from: bool = False
    tie_to: bool = False

    @property
    def is_rest(self) -> bool:
        return not self.pitches

    @property
    def offset(self) -> Fraction:
        return self.onset + self.duration


def meter_beats(meter: str) -> Fraction:
    try:
        numerator, denominator = (int(value) for value in meter.split("/", 1))
        return Fraction(numerator * 4, denominator)
    except Exception:
        return Fraction(4)
