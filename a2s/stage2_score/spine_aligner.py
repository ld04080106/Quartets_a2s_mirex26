"""Align independently reconstructed voices on a shared score-time grid."""

from __future__ import annotations

from fractions import Fraction

from .score_grid import Segment


def align_spines(
    spines: list[list[Segment]],
) -> list[tuple[Fraction, list[Segment | None]]]:
    """Return one row for every onset appearing in any spine.

    A missing voice at a row is represented by ``None`` and is written as a
    Humdrum null token. Rest filling guarantees that sustained score time is
    still explicitly covered within each individual spine.
    """

    by_spine = [{segment.onset: segment for segment in spine} for spine in spines]
    times = sorted({onset for mapping in by_spine for onset in mapping})
    return [(time, [mapping.get(time) for mapping in by_spine]) for time in times]

