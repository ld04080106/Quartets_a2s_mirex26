from __future__ import annotations

from dataclasses import replace
from statistics import median

from a2s.data.note_event import NoteEvent


def _weighted_median(values: list[tuple[float, float]]) -> float:
    ordered = sorted(values)
    total = sum(max(weight, 1e-6) for _, weight in ordered)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += max(weight, 1e-6)
        if cumulative >= total / 2:
            return value
    return ordered[-1][0]


def align_event_boundaries(
    notes: list[NoteEvent], tolerance_beat: float = 0.0,
    max_cluster_span_beat: float | None = None,
    min_cluster_size: int = 2,
) -> list[NoteEvent]:
    """Cluster nearby onset/offset boundaries before metric quantization.

    Both attacks and releases participate because an offset in one voice often
    coincides with an onset in another. A cluster may never contain both ends of
    the same note, preventing short notes from collapsing to zero duration.
    """
    copied = [replace(note) for note in notes]
    if tolerance_beat <= 0 or len(copied) < 2:
        return copied
    maximum_span = max_cluster_span_beat or tolerance_beat * 2
    boundaries: list[tuple[float, int, str, float]] = []
    for index, note in enumerate(copied):
        weight = max(0.05, float(note.confidence))
        if note.onset_beat is not None:
            boundaries.append((float(note.onset_beat), index, "onset", weight))
        if note.offset_beat is not None:
            boundaries.append((float(note.offset_beat), index, "offset", weight))
    boundaries.sort(key=lambda item: item[0])

    clusters: list[list[tuple[float, int, str, float]]] = []
    current: list[tuple[float, int, str, float]] = []
    for boundary in boundaries:
        if not current:
            current = [boundary]
            continue
        candidate_values = [item[0] for item in current] + [boundary[0]]
        center = median(candidate_values)
        note_indices = {item[1] for item in current}
        can_join = (
            boundary[1] not in note_indices
            and boundary[0] - current[0][0] <= maximum_span
            and abs(boundary[0] - center) <= tolerance_beat
        )
        if can_join:
            current.append(boundary)
        else:
            clusters.append(current)
            current = [boundary]
    if current:
        clusters.append(current)

    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            continue
        center = _weighted_median([(item[0], item[3]) for item in cluster])
        for _, note_index, kind, _ in cluster:
            if kind == "onset":
                copied[note_index].onset_beat = center
            else:
                copied[note_index].offset_beat = center
    for note in copied:
        if note.onset_beat is not None and note.offset_beat is not None:
            note.duration_beat = note.offset_beat - note.onset_beat
    return copied
