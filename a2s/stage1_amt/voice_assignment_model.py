"""Learned quartet voice labelling with onset-level assignment constraints."""

from __future__ import annotations

import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path

from a2s.data.note_event import NoteEventSequence, VOICES
from a2s.stage1_amt.voice_assignment_features import FEATURE_NAMES, feature_rows, notes_by_onset_group


VOICE_TO_ID = {voice: index for index, voice in enumerate(VOICES)}
ID_TO_VOICE = {index: voice for voice, index in VOICE_TO_ID.items()}


@dataclass
class VoiceAssignmentModel:
    estimator: object
    feature_names: list[str]
    onset_group_sec: float = 0.04
    constrained_onsets: bool = True
    model_type: str = "sklearn"

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "VoiceAssignmentModel":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError(f"unexpected voice assignment model payload: {type(model)!r}")
        return model

    def predict_sequence(self, sequence: NoteEventSequence) -> NoteEventSequence:
        """Assign one quartet voice to every note, mutating and returning ``sequence``."""
        if not sequence.notes:
            return sequence
        rows = feature_rows(sequence.notes, self.onset_group_sec)
        if hasattr(self.estimator, "predict_proba"):
            probabilities = self.estimator.predict_proba(rows)
            labels = _decode_with_constraints(probabilities, notes_by_onset_group(sequence.notes, self.onset_group_sec)) if self.constrained_onsets else [
                ID_TO_VOICE[int(max(range(len(row)), key=lambda i: row[i]))] for row in probabilities
            ]
        else:
            labels = [ID_TO_VOICE[int(value)] for value in self.estimator.predict(rows)]
        for note, voice in zip(sequence.notes, labels):
            note.voice = voice
        sequence.metadata["voice_assignment_model"] = {
            "model_type": self.model_type,
            "constrained_onsets": self.constrained_onsets,
            "onset_group_sec": self.onset_group_sec,
        }
        return sequence


def _decode_with_constraints(probabilities, groups: list[list[int]]) -> list[str]:
    """Find the best one-note-per-voice assignment for compact onset groups.

    Dense groups larger than four notes may contain double stops, so they fall
    back to independent classification instead of forcing an impossible bijection.
    """
    labels = ["unknown"] * len(probabilities)
    for group in groups:
        if len(group) <= len(VOICES):
            best_score = float("-inf")
            best_assignment: tuple[int, ...] | None = None
            for assignment in itertools.permutations(range(len(VOICES)), len(group)):
                score = sum(float(probabilities[note_index][voice_index]) for note_index, voice_index in zip(group, assignment))
                if score > best_score:
                    best_score = score
                    best_assignment = assignment
            assert best_assignment is not None
            for note_index, voice_index in zip(group, best_assignment):
                labels[note_index] = ID_TO_VOICE[int(voice_index)]
        else:
            for note_index in group:
                labels[note_index] = ID_TO_VOICE[int(max(range(len(VOICES)), key=lambda i: probabilities[note_index][i]))]
    return labels
