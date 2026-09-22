from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

VOICES = ("violin_1", "violin_2", "viola", "cello")
ALLOWED_VOICES = set(VOICES) | {"unknown"}


@dataclass
class NoteEvent:
    pitch: int
    voice: str = "unknown"
    onset_beat: float | None = None
    offset_beat: float | None = None
    duration_beat: float | None = None
    onset_sec: float | None = None
    offset_sec: float | None = None
    velocity: int | None = None
    confidence: float = 1.0
    pitch_name: str | None = None
    measure_index: int | None = None
    beat_in_measure: float | None = None
    original_token: str | None = None
    is_tied_start: bool = False
    is_tied_stop: bool = False

    def validate(self) -> None:
        if not 0 <= int(self.pitch) <= 127:
            raise ValueError(f"pitch out of MIDI range: {self.pitch}")
        if self.voice not in ALLOWED_VOICES:
            raise ValueError(f"unsupported voice: {self.voice}")
        if self.velocity is not None and not 0 <= int(self.velocity) <= 127:
            raise ValueError(f"velocity out of range: {self.velocity}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")
        for onset_name, offset_name in (("onset_beat", "offset_beat"), ("onset_sec", "offset_sec")):
            onset, offset = getattr(self, onset_name), getattr(self, offset_name)
            if onset is not None and offset is not None and offset <= onset:
                raise ValueError(f"{offset_name} must be greater than {onset_name}")
        if self.duration_beat is not None and self.duration_beat <= 0:
            raise ValueError("duration_beat must be positive")

    def normalized(self) -> "NoteEvent":
        self.pitch = int(self.pitch)
        self.confidence = float(self.confidence)
        if self.velocity is not None:
            self.velocity = int(self.velocity)
        if self.duration_beat is None and self.onset_beat is not None and self.offset_beat is not None:
            self.duration_beat = self.offset_beat - self.onset_beat
        if self.offset_beat is None and self.onset_beat is not None and self.duration_beat is not None:
            self.offset_beat = self.onset_beat + self.duration_beat
        self.validate()
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoteEvent":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in data.items() if key in allowed}).normalized()


@dataclass
class NoteEventSequence:
    sample_id: str
    notes: list[NoteEvent] = field(default_factory=list)
    meter: str | None = None
    key: str | None = None
    tempo_bpm: float | None = None
    staves: list[str] = field(default_factory=lambda: list(VOICES))
    source_model: str = "unknown"
    num_measures: int | None = None
    composer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id is required")
        if not isinstance(self.notes, list):
            raise ValueError("notes must be a list")
        for note in self.notes:
            note.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["schema_version"] = "1.0"
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoteEventSequence":
        sequence = cls(
            sample_id=str(data["sample_id"]),
            notes=[NoteEvent.from_dict(item) for item in data.get("notes", [])],
            meter=data.get("meter"), key=data.get("key"), tempo_bpm=data.get("tempo_bpm"),
            staves=list(data.get("staves") or VOICES), source_model=data.get("source_model", "unknown"),
            num_measures=data.get("num_measures"), composer=data.get("composer"),
            metadata=dict(data.get("metadata") or {}),
        )
        sequence.validate()
        return sequence
