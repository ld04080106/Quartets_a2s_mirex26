from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScoreMetadata:
    meter: str = "4/4"
    key: str = "C major"
    key_signature: str | None = None
    tempo_bpm: float = 120.0
    instruments: list[str] = field(default_factory=list)
    kern_indices: list[int] = field(default_factory=list)
    header_lines: list[str] = field(default_factory=list)


def parse_kern_metadata(source: str | Path) -> ScoreMetadata:
    raw = str(source)
    path = Path(raw)
    is_text = "\n" in raw or "\t" in raw
    text = raw if is_text else path.read_text(encoding="utf-8-sig", errors="replace")
    meta = ScoreMetadata()
    active_indices: list[int] = []
    instruments: dict[int, str] = {}
    for line in text.replace("\r", "").splitlines():
        fields = line.split("\t")
        if line.startswith("**") and "**kern" in fields:
            active_indices = [i for i, token in enumerate(fields) if token == "**kern"]
            meta.kern_indices = active_indices[:]
        if line.startswith("*") and not line.startswith("*-"):
            meta.header_lines.append(line)
        for index in active_indices:
            if index >= len(fields):
                continue
            token = fields[index]
            if token.startswith("*I") and not token.startswith("*IC"):
                instruments[index] = token[2:].strip().lower()
            elif re.fullmatch(r"\*M\d+/\d+", token):
                meta.meter = token[2:]
            elif re.fullmatch(r"\*MM\d+(?:\.\d+)?", token):
                meta.tempo_bpm = float(token[3:])
            elif re.fullmatch(r"\*k\[[^]]*\]", token):
                meta.key_signature = token
            elif re.fullmatch(r"\*[A-Ga-g][#-]?:", token):
                tonic = token[1:-1].replace("-", "b")
                meta.key = f"{tonic.upper()} major" if token[1].isupper() else f"{tonic.title()} minor"
        if line.startswith("="):
            break
    meta.instruments = [instruments.get(i, "unknown") for i in meta.kern_indices]
    return meta


def quartet_voice_names(meta: ScoreMetadata) -> list[str]:
    names: list[str | None] = []
    violin_positions: list[int] = []
    for inst in meta.instruments:
        value = inst.lower()
        if "cello" in value or "violonc" in value:
            names.append("cello")
        elif "viola" in value:
            names.append("viola")
        elif "viol" in value or "flt" in value or "flute" in value:
            violin_positions.append(len(names))
            names.append(None)
        else:
            names.append(None)
    # Kern datasets commonly store bottom staff first; the later violin is violin I.
    unresolved = [i for i, name in enumerate(names) if name is None]
    violin_like = violin_positions + [i for i in unresolved if i not in violin_positions]
    if violin_like:
        ordered = sorted(violin_like)
        for rank, index in enumerate(reversed(ordered)):
            names[index] = "violin_1" if rank == 0 else "violin_2"
    defaults = list(reversed(("violin_1", "violin_2", "viola", "cello")))
    return [name or defaults[min(i, 3)] for i, name in enumerate(names)]
