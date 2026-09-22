from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from a2s.stage2_score.duration_spelling import token_duration


@dataclass
class KernValidation:
    valid: bool
    issues: list[str] = field(default_factory=list)
    spine_count: int = 0
    barline_consistent: bool = True


def validate_kern_text(text: str) -> KernValidation:
    lines = [line for line in text.replace("\r", "").splitlines() if line.strip()]
    if not lines:
        return KernValidation(False, ["empty_file"])
    header = lines[0].split("\t")
    issues: list[str] = []
    if not header or "**kern" not in header or any(not token.startswith("**") for token in header):
        issues.append("invalid_exclusive_interpretation")
    columns = len(header)
    kern_indices = [index for index, token in enumerate(header) if token == "**kern"]
    times = [Fraction(0)] * len(kern_indices)
    barline_consistent = True
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t")
        if len(fields) != columns:
            issues.append(f"spine_count_line_{line_number}")
            continue
        if line.startswith("="):
            if len(set(times)) != 1:
                barline_consistent = False
            continue
        if line.startswith(("*", "!")):
            continue
        for local_index, field_index in enumerate(kern_indices):
            token = fields[field_index]
            if token == ".":
                continue
            first = token.split()[0]
            try:
                times[local_index] += token_duration(first.lstrip("[").rstrip("]_"))
            except (ValueError, ZeroDivisionError):
                issues.append(f"bad_duration_line_{line_number}_spine_{field_index + 1}")
    if not lines[-1].startswith("*-"):
        issues.append("missing_terminator")
    if len(set(times)) != 1:
        issues.append("unequal_spine_duration")
    if not barline_consistent:
        issues.append("barline_time_mismatch")
    return KernValidation(not issues, issues, columns, barline_consistent)
