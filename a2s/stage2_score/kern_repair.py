from __future__ import annotations


def safe_fallback_kern(meter: str = "4/4", key: str = "C major", tempo_bpm: float = 120.0) -> str:
    columns = 4
    repeat = lambda value: "\t".join([value] * columns)
    numerator, denominator = (int(value) for value in meter.split("/", 1))
    reciprocal = str(denominator)
    rest = reciprocal + ("r" if numerator == 1 else "." + "r" if numerator == 3 else "r")
    # One whole-measure rest is unambiguous for the common 4/4 case; repeated quarter rests cover others.
    data = [repeat("**kern"), repeat(f"*M{meter}"), repeat(f"*MM{tempo_bpm:g}")]
    if meter == "4/4":
        data.append(repeat("1r"))
    else:
        data.extend(repeat(reciprocal + "r") for _ in range(numerator))
    data.extend((repeat("=="), repeat("*-")))
    return "\n".join(data) + "\n"


def repair_kern(text: str) -> str:
    lines = [line for line in text.replace("\r", "").splitlines() if line.strip()]
    if not lines or not lines[0].startswith("**kern"):
        return safe_fallback_kern()
    columns = len(lines[0].split("\t"))
    fixed: list[str] = []
    for line in lines:
        fields = line.split("\t")
        if len(fields) < columns:
            fields.extend(["."] * (columns - len(fields)))
        fixed.append("\t".join(fields[:columns]))
    if not fixed[-1].startswith("*-"):
        fixed.append("\t".join(["*-"] * columns))
    return "\n".join(fixed) + "\n"
