from __future__ import annotations

import csv
from pathlib import Path


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_failures(path: str | Path, failures: list[dict[str, str]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "stage", "error"))
        writer.writeheader()
        writer.writerows(failures)
