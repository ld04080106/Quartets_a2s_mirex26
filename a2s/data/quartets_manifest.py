from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .kern_parser import count_measures
from .metadata_parser import parse_kern_metadata

FIELDS = (
    "sample_id", "composer", "piece_id", "excerpt_id", "split",
    "audio_path", "kern_path", "metadata_path", "duration_sec", "duration_warning",
    "num_measures", "meter", "key", "has_audio", "has_kern", "has_metadata",
)


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _relative(path: Path | None, project_root: Path) -> str:
    if path is None:
        return ""
    return Path(os.path.relpath(path.resolve(), project_root.resolve())).as_posix()


def audio_duration(path: Path) -> float:
    import soundfile as sf
    return float(sf.info(str(path)).duration)


def _read_existing_splits(root: Path) -> dict[str, str]:
    for name in ("splits.csv", "split.csv"):
        path = root / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.DictReader(handle)
            result = {}
            for row in rows:
                sample_id = row.get("sample_id") or row.get("id")
                split = (row.get("split") or "").lower()
                if sample_id and split:
                    result[sample_id] = "valid" if split in {"val", "validation", "dev"} else split
            return result
    return {}


def _directory_split(path: Path | None) -> str | None:
    if path is None:
        return None
    for part in path.parts:
        value = part.lower()
        if value in {"train", "test"}:
            return value
        if value in {"val", "valid", "validation", "dev"}:
            return "valid"
    return None


def _infer_piece_excerpt(sample_id: str) -> tuple[str, str]:
    cleaned = re.sub(r"^(train|test|val|valid)[_-]", "", sample_id, flags=re.I)
    match = re.match(r"(.+?)[_-](?:excerpt|segment|chunk)[_-]?(\d+)$", cleaned, flags=re.I)
    if match:
        return match.group(1), match.group(2)
    parts = re.split(r"[_-]", cleaned)
    if len(parts) == 1 and cleaned.isdigit():
        return "unknown", cleaned
    if len(parts) >= 2 and parts[-1].isdigit():
        piece = "_".join(parts[:-1]) or "unknown"
        return piece, parts[-1]
    return cleaned or "unknown", "unknown"


def _composer_from_files(metadata: Path | None, kern: Path | None, dataset_root: Path) -> str:
    if metadata and metadata.suffix.lower() == ".json":
        try:
            data = json.loads(metadata.read_text(encoding="utf-8-sig"))
            for key in ("composer", "composer_name", "author"):
                if data.get(key):
                    return str(data[key]).strip()
        except (OSError, ValueError, TypeError):
            pass
    if kern and kern.exists():
        text = kern.read_text(encoding="utf-8-sig", errors="replace")
        for line in text.splitlines():
            if line.lower().startswith(("!!!com:", "!!!composer:")):
                return line.split(":", 1)[1].strip() or "unknown"
    ignored = {"audio", "kern", "metadata", "train", "test", "val", "valid"}
    for path in (metadata, kern):
        if path:
            try:
                relative_parts = path.resolve().relative_to(dataset_root.resolve()).parts[:-1]
            except ValueError:
                relative_parts = ()
            for part in reversed(relative_parts):
                if part.lower() not in ignored:
                    return part
    return "unknown"


def _deterministic_split(group: str, config: dict[str, Any]) -> str:
    seed = int(config.get("seed", 42))
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    train_ratio = float(config.get("train_ratio", 0.8))
    valid_ratio = float(config.get("valid_ratio", 0.1))
    if value < train_ratio:
        return "train"
    if value < train_ratio + valid_ratio:
        return "valid"
    return "test"


def _index_files(root: Path, directory: str, extensions: set[str]) -> dict[str, Path]:
    base = root / directory
    if not base.exists():
        return {}
    indexed: dict[str, Path] = {}
    for path in sorted(item for item in base.rglob("*") if item.is_file() and item.suffix.lower() in extensions):
        indexed.setdefault(path.stem, path)
    return indexed


def prepare_quartets_dataset(config: dict[str, Any], project_root: str | Path) -> tuple[list[dict[str, object]], dict[str, Any], list[dict[str, str]]]:
    project = Path(project_root).resolve()
    dataset_config = config.get("dataset", {})
    root_value = Path(dataset_config.get("root_dir", "data/raw/quartets"))
    root = root_value if root_value.is_absolute() else project / root_value
    root = root.resolve()
    audio_ext = str(dataset_config.get("audio_ext", ".flac")).lower()
    kern_ext = str(dataset_config.get("kern_ext", ".krn")).lower()
    metadata_ext = str(dataset_config.get("metadata_ext", ".json")).lower()
    audio_files = _index_files(root, "audio", {audio_ext})
    kern_files = _index_files(root, "kern", {kern_ext})
    metadata_files = _index_files(root, "metadata", {metadata_ext, ".json", ".krn"})
    sample_ids = sorted(set(audio_files) | set(kern_files) | set(metadata_files))
    existing_splits = _read_existing_splits(root)
    split_config = config.get("split", {})
    max_duration = float(config.get("audio", {}).get("max_duration_sec", 30.0))
    failures: list[dict[str, str]] = []
    rows: list[dict[str, object]] = []

    logger = logging.getLogger("a2s")
    for row_index, sample_id in enumerate(sample_ids, start=1):
        if row_index == 1 or row_index % 1000 == 0:
            logger.info("preparing sample %d/%d", row_index, len(sample_ids))
        audio = audio_files.get(sample_id)
        kern = kern_files.get(sample_id)
        metadata = metadata_files.get(sample_id)
        piece_id, excerpt_id = _infer_piece_excerpt(sample_id)
        composer = _composer_from_files(metadata, kern, root)
        directory_split = _directory_split(audio or kern or metadata)
        split = existing_splits.get(sample_id) or directory_split
        if split not in {"train", "valid", "test"}:
            group = f"{composer}:{piece_id}" if piece_id != "unknown" else sample_id
            split = _deterministic_split(group, split_config)
        duration = -1.0
        if audio:
            try:
                duration = audio_duration(audio)
            except Exception as exc:
                failures.append({"sample_id": sample_id, "stage": "audio_duration", "error": str(exc)})
        else:
            failures.append({"sample_id": sample_id, "stage": "prepare", "error": "missing audio"})
        meter = key = ""
        num_measures = 0
        if kern:
            try:
                text = kern.read_text(encoding="utf-8-sig", errors="replace")
                meta = parse_kern_metadata(text)
                if len(meta.kern_indices) != 4:
                    raise ValueError(f"expected 4 **kern spines, found {len(meta.kern_indices)}")
                meter_match = re.search(r"(?:^|\t)\*M(\d+/\d+)(?:\t|$)", text, flags=re.M)
                key_match = re.search(r"(?:^|\t)(\*k\[[^]]*\])(?:\t|$)", text, flags=re.M)
                meter = meter_match.group(1) if meter_match else ""
                key = key_match.group(1) if key_match else ""
                num_measures = count_measures(text)
            except Exception as exc:
                failures.append({"sample_id": sample_id, "stage": "kern_parse", "error": str(exc)})
        else:
            failures.append({"sample_id": sample_id, "stage": "prepare", "error": "missing kern"})
        rows.append({
            "sample_id": sample_id, "composer": composer or "unknown", "piece_id": piece_id,
            "excerpt_id": excerpt_id, "split": split, "audio_path": _relative(audio, project),
            "kern_path": _relative(kern, project), "metadata_path": _relative(metadata, project),
            "duration_sec": round(duration, 6), "duration_warning": _bool(duration > max_duration),
            "num_measures": num_measures, "meter": meter, "key": key,
            "has_audio": _bool(audio is not None), "has_kern": _bool(kern is not None),
            "has_metadata": _bool(metadata is not None),
        })

    split_counts = Counter(str(row["split"]) for row in rows)
    composer_counts = Counter(str(row["composer"]) for row in rows)
    report = {
        "num_samples_total": len(rows), "num_train": split_counts["train"],
        "num_valid": split_counts["valid"], "num_test": split_counts["test"],
        "missing_audio": sum(row["has_audio"] == "false" for row in rows),
        "missing_kern": sum(row["has_kern"] == "false" for row in rows),
        "missing_metadata": sum(row["has_metadata"] == "false" for row in rows),
        "duration_over_30s": sum(row["duration_warning"] == "true" for row in rows),
        "composer_counts": dict(sorted(composer_counts.items())), "failed_samples": failures,
        "used_existing_split_file": bool(existing_splits),
    }
    return rows, report, failures


def write_split_manifests(rows: list[dict[str, object]], out_dir: str | Path) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        with (target / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(row for row in rows if row["split"] == split)
