from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _resolve(project_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _oracle_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        notes = [
            {
                "voice": note.get("voice"),
                "pitch": note.get("pitch"),
                "onset_beat": note.get("onset_beat"),
                "offset_beat": note.get("offset_beat"),
                "duration_beat": note.get("duration_beat"),
            }
            for note in payload.get("notes", [])
        ]
        canonical = {
            "meter": payload.get("meter"),
            "key": payload.get("key"),
            "num_measures": payload.get("num_measures"),
            "notes": notes,
        }
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
    except (OSError, ValueError, TypeError):
        return None


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _keeper_rank(row: dict[str, str]) -> tuple[Any, ...]:
    # Preserve holdout samples if the same bytes leaked into another split.
    split_rank = {"test": 0, "valid": 1, "train": 2}
    structured = row.get("piece_id", "unknown") not in {"", "unknown"}
    complete = row.get("has_audio") == "true" and row.get("has_kern") == "true"
    specificity = row.get("sample_id", "").count("_")
    return (
        split_rank.get(row.get("split", ""), 3),
        0 if complete else 1,
        0 if structured else 1,
        -specificity,
        row.get("sample_id", ""),
    )


def deduplicate_manifest_rows(
    rows: list[dict[str, str]],
    project_root: str | Path,
    oracle_root: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    project = Path(project_root).resolve()
    logger = logging.getLogger("a2s")
    oracle = Path(oracle_root)
    if not oracle.is_absolute():
        oracle = (project / oracle).resolve()
    dsu = _DisjointSet(len(rows))
    audio_paths: list[Path | None] = []
    audio_sizes: list[int | None] = []
    audio_hashes: dict[int, str] = {}
    oracle_hashes: dict[int, str] = {}
    unhashable: list[str] = []

    # Hash only size+duration collision candidates; this avoids reading every
    # audio file in a large manifest while still proving byte identity.
    size_duration_groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        path = _resolve(project, row.get("audio_path"))
        audio_paths.append(path)
        if path and path.is_file():
            size = path.stat().st_size
            audio_sizes.append(size)
            size_duration_groups[(size, row.get("duration_sec", ""))].append(index)
        else:
            audio_sizes.append(None)
            unhashable.append(row.get("sample_id", str(index)))

    audio_duplicate_groups = 0
    hash_candidate_count = sum(len(group) for group in size_duration_groups.values() if len(group) > 1)
    logger.info(
        "dedup scan: rows=%d audio_hash_candidates=%d unhashable_audio=%d",
        len(rows), hash_candidate_count, len(unhashable),
    )
    hashed_count = 0
    for candidates in size_duration_groups.values():
        if len(candidates) < 2:
            continue
        by_hash: dict[str, list[int]] = defaultdict(list)
        for index in candidates:
            digest = _sha256(audio_paths[index])  # type: ignore[arg-type]
            audio_hashes[index] = digest
            by_hash[digest].append(index)
            hashed_count += 1
            if hashed_count % 1000 == 0:
                logger.info("hashed duplicate candidate %d/%d", hashed_count, hash_candidate_count)
        for duplicate_indices in by_hash.values():
            if len(duplicate_indices) < 2:
                continue
            audio_duplicate_groups += 1
            for index in duplicate_indices[1:]:
                dsu.union(duplicate_indices[0], index)

    # Fallback for aliases whose audio is not present on this machine. Prefer an
    # exact duration match. For broken manifest rows where duration is unknown
    # (-1), require the same split and excerpt id in addition to an identical
    # oracle score. This retains distinct performances of the same score.
    fallback_candidates: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        oracle_path = oracle / row.get("split", "") / f"{row.get('sample_id', '')}.json"
        digest = _oracle_fingerprint(oracle_path)
        if digest:
            oracle_hashes[index] = digest
            duration = row.get("duration_sec", "")
            try:
                duration_valid = float(duration) >= 0.0
            except (TypeError, ValueError):
                duration_valid = False
            if duration_valid:
                fallback_candidates[("duration", digest, duration)].append(index)
            excerpt_id = row.get("excerpt_id", "")
            if excerpt_id:
                fallback_candidates[
                    ("alias", digest, row.get("split", ""), excerpt_id)
                ].append(index)
    fallback_duplicate_groups = 0
    for candidates in fallback_candidates.values():
        if len(candidates) < 2 or not any(audio_sizes[index] is None for index in candidates):
            continue
        roots_before = {dsu.find(index) for index in candidates}
        if len(roots_before) < 2:
            continue
        fallback_duplicate_groups += 1
        for index in candidates[1:]:
            dsu.union(candidates[0], index)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[dsu.find(index)].append(index)
    duplicate_components = [indices for indices in components.values() if len(indices) > 1]
    removed_indices: set[int] = set()
    removed_rows: list[dict[str, str]] = []
    cross_split_groups = 0
    kern_conflict_groups = 0
    kern_byte_difference_groups = 0
    group_summaries = []
    for indices in duplicate_components:
        keeper = min(indices, key=lambda index: _keeper_rank(rows[index]))
        splits = {rows[index].get("split", "") for index in indices}
        cross_split_groups += len(splits) > 1
        kern_hashes = set()
        for index in indices:
            kern_path = _resolve(project, rows[index].get("kern_path"))
            if kern_path and kern_path.is_file():
                kern_hashes.add(_sha256(kern_path))
        kern_byte_difference = len(kern_hashes) > 1
        kern_byte_difference_groups += kern_byte_difference
        semantic_hashes = {oracle_hashes[index] for index in indices if index in oracle_hashes}
        # Oracle fingerprints ignore bar numbers, headers and formatting. Use
        # raw **kern bytes only when semantic oracle data is unavailable.
        if all(index in oracle_hashes for index in indices):
            kern_conflict = len(semantic_hashes) > 1
        else:
            kern_conflict = kern_byte_difference
        kern_conflict_groups += kern_conflict
        method = "audio_sha256" if any(index in audio_hashes for index in indices) else "oracle_duration_fallback"
        group_summaries.append({
            "kept_sample_id": rows[keeper].get("sample_id"),
            "kept_split": rows[keeper].get("split"),
            "members": [rows[index].get("sample_id") for index in indices],
            "splits": sorted(splits),
            "method": method,
            "kern_content_conflict": kern_conflict,
            "kern_byte_difference": kern_byte_difference,
        })
        for index in indices:
            if index == keeper:
                continue
            removed_indices.add(index)
            removed_rows.append({
                "removed_sample_id": rows[index].get("sample_id", ""),
                "removed_split": rows[index].get("split", ""),
                "kept_sample_id": rows[keeper].get("sample_id", ""),
                "kept_split": rows[keeper].get("split", ""),
                "method": method,
                "kern_content_conflict": str(kern_conflict).lower(),
            })

    deduplicated = [row for index, row in enumerate(rows) if index not in removed_indices]
    before = Counter(row.get("split", "") for row in rows)
    after = Counter(row.get("split", "") for row in deduplicated)
    report = {
        "num_input_rows": len(rows),
        "num_output_rows": len(deduplicated),
        "num_removed_rows": len(removed_indices),
        "num_duplicate_groups": len(duplicate_components),
        "audio_sha256_groups": audio_duplicate_groups,
        "oracle_duration_fallback_groups": fallback_duplicate_groups,
        "cross_split_duplicate_groups": cross_split_groups,
        "kern_content_conflict_groups": kern_conflict_groups,
        "kern_byte_difference_groups": kern_byte_difference_groups,
        "counts_before": dict(sorted(before.items())),
        "counts_after": dict(sorted(after.items())),
        "unhashable_audio_count": len(unhashable),
        "unhashable_audio_samples": unhashable,
        "duplicate_groups": group_summaries,
    }
    return deduplicated, removed_rows, report
