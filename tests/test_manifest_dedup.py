from __future__ import annotations

import json
from pathlib import Path

from a2s.data.manifest_dedup import deduplicate_manifest_rows


def _row(sample_id: str, split: str, audio: str, piece_id: str = "unknown") -> dict[str, str]:
    return {
        "sample_id": sample_id, "split": split, "audio_path": audio,
        "kern_path": "", "duration_sec": "10.0", "piece_id": piece_id,
        "has_audio": "true", "has_kern": "true",
    }


def test_audio_hash_dedup_prefers_holdout_and_structured_id(tmp_path: Path) -> None:
    audio_a = tmp_path / "a.flac"
    audio_b = tmp_path / "b.flac"
    audio_a.write_bytes(b"same-audio")
    audio_b.write_bytes(b"same-audio")
    rows = [
        _row("train_000001", "train", "a.flac"),
        _row("test_piece_000001", "test", "b.flac", piece_id="piece"),
    ]
    deduped, removed, report = deduplicate_manifest_rows(rows, tmp_path, tmp_path / "oracle")
    assert [row["sample_id"] for row in deduped] == ["test_piece_000001"]
    assert removed[0]["removed_sample_id"] == "train_000001"
    assert report["cross_split_duplicate_groups"] == 1
    assert report["num_removed_rows"] == 1


def test_oracle_duration_fallback_for_missing_audio_alias(tmp_path: Path) -> None:
    oracle_root = tmp_path / "oracle"
    notes = {
        "meter": "4/4", "key": "*k[]", "num_measures": 1,
        "notes": [{
            "voice": "violin_1", "pitch": 72, "onset_beat": 0.0,
            "offset_beat": 1.0, "duration_beat": 1.0,
        }],
    }
    for split, sample_id in (("valid", "val_000001"), ("valid", "val_0000_000001")):
        target = oracle_root / split / f"{sample_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({**notes, "sample_id": sample_id}), encoding="utf-8")
    existing = tmp_path / "nested.flac"
    existing.write_bytes(b"audio")
    rows = [
        _row("val_000001", "valid", "missing.flac"),
        _row("val_0000_000001", "valid", "nested.flac", piece_id="0000"),
    ]
    deduped, removed, report = deduplicate_manifest_rows(rows, tmp_path, oracle_root)
    assert [row["sample_id"] for row in deduped] == ["val_0000_000001"]
    assert removed[0]["method"] == "oracle_duration_fallback"
    assert report["oracle_duration_fallback_groups"] == 1


def test_oracle_alias_fallback_accepts_unknown_missing_duration(tmp_path: Path) -> None:
    oracle_root = tmp_path / "oracle"
    notes = {
        "meter": "4/4", "key": "*k[]", "num_measures": 1,
        "notes": [{
            "voice": "violin_1", "pitch": 72, "onset_beat": 0.0,
            "offset_beat": 1.0, "duration_beat": 1.0,
        }],
    }
    for sample_id in ("train_000296", "train_0000_000296"):
        target = oracle_root / "train" / f"{sample_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({**notes, "sample_id": sample_id}), encoding="utf-8")
    existing = tmp_path / "nested.flac"
    existing.write_bytes(b"audio")
    missing = _row("train_000296", "train", "")
    missing.update({"duration_sec": "-1.0", "excerpt_id": "000296", "has_audio": "false"})
    structured = _row("train_0000_000296", "train", "nested.flac", piece_id="0000")
    structured["excerpt_id"] = "000296"
    deduped, removed, report = deduplicate_manifest_rows(
        [missing, structured], tmp_path, oracle_root
    )
    assert [row["sample_id"] for row in deduped] == ["train_0000_000296"]
    assert removed[0]["removed_sample_id"] == "train_000296"
    assert report["oracle_duration_fallback_groups"] == 1
    assert report["kern_content_conflict_groups"] == 0
