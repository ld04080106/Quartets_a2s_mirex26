from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from _common import read_manifest
from a2s.data.metadata_parser import parse_kern_metadata
from a2s.data.note_event import NoteEventSequence
from a2s.stage1_amt.yourmt3_dataset import oracle_notes_in_seconds, voice_programs_from_kern
from a2s.utils.config import deep_get, load_config
from a2s.utils.json_io import load_json, save_json
from a2s.utils.logging import configure_logging


def _path(project: Path, value: str) -> Path:
    expanded = Path(os.path.expandvars(value)).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project / expanded).resolve()


def _resolve_yourmt3_source(project: Path, value: str | None) -> Path:
    tried: list[Path] = []
    raw = value or "third_party/YourMT3"
    primary = _path(project, raw)
    candidates = [primary]
    if primary.name == "YourMT":
        candidates.append(primary.with_name("YourMT3"))
    env_value = os.environ.get("YOURMT3_SOURCE_DIR")
    if env_value:
        candidates.insert(0, _path(project, env_value))
    for candidate in candidates:
        if candidate in tried:
            continue
        tried.append(candidate)
        if (candidate / "amt" / "src" / "utils" / "note_event_dataclasses.py").is_file():
            return candidate
    tried_text = "\n".join(f"  - {path}" for path in tried)
    raise SystemExit("YourMT3 source is incomplete. Tried:\n" + tried_text)


def _convert_audio(source: Path, target: Path, sample_rate: int) -> int:
    import soundfile as sf
    import torch
    import torchaudio

    samples, source_rate = sf.read(str(source), dtype="float32", always_2d=True)
    audio = torch.from_numpy(samples).transpose(0, 1).mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    audio = audio.clamp(-1.0, 1.0).to(torch.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), audio.squeeze(0).cpu().numpy(), sample_rate, subtype="PCM_16")
    return int(audio.shape[-1])


def _wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes()


def _shift_records(records: list[dict[str, Any]], seconds: float) -> list[dict[str, Any]]:
    if not seconds:
        return records
    shifted = []
    for item in records:
        changed = dict(item)
        changed["onset"] = float(changed["onset"]) + seconds
        changed["offset"] = float(changed["offset"]) + seconds
        shifted.append(changed)
    return shifted


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Quartets in native YourMT3 training format.")
    parser.add_argument("--config", default="configs/stage1_yourmt3_synth_finetune_h800_4gpu_pad05_nops.yaml")
    parser.add_argument("--splits", default="train,valid,test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--source_dir")
    parser.add_argument("--manifest_dir")
    parser.add_argument("--prepared_dir")
    parser.add_argument("--data_home")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    source_dir = _resolve_yourmt3_source(
        project,
        args.source_dir or deep_get(config, "yourmt3.source_dir", "third_party/YourMT3"),
    )
    source_python = source_dir / "amt" / "src"
    if not (source_python / "utils" / "note_event_dataclasses.py").exists():
        raise SystemExit(f"YourMT3 source is incomplete: {source_dir}")
    sys.path.insert(0, str(source_python))
    import numpy as np
    from utils.midi import note_event2midi
    from utils.note2event import mix_notes, note2note_event
    from utils.note_event_dataclasses import Note

    data_home = _path(project, args.data_home or deep_get(config, "yourmt3.data_home", "data/yourmt3_data"))
    prepared = _path(project, args.prepared_dir or deep_get(config, "yourmt3.prepared_dir", "data/yourmt3_quartets"))
    manifest_dir = _path(project, args.manifest_dir or deep_get(config, "data.manifest_dir", "data/manifests_dedup"))
    oracle_root = _path(project, deep_get(config, "data.oracle_events_dir", "data/oracle_events"))
    reports_dir = _path(project, deep_get(config, "output.reports_dir", "data/reports"))
    sample_rate = int(deep_get(config, "audio.sample_rate", 16000))
    default_tempo = float(deep_get(config, "data.default_tempo_bpm", 120.0))
    jobs = args.jobs or int(deep_get(config, "data.prepare_jobs", 4))
    reports_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(reports_dir / "yourmt3_prepare.log")
    split_names = [item.strip() for item in args.splits.split(",") if item.strip()]
    index_dir = data_home / "yourmt3_indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    report: dict[str, Any] = {"splits": {}, "failed_samples": failures, "unknown_instruments": {}}

    def process(row: dict[str, str], split: str) -> tuple[dict[str, Any], list[str]]:
        sample_id = row["sample_id"]
        oracle_sample_id = row.get("oracle_sample_id") or row.get("original_sample_id") or sample_id
        audio_source = _path(project, row.get("audio_path", ""))
        kern_path = _path(project, row.get("kern_path", ""))
        oracle_path = oracle_root / split / f"{oracle_sample_id}.json"
        if not audio_source.is_file():
            raise FileNotFoundError(f"audio not found: {audio_source}")
        if not kern_path.is_file() or not oracle_path.is_file():
            raise FileNotFoundError(f"annotation missing: {kern_path} / {oracle_path}")
        base = prepared / split / sample_id
        wav_path = base.with_suffix(".wav")
        notes_path = base.with_name(base.name + "_notes.npy")
        events_path = base.with_name(base.name + "_note_events.npy")
        midi_path = base.with_suffix(".mid")
        if args.skip_existing and wav_path.is_file():
            n_frames = _wav_frames(wav_path)
        else:
            n_frames = _convert_audio(audio_source, wav_path, sample_rate)
        sequence = NoteEventSequence.from_dict(load_json(oracle_path))
        metadata = parse_kern_metadata(kern_path)
        tempo_value = row.get("tempo_bpm") or row.get("synthetic_tempo_bpm")
        tempo = float(tempo_value or metadata.tempo_bpm or sequence.tempo_bpm or default_tempo)
        programs_by_voice, warnings = voice_programs_from_kern(kern_path)
        records = oracle_notes_in_seconds(sequence, tempo, programs_by_voice)
        leading_silence_sec = float(row.get("leading_silence_sec") or row.get("label_shift_sec") or 0.0)
        records = _shift_records(records, leading_silence_sec)
        notes = [Note(**{key: item[key] for key in ("is_drum", "program", "onset", "offset", "pitch", "velocity")}) for item in records]
        notes = mix_notes([notes], sort=True, trim_overlap=True, fix_offset=True)
        note_events = note2note_event(notes, sort=True)
        programs = sorted({int(note.program) for note in notes})
        duration_sec = n_frames / sample_rate
        payload = {"sample_id": sample_id, "program": programs, "is_drum": [0] * len(programs), "duration_sec": duration_sec}
        base.parent.mkdir(parents=True, exist_ok=True)
        np.save(notes_path, {**payload, "notes": notes}, allow_pickle=True)
        np.save(events_path, {**payload, "note_events": note_events}, allow_pickle=True)
        note_event2midi(list(note_events), str(midi_path))
        return ({
            "sample_id": sample_id, "n_frames": n_frames,
            "mix_audio_file": str(wav_path), "notes_file": str(notes_path),
            "note_events_file": str(events_path), "midi_file": str(midi_path),
            "program": programs, "is_drum": [0] * len(programs),
        }, warnings)

    for split in split_names:
        rows = read_manifest(manifest_dir / f"{split}.csv")
        rows = rows[args.start:]
        if args.limit:
            rows = rows[:args.limit]
        results: dict[int, dict[str, Any]] = {}
        warning_count = 0
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = {pool.submit(process, row, split): (index, row) for index, row in enumerate(rows)}
            for completed, future in enumerate(as_completed(futures), start=1):
                index, row = futures[future]
                try:
                    entry, warnings = future.result()
                    results[index] = entry
                    warning_count += len(warnings)
                    for warning in warnings:
                        report["unknown_instruments"][warning] = report["unknown_instruments"].get(warning, 0) + 1
                except Exception as exc:  # keep the full preparation batch alive
                    failures.append({"sample_id": row.get("sample_id", ""), "split": split, "error": f"{type(exc).__name__}: {exc}"})
                    logger.exception("failed preparing %s", row.get("sample_id"))
                if completed == 1 or completed % 500 == 0:
                    logger.info("YourMT3 preparation %s: %d/%d", split, completed, len(rows))
        ordered = [results[index] for index in sorted(results)]
        yourmt3_split = "validation" if split == "valid" else split
        index_payload = {str(index): entry for index, entry in enumerate(ordered)}
        (index_dir / f"quartets_{yourmt3_split}_file_list.json").write_text(
            json.dumps(index_payload, indent=2), encoding="utf-8"
        )
        report["splits"][split] = {"requested": len(rows), "success": len(ordered), "failed": len(rows) - len(ordered), "warnings": warning_count}

    with (reports_dir / "yourmt3_prepare_failed_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "split", "error"))
        writer.writeheader(); writer.writerows(failures)
    report["num_failed"] = len(failures)
    report["data_home"] = str(data_home)
    save_json(reports_dir / "yourmt3_prepare_report.json", report)
    if failures:
        logger.error("YourMT3 preparation completed with %d failures", len(failures))
        raise SystemExit(1)
    logger.info("YourMT3 preparation complete")


if __name__ == "__main__":
    main()
