#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from _common import read_manifest
from a2s.data.note_event import NoteEventSequence
from a2s.stage1_amt.yourmt3_dataset import oracle_notes_in_seconds, voice_programs_from_kern
from a2s.utils.config import deep_get, load_config
from a2s.utils.json_io import load_json, save_json
from a2s.utils.logging import configure_logging


FIELDNAMES = [
    "sample_id", "original_sample_id", "oracle_sample_id", "composer", "piece_id", "excerpt_id", "split",
    "audio_path", "midi_path", "kern_path", "metadata_path", "duration_sec", "tempo_bpm",
    "tempo_scale", "leading_silence_sec", "soundfont_path", "velocity_seed", "num_measures", "meter", "key",
    "has_audio", "has_kern", "has_metadata", "is_synthetic",
]


def _project_path(project: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def _rel(project: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise FileNotFoundError(f"command not found: {command}")


def _write_midi(
    records: list[dict[str, Any]],
    midi_path: Path,
    tempo_bpm: float,
    velocity_jitter: int,
    rng: random.Random,
    leading_silence_sec: float = 0.0,
) -> None:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    instruments: dict[int, Any] = {}
    for record in records:
        program = int(record["program"])
        if program not in instruments:
            instruments[program] = pretty_midi.Instrument(program=program)
        velocity = 80 + rng.randint(-velocity_jitter, velocity_jitter)
        velocity = max(1, min(127, velocity))
        start = float(record["onset"]) + leading_silence_sec
        end = float(record["offset"]) + leading_silence_sec
        if end <= start:
            continue
        instruments[program].notes.append(pretty_midi.Note(velocity, int(record["pitch"]), start, end))
    for instrument in instruments.values():
        instrument.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
        midi.instruments.append(instrument)
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(midi_path))


def _render_audio(
    midi_path: Path,
    wav_path: Path,
    flac_path: Path,
    soundfont: Path,
    fluidsynth_cmd: str,
    ffmpeg_cmd: str,
    gain: float,
    sample_rate: int,
) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    flac_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        fluidsynth_cmd, "-ni", "-g", str(gain), "-r", str(sample_rate),
        "-F", str(wav_path), str(soundfont), str(midi_path),
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run([
        ffmpeg_cmd, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(wav_path), "-ar", str(sample_rate), "-ac", "1", str(flac_path),
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _audio_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render aligned synthetic Quartet audio from oracle **kern note events.")
    parser.add_argument("--config", default="configs/stage1_yourmt3_synth_data_pad05.yaml")
    parser.add_argument("--splits")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    manifest_dir = _project_path(project, deep_get(config, "dataset.manifest_dir", "data/manifests_dedup"))
    oracle_root = _project_path(project, deep_get(config, "dataset.oracle_events_dir", "data/oracle_events"))
    out_dir = _project_path(project, deep_get(config, "synthetic.out_dir", "data/synthetic_quartets"))
    synth_manifest_dir = _project_path(project, deep_get(config, "synthetic.manifest_dir", "data/manifests_synth"))
    reports_dir = _project_path(project, deep_get(config, "output.reports_dir", "data/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    synth_manifest_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(reports_dir / "render_synthetic_quartets.log")

    fluidsynth_cmd = str(deep_get(config, "synthetic.fluidsynth_cmd", "fluidsynth"))
    ffmpeg_cmd = str(deep_get(config, "synthetic.ffmpeg_cmd", "ffmpeg"))
    _require_command(fluidsynth_cmd)
    _require_command(ffmpeg_cmd)
    soundfonts = [_project_path(project, path) for path in deep_get(config, "synthetic.soundfonts", [])]
    if not soundfonts:
        raise SystemExit("configure at least one synthetic.soundfonts entry")
    missing_soundfonts = [str(path) for path in soundfonts if not path.is_file()]
    if missing_soundfonts:
        raise SystemExit("missing soundfont(s):\n" + "\n".join(missing_soundfonts))

    split_names = args.splits.split(",") if args.splits else deep_get(config, "synthetic.splits", ["train", "valid"])
    split_names = [split.strip() for split in split_names if split.strip()]
    variants_per_sample = int(deep_get(config, "synthetic.variants_per_sample", 1))
    base_tempo = float(deep_get(config, "synthetic.base_tempo_bpm", 120.0))
    tempo_jitter = float(deep_get(config, "synthetic.tempo_jitter_pct", 0.06))
    velocity_jitter = int(deep_get(config, "synthetic.velocity_jitter", 10))
    leading_silence_sec = float(deep_get(config, "synthetic.leading_silence_sec", 0.0))
    if leading_silence_sec < 0:
        raise SystemExit("synthetic.leading_silence_sec must be >= 0")
    sample_rate = int(deep_get(config, "audio.sample_rate", 16000))
    keep_wav = bool(deep_get(config, "synthetic.keep_wav", False))
    gain = float(deep_get(config, "synthetic.fluidsynth_gain", 0.7))
    seed = int(deep_get(config, "synthetic.random_seed", 42))

    failures: list[dict[str, str]] = []
    report: dict[str, Any] = {"splits": {}, "failed_samples": failures, "soundfonts": [str(path) for path in soundfonts]}

    def render_one(row: dict[str, str], split: str, variant: int, source_index: int) -> dict[str, Any]:
        original_id = row["sample_id"]
        sample_id = f"{original_id}__syn{variant:02d}"
        rng = random.Random(seed + source_index * 1009 + variant * 9173)
        tempo_scale = 1.0 + rng.uniform(-tempo_jitter, tempo_jitter)
        tempo_bpm = base_tempo * tempo_scale
        soundfont = soundfonts[(source_index + variant) % len(soundfonts)]
        oracle_path = oracle_root / split / f"{original_id}.json"
        kern_path = _project_path(project, row["kern_path"])
        if not oracle_path.is_file():
            raise FileNotFoundError(f"missing oracle: {oracle_path}")
        sequence = NoteEventSequence.from_dict(load_json(oracle_path))
        programs_by_voice, _warnings = voice_programs_from_kern(kern_path)
        records = oracle_notes_in_seconds(sequence, tempo_bpm, programs_by_voice)
        if not records:
            raise ValueError(f"no notes after oracle conversion: {original_id}")
        base = out_dir / split / sample_id
        midi_path = base.with_suffix(".mid")
        wav_path = base.with_suffix(".wav")
        flac_path = base.with_suffix(".flac")
        if not (args.skip_existing and flac_path.is_file()):
            _write_midi(records, midi_path, tempo_bpm, velocity_jitter, rng, leading_silence_sec)
            _render_audio(midi_path, wav_path, flac_path, soundfont, fluidsynth_cmd, ffmpeg_cmd, gain, sample_rate)
            if not keep_wav and wav_path.exists():
                wav_path.unlink()
        duration = _audio_duration(flac_path)
        return {
            "sample_id": sample_id,
            "original_sample_id": original_id,
            "oracle_sample_id": original_id,
            "composer": row.get("composer", "unknown"),
            "piece_id": row.get("piece_id", ""),
            "excerpt_id": row.get("excerpt_id", ""),
            "split": split,
            "audio_path": _rel(project, flac_path),
            "midi_path": _rel(project, midi_path),
            "kern_path": row.get("kern_path", ""),
            "metadata_path": row.get("metadata_path", ""),
            "duration_sec": f"{duration:.6f}",
            "tempo_bpm": f"{tempo_bpm:.6f}",
            "tempo_scale": f"{tempo_scale:.8f}",
            "leading_silence_sec": f"{leading_silence_sec:.6f}",
            "soundfont_path": _rel(project, soundfont),
            "velocity_seed": str(seed + source_index * 1009 + variant * 9173),
            "num_measures": row.get("num_measures", ""),
            "meter": row.get("meter", ""),
            "key": row.get("key", ""),
            "has_audio": "true",
            "has_kern": row.get("has_kern", "true"),
            "has_metadata": row.get("has_metadata", "false"),
            "is_synthetic": "true",
        }

    for split in split_names:
        rows = read_manifest(manifest_dir / f"{split}.csv")
        if args.limit:
            rows = rows[:args.limit]
        futures = {}
        rendered: dict[tuple[int, int], dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            for index, row in enumerate(rows):
                for variant in range(variants_per_sample):
                    futures[pool.submit(render_one, row, split, variant, index)] = (index, variant, row)
            for completed, future in enumerate(as_completed(futures), start=1):
                index, variant, row = futures[future]
                try:
                    rendered[(index, variant)] = future.result()
                except Exception as exc:
                    failures.append({"sample_id": row.get("sample_id", ""), "split": split, "error": f"{type(exc).__name__}: {exc}"})
                    logger.exception("failed rendering %s variant %d", row.get("sample_id"), variant)
                if completed == 1 or completed % 500 == 0:
                    logger.info("synthetic render %s: %d/%d", split, completed, len(futures))
        ordered = [rendered[key] for key in sorted(rendered)]
        with (synth_manifest_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(ordered)
        report["splits"][split] = {
            "source_rows": len(rows),
            "variants_per_sample": variants_per_sample,
            "rendered": len(ordered),
            "failed": len(rows) * variants_per_sample - len(ordered),
        }

    with (reports_dir / "render_synthetic_quartets_failed_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "split", "error"))
        writer.writeheader()
        writer.writerows(failures)
    report["num_failed"] = len(failures)
    save_json(reports_dir / "render_synthetic_quartets_report.json", report)
    if failures:
        raise SystemExit(1)
    logger.info("synthetic render complete")


if __name__ == "__main__":
    main()
