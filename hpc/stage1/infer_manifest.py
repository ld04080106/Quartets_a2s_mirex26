#!/usr/bin/env python3
"""Load an AMT checkpoint once and transcribe one manifest (optionally sharded)."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a2s.data.metadata_parser import parse_kern_metadata  # noqa: E402
from a2s.stage1_amt.output_normalizer import load_amt_output  # noqa: E402
from a2s.stage1_amt.postprocess_mt3 import postprocess  # noqa: E402
from a2s.utils.config import deep_get, load_config  # noqa: E402
from a2s.utils.json_io import save_json  # noqa: E402
from a2s.utils.logging import configure_logging  # noqa: E402
from scripts._common import read_manifest, write_failures  # noqa: E402


def _pad_audio_leading(source: Path, seconds: float, work_dir: Path) -> Path:
    if seconds <= 0:
        return source
    import soundfile as sf
    import numpy as np

    samples, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
    pad_frames = int(round(seconds * sample_rate))
    if pad_frames <= 0:
        return source
    padded = np.concatenate(
        [np.zeros((pad_frames, samples.shape[1]), dtype=samples.dtype), samples],
        axis=0,
    )
    target = work_dir / f"{source.stem}_leadpad_{seconds:.3f}.wav"
    sf.write(str(target), padded, sample_rate)
    return target


def _shift_sequence_seconds(sequence, seconds: float):
    if not seconds:
        return sequence
    kept = []
    for note in sequence.notes:
        if note.onset_sec is None or note.offset_sec is None:
            kept.append(note)
            continue
        note.onset_sec = max(0.0, float(note.onset_sec) - seconds)
        note.offset_sec = max(note.onset_sec + 1e-6, float(note.offset_sec) - seconds)
        kept.append(note)
    sequence.notes = kept
    sequence.metadata["inference_leading_silence_sec_removed"] = float(seconds)
    return sequence


def _path(value: str, env_name: str | None = None) -> Path:
    if env_name and os.environ.get(env_name):
        value = os.environ[env_name]
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return expanded if expanded.is_absolute() else PROJECT_ROOT / expanded


def _build_model(config: dict):
    backend = str(deep_get(config, "hpc_stage1.backend", ""))
    if backend == "yourmt3":
        from adapters.yourmt3 import YourMT3

        source = _path(
            str(deep_get(config, "hpc_stage1.source_dir", "third_party/YourMT3")),
            "YOURMT3_SOURCE_DIR",
        )
        return backend, YourMT3(
            source_dir=source,
            model_name=str(deep_get(config, "hpc_stage1.model_name", "yptf_moe_multi_nops")),
            precision=str(deep_get(config, "hpc_stage1.precision", "16")),
            device=str(deep_get(config, "hpc_stage1.device", "cuda")),
            inference_batch_size=int(os.environ.get(
                "YOURMT3_INFERENCE_BATCH_SIZE",
                deep_get(config, "hpc_stage1.inference_batch_size", 1),
            )),
            checkpoint_path=deep_get(config, "hpc_stage1.checkpoint_path", None),
        )
    raise ValueError(f"unknown hpc_stage1.backend: {backend!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--midi_dir", required=True)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard_index < num_shards")

    config = load_config(args.config)
    out_dir, midi_dir = Path(args.out_dir), Path(args.midi_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(out_dir / "infer_stage1.log")
    rows = read_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = rows[args.shard_index :: args.num_shards]
    logger.info("loading Stage 1 backend for shard %d/%d", args.shard_index, args.num_shards)
    backend, model = _build_model(config)
    logger.info("backend loaded: %s; samples=%d", backend, len(rows))
    leading_silence_sec = float(deep_get(config, "hpc_stage1.leading_silence_sec", 0.0))

    failures: list[dict[str, str]] = []
    total_notes = empty_outputs = skipped = unknown_before = unknown_after = 0
    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        json_path, midi_path = out_dir / f"{sample_id}.json", midi_dir / f"{sample_id}.mid"
        if args.skip_existing and json_path.exists():
            skipped += 1
            continue
        try:
            audio_path = _path(row["audio_path"])
            if not audio_path.exists():
                raise FileNotFoundError(f"missing audio: {audio_path}")
            with tempfile.TemporaryDirectory(prefix="a2s_stage1_pad_") as tmp:
                model_audio_path = _pad_audio_leading(audio_path, leading_silence_sec, Path(tmp))
                model.transcribe_to_midi(model_audio_path, midi_path)
            sequence = load_amt_output(midi_path, sample_id, "midi", backend)
            sequence = _shift_sequence_seconds(sequence, leading_silence_sec)
            unknown_before += sum(note.voice == "unknown" for note in sequence.notes)
            metadata_value = row.get("metadata_path")
            if metadata_value:
                metadata_path = _path(metadata_value)
                if metadata_path.exists():
                    meta = parse_kern_metadata(metadata_path)
                    sequence.meter = meta.meter
                    sequence.key = meta.key_signature or meta.key
                    sequence.tempo_bpm = meta.tempo_bpm
                    sequence.metadata["tonal_key"] = meta.key
            sequence.num_measures = int(row.get("num_measures") or 0) or None
            sequence = postprocess(
                sequence,
                assign_unknown_voices=bool(deep_get(config, "postprocess.assign_unknown_voices", True)),
                min_confidence=float(deep_get(config, "postprocess.min_confidence", 0.0)),
                transpose_semitones=int(deep_get(config, "postprocess.transpose_semitones", 0)),
            )
            unknown_after += sum(note.voice == "unknown" for note in sequence.notes)
            save_json(json_path, sequence.to_dict())
            total_notes += len(sequence.notes)
            empty_outputs += not sequence.notes
            if index == 1 or index % 100 == 0:
                logger.info("transcribed %d/%d (%s)", index, len(rows), sample_id)
        except Exception as exc:
            logger.exception("Stage 1 failed: %s", sample_id)
            failures.append({"sample_id": sample_id, "stage": "stage1", "error": str(exc)})

    write_failures(out_dir / "failed_samples.csv", failures)
    save_json(out_dir / "stage1_infer_report.json", {
        "backend": backend,
        "manifest": str(Path(args.manifest).resolve()),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_files": len(rows),
        "num_success": len(rows) - len(failures) - skipped,
        "num_failed": len(failures),
        "num_skipped_existing": skipped,
        "total_notes": total_notes,
        "empty_output_count": empty_outputs,
        "unknown_voices_before_postprocess": unknown_before,
        "unknown_voices_after_postprocess": unknown_after,
        "failed_samples": failures,
    })


if __name__ == "__main__":
    main()
