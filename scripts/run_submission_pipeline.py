"""MIREX entry pipeline: audio -> YourMT3 -> voices -> MIDI correction -> **kern.

All model-private formats end at their adapters.  The rest of the pipeline uses
``NoteEventSequence`` and isolates failures per sample so one bad excerpt never
prevents the remaining required outputs from being written.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from a2s.data.metadata_parser import ScoreMetadata, parse_kern_metadata
from a2s.data.note_event import NoteEventSequence
from a2s.evaluation.kern_validity import validate_kern_text
from a2s.stage1_amt.output_normalizer import load_amt_output
from a2s.stage1_amt.postprocess_mt3 import postprocess
from a2s.stage1_amt.voice_assignment import assign_voices
from a2s.stage1_amt.voice_assignment_model import VoiceAssignmentModel
from a2s.stage2_score.kern_repair import repair_kern, safe_fallback_kern
from a2s.stage2_score.midi_tokenizer import beats_to_seconds, midi_tokens_to_sequence, sequence_to_midi_tokens
from a2s.stage2_score.rule_based_converter import RuleBasedConverter
from a2s.stage2_score.staves_header import apply_staves_header
from a2s.stage2_score.tie_merger import merge_tied_notes
from a2s.utils.config import deep_get, load_config
from a2s.utils.json_io import save_json
from a2s.utils.logging import configure_logging
from a2s.utils.midi_io import events_to_midi


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _expand_env_defaults(value: str) -> str:
    """Expand ${NAME:-default} plus ordinary environment variables."""
    import re

    def repl(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name) or default or ""

    return os.path.expandvars(re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", repl, value))


def _path(value: str | Path | None, base: Path = PROJECT_ROOT) -> Path | None:
    if value in (None, ""):
        return None
    expanded = Path(_expand_env_defaults(str(value))).expanduser()
    return expanded if expanded.is_absolute() else base / expanded


def _find_metadata(
    metadata_source: Path | None, sample_id: str, relative_audio: Path | None = None,
) -> ScoreMetadata:
    if metadata_source is None:
        raise FileNotFoundError("Staves-Informed metadata is required")
    if metadata_source.is_file():
        candidates = [metadata_source]
    else:
        candidates: list[Path] = []
        if relative_audio is not None:
            for suffix in (".krn", ".kern", ".json", ".txt"):
                candidates.append(metadata_source / relative_audio.with_suffix(suffix))
        for suffix in (".krn", ".kern", ".json", ".txt"):
            candidates.append(metadata_source / f"{sample_id}{suffix}")
        if not any(path.is_file() for path in candidates):
            candidates.extend(
                path for path in metadata_source.rglob(f"{sample_id}.*")
                if path.suffix.lower() in {".krn", ".kern", ".json", ".txt"}
            )
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix.lower() in {".krn", ".kern", ".txt"}:
            metadata = parse_kern_metadata(path)
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            header = payload.get("header") or payload.get("kern_header")
            if isinstance(header, list):
                header = "\n".join(str(line) for line in header)
            if isinstance(header, str) and "**kern" in header:
                metadata = parse_kern_metadata(header)
            else:
                raise ValueError(
                    f"metadata JSON for {sample_id!r} must contain a four-spine "
                    "'header' or 'kern_header'"
                )
        else:
            continue
        if len(metadata.kern_indices) != 4 or not metadata.header_lines:
            raise ValueError(
                f"metadata for {sample_id!r} must contain a valid header with exactly "
                f"four **kern spines: {path}"
            )
        return metadata
    raise FileNotFoundError(
        f"no Staves-Informed metadata found for sample {sample_id!r} under {metadata_source}"
    )


def _apply_metadata(sequence: NoteEventSequence, metadata: ScoreMetadata) -> NoteEventSequence:
    sequence.meter = metadata.meter or sequence.meter or "4/4"
    sequence.key = metadata.key_signature or metadata.key or sequence.key or "C major"
    sequence.tempo_bpm = metadata.tempo_bpm or sequence.tempo_bpm or 120.0
    sequence.metadata["tonal_key"] = metadata.key
    return sequence


def _pad_audio_leading(source: Path, seconds: float, work_dir: Path) -> Path:
    if seconds <= 0:
        return source
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
    frames = int(round(seconds * sample_rate))
    if frames <= 0:
        return source
    target = work_dir / f"{source.stem}_leadpad_{seconds:.3f}.wav"
    sf.write(str(target), np.concatenate([np.zeros((frames, samples.shape[1]), dtype=samples.dtype), samples]), sample_rate)
    return target


def _shift_sequence_seconds(sequence: NoteEventSequence, seconds: float) -> NoteEventSequence:
    if seconds <= 0:
        return sequence
    for note in sequence.notes:
        if note.onset_sec is None or note.offset_sec is None:
            continue
        note.onset_sec = max(0.0, float(note.onset_sec) - seconds)
        note.offset_sec = max(note.onset_sec + 1e-6, float(note.offset_sec) - seconds)
    sequence.metadata["inference_leading_silence_sec_removed"] = float(seconds)
    return sequence


def _transpose_sequence(sequence: NoteEventSequence, semitones: int) -> NoteEventSequence:
    if semitones == 0:
        return sequence
    kept = []
    for note in sequence.notes:
        pitch = int(note.pitch) + semitones
        if 0 <= pitch <= 127:
            note.pitch = pitch
            kept.append(note)
    sequence.notes = kept
    sequence.metadata["postprocess_transpose_semitones"] = int(semitones)
    return sequence


def _load_yourmt3(config: dict[str, Any]):
    sys.path.insert(0, str(PROJECT_ROOT / "hpc" / "stage1"))
    from adapters.yourmt3 import YourMT3

    source_dir = _path(deep_get(config, "stage1.source_dir", "third_party/YourMT3"))
    checkpoint = deep_get(config, "stage1.checkpoint_path", None)
    precision = str(deep_get(config, "stage1.precision", "auto"))
    if precision == "auto":
        import torch

        precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "16"
    return YourMT3(
        source_dir=source_dir,
        model_name=str(deep_get(config, "stage1.model_name", "quartets_yptf_synth_finetune_h800_4gpu_pad05_nops_v1")),
        precision=precision,
        device=str(deep_get(config, "stage1.device", "cuda")),
        inference_batch_size=int(deep_get(config, "stage1.inference_batch_size", 1)),
        checkpoint_path=checkpoint,
    )


def _transcribe_stage1(model, audio_path: Path, sample_id: str, config: dict[str, Any], midi_dir: Path) -> NoteEventSequence:
    leading = float(deep_get(config, "stage1.leading_silence_sec", 0.0))
    midi_path = midi_dir / f"{sample_id}.mid"
    with tempfile.TemporaryDirectory(prefix="a2s_submission_pad_") as tmp:
        model_audio = _pad_audio_leading(audio_path, leading, Path(tmp))
        model.transcribe_to_midi(model_audio, midi_path)
    sequence = load_amt_output(midi_path, sample_id, "midi", "yourmt3_submission")
    sequence = _shift_sequence_seconds(sequence, leading)
    sequence = _transpose_sequence(sequence, int(deep_get(config, "stage1.transpose_semitones", 0)))
    return postprocess(
        sequence,
        assign_unknown_voices=False,
        min_confidence=float(deep_get(config, "stage1.min_confidence", 0.0)),
    )


def _load_stage2_model(config: dict[str, Any]):
    if not bool(deep_get(config, "stage2_midi.enabled", True)):
        return None
    from a2s.stage2_score.midi_correction_model import MidiCorrectionModel

    checkpoint = _path(deep_get(config, "stage2_midi.checkpoint", None))
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"missing Stage2 MIDI checkpoint: {checkpoint}")
    return MidiCorrectionModel(
        str(checkpoint), str(deep_get(config, "stage2_midi.device", "cuda"))
    )


def _stage2_correct(sequence: NoteEventSequence, model, config: dict[str, Any]) -> NoteEventSequence:
    if model is None:
        return sequence
    subdivisions = int(deep_get(config, "stage2_midi.subdivisions_per_beat", 24))
    input_tokens = [
        token for token in sequence_to_midi_tokens(sequence, subdivisions)
        if token not in {"<BOS_MIDI>", "<EOS_MIDI>"}
    ]
    output_tokens = model.generate(input_tokens, int(deep_get(config, "stage2_midi.max_decode_tokens", 3072)))
    corrected = midi_tokens_to_sequence(
        output_tokens,
        sequence.sample_id,
        meter=sequence.meter,
        key=sequence.key,
        tempo_bpm=sequence.tempo_bpm or float(deep_get(config, "stage2.default_tempo_bpm", 120.0)),
    )
    if not corrected.notes:
        raise ValueError("Stage2 MIDI model produced no notes")
    corrected.metadata.update(sequence.metadata)
    corrected.source_model = "stage2_midi_long_v2_submission"
    return corrected


def _convert_to_kern(
    sequence: NoteEventSequence, converter: RuleBasedConverter, metadata: ScoreMetadata | None = None,
) -> str:
    sequence.notes = merge_tied_notes(sequence.notes)
    text = converter.convert(sequence)
    if metadata is not None:
        text = apply_staves_header(text, metadata)
    validation = validate_kern_text(text)
    if not validation.valid:
        text = repair_kern(text)
        validation = validate_kern_text(text)
    if not validation.valid:
        raise ValueError("invalid kern after repair: " + ",".join(validation.issues))
    return text


def _audio_inputs(input_path: Path, exts: list[str]) -> list[Path]:
    allowed = {ext.lower() for ext in exts}
    if input_path.is_file():
        if input_path.suffix.lower() not in allowed:
            raise ValueError(f"unsupported input audio extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"input audio path not found: {input_path}")
    priority = {ext: index for index, ext in enumerate([".flac", ".wav", ".ogg", ".mp3"])}
    candidates = sorted(
        (path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in allowed),
        key=lambda path: (path.stem, priority.get(path.suffix.lower(), 99), path.as_posix()),
    )
    selected: dict[str, Path] = {}
    for path in candidates:
        previous = selected.setdefault(path.stem, path)
        if previous != path and previous.suffix.lower() == path.suffix.lower():
            raise ValueError(f"duplicate audio basename {path.stem!r}: {previous} and {path}")
    return list(selected.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Submission-grade A2S pipeline: audio -> YourMT3 -> voice model -> Stage2 MIDI -> **kern.")
    parser.add_argument("--config", default="configs/pipeline_submission.yaml")
    parser.add_argument("--input_audio_dir", required=True, help="Input audio file or directory (kept for wrapper compatibility).")
    parser.add_argument("--output_kern_dir", required=True)
    parser.add_argument(
        "--metadata_dir",
        required=True,
        help="Required Staves-Informed header file or directory.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--validate_inputs_only",
        action="store_true",
        help="Validate audio/metadata pairing without loading model checkpoints.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    started_at = time.perf_counter()
    input_path = Path(args.input_audio_dir)
    output_dir = Path(args.output_kern_dir)
    metadata_dir = Path(args.metadata_dir) if args.metadata_dir else None
    logs_dir = output_dir / "logs"
    work_dir = logs_dir / str(deep_get(config, "pipeline.work_subdir", "work"))
    stage1_events_dir = work_dir / "stage1_events"
    stage1_midi_dir = work_dir / "stage1_midi"
    voice_events_dir = work_dir / "voice_events"
    stage2_events_dir = work_dir / "stage2_events"
    stage2_midi_dir = work_dir / "stage2_midi"
    draft_kern_dir = work_dir / "draft_kern"
    for directory in (output_dir, logs_dir, work_dir, stage1_events_dir, stage1_midi_dir, voice_events_dir, stage2_events_dir, stage2_midi_dir, draft_kern_dir):
        directory.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(logs_dir / "transcription.log")
    save_intermediates = bool(deep_get(config, "pipeline.save_intermediates", False))

    audio_paths = _audio_inputs(input_path, list(deep_get(config, "submission.audio_exts", [".flac", ".wav"])))
    if args.limit:
        audio_paths = audio_paths[: args.limit]
    if not audio_paths:
        raise SystemExit(f"no supported audio files found under {input_path}")

    if metadata_dir is None or not metadata_dir.exists():
        raise SystemExit(f"Staves-Informed metadata path not found: {metadata_dir}")
    if metadata_dir.is_file() and len(audio_paths) != 1:
        raise SystemExit(
            "a single metadata file may only be used with one input audio file; "
            "provide a metadata directory for multiple inputs"
        )
    metadata_by_sample: dict[str, ScoreMetadata] = {}
    metadata_errors: list[str] = []
    for audio_path in audio_paths:
        relative_audio = audio_path.relative_to(input_path) if input_path.is_dir() else None
        try:
            metadata_by_sample[audio_path.stem] = _find_metadata(
                metadata_dir, audio_path.stem, relative_audio
            )
        except Exception as exc:
            metadata_errors.append(f"{audio_path.stem}: {type(exc).__name__}: {exc}")
    if metadata_errors:
        details = "\n  - ".join(metadata_errors)
        raise SystemExit(f"Staves-Informed metadata validation failed:\n  - {details}")
    if args.validate_inputs_only:
        logger.info(
            "input validation passed: %d audio file(s), %d metadata header(s)",
            len(audio_paths),
            len(metadata_by_sample),
        )
        return

    failures: list[dict[str, str]] = []
    init_errors: list[str] = []
    timings = {"initialization": 0.0, "stage1": 0.0, "voice_assignment": 0.0, "stage2_and_kern": 0.0}
    initialization_started = time.perf_counter()
    try:
        stage1_model = _load_yourmt3(config)
    except Exception as exc:
        stage1_model = None
        init_errors.append(f"stage1_init: {type(exc).__name__}: {exc}")
        logger.exception("Stage1 initialization failed; all samples will use fallback")
    try:
        voice_model_path = _path(deep_get(config, "voice_assignment.model_path", "outputs/voice_assignment/model.pkl"))
        voice_model = VoiceAssignmentModel.load(voice_model_path) if bool(deep_get(config, "voice_assignment.enabled", True)) else None
    except Exception as exc:
        voice_model = None
        init_errors.append(f"voice_model_init: {type(exc).__name__}: {exc}")
        logger.exception("Voice assignment initialization failed; rule voices will be used")
    try:
        stage2_model = _load_stage2_model(config)
    except Exception as exc:
        stage2_model = None
        init_errors.append(f"stage2_init: {type(exc).__name__}: {exc}")
        logger.exception("Stage2 MIDI initialization failed; rule-based Stage2 fallback will be used")

    required = set(deep_get(config, "submission.required_components", ["stage1", "voice_assignment", "stage2_midi"]))
    unavailable = {
        name for name, model in (
            ("stage1", stage1_model), ("voice_assignment", voice_model), ("stage2_midi", stage2_model)
        ) if name in required and model is None
    }
    if unavailable:
        raise RuntimeError("required submission components unavailable: " + ", ".join(sorted(unavailable)))
    timings["initialization"] = time.perf_counter() - initialization_started

    converter = RuleBasedConverter(
        subdivisions_per_beat=int(deep_get(config, "stage2.quantization.subdivisions_per_beat", 4)),
        default_tempo_bpm=float(deep_get(config, "stage2.default_tempo_bpm", 120.0)),
        alignment_tolerance_beat=float(deep_get(config, "stage2.alignment.tolerance_beat", 0.12)),
        alignment_max_span_beat=float(deep_get(config, "stage2.alignment.max_cluster_span_beat", 0.20)),
    )

    for index, audio_path in enumerate(audio_paths, start=1):
        sample_id = audio_path.stem
        relative_audio = None
        if input_path.is_dir():
            relative_audio = audio_path.relative_to(input_path)
        metadata = metadata_by_sample[sample_id]
        final_text: str | None = None
        logger.info("processing %d/%d: %s", index, len(audio_paths), sample_id)
        stage_started = time.perf_counter()
        try:
            if stage1_model is None:
                raise RuntimeError("Stage1 model is unavailable")
            sequence = _transcribe_stage1(stage1_model, audio_path, sample_id, config, stage1_midi_dir)
            sequence = _apply_metadata(sequence, metadata)
            if save_intermediates:
                save_json(stage1_events_dir / f"{sample_id}.json", sequence.to_dict())
        except Exception as exc:
            logger.exception("Stage1 failed for %s", sample_id)
            failures.append({"sample_id": sample_id, "stage": "stage1", "error": str(exc)})
            sequence = NoteEventSequence(
                sample_id=sample_id,
                notes=[],
                meter=metadata.meter,
                key=metadata.key_signature or metadata.key,
                tempo_bpm=metadata.tempo_bpm,
                source_model="stage1_failure_fallback",
                metadata={"tonal_key": metadata.key},
            )
        finally:
            timings["stage1"] += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        try:
            if voice_model is not None and sequence.notes:
                sequence = voice_model.predict_sequence(sequence)
            elif sequence.notes and any(note.voice == "unknown" for note in sequence.notes):
                sequence.notes = assign_voices(sequence.notes)
                sequence.metadata["voice_assignment_fallback"] = "rule_based"
            if save_intermediates:
                save_json(voice_events_dir / f"{sample_id}.json", sequence.to_dict())
        except Exception as exc:
            logger.exception("Voice assignment failed for %s", sample_id)
            failures.append({"sample_id": sample_id, "stage": "voice_assignment", "error": str(exc)})
            if sequence.notes and any(note.voice == "unknown" for note in sequence.notes):
                sequence.notes = assign_voices(sequence.notes)
                sequence.metadata["voice_assignment_fallback"] = "rule_based_after_error"
        finally:
            timings["voice_assignment"] += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        try:
            corrected = _stage2_correct(sequence, stage2_model, config)
            if save_intermediates:
                save_json(stage2_events_dir / f"{sample_id}.json", corrected.to_dict())
                events_to_midi(beats_to_seconds(corrected, corrected.tempo_bpm), stage2_midi_dir / f"{sample_id}.mid")
            final_text = _convert_to_kern(corrected, converter, metadata)
            if save_intermediates:
                (draft_kern_dir / f"{sample_id}.krn").write_text(final_text, encoding="utf-8")
        except Exception as exc:
            logger.exception("Stage2 MIDI/rule conversion failed for %s", sample_id)
            failures.append({"sample_id": sample_id, "stage": "stage2", "error": str(exc)})
            try:
                final_text = _convert_to_kern(sequence, converter, metadata)
            except Exception as fallback_exc:
                logger.exception("Rule fallback failed for %s", sample_id)
                failures.append({"sample_id": sample_id, "stage": "fallback", "error": str(fallback_exc)})
                final_text = safe_fallback_kern(sequence.meter or metadata.meter, sequence.key or metadata.key, sequence.tempo_bpm or metadata.tempo_bpm)
                final_text = apply_staves_header(final_text, metadata)
        finally:
            timings["stage2_and_kern"] += time.perf_counter() - stage_started

        final_validation = validate_kern_text(final_text)
        if not final_validation.valid:
            failures.append({"sample_id": sample_id, "stage": "final_validation", "error": ",".join(final_validation.issues)})
            # Staves-Informed evaluation requires the supplied header even when
            # every musical stage failed, so project the final legal rest score.
            final_text = apply_staves_header(safe_fallback_kern(), metadata)

        (output_dir / f"{sample_id}.krn").write_text(final_text, encoding="utf-8")
        if not save_intermediates:
            (stage1_midi_dir / f"{sample_id}.mid").unlink(missing_ok=True)

    if not save_intermediates:
        shutil.rmtree(work_dir, ignore_errors=True)

    report = {
        "config": str(Path(args.config).resolve()),
        "input_audio_path": str(input_path.resolve()),
        "output_kern_dir": str(output_dir.resolve()),
        "metadata_dir": str(metadata_dir.resolve()) if metadata_dir else None,
        "num_inputs": len(audio_paths),
        "num_outputs": sum((output_dir / f"{path.stem}.krn").exists() for path in audio_paths),
        "init_errors": init_errors,
        "num_failures": len(failures),
        "failures": failures,
        "pipeline": deep_get(config, "submission.name", "a2s_submission"),
        "stage1_model_available": stage1_model is not None,
        "voice_model_available": voice_model is not None,
        "stage2_model_available": stage2_model is not None,
        "work_dir": str(work_dir) if save_intermediates else None,
        "save_intermediates": save_intermediates,
        "timing_seconds": timings,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    try:
        import torch

        report["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        report["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        report["peak_gpu_memory_bytes"] = None
        report["gpu_name"] = None
    save_json(logs_dir / "pipeline_report.json", report)
    with (logs_dir / "failed_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=("sample_id", "stage", "error"))
        writer.writeheader()
        writer.writerows(failures)
    logger.info("submission pipeline complete: %s", json.dumps({k: report[k] for k in ("num_inputs", "num_outputs", "num_failures")}))


if __name__ == "__main__":
    main()
