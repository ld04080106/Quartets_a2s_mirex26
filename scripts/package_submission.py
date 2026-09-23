from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
from a2s.utils.config import deep_get, load_config
from scripts.run_submission_pipeline import _path


# Keep the competition archive independent of research scripts and old ablation
# configs.  GitHub contains the reproducible training path; this list is the
# smaller runtime closure used by the evaluator.
RUNTIME_FILES = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "a2s/__init__.py",
    "a2s/data/__init__.py",
    "a2s/data/metadata_parser.py",
    "a2s/data/note_event.py",
    "a2s/evaluation/__init__.py",
    "a2s/evaluation/kern_validity.py",
    "a2s/stage1_amt/__init__.py",
    "a2s/stage1_amt/output_normalizer.py",
    "a2s/stage1_amt/postprocess_mt3.py",
    "a2s/stage1_amt/voice_assignment.py",
    "a2s/stage1_amt/voice_assignment_features.py",
    "a2s/stage1_amt/voice_assignment_model.py",
    "a2s/stage2_score/__init__.py",
    "a2s/stage2_score/barline_repair.py",
    "a2s/stage2_score/duration_spelling.py",
    "a2s/stage2_score/event_quantizer.py",
    "a2s/stage2_score/kern_repair.py",
    "a2s/stage2_score/kern_writer.py",
    "a2s/stage2_score/midi_correction_model.py",
    "a2s/stage2_score/midi_tokenizer.py",
    "a2s/stage2_score/rest_filler.py",
    "a2s/stage2_score/rule_based_converter.py",
    "a2s/stage2_score/score_grid.py",
    "a2s/stage2_score/spine_aligner.py",
    "a2s/stage2_score/staves_header.py",
    "a2s/stage2_score/temporal_alignment.py",
    "a2s/stage2_score/tie_merger.py",
    "a2s/utils/__init__.py",
    "a2s/utils/config.py",
    "a2s/utils/json_io.py",
    "a2s/utils/logging.py",
    "a2s/utils/midi_io.py",
    "configs/pipeline_submission.yaml",
    "scripts/_bootstrap.py",
    "scripts/check_submission_assets.py",
    "scripts/check_submission_outputs.py",
    "scripts/ensure_model_assets.py",
    "scripts/package_precomputed_kern.py",
    "scripts/run_submission_pipeline.py",
    "hpc/stage1/adapters/yourmt3.py",
    "SUBMISSION.md",
    "RESOURCE_DECLARATION.md",
    "requirements_submission.txt",
    "transcription.sh",
)


def _copy_runtime_assets(config: dict, project: Path, staging: Path) -> None:
    source_dir = _path(deep_get(config, "stage1.source_dir", ""))
    if source_dir is None or not (source_dir / "model_helper.py").is_file():
        raise FileNotFoundError(f"YourMT3 source is missing: {source_dir}")
    target_source = staging / "third_party" / "YourMT3"
    shutil.copytree(
        source_dir,
        target_source,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", "*.pyc", ".coverage", ".DS_Store",
            "*.a2s_original", "model_output", "wandb", "lightning_logs",
            "logs", "extras", "tests",
        ),
    )
    checkpoint = Path(str(deep_get(config, "stage1.checkpoint_path", "")))
    checkpoint = checkpoint if checkpoint.is_absolute() else source_dir / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected YourMT3 checkpoint is missing: {checkpoint}")
    checkpoint_target = target_source / checkpoint.relative_to(source_dir)
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, checkpoint_target)

    for dotted_name in ("voice_assignment.model_path", "stage2_midi.checkpoint"):
        source = _path(deep_get(config, dotted_name, ""))
        if source is None or not source.is_file():
            raise FileNotFoundError(f"selected asset is missing ({dotted_name}): {source}")
        relative = source.relative_to(project)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_runtime_code(project: Path, staging: Path) -> None:
    """Copy only files needed by the submission entry point."""

    for relative_name in RUNTIME_FILES:
        source = project / relative_name
        if not source.is_file():
            raise FileNotFoundError(f"required submission source is missing: {source}")
        target = staging / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    # The evaluator sees the concise runtime instructions at archive root; the
    # longer GitHub README documents training files that are intentionally absent.
    shutil.copy2(
        project / "SUBMISSION.md",
        staging / "README.md",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_submission.yaml")
    parser.add_argument("--out", default="outputs/a2s_mirex_v2_submission")
    parser.add_argument("--include_runtime_assets", action="store_true", help="Copy only the selected YourMT3, voice and Stage2 assets.")
    parser.add_argument("--format", choices=("tar.gz", "zip"), default="tar.gz")
    args = parser.parse_args()
    config = load_config(args.config)
    project = Path(__file__).resolve().parents[1]
    staging = Path(args.out)
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite {staging}")
    staging.mkdir(parents=True)
    _copy_runtime_code(project, staging)
    if args.include_runtime_assets:
        _copy_runtime_assets(config, project, staging)
    archive_format = "gztar" if args.format == "tar.gz" else "zip"
    archive = Path(shutil.make_archive(str(staging), archive_format, staging.parent, staging.name))
    checksum = archive.with_name(archive.name + ".sha256")
    checksum.write_text(f"{_sha256(archive)}  {archive.name}\n", encoding="ascii")
    print(archive)
    print(checksum)


if __name__ == "__main__":
    main()
