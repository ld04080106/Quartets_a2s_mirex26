from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _common import read_manifest
from a2s.data.metadata_parser import parse_kern_metadata, quartet_voice_names
from a2s.data.note_event import VOICES
from a2s.evaluation.kern_validity import validate_kern_text
from a2s.evaluation.wer_ler import line_error_rate, word_error_rate
from a2s.utils.config import load_config
from a2s.utils.json_io import save_json
from a2s.utils.logging import configure_logging


def _index(directory: Path) -> dict[str, Path]:
    return {path.stem: path for path in directory.rglob("*.krn")}


def _score_body(text: str) -> str:
    """Select and order four **kern spines, excluding headers and dynamics."""
    metadata = parse_kern_metadata(text)
    indices = metadata.kern_indices
    if len(indices) != 4:
        return text
    voices = quartet_voice_names(metadata)
    by_voice = {voice: index for voice, index in zip(voices, indices)}
    selected = [by_voice.get(voice) for voice in VOICES]
    if any(index is None for index in selected):
        selected = list(indices)
    selected = [int(index) for index in selected]
    body: list[str] = []
    for raw_line in text.replace("\r", "").splitlines():
        if not raw_line or raw_line.startswith("!!!"):
            continue
        fields = raw_line.split("\t")
        if len(fields) <= max(selected):
            continue
        tokens = [fields[index] for index in selected]
        if all(token.startswith("*") for token in tokens):
            continue
        if all(token.startswith("!") for token in tokens):
            continue
        tokens = ["=" if token.startswith("=") else token for token in tokens]
        body.append("\t".join(tokens))
    return "\n".join(body)


def _manifest_gt(path: Path, project_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for row in read_manifest(path):
        value = row.get("kern_path")
        if value:
            source = Path(value)
            result[row["sample_id"]] = source if source.is_absolute() else project_root / source
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_midi_pred_events_long_v2_infer.yaml")
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir")
    parser.add_argument("--manifest", help="Use ground-truth paths from a split manifest.")
    parser.add_argument("--limit", type=int, help="Optional manifest-prefix smoke-test limit.")
    parser.add_argument(
        "--pred_only",
        action="store_true",
        help="Evaluate only samples with a predicted .krn. Useful for smoke-test subsets.",
    )
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()
    load_config(args.config)
    logger = configure_logging(Path(args.out_json).with_suffix(".log"))
    pred = _index(Path(args.pred_dir))
    project_root = Path(__file__).resolve().parents[1]
    if args.manifest:
        gt = _manifest_gt(Path(args.manifest), project_root)
    elif args.gt_dir:
        gt = _index(Path(args.gt_dir))
    else:
        parser.error("--gt_dir or --manifest is required")
    if args.limit:
        gt = dict(list(gt.items())[:args.limit])
    if args.pred_only:
        gt = {sample_id: path for sample_id, path in gt.items() if sample_id in pred}
    rows = []
    missing_ground_truth = []
    for index, (sample_id, ref_path) in enumerate(gt.items(), start=1):
        if index == 1 or index % 500 == 0:
            logger.info("evaluating Stage 2 file %d/%d", index, len(gt))
        if not ref_path.exists():
            missing_ground_truth.append(sample_id)
            continue
        hyp_path = pred.get(sample_id)
        ref = ref_path.read_text(encoding="utf-8-sig", errors="replace")
        hyp = hyp_path.read_text(encoding="utf-8-sig", errors="replace") if hyp_path else ""
        ref_body, hyp_body = _score_body(ref), _score_body(hyp) if hyp else ""
        validity = validate_kern_text(hyp)
        rows.append({
            "sample_id": sample_id, "valid": validity.valid,
            "wer": word_error_rate(ref_body, hyp_body), "ler": line_error_rate(ref_body, hyp_body),
            "raw_wer": word_error_rate(ref, hyp), "raw_ler": line_error_rate(ref, hyp),
            "empty": not hyp.strip(), "length": len(hyp.split()),
            "barline_consistent": validity.barline_consistent, "spine_count_consistent": validity.spine_count == 4,
        })
    count = len(rows)
    summary = {
        "evaluated": count,
        "valid_kern_rate": sum(row["valid"] for row in rows) / max(1, count),
        "render_success_rate": None,
        "WER": sum(row["wer"] for row in rows) / max(1, count),
        "LER": sum(row["ler"] for row in rows) / max(1, count),
        "raw_WER": sum(row["raw_wer"] for row in rows) / max(1, count),
        "raw_LER": sum(row["raw_ler"] for row in rows) / max(1, count),
        "empty_output_rate": sum(row["empty"] for row in rows) / max(1, count),
        "avg_output_length": sum(row["length"] for row in rows) / max(1, count),
        "barline_consistency": sum(row["barline_consistent"] for row in rows) / max(1, count),
        "spine_count_consistency": sum(row["spine_count_consistent"] for row in rows) / max(1, count),
        "missing_ground_truth": missing_ground_truth, "per_sample": rows,
    }
    save_json(args.out_json, summary)
    logger.info(
        "Stage 2 evaluation: evaluated=%d valid=%.4f WER=%.4f LER=%.4f missing_gt=%d",
        count, summary["valid_kern_rate"], summary["WER"], summary["LER"],
        len(missing_ground_truth),
    )


if __name__ == "__main__":
    main()
