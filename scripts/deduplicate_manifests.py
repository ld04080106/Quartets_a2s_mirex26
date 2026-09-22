from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from _common import read_manifest
from a2s.data.manifest_dedup import deduplicate_manifest_rows
from a2s.data.quartets_manifest import write_split_manifests
from a2s.utils.config import load_config
from a2s.utils.json_io import save_json
from a2s.utils.logging import configure_logging


def _project_path(project: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate manifests without modifying raw data.")
    parser.add_argument("--config", default="configs/data_quartets.yaml")
    parser.add_argument("--manifest_dir", default="data/manifests")
    parser.add_argument("--out_dir", default="data/manifests_dedup")
    parser.add_argument("--report", default="data/reports/manifest_dedup_report.json")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    manifest_dir = _project_path(project, args.manifest_dir)
    out_dir = _project_path(project, args.out_dir)
    report_path = _project_path(project, args.report)
    oracle_dir = _project_path(
        project, config.get("output", {}).get("oracle_events_dir", "data/oracle_events")
    )
    logger = configure_logging(report_path.with_suffix(".log"))
    rows = []
    for split in ("train", "valid", "test"):
        split_rows = read_manifest(manifest_dir / f"{split}.csv")
        for row in split_rows:
            row["split"] = split
        rows.extend(split_rows)
    deduplicated, removed, report = deduplicate_manifest_rows(rows, project, oracle_dir)
    write_split_manifests(deduplicated, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "removed_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "removed_sample_id", "removed_split", "kept_sample_id", "kept_split",
            "method", "kern_content_conflict",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(removed)
    save_json(report_path, report)
    logger.info(
        "manifest dedup complete: input=%d output=%d removed=%d groups=%d cross_split=%d",
        report["num_input_rows"], report["num_output_rows"], report["num_removed_rows"],
        report["num_duplicate_groups"], report["cross_split_duplicate_groups"],
    )


if __name__ == "__main__":
    main()
