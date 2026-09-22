from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from _common import write_failures
from a2s.data.quartets_manifest import prepare_quartets_dataset, write_split_manifests
from a2s.utils.config import load_config
from a2s.utils.json_io import save_json
from a2s.utils.logging import configure_logging


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize Quartets data into split manifests.")
    parser.add_argument("--config", default="configs/data_quartets.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    output = config.get("output", {})
    manifest_dir = _project_path(project_root, output.get("manifest_dir", "data/manifests"))
    reports_dir = _project_path(project_root, output.get("reports_dir", "data/reports"))
    logger = configure_logging(reports_dir / "prepare_quartets.log", args.verbose)
    rows, report, failures = prepare_quartets_dataset(config, project_root)
    write_split_manifests(rows, manifest_dir)
    write_failures(manifest_dir / "failed_samples.csv", failures)
    save_json(reports_dir / "prepare_quartets_report.json", report)
    logger.info(
        "prepared %d samples: train=%d valid=%d test=%d failures=%d",
        report["num_samples_total"], report["num_train"], report["num_valid"],
        report["num_test"], len(failures),
    )


if __name__ == "__main__":
    main()
