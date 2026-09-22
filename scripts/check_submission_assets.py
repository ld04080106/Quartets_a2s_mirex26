from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from a2s.utils.config import deep_get, load_config
from scripts.run_submission_pipeline import _path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check required assets for the MIREX submission pipeline.")
    parser.add_argument("--config", default="configs/pipeline_submission.yaml")
    parser.add_argument("--out_json")
    args = parser.parse_args()

    config = load_config(args.config)
    source_dir = _path(deep_get(config, "stage1.source_dir", ""))
    stage1_checkpoint = Path(str(deep_get(config, "stage1.checkpoint_path", "")))
    if source_dir is not None and not stage1_checkpoint.is_absolute():
        stage1_checkpoint = source_dir / stage1_checkpoint
    checks = {
        "yourmt3_model_helper": source_dir / "model_helper.py" if source_dir else None,
        "yourmt3_amt_source": source_dir / "amt" / "src" if source_dir else None,
        "yourmt3_checkpoint": stage1_checkpoint,
        "voice_assignment_model": _path(deep_get(config, "voice_assignment.model_path", "")),
        "stage2_midi_checkpoint": _path(deep_get(config, "stage2_midi.checkpoint", "")),
    }
    report = {
        name: {"path": str(path), "exists": bool(path and path.exists())}
        for name, path in checks.items()
    }
    modules = (
        "torch", "torchaudio", "soundfile", "yaml", "pretty_midi", "mido", "librosa",
        "sklearn", "pytorch_lightning", "transformers", "einops", "mir_eval",
        "deprecated", "dotenv", "wandb", "cosine_annealing_warmup",
    )
    report["python"] = {"executable": sys.executable, "version": sys.version}
    report["modules"] = {name: importlib.util.find_spec(name) is not None for name in modules}
    try:
        import torch

        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except ImportError:
        report["cuda"] = {"available": False, "device_count": 0, "devices": []}
    asset_ok = all(item["exists"] for name, item in report.items() if name in checks)
    report["ok"] = asset_ok and all(report["modules"].values()) and report["cuda"]["available"]
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        target = Path(args.out_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
