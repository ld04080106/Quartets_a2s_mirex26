#!/usr/bin/env python3
"""Audit an existing cluster environment without loading a YourMT3 checkpoint."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path


MODULES = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "numpy": "numpy",
    "pkg_resources": "setuptools",
    "pytorch_lightning": "pytorch-lightning",
    "transformers": "transformers",
    "librosa": "librosa",
    "einops": "einops",
    "mido": "mido",
    "mir_eval": "mir_eval",
    "deprecated": "Deprecated",
    "dotenv": "python-dotenv",
    "yaml": "PyYAML",
    "pretty_midi": "pretty_midi",
    "soundfile": "soundfile",
    "wandb": "wandb",
    "cosine_annealing_warmup": "cosine-annealing-warmup",
}


def version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--out_json", default="outputs/yourmt3_runtime_check.json")
    parser.add_argument("--import_model_helper", action="store_true")
    args = parser.parse_args()
    source = Path(args.source_dir).resolve()
    modules = {}
    for module_name, distribution in MODULES.items():
        spec = importlib.util.find_spec(module_name)
        modules[module_name] = {
            "available": spec is not None,
            "version": version(distribution) if spec is not None else None,
        }
    cuda = {"available": False, "device_count": 0, "devices": []}
    if modules["torch"]["available"]:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    checkpoints = [
        {"path": str(path.relative_to(source)), "bytes": path.stat().st_size}
        for path in sorted(source.rglob("*.ckpt"))
    ]
    model_helper = {"requested": args.import_model_helper, "success": None, "error": None}
    if args.import_model_helper:
        sys.path.insert(0, str(source / "amt" / "src"))
        sys.path.insert(0, str(source))
        try:
            importlib.import_module("model_helper")
            model_helper["success"] = True
        except Exception as exc:
            model_helper["success"] = False
            model_helper["error"] = f"{type(exc).__name__}: {exc}"
    missing = [name for name, details in modules.items() if not details["available"]]
    compatibility_warnings = []
    numpy_version = modules["numpy"]["version"]
    if numpy_version and int(numpy_version.split(".", 1)[0]) >= 2:
        compatibility_warnings.append(
            f"YourMT3 Space pins numpy==1.26.4, but this environment has numpy=={numpy_version}."
        )
    transformers_version = modules["transformers"]["version"]
    if transformers_version and transformers_version != "4.45.1":
        compatibility_warnings.append(
            f"YourMT3 Space pins transformers==4.45.1, but this environment has {transformers_version}."
        )
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "source_dir": str(source),
        "source_assets": {
            "model_helper": (source / "model_helper.py").exists(),
            "amt_src": (source / "amt" / "src").is_dir(),
            "checkpoint_count": len(checkpoints),
        },
        "checkpoints": checkpoints,
        "modules": modules,
        "missing_modules": missing,
        "compatibility_warnings": compatibility_warnings,
        "cuda": cuda,
        "model_helper_import": model_helper,
    }
    target = Path(args.out_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if missing or not all(report["source_assets"].values()) or not cuda["available"]:
        raise SystemExit(2)
    if args.import_model_helper and not model_helper["success"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
