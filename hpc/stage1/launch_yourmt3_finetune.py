from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from a2s.utils.config import deep_get, load_config  # noqa: E402
from hpc.stage1.patch_yourmt3_finetune import patch_source  # noqa: E402


def _path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def _resolve_yourmt3_source(value: str | None) -> Path:
    tried: list[Path] = []
    raw = value or "third_party/YourMT3"
    primary = _path(raw)
    candidates = [primary]
    if primary.name == "YourMT":
        candidates.append(primary.with_name("YourMT3"))
    env_value = os.environ.get("YOURMT3_SOURCE_DIR")
    if env_value:
        candidates.insert(0, _path(env_value))
    for candidate in candidates:
        if candidate in tried:
            continue
        tried.append(candidate)
        if (candidate / "amt" / "src" / "utils" / "note_event_dataclasses.py").is_file():
            return candidate
    tried_text = "\n".join(f"  - {path}" for path in tried)
    raise SystemExit("YourMT3 source is incomplete. Tried:\n" + tried_text)


def _resolve_checkpoint(config: dict, source: Path) -> Path:
    configured = str(deep_get(config, "yourmt3.pretrained_checkpoint", ""))
    checkpoint = _path(configured) if configured else Path()
    if checkpoint.is_file():
        return checkpoint
    return source / "amt" / "logs" / "2024" / "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops" / "checkpoints" / "last.ckpt"


def _visible_cuda_count() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
        return 0
    except Exception:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None or visible.strip() == "":
            return None
        if visible.strip() == "-1":
            return 0
        return len([item for item in visible.split(",") if item.strip()])


def _bf16_supported() -> bool | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return None


def _resolve_gpu_request(train: dict, *, dry_run: bool) -> tuple[str, str, dict]:
    requested_raw = os.environ.get("YOURMT3_NUM_GPUS", train.get("num_gpus", 3))
    requested = str(requested_raw)
    strategy = str(os.environ.get("YOURMT3_STRATEGY", train.get("strategy", "ddp")))
    visible = _visible_cuda_count()
    adjustment = {
        "configured_num_gpus": requested,
        "configured_strategy": strategy,
        "visible_cuda_devices": visible,
        "effective_num_gpus": requested,
        "effective_strategy": strategy,
        "warnings": [],
    }
    try:
        requested_int = int(requested)
    except ValueError:
        requested_int = None
    if visible is not None and visible <= 0 and not dry_run:
        raise SystemExit("No CUDA GPU is visible to this job; check scheduler allocation or CUDA_VISIBLE_DEVICES.")
    if requested_int is not None and visible is not None and visible > 0 and requested_int > visible:
        adjustment["warnings"].append(
            f"requested {requested_int} GPU(s), but only {visible} visible; using {visible}"
        )
        requested = str(visible)
    try:
        effective_int = int(requested)
    except ValueError:
        effective_int = visible
    if effective_int == 1 and strategy == "ddp":
        adjustment["warnings"].append("single visible GPU; switching strategy from ddp to auto")
        strategy = "auto"
    adjustment["effective_num_gpus"] = requested
    adjustment["effective_strategy"] = strategy
    return requested, strategy, adjustment


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch reproducible YourMT3 quartet fine-tuning.")
    parser.add_argument("--config", default="configs/stage1_yourmt3_synth_finetune_h800_4gpu_pad05_nops.yaml")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_patch", action="store_true")
    args = parser.parse_args()
    config = load_config(_path(args.config))
    source = _resolve_yourmt3_source(deep_get(config, "yourmt3.source_dir", "third_party/YourMT3"))
    data_home = _path(deep_get(config, "yourmt3.data_home", "data/yourmt3_data"))
    checkpoint = _resolve_checkpoint(config, source)
    if not checkpoint.is_file():
        raise SystemExit(f"pretrained checkpoint not found: {checkpoint}")
    required = [
        data_home / "yourmt3_indexes" / "quartets_train_file_list.json",
        data_home / "yourmt3_indexes" / "quartets_validation_file_list.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("prepare YourMT3 data first; missing:\n" + "\n".join(missing))
    if not args.no_patch:
        patch_source(source)

    train = config.get("train", {})
    precision = str(train.get("precision", "16"))
    if precision.startswith("bf16") and _bf16_supported() is False:
        raise SystemExit("This visible GPU/runtime does not support bf16; use precision: '16'")
    num_gpus, strategy, gpu_adjustment = _resolve_gpu_request(train, dry_run=args.dry_run)
    for warning in gpu_adjustment["warnings"]:
        print(f"A2S GPU warning: {warning}", file=sys.stderr)
    command = [
        sys.executable, "amt/src/train.py", str(train.get("experiment_id", "quartets_yptf_finetune")),
        "-p", str(train.get("project", "2026_quartets")), "-d", "quartets",
        "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26",
        "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe", "-wf", "4",
        "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope",
        "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1",
        "-pr", precision, "-st", strategy,
        "-g", num_gpus, "-bsz",
        str(train.get("sub_batch_size", 1)), str(train.get("local_batch_size", 1)),
        "-se", str(train.get("samples_per_epoch", 24000)),
        "-it", str(train.get("max_steps", 10000)),
        "-vit", str(train.get("val_interval_steps", 1000)),
        "-lr", str(train.get("learning_rate", 0.0001)),
        "-o", str(train.get("optimizer", "AdamWScale")),
        "-s", str(train.get("scheduler", "cosine")), "-wb", "disabled",
        "-amp", str(train.get("amplitude_min", 0.8)), str(train.get("amplitude_max", 1.1)),
        "-iaug", "1.0", "-xk", "0", "-ps",
        str(train.get("pitch_shift_min", -1)), str(train.get("pitch_shift_max", 1)),
    ]
    env = os.environ.copy()
    env["YOURMT3_DATA_HOME"] = str(data_home)
    env["YOURMT3_INIT_CHECKPOINT"] = str(checkpoint)
    env["YOURMT3_VALIDATION_BATCH_SIZE"] = str(train.get("validation_batch_size", 1))
    env.setdefault("WANDB_MODE", "disabled")
    report = {
        "source_dir": str(source), "data_home": str(data_home),
        "pretrained_checkpoint": str(checkpoint), "command": command,
        "gpu_adjustment": gpu_adjustment,
        "environment": {
            "YOURMT3_DATA_HOME": str(data_home), "YOURMT3_INIT_CHECKPOINT": str(checkpoint),
            "YOURMT3_VALIDATION_BATCH_SIZE": env["YOURMT3_VALIDATION_BATCH_SIZE"],
        },
    }
    report_path = _path(deep_get(config, "output.launch_report", "outputs/yourmt3_finetune/launch.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("YourMT3 command:")
    print(shlex.join(command))
    if not args.dry_run:
        raise SystemExit(subprocess.call(command, cwd=source, env=env))


if __name__ == "__main__":
    main()
