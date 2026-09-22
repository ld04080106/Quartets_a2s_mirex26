from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PRESET_MARKER = "# A2S_QUARTETS_PRESET_V1"
TRAIN_MARKER = "# A2S_INIT_CHECKPOINT_V1"
WANDB_MARKER = "# A2S_WANDB_GUARD_V1"
VALIDATION_BSZ_MARKER = "# A2S_VALIDATION_BSZ_GUARD_V1"


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".a2s_original")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_source(source_dir: Path) -> dict:
    source_dir = source_dir.resolve()
    src = source_dir / "amt" / "src"
    data_presets = src / "config" / "data_presets.py"
    config_path = src / "config" / "config.py"
    train_path = src / "train.py"
    ymt3_path = src / "model" / "ymt3.py"
    for path in (data_presets, config_path, train_path, ymt3_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        _backup(path)
    changed = []

    text = data_presets.read_text(encoding="utf-8")
    quartet_preset = f'''\n\n{PRESET_MARKER}\ndata_preset_single_cfg["quartets"] = {{\n    "eval_vocab": [GM_INSTR_FULL],\n    "dataset_name": "quartets",\n    "train_split": "train",\n    "validation_split": "validation",\n    "test_split": "test",\n    "has_stem": False,\n}}\n'''
    bad_quartet_preset = f'''\n\n{PRESET_MARKER}\n+data_preset_single_cfg["quartets"] = {{\n+    "eval_vocab": [GM_INSTR_FULL],\n+    "dataset_name": "quartets",\n+    "train_split": "train",\n+    "validation_split": "validation",\n+    "test_split": "test",\n+    "has_stem": False,\n+}}\n+'''
    text = text.replace(bad_quartet_preset, quartet_preset)
    if PRESET_MARKER not in text:
        text += quartet_preset
    data_presets.write_text(text, encoding="utf-8")
    changed.append(str(data_presets))

    text = config_path.read_text(encoding="utf-8")
    if "import os\n" not in text[:200]:
        text = text.replace('"""config.py"""\n', '"""config.py"""\nimport os\n', 1)
    text = text.replace(
        '"data_home": "../../data", # path to the data directory. If using relative path, it is relative to /src directory.',
        '"data_home": os.environ.get("YOURMT3_DATA_HOME", "../../data"), # A2S override',
    )
    text = text.replace(
        '"validation": 64, # validation batch size is per GPU in DDP mode',
        '"validation": int(os.environ.get("YOURMT3_VALIDATION_BATCH_SIZE", "1")), # A2S override',
    )
    config_path.write_text(text, encoding="utf-8")
    changed.append(str(config_path))

    text = train_path.read_text(encoding="utf-8")
    if "import os\n" not in text[:500]:
        text = text.replace("import argparse\n", "import argparse\nimport os\n", 1)
    text = text.replace(
        "    if trainer.global_rank == 0:\n        wandb_logger.experiment.config.update",
        "    if trainer.global_rank == 0 and wandb_logger is not None:\n        wandb_logger.experiment.config.update",
    )
    unguarded_watch = "    wandb_logger.watch(model, log='gradients', log_freq=5000)\n"
    guarded_watch = (
        f"    {WANDB_MARKER}\n"
        "    if wandb_logger is not None:\n"
        "        wandb_logger.watch(model, log='gradients', log_freq=5000)\n"
    )
    watch_variants = [
        (
            "    if wandb_logger is not None:\n"
            "        if wandb_logger is not None:\n"
            "        if wandb_logger is not None:\n"
            "        wandb_logger.watch(model, log='gradients', log_freq=5000)\n"
        ),
        (
            "    if wandb_logger is not None:\n"
            "        if wandb_logger is not None:\n"
            "        wandb_logger.watch(model, log='gradients', log_freq=5000)\n"
        ),
        (
            "    if wandb_logger is not None:\n"
            "        wandb_logger.watch(model, log='gradients', log_freq=5000)\n"
        ),
    ]
    for variant in watch_variants:
        if variant in text:
            text = text.replace(variant, guarded_watch, 1)
            break
    duplicate_marker = f"    {WANDB_MARKER}\n    {WANDB_MARKER}\n"
    while duplicate_marker in text:
        text = text.replace(duplicate_marker, f"    {WANDB_MARKER}\n")
    if WANDB_MARKER not in text:
        if unguarded_watch not in text:
            raise RuntimeError("YourMT3 train.py wandb watch layout changed; refusing an unsafe patch")
        text = text.replace(unguarded_watch, guarded_watch, 1)
    old = '''    # last_ckpt_path can be None
    if dir_info["last_ckpt_path"] is not None:
        checkpoint = torch.load(dir_info["last_ckpt_path"])
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        trainer.fit(model, datamodule=dm)
    else:
        trainer.fit(model, ckpt_path=dir_info["last_ckpt_path"], datamodule=dm)
'''
    new = f'''    {TRAIN_MARKER}
    # Resume an A2S fine-tuning run when its own last.ckpt exists. Otherwise
    # initialize model weights from the downloaded YourMT3 checkpoint while
    # starting a fresh optimizer/scheduler state.
    if dir_info["last_ckpt_path"] is not None:
        trainer.fit(model, ckpt_path=dir_info["last_ckpt_path"], datamodule=dm)
    else:
        init_checkpoint = os.environ.get("YOURMT3_INIT_CHECKPOINT")
        if init_checkpoint:
            checkpoint = torch.load(init_checkpoint, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            incompatible = model.load_state_dict(state_dict, strict=False)
            print(f"A2S initialized from {{init_checkpoint}}; missing={{len(incompatible.missing_keys)}} "
                  f"unexpected={{len(incompatible.unexpected_keys)}}")
        trainer.fit(model, datamodule=dm)
'''
    if TRAIN_MARKER not in text:
        if old not in text:
            raise RuntimeError("YourMT3 train.py layout changed; refusing an unsafe patch")
        text = text.replace(old, new, 1)
    train_path.write_text(text, encoding="utf-8")
    changed.append(str(train_path))

    text = ymt3_path.read_text(encoding="utf-8")
    validation_old = '''        if self.task_manager.num_decoding_channels == 1:
            bsz = self.shared_cfg["BSZ"]["validation"]
        else:
            bsz = self.shared_cfg["BSZ"]["validation"] // self.task_manager.num_decoding_channels * 3
'''
    validation_new = f'''        {VALIDATION_BSZ_MARKER}
        if self.task_manager.num_decoding_channels == 1:
            bsz = self.shared_cfg["BSZ"]["validation"]
        else:
            bsz = self.shared_cfg["BSZ"]["validation"] // self.task_manager.num_decoding_channels * 3
        bsz = max(1, int(bsz))
'''
    if VALIDATION_BSZ_MARKER not in text:
        if validation_old not in text:
            raise RuntimeError("YourMT3 ymt3.py validation batch layout changed; refusing an unsafe patch")
        text = text.replace(validation_old, validation_new, 1)
    ymt3_path.write_text(text, encoding="utf-8")
    changed.append(str(ymt3_path))
    return {"source_dir": str(source_dir), "changed": changed, "patched": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply idempotent A2S fine-tuning hooks to YourMT3.")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    result = patch_source(Path(args.source_dir))
    if args.report:
        target = Path(args.report); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
