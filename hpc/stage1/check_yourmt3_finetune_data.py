from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from a2s.utils.config import deep_get, load_config  # noqa: E402


def _path(value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def _resolve_yourmt3_source(value: str | None) -> Path:
    tried: list[Path] = []
    raw = value or "hpc_assets/stage1/sources/YourMT3"
    primary = _path(raw)
    candidates = [primary]
    if primary.name == "YourMT":
        candidates.append(primary.with_name("YourMT3"))
    candidates.append((PROJECT / "hpc_assets" / "stage1" / "sources" / "YourMT3").resolve())
    env_value = os.environ.get("YOURMT3_SOURCE_DIR")
    if env_value:
        candidates.append(_path(env_value))
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
    fallback = source / "amt" / "logs" / "2024" / "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops" / "checkpoints" / "last.ckpt"
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prepared YourMT3 quartet indexes without training.")
    parser.add_argument("--config", default="configs/stage1_yourmt3_synth_finetune_h800_4gpu_pad05_nops.yaml")
    parser.add_argument("--full", action="store_true", help="Check every entry instead of a small sample.")
    parser.add_argument("--out", default="data/reports/yourmt3_finetune_preflight.json")
    parser.add_argument("--source_dir")
    parser.add_argument("--data_home")
    args = parser.parse_args()
    config = load_config(_path(args.config))
    source = _resolve_yourmt3_source(args.source_dir or deep_get(config, "yourmt3.source_dir", "hpc_assets/stage1/sources/YourMT3"))
    data_home = _path(args.data_home or deep_get(config, "yourmt3.data_home", "data/yourmt3_data"))
    checkpoint = _resolve_checkpoint(config, source)
    source_python = source / "amt" / "src"
    if not (source_python / "utils" / "note_event_dataclasses.py").is_file():
        raise SystemExit(f"YourMT3 source is incomplete: {source}")
    sys.path.insert(0, str(source_python))
    import numpy as np
    import utils.note_event_dataclasses  # noqa: F401 - required for pickle loading

    report = {"source_dir": str(source), "checkpoint": str(checkpoint), "splits": {}, "errors": []}
    if not checkpoint.is_file(): report["errors"].append(f"checkpoint missing: {checkpoint}")
    for split in ("train", "validation", "test"):
        index_path = data_home / "yourmt3_indexes" / f"quartets_{split}_file_list.json"
        if not index_path.is_file():
            report["splits"][split] = {"count": 0, "missing_index": True}
            if split != "test": report["errors"].append(f"index missing: {index_path}")
            continue
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        entries = list(payload.values())
        selected = entries if args.full or len(entries) <= 8 else entries[:4] + entries[-4:]
        checked = 0; notes = 0
        for entry in selected:
            try:
                paths = [Path(entry[key]) for key in ("mix_audio_file", "notes_file", "note_events_file", "midi_file")]
                if not all(path.is_file() for path in paths): raise FileNotFoundError(paths)
                with wave.open(str(paths[0]), "rb") as handle:
                    if handle.getframerate() != 16000 or handle.getnchannels() != 1:
                        raise ValueError(f"bad WAV format: {paths[0]}")
                note_payload = np.load(paths[1], allow_pickle=True).tolist()
                event_payload = np.load(paths[2], allow_pickle=True).tolist()
                if not isinstance(note_payload.get("notes"), list) or not isinstance(event_payload.get("note_events"), list):
                    raise ValueError("invalid npy payload")
                notes += len(note_payload["notes"]); checked += 1
            except Exception as exc:
                report["errors"].append(f"{split}:{entry.get('sample_id')}: {type(exc).__name__}: {exc}")
        report["splits"][split] = {"count": len(entries), "checked": checked, "checked_notes": notes, "missing_index": False}
    target = _path(args.out); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["errors"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
