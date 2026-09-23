"""Fetch missing model assets from immutable, checksum-pinned URLs.

Existing files are never downloaded again. A configured SHA-256 is verified for
both cached and newly downloaded files, and a temporary file is atomically moved
into place only after verification succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from a2s.utils.config import deep_get, load_config
from scripts.run_submission_pipeline import _expand_env_defaults, _path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    source_dir = _path(deep_get(config, "stage1.source_dir", "third_party/YourMT3"))
    stage1_value = Path(str(deep_get(config, "stage1.checkpoint_path", "")))
    stage1_path = stage1_value if stage1_value.is_absolute() else source_dir / stage1_value
    return [
        {
            "name": "stage1_checkpoint",
            "path": stage1_path,
            "url": deep_get(config, "stage1.checkpoint_url", ""),
            "sha256": deep_get(config, "stage1.checkpoint_sha256", ""),
        },
        {
            "name": "voice_assignment_model",
            "path": _path(deep_get(config, "voice_assignment.model_path", "")),
            "url": deep_get(config, "voice_assignment.model_url", ""),
            "sha256": deep_get(config, "voice_assignment.model_sha256", ""),
        },
        {
            "name": "stage2_midi_checkpoint",
            "path": _path(deep_get(config, "stage2_midi.checkpoint", "")),
            "url": deep_get(config, "stage2_midi.checkpoint_url", ""),
            "sha256": deep_get(config, "stage2_midi.checkpoint_sha256", ""),
        },
    ]


def _download(url: str, target: Path, expected: str, timeout: int, retries: int) -> None:
    token = os.environ.get("A2S_ASSET_BEARER_TOKEN", "")
    headers = {"User-Agent": "quartets-a2s-mirex2026/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        for attempt in range(1, retries + 1):
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                actual = _sha256(temporary)
                if actual.lower() != expected.lower():
                    raise ValueError(f"SHA-256 mismatch: expected {expected}, got {actual}")
                os.replace(temporary, target)
                return
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == retries:
                    raise
                time.sleep(min(2 ** (attempt - 1), 4))
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download missing, checksum-pinned model assets.")
    parser.add_argument("--config", default="configs/pipeline_submission.yaml")
    parser.add_argument("--out_json")
    args = parser.parse_args()
    config = load_config(args.config)
    section = config.get("asset_download", {})
    enabled = bool(section.get("enabled", True))
    timeout = int(section.get("timeout_sec", 120))
    retries = max(1, int(section.get("retries", 3)))
    report: dict[str, Any] = {"enabled": enabled, "assets": {}, "ok": True}

    for spec in _asset_specs(config):
        name = spec["name"]
        target = spec["path"]
        url = _expand_env_defaults(str(spec["url"] or ""))
        expected = _expand_env_defaults(str(spec["sha256"] or "")).strip().lower()
        item = {"path": str(target), "downloaded": False, "verified": False}
        try:
            if target is None:
                raise ValueError("asset path is not configured")
            if target.is_file():
                if expected:
                    actual = _sha256(target)
                    if actual != expected:
                        raise ValueError(f"cached SHA-256 mismatch: expected {expected}, got {actual}")
                    item["verified"] = True
                else:
                    item["verified"] = None
            else:
                if not enabled:
                    raise FileNotFoundError(f"asset is missing and automatic download is disabled: {target}")
                if not url or not expected:
                    raise FileNotFoundError(
                        f"asset is missing; configure both URL and SHA-256 for {name}: {target}"
                    )
                _download(url, target, expected, timeout, retries)
                item.update({"downloaded": True, "verified": True})
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            report["ok"] = False
        report["assets"][name] = item

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        output = Path(args.out_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
