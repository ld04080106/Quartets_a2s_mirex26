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
import sys
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
            "bytes": deep_get(config, "stage1.checkpoint_bytes", None),
        },
        {
            "name": "voice_assignment_model",
            "path": _path(deep_get(config, "voice_assignment.model_path", "")),
            "url": deep_get(config, "voice_assignment.model_url", ""),
            "sha256": deep_get(config, "voice_assignment.model_sha256", ""),
            "bytes": deep_get(config, "voice_assignment.model_bytes", None),
        },
        {
            "name": "stage2_midi_checkpoint",
            "path": _path(deep_get(config, "stage2_midi.checkpoint", "")),
            "url": deep_get(config, "stage2_midi.checkpoint_url", ""),
            "sha256": deep_get(config, "stage2_midi.checkpoint_sha256", ""),
            "bytes": deep_get(config, "stage2_midi.checkpoint_bytes", None),
        },
    ]


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown size"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def _confirm_download(specs: list[dict[str, Any]], assume_yes: bool) -> bool:
    print("The following required model assets are missing:")
    for spec in specs:
        print(f"  - {spec['name']}: {_human_bytes(spec.get('bytes'))}")
        print(f"    target: {spec['path']}")
    total = sum(int(spec.get("bytes") or 0) for spec in specs)
    if total:
        print(f"Total download size: {_human_bytes(total)}")
    if assume_yes:
        print("Download confirmed automatically (non-interactive mode or --yes).")
        return True
    override = os.environ.get("A2S_ASSET_AUTO_CONFIRM", "").strip().lower()
    if override in {"1", "true", "yes", "y"}:
        print("Download confirmed by A2S_ASSET_AUTO_CONFIRM.")
        return True
    if override in {"0", "false", "no", "n"}:
        print("Download declined by A2S_ASSET_AUTO_CONFIRM.")
        return False
    if not sys.stdin.isatty():
        print("Download confirmed automatically (stdin is not interactive).")
        return True
    try:
        answer = input("Download now? [Y/n]: ").strip().lower()
    except EOFError:
        answer = ""
    return answer in {"", "y", "yes"}


def _download(
    name: str,
    url: str,
    target: Path,
    expected: str,
    expected_bytes: int | None,
    timeout: int,
    retries: int,
) -> None:
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
                print(f"Downloading {name} (attempt {attempt}/{retries})")
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
                    header_length = response.headers.get("Content-Length")
                    total = int(header_length) if header_length and header_length.isdigit() else expected_bytes
                    downloaded = 0
                    started = time.monotonic()
                    last_update = started
                    last_percent = -1
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        percent = int(downloaded * 100 / total) if total else -1
                        should_update = (
                            now - last_update >= 1.0
                            or (total is not None and downloaded >= total)
                            or (percent >= 0 and percent >= last_percent + 5)
                        )
                        if should_update:
                            elapsed = max(now - started, 1e-6)
                            speed = _human_bytes(int(downloaded / elapsed)) + "/s"
                            if total:
                                message = (
                                    f"  {percent:3d}%  {_human_bytes(downloaded)} / "
                                    f"{_human_bytes(total)}  {speed}"
                                )
                            else:
                                message = f"  {_human_bytes(downloaded)}  {speed}"
                            if sys.stdout.isatty():
                                print("\r" + message.ljust(78), end="", flush=True)
                            else:
                                print(message, flush=True)
                            last_update = now
                            last_percent = percent
                    if sys.stdout.isatty():
                        print()
                if expected_bytes is not None and temporary.stat().st_size != expected_bytes:
                    raise ValueError(
                        f"size mismatch: expected {expected_bytes}, got {temporary.stat().st_size}"
                    )
                print(f"Verifying SHA-256 for {name} ...", flush=True)
                actual = _sha256(temporary)
                if actual.lower() != expected.lower():
                    raise ValueError(f"SHA-256 mismatch: expected {expected}, got {actual}")
                os.replace(temporary, target)
                print(f"Installed {name}: {target}")
                return
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                if attempt == retries:
                    raise
                print(f"Download attempt failed for {name}: {exc}; retrying ...", file=sys.stderr)
                time.sleep(min(2 ** (attempt - 1), 4))
    finally:
        temporary.unlink(missing_ok=True)


def _write_report(report: dict[str, Any], output_path: str | None) -> None:
    """Print the asset report and optionally persist it for submission diagnostics."""
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download missing, checksum-pinned model assets.")
    parser.add_argument("--config", default="configs/pipeline_submission.yaml")
    parser.add_argument("--out_json")
    parser.add_argument("--yes", action="store_true", help="Download missing assets without prompting.")
    args = parser.parse_args()
    config = load_config(args.config)
    section = config.get("asset_download", {})
    enabled = bool(section.get("enabled", True))
    timeout = int(section.get("timeout_sec", 120))
    retries = max(1, int(section.get("retries", 3)))
    report: dict[str, Any] = {"enabled": enabled, "assets": {}, "ok": True}

    specs = _asset_specs(config)
    for spec in specs:
        spec["url"] = _expand_env_defaults(str(spec["url"] or ""))
        spec["sha256"] = _expand_env_defaults(str(spec["sha256"] or "")).strip().lower()
        if spec.get("bytes") is not None:
            spec["bytes"] = int(spec["bytes"])
    missing = [spec for spec in specs if spec["path"] is None or not spec["path"].is_file()]
    if missing and enabled and not _confirm_download(specs=missing, assume_yes=args.yes):
        report["ok"] = False
        report["cancelled"] = True
        for spec in missing:
            report["assets"][spec["name"]] = {
                "path": str(spec["path"]),
                "downloaded": False,
                "verified": False,
                "error": "download cancelled by user",
            }
        print("Model download cancelled.", file=sys.stderr)
        _write_report(report, args.out_json)
        raise SystemExit(2)

    for spec in specs:
        name = spec["name"]
        target = spec["path"]
        url = spec["url"]
        expected = spec["sha256"]
        item = {"path": str(target), "downloaded": False, "verified": False}
        try:
            if target is None:
                raise ValueError("asset path is not configured")
            if target.is_file():
                print(f"Checking cached asset: {name}")
                if expected:
                    actual = _sha256(target)
                    if actual != expected:
                        raise ValueError(f"cached SHA-256 mismatch: expected {expected}, got {actual}")
                    item["verified"] = True
                    print(f"  OK: {target}")
                else:
                    item["verified"] = None
            else:
                if not enabled:
                    raise FileNotFoundError(f"asset is missing and automatic download is disabled: {target}")
                if not url or not expected:
                    raise FileNotFoundError(
                        f"asset is missing; configure both URL and SHA-256 for {name}: {target}"
                    )
                _download(name, url, target, expected, spec.get("bytes"), timeout, retries)
                item.update({"downloaded": True, "verified": True})
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            report["ok"] = False
        report["assets"][name] = item

    _write_report(report, args.out_json)
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
