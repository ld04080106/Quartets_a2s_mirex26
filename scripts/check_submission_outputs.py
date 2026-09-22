from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from a2s.evaluation.kern_validity import validate_kern_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MIREX one-audio/one-valid-kern output contract.")
    parser.add_argument("--input_audio", required=True, help="Input audio file or directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--out_json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    source = Path(args.input_audio)
    audio_paths = [source] if source.is_file() else [
        path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in {".flac", ".wav"}
    ]
    priority = {".flac": 0, ".wav": 1}
    audio_paths.sort(key=lambda path: (path.stem, priority[path.suffix.lower()], path.as_posix()))
    selected: dict[str, Path] = {}
    duplicate_basenames: list[str] = []
    for path in audio_paths:
        previous = selected.setdefault(path.stem, path)
        if previous != path and previous.suffix.lower() == path.suffix.lower():
            duplicate_basenames.append(path.stem)
    audio_paths = list(selected.values())
    if args.limit:
        audio_paths = audio_paths[: args.limit]
    expected = {path.stem for path in audio_paths}
    output_dir = Path(args.output_dir)
    output_paths = sorted(output_dir.glob("*.krn"))
    actual = {path.stem for path in output_paths}
    invalid: dict[str, list[str]] = {}
    for path in output_paths:
        result = validate_kern_text(path.read_text(encoding="utf-8-sig", errors="replace"))
        if not result.valid:
            invalid[path.name] = result.issues

    report = {
        "num_audio_inputs": len(audio_paths),
        "num_unique_audio_basenames": len(expected),
        "num_kern_outputs": len(output_paths),
        "missing_outputs": sorted(expected - actual),
        "extra_outputs": sorted(actual - expected),
        "duplicate_audio_basenames": duplicate_basenames,
        "invalid_outputs": invalid,
    }
    report["ok"] = not any((
        report["missing_outputs"], report["extra_outputs"], duplicate_basenames, invalid,
    )) and len(output_paths) == len(audio_paths)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        target = Path(args.out_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
