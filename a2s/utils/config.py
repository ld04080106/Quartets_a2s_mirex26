from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load --config files; install requirements.txt") from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data


def deep_get(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value
