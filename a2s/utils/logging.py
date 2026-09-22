from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: str | Path | None = None, verbose: bool = False) -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        target = Path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(target, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("a2s")
