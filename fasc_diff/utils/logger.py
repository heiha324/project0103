from __future__ import annotations

import logging
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def setup_logger(name: str, log_path: str | Path) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    return logger


def log_message(msg: str, logger: logging.Logger | None, *, console: bool, use_tqdm: bool) -> None:
    if logger is not None:
        logger.info(msg)
    if console:
        if tqdm is not None and use_tqdm:
            tqdm.write(msg)
        else:
            print(msg, flush=True)

