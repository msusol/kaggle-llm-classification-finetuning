#!/usr/bin/env python3
"""Shared logging setup for all project scripts."""
import logging
import sys
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

LOGS_DIR = Path(__file__).parent.parent / "logs"
FMT = "%(asctime)s [%(levelname)s] %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


class _TqdmStream:
    """Route log output through tqdm.write so progress bars aren't clobbered."""
    @staticmethod
    def write(msg):
        tqdm.write(msg, end="")

    @staticmethod
    def flush():
        pass


def setup_logging(script_path: str, level: int = logging.INFO) -> logging.Logger:
    """Configure root logger: timestamped stdout (tqdm-safe) + log file.

    Args:
        script_path: pass __file__ from the calling script.
        level: logging level (default INFO).

    Returns:
        Root logger (use logging.info/warning/error directly after calling this).
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    stem = Path(script_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"{stem}_{timestamp}.log"

    formatter = logging.Formatter(FMT, datefmt=DATEFMT)

    stdout_handler = logging.StreamHandler(_TqdmStream())
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)

    logging.info("Log file: %s", log_file)
    return root
