from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%H:%M:%S"


def get_logger(name: str, level: str = "INFO", save_dir: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(_LOG_FMT, _DATE_FMT))
    logger.addHandler(sh)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(save_dir, f"run_{ts}.log"))
        fh.setFormatter(logging.Formatter(_LOG_FMT, _DATE_FMT))
        logger.addHandler(fh)

    return logger
