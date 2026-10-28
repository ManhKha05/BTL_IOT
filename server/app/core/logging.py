import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure unified application logger."""
    logger = logging.getLogger("assistive_ai")
    
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = setup_logging()
