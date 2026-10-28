"""
Core module: configuration and logging setup.
"""
from app.core.config import settings
from app.core.logging import setup_logging, logger

__all__ = ["settings", "setup_logging", "logger"]
