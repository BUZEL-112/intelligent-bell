"""Extensive structured logging utilities for SpecCache.

Provides:
- get_logger: Returns configured loggers under the speccache hierarchy.
- configure_logging: Configures log level, handlers, and formats.
- Environment variable overrides:
  * SPECCACHE_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO).
  * SPECCACHE_LOG_FILE: Optional path to write persistent log output.
"""

import logging
import os
import sys

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_INITIALIZED = False


def _get_env_log_level() -> int:
    """Resolve logging level from SPECCACHE_LOG_LEVEL environment variable."""
    env_level = os.environ.get("SPECCACHE_LOG_LEVEL", "INFO").strip().upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(env_level, logging.INFO)


def configure_logging(
    level: int | None = None,
    log_format: str | None = None,
    date_format: str | None = None,
    log_file: str | None = None,
) -> None:
    """Configure SpecCache logging handlers and formatters.

    Args:
        level: Optional explicit logging level. If None, reads SPECCACHE_LOG_LEVEL.
        log_format: Custom log line format string.
        date_format: Custom date format string.
        log_file: Optional file path to append logs to.
    """
    global _INITIALIZED

    effective_level = level if level is not None else _get_env_log_level()
    fmt = log_format or DEFAULT_LOG_FORMAT
    dfmt = date_format or DEFAULT_DATE_FORMAT

    root_logger = logging.getLogger("speccache")
    root_logger.setLevel(effective_level)

    # Avoid adding duplicate handlers if already configured
    if not root_logger.handlers:
        formatter = logging.Formatter(fmt=fmt, datefmt=dfmt)

        # Console handler to stdout for Colab and container stdout capture
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(effective_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Optional file handler from argument or environment variable
        target_file = log_file or os.environ.get("SPECCACHE_LOG_FILE")
        if target_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
                file_handler = logging.FileHandler(target_file, mode="a", encoding="utf-8")
                file_handler.setLevel(effective_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            except Exception as e:
                sys.stderr.write(f"Warning: Failed to initialize log file {target_file}: {e}\n")

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Obtain a structured logger for a module under the speccache hierarchy.

    Args:
        name: Module name, typically __name__.

    Returns:
        logging.Logger instance with speccache handlers configured.
    """
    if not _INITIALIZED:
        configure_logging()

    if not name.startswith("speccache"):
        logger_name = f"speccache.{name}"
    else:
        logger_name = name

    return logging.getLogger(logger_name)
