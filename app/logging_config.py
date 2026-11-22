"""Logging setup for the churn API."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once with a consistent format.

    Args:
        level: Logging level name, for example ``"INFO"`` or ``"DEBUG"``.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )
