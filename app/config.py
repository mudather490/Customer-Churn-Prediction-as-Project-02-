"""Application configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "customer_churn_model.joblib"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Runtime settings.

    Attributes:
        model_path: Location of the joblib artifact containing the fitted pipeline.
        risk_low_max: Upper bound (exclusive) of the application-level Low risk band.
        risk_high_min: Lower bound (inclusive) of the application-level High risk band.
        max_batch_size: Maximum number of customers accepted by /predict/batch.
        log_level: Python logging level name.
    """

    model_path: Path = DEFAULT_MODEL_PATH
    risk_low_max: float = 0.30
    risk_high_min: float = 0.60
    max_batch_size: int = 1000
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_low_max < self.risk_high_min <= 1.0:
            raise ValueError(
                "Risk bands must satisfy 0 <= RISK_LOW_MAX < RISK_HIGH_MIN <= 1"
            )
        if self.max_batch_size < 1:
            raise ValueError("MAX_BATCH_SIZE must be at least 1")
        if self.log_level.upper() not in logging.getLevelNamesMapping():
            raise ValueError(f"Unknown LOG_LEVEL: {self.log_level!r}")

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables with safe defaults."""
        raw_path = os.getenv("CHURN_MODEL_PATH", str(DEFAULT_MODEL_PATH)).strip()
        model_path = Path(raw_path).expanduser()
        if not model_path.is_absolute():
            model_path = (PROJECT_ROOT / model_path).resolve()

        return cls(
            model_path=model_path,
            risk_low_max=_env_float("RISK_LOW_MAX", 0.30),
            risk_high_min=_env_float("RISK_HIGH_MIN", 0.60),
            max_batch_size=_env_int("MAX_BATCH_SIZE", 1000),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
