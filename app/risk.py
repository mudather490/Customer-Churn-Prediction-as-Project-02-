"""Application-level churn risk bands.

These bands are a product/UX categorization. They are NOT outputs of the
trained model and are independent of the tuned classification threshold
(0.65 in the current artifact). A customer can therefore be "High" risk
while still being classified as "No Churn" if the probability is below the
decision threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    """Coarse probability band used for triage."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


@dataclass(frozen=True)
class RiskBands:
    """Probability bands for application-level risk levels.

    Bands:
        Low:    probability < low_max
        Medium: low_max <= probability < high_min
        High:   probability >= high_min
    """

    low_max: float = 0.30
    high_min: float = 0.60

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_max < self.high_min <= 1.0:
            raise ValueError("Risk bands must satisfy 0 <= low_max < high_min <= 1")

    def classify(self, probability: float) -> RiskLevel:
        """Return the risk level for a churn probability in [0, 1]."""
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be within [0, 1], got {probability}")
        if probability < self.low_max:
            return RiskLevel.LOW
        if probability < self.high_min:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH

    def as_dict(self) -> dict[str, Any]:
        """Serialize the bands for API documentation responses."""
        return {
            "low": f"p < {self.low_max:.2f}",
            "medium": f"{self.low_max:.2f} <= p < {self.high_min:.2f}",
            "high": f"p >= {self.high_min:.2f}",
        }
