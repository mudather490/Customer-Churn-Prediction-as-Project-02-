"""Tests for threshold application and risk-band independence."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.model_service import decide
from app.risk import RiskBands, RiskLevel


def test_decision_threshold_boundaries() -> None:
    assert decide(0.0, 0.65) == 0
    assert decide(0.649999, 0.65) == 0
    assert decide(0.65, 0.65) == 1
    assert decide(0.650001, 0.65) == 1
    assert decide(1.0, 0.65) == 1


def test_threshold_is_read_from_the_artifact(
    client: TestClient, high_risk_customer: dict[str, object]
) -> None:
    model_info = client.get("/model-info").json()
    prediction = client.post("/predict", json=high_risk_customer).json()

    assert model_info["threshold"] == 0.65
    assert prediction["threshold"] == model_info["threshold"]


def test_probability_below_threshold_is_no_churn(
    client: TestClient, high_risk_below_threshold_customer: dict[str, object]
) -> None:
    body = client.post("/predict", json=high_risk_below_threshold_customer).json()

    assert body["churn_probability"] < body["threshold"]
    assert body["prediction"] == 0


def test_probability_at_or_above_threshold_is_churn(
    client: TestClient, borderline_churn_customer: dict[str, object]
) -> None:
    body = client.post("/predict", json=borderline_churn_customer).json()

    assert body["churn_probability"] >= body["threshold"]
    assert body["prediction"] == 1


def test_risk_bands_are_independent_of_the_threshold() -> None:
    bands = RiskBands(low_max=0.30, high_min=0.60)

    # A High application risk band can still be below the churn threshold.
    assert bands.classify(0.62) is RiskLevel.HIGH
    assert decide(0.62, 0.65) == 0

    # Conversely, the same threshold applies regardless of the band.
    assert decide(0.65, 0.65) == 1


def test_risk_band_boundaries() -> None:
    bands = RiskBands(low_max=0.30, high_min=0.60)

    assert bands.classify(0.0) is RiskLevel.LOW
    assert bands.classify(0.299999) is RiskLevel.LOW
    assert bands.classify(0.30) is RiskLevel.MEDIUM
    assert bands.classify(0.599999) is RiskLevel.MEDIUM
    assert bands.classify(0.60) is RiskLevel.HIGH
    assert bands.classify(1.0) is RiskLevel.HIGH


def test_invalid_risk_bands_are_rejected() -> None:
    with pytest.raises(ValueError):
        RiskBands(low_max=0.60, high_min=0.30)
    with pytest.raises(ValueError):
        RiskBands(low_max=-0.10, high_min=0.60)
    with pytest.raises(ValueError):
        RiskBands(low_max=0.30, high_min=1.10)
    with pytest.raises(ValueError):
        RiskBands().classify(1.01)
