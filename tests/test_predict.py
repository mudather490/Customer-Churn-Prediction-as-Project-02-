"""Tests for single-customer predictions."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

REQUIRED_FIELDS = {
    "prediction",
    "prediction_label",
    "churn_probability",
    "risk_level",
    "threshold",
}


def expected_risk(probability: float) -> str:
    """Mirror the documented application-level bands."""
    if probability < 0.30:
        return "Low"
    if probability < 0.60:
        return "Medium"
    return "High"


def test_valid_low_risk_prediction(
    client: TestClient, low_risk_customer: dict[str, object]
) -> None:
    response = client.post("/predict", json=low_risk_customer)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == REQUIRED_FIELDS
    assert body["prediction"] == 0
    assert body["prediction_label"] == "No Churn"
    assert body["risk_level"] == "Low"
    assert body["threshold"] == 0.65
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_valid_high_risk_prediction(
    client: TestClient, high_risk_customer: dict[str, object]
) -> None:
    response = client.post("/predict", json=high_risk_customer)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["prediction_label"] == "Churn"
    assert body["risk_level"] == "High"
    assert body["threshold"] == 0.65
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_mid_risk_prediction(
    client: TestClient, mid_risk_customer: dict[str, object]
) -> None:
    body = client.post("/predict", json=mid_risk_customer).json()

    assert 0.30 <= body["churn_probability"] < 0.60
    assert body["risk_level"] == "Medium"
    assert body["prediction"] == 0
    assert body["prediction_label"] == "No Churn"


def test_high_risk_band_is_independent_of_threshold(
    client: TestClient, high_risk_below_threshold_customer: dict[str, object]
) -> None:
    body = client.post("/predict", json=high_risk_below_threshold_customer).json()

    # High application risk, but below the tuned 0.65 churn threshold.
    assert 0.60 <= body["churn_probability"] < 0.65
    assert body["risk_level"] == "High"
    assert body["prediction"] == 0
    assert body["prediction_label"] == "No Churn"


def test_total_charges_null_is_accepted(
    client: TestClient, high_risk_customer: dict[str, object]
) -> None:
    payload = dict(high_risk_customer, TotalCharges=None)
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in (0, 1)


def test_customer_id_is_ignored(
    client: TestClient, high_risk_customer: dict[str, object]
) -> None:
    without_id = client.post("/predict", json=high_risk_customer)
    with_id = client.post(
        "/predict", json={**high_risk_customer, "customerID": "7590-VHVEG"}
    )

    assert with_id.status_code == 200
    assert with_id.json() == without_id.json()


def test_repeated_identical_inputs_are_consistent(
    client: TestClient, mid_risk_customer: dict[str, object]
) -> None:
    first = client.post("/predict", json=mid_risk_customer)
    second = client.post("/predict", json=mid_risk_customer)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()


def test_probability_is_always_between_zero_and_one(
    client: TestClient,
    low_risk_customer: dict[str, object],
    mid_risk_customer: dict[str, object],
    high_risk_customer: dict[str, object],
    high_risk_below_threshold_customer: dict[str, object],
    borderline_churn_customer: dict[str, object],
) -> None:
    customers = [
        low_risk_customer,
        mid_risk_customer,
        high_risk_customer,
        high_risk_below_threshold_customer,
        borderline_churn_customer,
    ]

    for customer in customers:
        body = client.post("/predict", json=customer).json()
        assert 0.0 <= body["churn_probability"] <= 1.0


def test_labels_and_threshold_are_consistent(
    client: TestClient,
    low_risk_customer: dict[str, object],
    mid_risk_customer: dict[str, object],
    high_risk_customer: dict[str, object],
    high_risk_below_threshold_customer: dict[str, object],
    borderline_churn_customer: dict[str, object],
) -> None:
    customers = [
        low_risk_customer,
        mid_risk_customer,
        high_risk_customer,
        high_risk_below_threshold_customer,
        borderline_churn_customer,
    ]

    for customer in customers:
        body: dict[str, Any] = client.post("/predict", json=customer).json()
        probability = body["churn_probability"]
        threshold = body["threshold"]

        assert threshold == 0.65
        assert body["prediction"] == int(probability >= threshold)
        assert body["prediction_label"] == (
            "Churn" if body["prediction"] == 1 else "No Churn"
        )
        assert body["risk_level"] == expected_risk(probability)
