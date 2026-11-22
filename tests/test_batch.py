"""Tests for batch prediction."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_batch_predictions_preserve_order(
    client: TestClient,
    low_risk_customer: dict[str, object],
    mid_risk_customer: dict[str, object],
    high_risk_customer: dict[str, object],
) -> None:
    customers = [high_risk_customer, low_risk_customer, mid_risk_customer]

    response = client.post("/predict/batch", json={"customers": customers})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3

    for customer, batch_item in zip(customers, body["predictions"]):
        single = client.post("/predict", json=customer).json()
        assert batch_item["prediction"] == single["prediction"]
        assert batch_item["prediction_label"] == single["prediction_label"]
        assert batch_item["risk_level"] == single["risk_level"]
        assert batch_item["threshold"] == 0.65
        assert batch_item["churn_probability"] == pytest.approx(
            single["churn_probability"], abs=1e-12
        )


def test_batch_repeated_identical_inputs_are_consistent(
    client: TestClient,
    low_risk_customer: dict[str, object],
    high_risk_customer: dict[str, object],
) -> None:
    payload = {"customers": [low_risk_customer, high_risk_customer]}

    first = client.post("/predict/batch", json=payload)
    second = client.post("/predict/batch", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()


def test_batch_probabilities_and_labels(
    client: TestClient,
    low_risk_customer: dict[str, object],
    mid_risk_customer: dict[str, object],
    high_risk_customer: dict[str, object],
    high_risk_below_threshold_customer: dict[str, object],
) -> None:
    customers = [
        low_risk_customer,
        mid_risk_customer,
        high_risk_customer,
        high_risk_below_threshold_customer,
    ]

    body = client.post("/predict/batch", json={"customers": customers}).json()

    for item in body["predictions"]:
        probability = item["churn_probability"]
        assert 0.0 <= probability <= 1.0
        assert item["prediction"] == int(probability >= item["threshold"])
        assert item["prediction_label"] == (
            "Churn" if item["prediction"] == 1 else "No Churn"
        )


def test_batch_accepts_null_total_charges(
    client: TestClient, high_risk_customer: dict[str, object]
) -> None:
    customer = dict(high_risk_customer, TotalCharges=None)

    response = client.post("/predict/batch", json={"customers": [customer]})

    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_batch_with_one_customer(
    client: TestClient, high_risk_customer: dict[str, object]
) -> None:
    response = client.post("/predict/batch", json={"customers": [high_risk_customer]})

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["predictions"][0]["prediction"] in (0, 1)
