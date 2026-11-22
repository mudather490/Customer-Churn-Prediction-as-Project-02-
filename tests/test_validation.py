"""Tests for request validation and error responses."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

FEATURE_NAMES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def error_fields(response) -> list[str]:
    return [item["field"] for item in response.json()["detail"]]


@pytest.mark.parametrize("field", FEATURE_NAMES)
def test_every_feature_is_required(
    client: TestClient, high_risk_customer: dict[str, object], field: str
) -> None:
    payload = dict(high_risk_customer)
    del payload[field]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert field in error_fields(response)
    assert any(
        item["field"] == field and item["type"] == "missing"
        for item in response.json()["detail"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gender", "Unknown"),
        ("Partner", "Maybe"),
        ("Dependents", "Maybe"),
        ("PhoneService", "Maybe"),
        ("MultipleLines", "Maybe"),
        ("InternetService", "Satellite"),
        ("OnlineSecurity", "Sometimes"),
        ("OnlineBackup", "Sometimes"),
        ("DeviceProtection", "Sometimes"),
        ("TechSupport", "Sometimes"),
        ("StreamingTV", "Sometimes"),
        ("StreamingMovies", "Sometimes"),
        ("Contract", "Weekly"),
        ("PaperlessBilling", "Maybe"),
        ("PaymentMethod", "Cash"),
    ],
)
def test_invalid_categorical_values_are_rejected(
    client: TestClient,
    high_risk_customer: dict[str, object],
    field: str,
    value: str,
) -> None:
    response = client.post("/predict", json={**high_risk_customer, field: value})

    assert response.status_code == 422
    assert field in error_fields(response)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenure", "many"),
        ("MonthlyCharges", "free"),
        ("TotalCharges", "unknown"),
        ("SeniorCitizen", "0"),
        ("TotalCharges", []),
    ],
)
def test_invalid_numeric_types_are_rejected(
    client: TestClient,
    high_risk_customer: dict[str, object],
    field: str,
    value: object,
) -> None:
    response = client.post("/predict", json={**high_risk_customer, field: value})

    assert response.status_code == 422
    assert field in error_fields(response)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenure", -1),
        ("tenure", 101),
        ("MonthlyCharges", -1),
        ("MonthlyCharges", 1001),
        ("TotalCharges", -1),
        ("SeniorCitizen", 2),
        ("SeniorCitizen", -1),
    ],
)
def test_out_of_range_numeric_values_are_rejected(
    client: TestClient,
    high_risk_customer: dict[str, object],
    field: str,
    value: float,
) -> None:
    response = client.post("/predict", json={**high_risk_customer, field: value})

    assert response.status_code == 422
    assert field in error_fields(response)


def test_unknown_fields_are_ignored(
    client: TestClient, high_risk_customer: dict[str, object]
) -> None:
    payload = {
        **high_risk_customer,
        "customerID": "7590-VHVEG",
        "some_future_field": "ignored",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200


def test_empty_batch_is_rejected(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"customers": []})

    assert response.status_code == 422
    assert "customers" in error_fields(response)


def test_batch_requires_customers_key(client: TestClient) -> None:
    response = client.post("/predict/batch", json={})

    assert response.status_code == 422
    assert "customers" in error_fields(response)


def test_invalid_customer_inside_batch_is_rejected(
    client: TestClient,
    high_risk_customer: dict[str, object],
    low_risk_customer: dict[str, object],
) -> None:
    invalid = {**high_risk_customer, "Contract": "Weekly"}

    response = client.post(
        "/predict/batch", json={"customers": [low_risk_customer, invalid]}
    )

    assert response.status_code == 422
    assert "customers.1.Contract" in error_fields(response)


def test_batch_size_limit(
    low_risk_customer: dict[str, object],
    mid_risk_customer: dict[str, object],
    high_risk_customer: dict[str, object],
) -> None:
    settings = Settings(
        model_path=Settings.from_env().model_path,
        max_batch_size=2,
    )
    limited_app = create_app(settings)

    with TestClient(limited_app) as limited_client:
        response = limited_client.post(
            "/predict/batch",
            json={
                "customers": [
                    low_risk_customer,
                    mid_risk_customer,
                    high_risk_customer,
                ]
            },
        )

    assert response.status_code == 413
    assert "maximum of 2" in response.json()["detail"]


def test_validation_errors_do_not_leak_internals(
    client: TestClient, high_risk_customer: dict[str, object]
) -> None:
    payload = {**high_risk_customer, "tenure": "many", "Contract": "Weekly"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "Traceback" not in response.text
    assert "site-packages" not in response.text
    assert "customer_churn_model.joblib" not in response.text
    assert "E:\\" not in response.text
    assert str(Path.cwd()) not in response.text
