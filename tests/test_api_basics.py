"""Tests for root, health, model-info, and OpenAPI endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "name": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "ok",
    }


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_available": True,
        "version": "1.0.0",
    }


def test_model_info(client: TestClient) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["project"] == "Customer Churn Prediction"
    assert body["model_name"] == "Logistic Regression"
    assert body["algorithm"] == "LogisticRegression"
    assert body["threshold"] == 0.65
    assert body["expected_feature_count"] == 19
    assert len(body["feature_columns"]) == 19
    assert body["target_mapping"] == {"No": 0, "Yes": 1}
    assert body["validation_metrics"] == {
        "precision": 0.6161,
        "recall": 0.6656,
        "f1_score": 0.6399,
    }
    assert "not final test-set" in body["metrics_note"]
    assert body["risk_bands"] == {
        "low": "p < 0.30",
        "medium": "0.30 <= p < 0.60",
        "high": "p >= 0.60",
    }
    assert "independent of the classification threshold" in body["risk_bands_note"]


def test_openapi_schema(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Customer Churn Prediction API"
    assert schema["info"]["version"] == "1.0.0"
    for path in ("/", "/health", "/model-info", "/predict", "/predict/batch"):
        assert path in schema["paths"]


def test_degraded_startup_reports_model_unavailable(
    tmp_path: Path, low_risk_customer: dict[str, object]
) -> None:
    missing_path = tmp_path / "missing-model.joblib"
    degraded_app = create_app(Settings(model_path=missing_path))

    with TestClient(degraded_app) as degraded_client:
        health = degraded_client.get("/health")
        assert health.status_code == 503
        assert health.json()["status"] == "degraded"
        assert health.json()["model_available"] is False

        assert degraded_client.get("/model-info").status_code == 503
        assert degraded_client.post("/predict", json=low_risk_customer).status_code == 503

        # Root remains available so operators can identify the service.
        assert degraded_client.get("/").status_code == 200

        for response in (
            health,
            degraded_client.get("/model-info"),
            degraded_client.post("/predict", json=low_risk_customer),
        ):
            assert "missing-model.joblib" not in response.text
            assert "Traceback" not in response.text
