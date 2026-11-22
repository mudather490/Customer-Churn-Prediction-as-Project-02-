"""Direct model-service and configuration tests."""

from __future__ import annotations

import pytest

from app.config import PROJECT_ROOT, Settings
from app.model_service import ModelService, PredictionError


def test_empty_batch_returns_empty(model_service: ModelService) -> None:
    assert model_service.predict([]) == []


def test_service_rejects_missing_feature(
    model_service: ModelService, low_risk_customer: dict[str, object]
) -> None:
    customer = dict(low_risk_customer)
    del customer["tenure"]

    with pytest.raises(PredictionError):
        model_service.predict_one(customer)


def test_service_rejects_non_numeric_numeric_field(
    model_service: ModelService, low_risk_customer: dict[str, object]
) -> None:
    customer = dict(low_risk_customer, tenure="many")

    with pytest.raises(PredictionError):
        model_service.predict_one(customer)


def test_service_batch_preserves_order(
    model_service: ModelService,
    low_risk_customer: dict[str, object],
    high_risk_customer: dict[str, object],
) -> None:
    results = model_service.predict([high_risk_customer, low_risk_customer])

    assert len(results) == 2
    assert results[0].prediction == 1
    assert results[1].prediction == 0
    assert results[0].churn_probability > results[1].churn_probability


def test_settings_resolve_relative_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHURN_MODEL_PATH", "models/customer_churn_model.joblib")

    settings = Settings.from_env()

    assert settings.model_path == (PROJECT_ROOT / "models" / "customer_churn_model.joblib").resolve()


def test_settings_reject_inverted_risk_bands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISK_LOW_MAX", "0.80")
    monkeypatch.setenv("RISK_HIGH_MIN", "0.60")

    with pytest.raises(ValueError):
        Settings.from_env()


def test_settings_reject_invalid_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "LOUD")

    with pytest.raises(ValueError):
        Settings.from_env()
