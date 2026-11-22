"""Tests that the existing artifact loads and is used unchanged."""

from __future__ import annotations

import joblib
import pytest
from sklearn.pipeline import Pipeline

from app.config import Settings
from app.model_service import (
    ModelArtifactNotFoundError,
    ModelLoadError,
    ModelService,
    ModelUnavailableError,
)

EXPECTED_FEATURES = [
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

EXPECTED_METRICS = {"precision": 0.6161, "recall": 0.6656, "f1_score": 0.6399}


def test_artifact_file_exists() -> None:
    path = Settings.from_env().model_path
    assert path.is_file(), f"Missing model artifact: {path}"


def test_artifact_bundle_contract() -> None:
    bundle = joblib.load(Settings.from_env().model_path)

    assert isinstance(bundle, dict)
    assert {
        "pipeline",
        "threshold",
        "target_mapping",
        "feature_columns",
        "model_name",
        "project",
        "validation_metrics",
    }.issubset(bundle)

    pipeline = bundle["pipeline"]
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocessor", "classifier"]
    assert pipeline.n_features_in_ == 19
    assert list(pipeline.feature_names_in_) == EXPECTED_FEATURES
    assert list(pipeline.classes_) == [0, 1]
    assert hasattr(pipeline, "predict_proba")

    classifier = pipeline.named_steps["classifier"]
    assert classifier.__class__.__name__ == "LogisticRegression"
    assert hasattr(classifier, "coef_"), "Artifact classifier is not fitted"

    assert bundle["threshold"] == 0.65
    assert bundle["target_mapping"] == {"No": 0, "Yes": 1}
    assert bundle["feature_columns"] == EXPECTED_FEATURES
    assert bundle["validation_metrics"] == EXPECTED_METRICS


def test_service_loads_and_exposes_metadata() -> None:
    service = ModelService(Settings.from_env().model_path)
    service.load()

    assert service.is_loaded
    assert service.threshold == 0.65
    assert list(service.feature_columns) == EXPECTED_FEATURES

    metadata = service.metadata()
    assert metadata["model_name"] == "Logistic Regression"
    assert metadata["algorithm"] == "LogisticRegression"
    assert metadata["threshold"] == 0.65
    assert metadata["expected_feature_count"] == 19
    assert metadata["feature_columns"] == EXPECTED_FEATURES
    assert metadata["target_mapping"] == {"No": 0, "Yes": 1}
    assert metadata["validation_metrics"] == EXPECTED_METRICS
    assert "not final test-set" in metadata["metrics_note"]


def test_load_does_not_retrain(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fit(*args: object, **kwargs: object) -> None:
        raise AssertionError("ModelService.load() must not call Pipeline.fit()")

    monkeypatch.setattr(Pipeline, "fit", fail_fit)

    service = ModelService(Settings.from_env().model_path)
    service.load()

    assert service.is_loaded


def test_missing_artifact_raises(tmp_path) -> None:
    service = ModelService(tmp_path / "missing.joblib")

    with pytest.raises(ModelArtifactNotFoundError):
        service.load()


def test_malformed_artifact_raises(tmp_path) -> None:
    path = tmp_path / "malformed.joblib"
    joblib.dump({"threshold": 0.65}, path)
    service = ModelService(path)

    with pytest.raises(ModelLoadError):
        service.load()


def test_predict_before_load_raises(low_risk_customer: dict[str, object]) -> None:
    service = ModelService(Settings.from_env().model_path)

    with pytest.raises(ModelUnavailableError):
        service.predict_one(low_risk_customer)
