"""Shared fixtures and realistic IBM Telco-style customers."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.model_service import ModelService

LOW_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 55.0,
    "TotalCharges": 3300.0,
}

HIGH_RISK_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 95.0,
}

# p ~= 0.450 -> Medium band, below the 0.65 decision threshold.
MID_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 50.0,
    "TotalCharges": 600.0,
}

# p ~= 0.620 -> High risk band but still below the 0.65 decision threshold.
HIGH_RISK_BELOW_THRESHOLD_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 70.0,
}

# p ~= 0.660 -> just above the 0.65 decision threshold.
BORDERLINE_CHURN_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 90.0,
    "TotalCharges": 1080.0,
}


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """Run the real application lifespan once for the test session."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def model_service() -> ModelService:
    """A separately loaded service for direct unit-level checks."""
    service = ModelService(Settings.from_env().model_path)
    service.load()
    return service


@pytest.fixture
def low_risk_customer() -> dict[str, object]:
    return dict(LOW_RISK_CUSTOMER)


@pytest.fixture
def high_risk_customer() -> dict[str, object]:
    return dict(HIGH_RISK_CUSTOMER)


@pytest.fixture
def mid_risk_customer() -> dict[str, object]:
    return dict(MID_RISK_CUSTOMER)


@pytest.fixture
def high_risk_below_threshold_customer() -> dict[str, object]:
    return dict(HIGH_RISK_BELOW_THRESHOLD_CUSTOMER)


@pytest.fixture
def borderline_churn_customer() -> dict[str, object]:
    return dict(BORDERLINE_CHURN_CUSTOMER)
