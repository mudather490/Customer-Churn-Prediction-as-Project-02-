"""Pydantic request and response schemas for the churn API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.risk import RiskLevel

Gender = Literal["Female", "Male"]
YesNo = Literal["Yes", "No"]
MultipleLinesType = Literal["No", "Yes", "No phone service"]
InternetServiceType = Literal["DSL", "Fiber optic", "No"]
InternetAddon = Literal["No", "Yes", "No internet service"]
ContractType = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodType = Literal[
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]

EXAMPLE_CUSTOMER = {
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


class CustomerFeatures(BaseModel):
    """The 19 raw features expected by the fitted preprocessing pipeline.

    ``customerID`` and any other unknown fields are ignored. ``TotalCharges``
    may be null; the fitted pipeline's median imputer handles that case.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        json_schema_extra={"examples": [EXAMPLE_CUSTOMER]},
    )

    gender: Gender = Field(..., description="Customer gender.")
    SeniorCitizen: int = Field(
        ...,
        strict=True,
        ge=0,
        le=1,
        description="1 if the customer is a senior citizen, otherwise 0.",
    )
    Partner: YesNo = Field(..., description="Whether the customer has a partner.")
    Dependents: YesNo = Field(..., description="Whether the customer has dependents.")
    tenure: int = Field(
        ..., ge=0, le=100, description="Months the customer has stayed with the company."
    )
    PhoneService: YesNo = Field(..., description="Whether the customer has phone service.")
    MultipleLines: MultipleLinesType = Field(
        ..., description="Whether the customer has multiple phone lines."
    )
    InternetService: InternetServiceType = Field(
        ..., description="Customer's internet service provider type."
    )
    OnlineSecurity: InternetAddon = Field(
        ..., description="Whether the customer has online security."
    )
    OnlineBackup: InternetAddon = Field(
        ..., description="Whether the customer has online backup."
    )
    DeviceProtection: InternetAddon = Field(
        ..., description="Whether the customer has device protection."
    )
    TechSupport: InternetAddon = Field(
        ..., description="Whether the customer has tech support."
    )
    StreamingTV: InternetAddon = Field(
        ..., description="Whether the customer streams TV."
    )
    StreamingMovies: InternetAddon = Field(
        ..., description="Whether the customer streams movies."
    )
    Contract: ContractType = Field(..., description="Customer's contract term.")
    PaperlessBilling: YesNo = Field(
        ..., description="Whether the customer uses paperless billing."
    )
    PaymentMethod: PaymentMethodType = Field(
        ..., description="Customer's payment method."
    )
    MonthlyCharges: float = Field(
        ..., ge=0, le=1000, description="Current monthly charges."
    )
    TotalCharges: Annotated[
        float | None,
        Field(
            ge=0,
            le=1_000_000,
            description=(
                "Total charges to date. Null is accepted and imputed by the "
                "fitted pipeline."
            ),
        ),
    ]


class PredictionResponse(BaseModel):
    """Churn prediction for one customer."""

    prediction: Literal[0, 1] = Field(
        ..., description="1 when churn is predicted, otherwise 0."
    )
    prediction_label: Literal["No Churn", "Churn"] = Field(
        ..., description="Human-readable prediction."
    )
    churn_probability: float = Field(
        ..., ge=0, le=1, description="Model probability of churn."
    )
    risk_level: RiskLevel = Field(
        ...,
        description=(
            "Application-level triage band. Independent of the classification "
            "threshold."
        ),
    )
    threshold: float = Field(
        ..., gt=0, lt=1, description="Classification threshold used for this prediction."
    )


class BatchPredictRequest(BaseModel):
    """A list of customers to score in one request."""

    model_config = ConfigDict(extra="ignore")

    customers: list[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Customers to score, preserving request order.",
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction results."""

    count: int = Field(..., ge=0, description="Number of predictions returned.")
    predictions: list[PredictionResponse] = Field(
        ..., description="Predictions in the same order as the request customers."
    )


class RootResponse(BaseModel):
    """API metadata."""

    name: str
    version: str
    status: Literal["ok"]


class HealthResponse(BaseModel):
    """Application and model availability."""

    status: Literal["ok", "degraded"]
    model_available: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Metadata read from the loaded artifact."""

    project: str
    model_name: str
    algorithm: str
    threshold: float
    expected_feature_count: int
    feature_columns: list[str]
    target_mapping: dict[str, int]
    validation_metrics: dict[str, float]
    metrics_note: str
    risk_bands: dict[str, str]
    risk_bands_note: str


class ValidationErrorItem(BaseModel):
    """A single sanitized validation error."""

    field: str
    message: str
    type: str


class ErrorResponse(BaseModel):
    """Standard error envelope used by the API."""

    error: str
    detail: str | list[ValidationErrorItem]
