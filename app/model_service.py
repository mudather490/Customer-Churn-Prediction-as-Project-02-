"""Load and serve the existing trained churn pipeline.

The artifact is a trusted joblib export from the training notebook. It is
loaded once by the application lifespan and reused for every request. No
training, fitting, threshold tuning, or preprocessing changes happen here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from app.risk import RiskBands, RiskLevel

logger = logging.getLogger(__name__)

NO_CHURN_LABEL = "No Churn"
CHURN_LABEL = "Churn"


class ModelError(Exception):
    """Base class for model-related application errors."""


class ModelArtifactNotFoundError(ModelError):
    """Raised when the configured model artifact does not exist."""


class ModelLoadError(ModelError):
    """Raised when the artifact is malformed or incompatible."""


class ModelUnavailableError(ModelError):
    """Raised when a prediction is requested before the model is loaded."""


class PredictionError(ModelError):
    """Raised when the pipeline returns an unexpected prediction shape."""


@dataclass(frozen=True)
class Prediction:
    """A single churn prediction."""

    prediction: int
    prediction_label: str
    churn_probability: float
    risk_level: RiskLevel


def decide(probability: float, threshold: float) -> int:
    """Apply the selected classification threshold.

    Args:
        probability: Churn probability from ``predict_proba``.
        threshold: Decision threshold stored in the model artifact.

    Returns:
        1 when the probability meets or exceeds the threshold, otherwise 0.
    """
    return int(probability >= threshold)


class ModelService:
    """Wrapper around the fitted sklearn pipeline and its metadata."""

    def __init__(self, artifact_path: Path, risk_bands: RiskBands | None = None) -> None:
        self._artifact_path = Path(artifact_path)
        self._risk_bands = risk_bands or RiskBands()
        self._pipeline: Pipeline | None = None
        self._threshold: float | None = None
        self._feature_columns: tuple[str, ...] = ()
        self._numeric_columns: tuple[str, ...] = ()
        self._target_mapping: dict[str, int] = {}
        self._validation_metrics: dict[str, float] = {}
        self._model_name = ""
        self._project = ""
        self._algorithm = ""

    @property
    def is_loaded(self) -> bool:
        """Whether the artifact has been loaded successfully."""
        return self._pipeline is not None

    @property
    def threshold(self) -> float:
        """Decision threshold read from the artifact."""
        self._ensure_loaded()
        assert self._threshold is not None
        return self._threshold

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Input feature names in the exact order expected by the pipeline."""
        self._ensure_loaded()
        return self._feature_columns

    @property
    def risk_bands(self) -> RiskBands:
        """Application-level risk bands."""
        return self._risk_bands

    def _ensure_loaded(self) -> None:
        if self._pipeline is None:
            raise ModelUnavailableError("The churn model is not loaded")

    def load(self) -> None:
        """Load and validate the artifact.

        Raises:
            ModelArtifactNotFoundError: The configured path does not exist.
            ModelLoadError: The artifact is not a compatible churn bundle.
        """
        logger.info("Loading churn model artifact from %s", self._artifact_path)
        if not self._artifact_path.is_file():
            raise ModelArtifactNotFoundError(
                f"Model artifact not found: {self._artifact_path}"
            )

        try:
            bundle = joblib.load(self._artifact_path)
        except Exception as exc:  # joblib/pickle errors vary by version
            raise ModelLoadError("Unable to deserialize the model artifact") from exc

        if not isinstance(bundle, dict):
            raise ModelLoadError("Model artifact must contain a dictionary bundle")

        required_keys = {
            "pipeline",
            "threshold",
            "target_mapping",
            "feature_columns",
            "model_name",
            "validation_metrics",
        }
        missing = sorted(required_keys.difference(bundle))
        if missing:
            raise ModelLoadError(
                "Model artifact is missing required keys: " + ", ".join(missing)
            )

        pipeline = bundle["pipeline"]
        if not isinstance(pipeline, Pipeline):
            raise ModelLoadError("Artifact 'pipeline' is not a sklearn Pipeline")
        if not hasattr(pipeline, "predict_proba"):
            raise ModelLoadError("Pipeline does not expose predict_proba")

        feature_columns = bundle["feature_columns"]
        if (
            not isinstance(feature_columns, (list, tuple))
            or not feature_columns
            or not all(isinstance(column, str) for column in feature_columns)
        ):
            raise ModelLoadError("Artifact 'feature_columns' must be a non-empty list of strings")
        feature_columns = tuple(feature_columns)
        if len(set(feature_columns)) != len(feature_columns):
            raise ModelLoadError("Artifact 'feature_columns' contains duplicates")

        n_features = getattr(pipeline, "n_features_in_", None)
        if n_features != len(feature_columns):
            raise ModelLoadError(
                "Pipeline feature count does not match artifact feature_columns"
            )

        try:
            classes = np.asarray(pipeline.classes_).tolist()
        except AttributeError as exc:
            raise ModelLoadError("Pipeline does not expose fitted classes") from exc
        if classes != [0, 1]:
            raise ModelLoadError(
                f"Expected binary classes [0, 1] with churn in column 1, got {classes}"
            )

        try:
            threshold = float(bundle["threshold"])
        except (TypeError, ValueError) as exc:
            raise ModelLoadError("Artifact 'threshold' must be numeric") from exc
        if not 0.0 < threshold < 1.0:
            raise ModelLoadError("Artifact 'threshold' must be between 0 and 1")

        target_mapping = bundle["target_mapping"]
        if (
            not isinstance(target_mapping, dict)
            or set(target_mapping.values()) != {0, 1}
        ):
            raise ModelLoadError("Artifact 'target_mapping' must map labels to 0 and 1")

        validation_metrics = bundle["validation_metrics"]
        if not isinstance(validation_metrics, dict):
            raise ModelLoadError("Artifact 'validation_metrics' must be a dictionary")
        try:
            metrics = {str(key): float(value) for key, value in validation_metrics.items()}
        except (TypeError, ValueError) as exc:
            raise ModelLoadError("Artifact validation metrics must be numeric") from exc

        preprocessor = pipeline.named_steps.get("preprocessor")
        if not isinstance(preprocessor, ColumnTransformer):
            raise ModelLoadError("Pipeline is missing a fitted ColumnTransformer preprocessor")
        if not hasattr(preprocessor, "transformers_"):
            raise ModelLoadError("ColumnTransformer preprocessor is not fitted")

        pipeline_features = getattr(pipeline, "feature_names_in_", None)
        if pipeline_features is not None and list(pipeline_features) != list(feature_columns):
            raise ModelLoadError(
                "Artifact feature_columns do not match the pipeline's fitted feature order"
            )

        self._pipeline = pipeline
        self._threshold = threshold
        self._feature_columns = feature_columns
        self._numeric_columns = self._derive_numeric_columns(pipeline, feature_columns)
        self._target_mapping = {str(key): int(value) for key, value in target_mapping.items()}
        self._validation_metrics = metrics
        self._model_name = str(bundle["model_name"])
        self._project = str(bundle.get("project", ""))
        self._algorithm = type(pipeline.named_steps["classifier"]).__name__

        logger.info(
            "Model loaded: name=%s algorithm=%s threshold=%.4f features=%d",
            self._model_name,
            self._algorithm,
            self._threshold,
            len(self._feature_columns),
        )

    @staticmethod
    def _derive_numeric_columns(
        pipeline: Pipeline, feature_columns: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Find numeric/passthrough columns so nulls can be passed as NaN.

        Categorical columns are left untouched; the fitted imputer and
        one-hot encoder inside the pipeline remain responsible for them.
        """
        preprocessor = pipeline.named_steps["preprocessor"]
        assert isinstance(preprocessor, ColumnTransformer)

        # ``transformers`` keeps the unfitted originals; ``transformers_``
        # contains the fitted clones actually used by transform/predict.
        fitted_transformers = getattr(preprocessor, "transformers_", preprocessor.transformers)
        numeric: set[str] = set()
        for _, transformer, columns in fitted_transformers:
            if isinstance(columns, str):
                continue
            if isinstance(transformer, str) and transformer == "passthrough":
                numeric.update(columns)
                continue
            steps = getattr(transformer, "named_steps", {})
            if any(hasattr(step, "scale_") and hasattr(step, "mean_") for step in steps.values()):
                numeric.update(columns)

        return tuple(column for column in feature_columns if column in numeric)

    def _prepare_frame(self, customers: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """Build a DataFrame in artifact feature order without preprocessing."""
        rows: list[dict[str, Any]] = []
        for index, customer in enumerate(customers):
            missing = [column for column in self._feature_columns if column not in customer]
            if missing:
                raise PredictionError(
                    f"Customer at index {index} is missing features: {', '.join(missing)}"
                )
            rows.append({column: customer[column] for column in self._feature_columns})

        frame = pd.DataFrame(rows, columns=list(self._feature_columns))
        for column in self._numeric_columns:
            series = frame[column].where(frame[column].notna(), np.nan)
            try:
                frame[column] = pd.to_numeric(series, errors="raise").astype("float64")
            except (TypeError, ValueError) as exc:
                raise PredictionError(
                    f"Feature {column!r} contains a non-numeric value"
                ) from exc
        return frame

    def predict(self, customers: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        """Predict churn for one or more customers.

        Args:
            customers: Feature mappings containing the artifact's feature columns.

        Returns:
            One :class:`Prediction` per customer, preserving input order.
        """
        self._ensure_loaded()
        assert self._pipeline is not None
        if not customers:
            return []

        frame = self._prepare_frame(customers)
        probabilities = self._pipeline.predict_proba(frame)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise PredictionError("Pipeline returned an unexpected probability shape")

        results: list[Prediction] = []
        for raw_probability in probabilities[:, 1]:
            probability = float(raw_probability)
            if not 0.0 <= probability <= 1.0:
                raise PredictionError("Pipeline returned a probability outside [0, 1]")
            prediction = decide(probability, self.threshold)
            results.append(
                Prediction(
                    prediction=prediction,
                    prediction_label=CHURN_LABEL if prediction == 1 else NO_CHURN_LABEL,
                    churn_probability=probability,
                    risk_level=self._risk_bands.classify(probability),
                )
            )
        return results

    def predict_one(self, customer: Mapping[str, Any]) -> Prediction:
        """Predict churn for a single customer."""
        return self.predict([customer])[0]

    def metadata(self) -> dict[str, Any]:
        """Return model metadata for the /model-info endpoint."""
        self._ensure_loaded()
        return {
            "project": self._project,
            "model_name": self._model_name,
            "algorithm": self._algorithm,
            "threshold": self.threshold,
            "expected_feature_count": len(self._feature_columns),
            "feature_columns": list(self._feature_columns),
            "target_mapping": dict(self._target_mapping),
            "validation_metrics": dict(self._validation_metrics),
            "metrics_note": (
                "Validation-set metrics from the original training workflow; "
                "they are not final test-set performance."
            ),
            "risk_bands": self._risk_bands.as_dict(),
            "risk_bands_note": (
                "Application-level triage bands, independent of the classification "
                "threshold. High risk does not always mean the customer is classified "
                "as churn."
            ),
        }
