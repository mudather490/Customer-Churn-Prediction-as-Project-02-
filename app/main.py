"""FastAPI application for customer churn prediction."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.config import Settings
from app.logging_config import configure_logging
from app.model_service import (
    ModelError,
    ModelLoadError,
    ModelService,
    ModelUnavailableError,
    Prediction,
    PredictionError,
)
from app.risk import RiskBands
from app.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    CustomerFeatures,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    RootResponse,
)

logger = logging.getLogger(__name__)

API_NAME = "Customer Churn Prediction API"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the trained artifact once for the lifetime of the application."""
    settings: Settings = app.state.settings
    service = ModelService(
        settings.model_path,
        RiskBands(low_max=settings.risk_low_max, high_min=settings.risk_high_min),
    )
    app.state.model_service = service
    app.state.model_ready = False
    app.state.model_error = None

    try:
        service.load()

        artifact_fields = set(service.feature_columns)
        schema_fields = set(CustomerFeatures.model_fields)
        missing = sorted(artifact_fields - schema_fields)
        unexpected = sorted(schema_fields - artifact_fields)
        if missing or unexpected:
            raise ModelLoadError(
                "API schema does not match artifact features "
                f"(missing={missing}, unexpected={unexpected})"
            )
    except ModelError as exc:
        # Keep the process alive so /health can report the degraded state.
        app.state.model_error = str(exc)
        logger.error("Model unavailable at startup: %s", exc)
    else:
        app.state.model_ready = True
        logger.info(
            "API ready: model=%s threshold=%.4f features=%d",
            service.metadata()["model_name"],
            service.threshold,
            len(service.feature_columns),
        )

    yield


def _get_ready_service(request: Request) -> ModelService:
    """Return the loaded model service or raise a sanitized 503."""
    service = getattr(request.app.state, "model_service", None)
    if not bool(getattr(request.app.state, "model_ready", False)) or not isinstance(
        service, ModelService
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable",
        )
    return service


def _to_response(result: Prediction, threshold: float) -> PredictionResponse:
    """Convert a service prediction into the API response schema."""
    return PredictionResponse(
        prediction=result.prediction,
        prediction_label=result.prediction_label,
        churn_probability=result.churn_probability,
        risk_level=result.risk_level,
        threshold=threshold,
    )


def _register_routes(application: FastAPI) -> None:
    @application.get(
        "/",
        response_model=RootResponse,
        tags=["metadata"],
        summary="API metadata",
    )
    def root() -> RootResponse:
        """Return the API name, version, and process status."""
        return RootResponse(name=API_NAME, version=__version__, status="ok")

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["metadata"],
        summary="Health check",
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": HealthResponse,
                "description": "Model is not available",
            }
        },
    )
    def health(request: Request, response: Response) -> HealthResponse:
        """Report application health and model availability."""
        ready = bool(getattr(request.app.state, "model_ready", False))
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(
                status="degraded", model_available=False, version=__version__
            )
        return HealthResponse(status="ok", model_available=True, version=__version__)

    @application.get(
        "/model-info",
        response_model=ModelInfoResponse,
        tags=["model"],
        summary="Model metadata",
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": ErrorResponse,
                "description": "Model is not available",
            }
        },
    )
    def model_info(request: Request) -> ModelInfoResponse:
        """Return the algorithm, threshold, features, and validation metrics."""
        service = _get_ready_service(request)
        return ModelInfoResponse(**service.metadata())

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        tags=["predictions"],
        summary="Predict churn for one customer",
        responses={
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "model": ErrorResponse,
                "description": "Invalid customer input",
            },
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": ErrorResponse,
                "description": "Model is not available",
            },
        },
    )
    def predict(customer: CustomerFeatures, request: Request) -> PredictionResponse:
        """Score one customer using the existing fitted pipeline."""
        service = _get_ready_service(request)
        try:
            result = service.predict_one(customer.model_dump())
        except PredictionError:
            logger.exception("Single prediction failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction could not be completed",
            ) from None
        return _to_response(result, service.threshold)

    @application.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["predictions"],
        summary="Predict churn for a batch of customers",
        responses={
            status.HTTP_413_CONTENT_TOO_LARGE: {
                "model": ErrorResponse,
                "description": "Batch exceeds the configured maximum",
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "model": ErrorResponse,
                "description": "Invalid batch input",
            },
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": ErrorResponse,
                "description": "Model is not available",
            },
        },
    )
    def predict_batch(
        payload: BatchPredictRequest, request: Request
    ) -> BatchPredictionResponse:
        """Score multiple customers in one call, preserving input order."""
        service = _get_ready_service(request)
        settings: Settings = request.app.state.settings
        if len(payload.customers) > settings.max_batch_size:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=(
                    "Batch size exceeds the maximum of "
                    f"{settings.max_batch_size} customers"
                ),
            )

        try:
            results = service.predict(
                [customer.model_dump() for customer in payload.customers]
            )
        except PredictionError:
            logger.exception("Batch prediction failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch prediction could not be completed",
            ) from None

        return BatchPredictionResponse(
            count=len(results),
            predictions=[_to_response(result, service.threshold) for result in results],
        )


def _register_error_handlers(application: FastAPI) -> None:
    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Return field-level validation errors without echoing input values."""
        errors = []
        for error in exc.errors():
            location = ".".join(
                str(part) for part in error.get("loc", ()) if part != "body"
            )
            errors.append(
                {
                    "field": location or "body",
                    "message": error.get("msg", "Invalid value"),
                    "type": error.get("type", "value_error"),
                }
            )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"error": "validation_error", "detail": errors},
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Return a consistent envelope for HTTP errors."""
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "http_error", "detail": exc.detail},
            headers=exc.headers,
        )

    @application.exception_handler(ModelUnavailableError)
    async def model_unavailable_handler(
        request: Request, exc: ModelUnavailableError
    ) -> JSONResponse:
        """Return 503 when a prediction races with model availability."""
        logger.warning("Model unavailable during request: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "model_unavailable", "detail": "Model is unavailable"},
        )

    @application.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Log unexpected errors and avoid exposing internals to clients."""
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "internal_error", "detail": "Internal server error"},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Optional settings override, primarily for tests.

    Returns:
        A configured FastAPI application.
    """
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)

    application = FastAPI(
        title=API_NAME,
        description=(
            "Predict customer churn using the existing trained Logistic "
            "Regression pipeline. The model and its preprocessing are loaded "
            "from a trusted joblib artifact; nothing is retrained at runtime."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.state.model_service = None
    application.state.model_ready = False
    application.state.model_error = None

    _register_routes(application)
    _register_error_handlers(application)
    return application


app = create_app()
