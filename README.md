# Customer Churn Prediction API

A production-ready FastAPI service built around the **existing trained Logistic Regression churn model**. The service loads the exported scikit-learn pipeline once at startup and exposes single-customer and batch prediction endpoints.

The model is **not retrained, re-tuned, replaced, or modified** by this application. The saved artifact remains the source of truth for preprocessing, feature order, the classification threshold, and validation metadata.

---

## Contents

- [Overview](#overview)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Model](#model)
- [Input features](#input-features)
- [Preprocessing](#preprocessing)
- [Classification threshold](#classification-threshold)
- [Validation metrics](#validation-metrics)
- [Application risk levels](#application-risk-levels)
- [API](#api)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [Environment variables](#environment-variables)
- [Example requests](#example-requests)
- [Testing](#testing)
- [Google Colab workspace](#google-colab-workspace)
- [Deployment preparation](#deployment-preparation)
- [Troubleshooting](#troubleshooting)
- [Design notes and limitations](#design-notes-and-limitations)
- [Model artifact provenance](#model-artifact-provenance)

---

## Overview

The application:

- Loads `models/customer_churn_model.joblib` once during the FastAPI lifespan.
- Uses the fitted pipeline directly: raw customer features in, churn probability out.
- Reads the decision threshold (`0.65`) from the artifact rather than hardcoding it in prediction logic.
- Reads the expected feature order from the artifact and validates it against the API schema at startup.
- Returns `prediction`, `prediction_label`, `churn_probability`, `risk_level`, and the threshold used.
- Supports batch prediction while preserving input order.
- Validates all 19 required features with Pydantic.
- Returns sanitized error responses without internal paths or stack traces.
- Starts in a degraded state if the artifact is missing, so `/health` can report the problem.

The service does not use a database and does not perform training at startup or request time.

---

## Project structure

```
customer-churn-prediction/
├── app/
│   ├── __init__.py            # API version
│   ├── config.py              # Environment-driven settings
│   ├── logging_config.py      # Logging setup
│   ├── main.py                # FastAPI app, lifespan, routes, error handlers
│   ├── model_service.py       # Artifact loading, prediction, threshold, risk
│   ├── risk.py                # Application-level risk bands
│   └── schemas.py             # Pydantic request/response schemas
├── models/
│   ├── .gitkeep               # Keeps the directory in version control
│   └── customer_churn_model.joblib   # Tracked trained artifact (9 KB)
├── notebooks/
│   └── Customer_Churn_Colab.ipynb    # Google Colab workspace
├── tests/
│   ├── conftest.py
│   ├── test_api_basics.py
│   ├── test_batch.py
│   ├── test_model_loading.py
│   ├── test_model_service.py
│   ├── test_predict.py
│   ├── test_threshold.py
│   └── test_validation.py
├── .env.example
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

---

## Dataset

The model was trained on the **IBM Telco Customer Churn** dataset schema. The dataset describes customer demographics, account tenure, subscribed services, contract type, billing preferences, and charges.

- **Target:** `Churn`
  - `No` → `0`
  - `Yes` → `1`
- **Excluded identifier:** `customerID` was excluded during training and is not required by this API.
- **Model input:** exactly 19 raw features (listed below).
- **Prediction target:** probability of class `1` (churn).

The original Colab notebook and training dataset are not part of this repository. This service consumes the exported model artifact produced by that workflow.

---

## Model

| Item | Value |
|---|---|
| Algorithm | Logistic Regression |
| Estimator | `sklearn.linear_model.LogisticRegression` |
| Preprocessor | Fitted `sklearn.compose.ColumnTransformer` |
| Pipeline | Fitted `sklearn.pipeline.Pipeline` |
| Artifact format | Joblib |
| Artifact path | `models/customer_churn_model.joblib` |
| Classification threshold | `0.65` |
| Input features | 19 |
| Transformed features | 45 |
| Output | Probability of churn |

The artifact is a dictionary containing:

| Key | Meaning |
|---|---|
| `pipeline` | Fitted preprocessing + Logistic Regression pipeline |
| `threshold` | Selected classification threshold (`0.65`) |
| `target_mapping` | `{"No": 0, "Yes": 1}` |
| `feature_columns` | Exact 19-feature input order |
| `model_name` | `Logistic Regression` |
| `project` | `Customer Churn Prediction` |
| `validation_metrics` | Validation precision, recall, and F1 |

Prediction logic:

```python
churn_probability = pipeline.predict_proba(input_data)[:, 1]
prediction = 1 if churn_probability >= threshold else 0
```

The API deliberately does **not** call `pipeline.predict()`, because that would use the default `0.5` cutoff instead of the selected `0.65` threshold.

---

## Input features

All 19 fields are required. Categorical values are validated against the exact categories used during training.

| # | Feature | Type | Accepted values / notes |
|---|---|---|---|
| 1 | `gender` | categorical | `Female`, `Male` |
| 2 | `SeniorCitizen` | integer | `0` or `1` |
| 3 | `Partner` | categorical | `Yes`, `No` |
| 4 | `Dependents` | categorical | `Yes`, `No` |
| 5 | `tenure` | integer | Months with the company; API range `0–100` |
| 6 | `PhoneService` | categorical | `Yes`, `No` |
| 7 | `MultipleLines` | categorical | `No`, `Yes`, `No phone service` |
| 8 | `InternetService` | categorical | `DSL`, `Fiber optic`, `No` |
| 9 | `OnlineSecurity` | categorical | `No`, `Yes`, `No internet service` |
| 10 | `OnlineBackup` | categorical | `No`, `Yes`, `No internet service` |
| 11 | `DeviceProtection` | categorical | `No`, `Yes`, `No internet service` |
| 12 | `TechSupport` | categorical | `No`, `Yes`, `No internet service` |
| 13 | `StreamingTV` | categorical | `No`, `Yes`, `No internet service` |
| 14 | `StreamingMovies` | categorical | `No`, `Yes`, `No internet service` |
| 15 | `Contract` | categorical | `Month-to-month`, `One year`, `Two year` |
| 16 | `PaperlessBilling` | categorical | `Yes`, `No` |
| 17 | `PaymentMethod` | categorical | `Bank transfer (automatic)`, `Credit card (automatic)`, `Electronic check`, `Mailed check` |
| 18 | `MonthlyCharges` | number | API range `0–1000` |
| 19 | `TotalCharges` | number or null | API range `0–1,000,000`; null is accepted |

Notes:

- `customerID` and any other unknown fields are accepted and ignored.
- `TotalCharges` may be `null`; the fitted pipeline's median imputer handles it.
- Missing fields, unknown categories, wrong types, and out-of-range values return `422`.

---

## Preprocessing

Preprocessing is entirely contained in the fitted artifact. The API does not manually scale, encode, or impute features.

| Feature group | Pipeline steps |
|---|---|
| Numeric: `tenure`, `MonthlyCharges`, `TotalCharges` | `SimpleImputer(strategy="median")` → `StandardScaler()` |
| Categorical: 15 columns | `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")` |
| Binary: `SeniorCitizen` | Passthrough |

The `ColumnTransformer` produces 45 transformed features for the Logistic Regression classifier.

Beyond Pydantic validation, the API's only input handling is marshalling JSON values into a DataFrame in the artifact's `feature_columns` order and converting numeric nulls to `NaN` so the fitted imputer can handle them.

Although the pipeline can impute missing categorical values, the API requires categorical fields to contain one of the documented values. This avoids silently replacing a missing category with the training-set mode.

---

## Classification threshold

The selected threshold is:

```
0.65
```

It is stored in the artifact and read at startup. The prediction rule is:

```text
prediction = 1 (Churn)     if churn_probability >= 0.65
prediction = 0 (No Churn)  otherwise
```

The threshold was selected during the original validation workflow. This application treats it as fixed and does not tune it at runtime.

The threshold is exposed in every prediction response and in `/model-info` for auditability.

---

## Validation metrics

These metrics are stored in the artifact and are **validation-set results from the original training workflow**. They are **not final test-set performance** and should not be presented as such.

| Metric | Value |
|---|---|
| Precision | `0.6161` |
| Recall | `0.6656` |
| F1 score | `0.6399` |
| Selected threshold | `0.65` |

The API does not recompute or fabricate performance metrics. `/model-info` returns these values with an explicit note that they are validation metrics.

---

## Application risk levels

`risk_level` is an **application-level triage category**, not an output of the trained model. It is independent of the `0.65` classification threshold.

| Risk level | Probability band |
|---|---|
| `Low` | `p < 0.30` |
| `Medium` | `0.30 <= p < 0.60` |
| `High` | `p >= 0.60` |

Because the bands and the threshold are independent, they can disagree:

- A customer with `p = 0.62` is **High risk** but is classified as **No Churn**, because `0.62 < 0.65`.
- A customer with `p = 0.66` is **High risk** and classified as **Churn**.

This behavior is intentional. Consumers should use `prediction` for the trained decision and `risk_level` for triage or outreach prioritization.

The bands can be changed with `RISK_LOW_MAX` and `RISK_HIGH_MIN`; they must satisfy `0 <= low_max < high_min <= 1`.

---

## API

Base URL for local development:

```
http://127.0.0.1:8000
```

Interactive documentation:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

### Endpoint summary

| Method | Path | Purpose | Success | Errors |
|---|---|---|---|---|
| `GET` | `/` | API name, version, status | `200` | — |
| `GET` | `/health` | Health and model availability | `200` | `503` |
| `GET` | `/model-info` | Model metadata and validation metrics | `200` | `503` |
| `POST` | `/predict` | Predict one customer | `200` | `422`, `503`, `500` |
| `POST` | `/predict/batch` | Predict a list of customers | `200` | `413`, `422`, `503`, `500` |

### `GET /`

```json
{
  "name": "Customer Churn Prediction API",
  "version": "1.0.0",
  "status": "ok"
}
```

### `GET /health`

Healthy:

```json
{
  "status": "ok",
  "model_available": true,
  "version": "1.0.0"
}
```

Degraded (HTTP `503`):

```json
{
  "status": "degraded",
  "model_available": false,
  "version": "1.0.0"
}
```

### `GET /model-info`

```json
{
  "project": "Customer Churn Prediction",
  "model_name": "Logistic Regression",
  "algorithm": "LogisticRegression",
  "threshold": 0.65,
  "expected_feature_count": 19,
  "feature_columns": [
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
    "TotalCharges"
  ],
  "target_mapping": {
    "No": 0,
    "Yes": 1
  },
  "validation_metrics": {
    "precision": 0.6161,
    "recall": 0.6656,
    "f1_score": 0.6399
  },
  "metrics_note": "Validation-set metrics from the original training workflow; they are not final test-set performance.",
  "risk_bands": {
    "low": "p < 0.30",
    "medium": "0.30 <= p < 0.60",
    "high": "p >= 0.60"
  },
  "risk_bands_note": "Application-level triage bands, independent of the classification threshold. High risk does not always mean the customer is classified as churn."
}
```

### `POST /predict`

Accepts one customer object containing the 19 features.

```json
{
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
  "TotalCharges": 95.0
}
```

Response:

```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "churn_probability": 0.9272416200659704,
  "risk_level": "High",
  "threshold": 0.65
}
```

### `POST /predict/batch`

Request:

```json
{
  "customers": [
    {
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
      "TotalCharges": 3300.0
    },
    {
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
      "TotalCharges": 95.0
    }
  ]
}
```

Response:

```json
{
  "count": 2,
  "predictions": [
    {
      "prediction": 0,
      "prediction_label": "No Churn",
      "churn_probability": 0.0150433133359287,
      "risk_level": "Low",
      "threshold": 0.65
    },
    {
      "prediction": 1,
      "prediction_label": "Churn",
      "churn_probability": 0.9272416200659704,
      "risk_level": "High",
      "threshold": 0.65
    }
  ]
}
```

Batch rules:

- `customers` must contain at least 1 item.
- The default maximum is `1,000` customers (`MAX_BATCH_SIZE`).
- The hard schema cap is `10,000` items.
- Results preserve request order.
- A batch over the configured limit returns `413`.
- An invalid customer anywhere in the batch returns `422` with a field path such as `customers.1.Contract`.

### Error responses

Validation error (`422`):

```json
{
  "error": "validation_error",
  "detail": [
    {
      "field": "Contract",
      "message": "Input should be 'Month-to-month', 'One year' or 'Two year'",
      "type": "literal_error"
    }
  ]
}
```

Model unavailable (`503`):

```json
{
  "error": "http_error",
  "detail": "Model is unavailable"
}
```

Batch too large (`413`):

```json
{
  "error": "http_error",
  "detail": "Batch size exceeds the maximum of 1000 customers"
}
```

Unexpected server error (`500`):

```json
{
  "error": "internal_error",
  "detail": "Internal server error"
}
```

Internal paths and tracebacks are logged server-side only; they are never returned to clients.

---

## Installation

Tested with **Python 3.13.14** on Windows. The pinned `scikit-learn==1.6.1` matches the artifact's training version.

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. Install runtime dependencies

```powershell
pip install -r requirements.txt
```

### 3. Model artifact

The trained artifact is included in this repository at:

```
models/customer_churn_model.joblib
```

A fresh clone is therefore runnable without extra provisioning. If you keep the artifact elsewhere, set `CHURN_MODEL_PATH` instead:

```powershell
$env:CHURN_MODEL_PATH = "D:\artifacts\customer_churn_model.joblib"
```

Relative `CHURN_MODEL_PATH` values are resolved from the project root.

### 4. Verify the artifact

The verified artifact in this workspace has SHA256:

```
C4D73339D823F3FBA77FECC7235DF7DAC31E9044D8B80B5332949C0D459D503E
```

PowerShell:

```powershell
Get-FileHash models\customer_churn_model.joblib -Algorithm SHA256
```

Only deploy an artifact exported from the trusted training workflow. Joblib/pickle files can execute code during deserialization.

---

## Running the API

Start the server:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For development with auto-reload:

```powershell
uvicorn app.main:app --reload
```

Using an environment file:

```powershell
Copy-Item .env.example .env
uvicorn app.main:app --env-file .env
```

`uvicorn[standard]` includes `python-dotenv`, which provides `--env-file` support.

At startup, the application:

1. Reads settings from the environment.
2. Loads the artifact once.
3. Validates the pipeline, threshold, classes, feature count, and feature order.
4. Verifies that the Pydantic schema matches the artifact features.
5. Logs the model name, threshold, and feature count.

If loading fails, the process stays up but `/health` returns `503` and prediction endpoints refuse requests.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `CHURN_MODEL_PATH` | `models/customer_churn_model.joblib` | Path to the trusted joblib artifact. Relative paths resolve from the project root. |
| `RISK_LOW_MAX` | `0.30` | Upper bound (exclusive) of the `Low` risk band. |
| `RISK_HIGH_MIN` | `0.60` | Lower bound (inclusive) of the `High` risk band. |
| `MAX_BATCH_SIZE` | `1000` | Maximum customers accepted by `/predict/batch`. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |

Risk bands must satisfy:

```
0 <= RISK_LOW_MAX < RISK_HIGH_MIN <= 1
```

Invalid settings fail fast at startup with a clear error.

---

## Example requests

Save the single-customer JSON as `customer.json`, then use either client.

### curl

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @customer.json
```

### PowerShell

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body (Get-Content .\customer.json -Raw)
```

### Batch

Save a `{ "customers": [ ... ] }` body as `batch.json`:

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d @batch.json
```

---

## Testing

Install test dependencies:

```powershell
pip install -r requirements-dev.txt
```

Run the suite:

```powershell
pytest
```

Current result:

```
92 passed
```

The suite covers:

1. Successful artifact loading and metadata validation.
2. Health endpoint, including degraded startup.
3. Model-info endpoint.
4. Valid low-, medium-, and high-risk predictions.
5. Invalid categorical and numeric input.
6. Missing required fields — all 19 features tested individually.
7. Batch prediction, ordering, and limits.
8. Correct application of the `0.65` threshold, including exact boundary behavior.
9. Probability output constrained to `[0, 1]`.
10. Deterministic repeated predictions.
11. Risk-band independence from the classification threshold.
12. Sanitized error responses with no paths or tracebacks.

Tests use realistic IBM Telco-style customers. No model performance metrics are fabricated.

Run a focused test file:

```powershell
pytest tests/test_threshold.py -q
```

---

## Google Colab workspace

`notebooks/Customer_Churn_Colab.ipynb` turns Google Colab into a complete workspace for this project.

Open it directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mudather490/Customer-Churn-Prediction-as-Project-02-/blob/main/notebooks/Customer_Churn_Colab.ipynb)

The notebook:

1. Clones the repository (or pulls the latest changes).
2. Installs the pinned dependencies from `requirements-dev.txt`.
3. Verifies the model artifact SHA256.
4. Runs the full pytest suite.
5. Calls the API in-process through FastAPI's test client.
6. Optionally serves Swagger UI on a temporary Colab URL.
7. Commits and pushes your edits back to GitHub using a token from Colab Secrets.

### Editing code in Colab

- Files appear in the Colab file browser under `/content/Customer-Churn-Prediction-as-Project-02-`.
- Double-click a `.py` file to edit it; Colab saves changes to the runtime disk.
- Re-run the test cell before pushing.
- Add a GitHub token with `repo` scope as a Colab secret named `GITHUB_TOKEN`, and enable notebook access to that secret.

Colab sessions are temporary. Push or download your changes before closing the tab.

---

## Deployment preparation

### Artifact provisioning

The trained artifact is committed at `models/customer_churn_model.joblib`, so a normal clone or image build already contains it. You can still override the path at deploy time:

- Keep the committed artifact for simple deployments.
- Mount a volume or Kubernetes secret containing a different `customer_churn_model.joblib`.
- Download a versioned artifact from object storage in CI/CD.
- Set `CHURN_MODEL_PATH` to the absolute runtime path when the artifact lives outside the project.

Verify the SHA256 hash before starting the service.

### Process management

Example production command:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Each worker loads the artifact once at startup. The artifact is small (about 9 KB), so multiple workers are inexpensive.

Do not use `--reload` in production.

### Health and readiness

- Use `GET /health` as the readiness probe.
- `200` means the model is loaded and ready.
- `503` means the process is running but the model is unavailable.
- Do not route prediction traffic when `/health` returns `503`.
- `GET /` remains available for service identification.

### Logging

The application logs to standard output with timestamps, levels, logger names, and messages. Collect logs with the platform's normal mechanism.

Customer payloads are not logged. Prediction failures are logged with server-side exception details; clients receive a generic `500`.

### Security hardening

This service intentionally does not implement authentication, authorization, or rate limiting. In production:

- Put the API behind a gateway or reverse proxy.
- Enforce TLS.
- Add authentication and rate limiting appropriate to your environment.
- Restrict batch sizes and request body sizes at the gateway as well as in the app.
- Treat the joblib artifact as trusted executable input.

### Dependency compatibility

- Keep `scikit-learn==1.6.1` unless the artifact is re-exported and retested.
- Keep `numpy` and `pandas` on the pinned versions unless the full suite is rerun.
- The API does not retrain or re-fit anything, so model behavior depends on the artifact and the pinned runtime.

### No database

The service is stateless apart from the loaded model. No database, migration, or external service is required.

---

## Troubleshooting

### `/health` returns `503`

Check the server logs for `Model unavailable at startup`. Common causes:

- `models/customer_churn_model.joblib` is missing.
- `CHURN_MODEL_PATH` points to the wrong location.
- The artifact is malformed or was exported with an incompatible scikit-learn version.
- The API schema and artifact feature lists do not match.

### `ModelArtifactNotFoundError` in logs

Verify the path:

```powershell
Test-Path models\customer_churn_model.joblib
```

Relative paths are resolved from the project root, not the current shell directory.

### scikit-learn version warning

Install the pinned version:

```powershell
pip install scikit-learn==1.6.1
```

Do not upgrade scikit-learn without re-exporting and retesting the artifact.

### `422` responses

Read the `detail` array. Each item names the failing field and the reason. Common causes:

- A categorical value is not one of the documented categories.
- A required field is missing.
- `SeniorCitizen` is not `0` or `1`.
- A numeric value is negative or outside the documented range.

### `413` on batch requests

Reduce the number of customers below `MAX_BATCH_SIZE`, or raise the limit deliberately in the environment.

### Port already in use

Choose another port:

```powershell
uvicorn app.main:app --port 8001
```

### `uvicorn --env-file` is not recognized

Ensure the runtime dependencies are installed, including `uvicorn[standard]`, which brings in `python-dotenv`.

---

## Design notes and limitations

- **Artifact is authoritative.** Threshold, feature order, model name, target mapping, and validation metrics come from the artifact.
- **No retraining.** `Pipeline.fit()` is never called by the application. A test guards this.
- **No preprocessing duplication.** Scaling, imputation, and one-hot encoding remain inside the fitted pipeline.
- **Startup validation.** The app checks binary classes `[0, 1]`, feature count, feature order, and schema/artifact alignment before marking itself ready.
- **Threshold vs. risk bands.** The `0.65` threshold drives `prediction`; the Low/Medium/High bands are application-level triage only.
- **Validation metrics are not test metrics.** They are reported as validation results from the original workflow.
- **No authentication or rate limiting.** Add these at the gateway for production use.
- **No model monitoring.** The service does not detect data drift, concept drift, or calibration changes.
- **No retraining pipeline.** Model refresh is an external workflow that produces a new artifact.
- **No database.** The service is stateless apart from the loaded model.

---

## Model artifact provenance

The existing artifact was verified during integration:

| Item | Value |
|---|---|
| Original location found | `.venv/customer_churn_model.joblib` |
| Deployment location | `models/customer_churn_model.joblib` |
| Size | 9,406 bytes |
| SHA256 | `C4D73339D823F3FBA77FECC7235DF7DAC31E9044D8B80B5332949C0D459D503E` |
| scikit-learn version | 1.6.1 |
| Classification threshold | 0.65 |
| Classes | `[0, 1]` |
| Input features | 19 |
| Transformed features | 45 |

The original file in `.venv` was left untouched; the deployment copy is byte-identical.

---

## API version

`1.0.0`
