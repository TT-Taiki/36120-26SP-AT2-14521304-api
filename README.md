# Sydney Weather ML API

FastAPI service for forecasting two weather-derived indices for Sydney, Australia:

- **Climate Comfort Index (CCI)** — regression forecasts for **t+1, t+2, and t+3 days**
- **Weather Hazard Category (WHC)** — multiclass classification forecast for **exactly t+7 days**

This repository contains the deployment-ready API for UTS 36120 Advanced Machine Learning Applications, Assessment Task 2.

## Live API

The API is deployed on Render:

https://three6120-26sp-at2-14521304-api.onrender.com

Interactive Swagger documentation:

https://three6120-26sp-at2-14521304-api.onrender.com/docs

> The deployed service may take additional time to respond after a period of inactivity depending on the Render service configuration.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information and available endpoints |
| GET | `/health` | Service and model-artifact health check |
| GET | `/model-metadata` | Production model metadata |
| GET | `/predict/index/comfort_climate` | Predict CCI for t+1, t+2, and t+3 |
| GET | `/predict/category/weather_hazard` | Predict WHC exactly t+7 |

Both prediction endpoints require a query parameter:

```text
date=YYYY-MM-DD
```

## Example Requests

### Climate Comfort Index

```bash
curl "https://three6120-26sp-at2-14521304-api.onrender.com/predict/index/comfort_climate?date=2026-09-15"
```

Example response:

```json
{
  "input_date": "2026-09-15",
  "predictions": {
    "2026-09-16": 72.07518601587175,
    "2026-09-17": 72.23845812562087,
    "2026-09-18": 72.24003410558134
  }
}
```

### Weather Hazard Category

```bash
curl "https://three6120-26sp-at2-14521304-api.onrender.com/predict/category/weather_hazard?date=2026-09-15"
```

Example response:

```json
{
  "input_date": "2026-09-15",
  "prediction": {
    "prediction_date": "2026-09-22",
    "class_id": 0,
    "class_name": "Low"
  }
}
```

WHC classes are:

| Class ID | Category |
|---:|---|
| 0 | Low |
| 1 | Moderate |
| 2 | High |
| 3 | Extreme |

## Architecture

The production inference workflow is:

```text
Open-Meteo Historical Weather API
            |
            v
weather-ml Python package
            |
            v
Feature engineering
            |
            v
Trained production models
            |
            v
FastAPI
            |
            v
Docker
            |
            v
Render
```

Historical weather observations are retrieved from Open-Meteo using the same ERA5-Seamless source used during model development. The custom `weather-ml` package performs feature engineering and production inference.

## Production Models

Four production models are included in `models/`:

```text
models/
├── cci_t1.joblib
├── cci_t2.joblib
├── cci_t3.joblib
├── whc_t7.joblib
└── metadata.json
```

The fixed model configurations selected during experimentation were refitted using all available pre-2026 labelled development data before deployment.

The locked 2024–2025 test results were used for final generalisation evaluation and are not recalculated from the production-refitted artefacts.

## Custom Python Package

Shared ML functionality is provided by the custom `weather-ml` package.

TestPyPI:

https://test.pypi.org/project/weather-ml/0.0.1/

The package contains reusable functionality for:

- target construction
- dataset processing
- feature engineering
- model training and prediction
- production inference

The API repository installs the package from TestPyPI through the dependency configuration in `pyproject.toml` and `uv.lock`.

## Local Development

### Requirements

- Python 3.12
- `uv`

Clone the repository:

```bash
git clone git@github.com:TT-Taiki/36120-26SP-AT2-14521304-api.git
cd 36120-26SP-AT2-14521304-api
```

Install the locked dependencies:

```bash
uv sync
```

Start the development server:

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Testing

Run the automated API tests:

```bash
uv run pytest -q
```

The test suite covers:

- root endpoint
- health endpoint
- model metadata
- CCI prediction response
- WHC prediction response
- missing query parameters
- invalid dates
- expected inference failures
- unexpected inference failures
- missing model artefacts
- missing metadata

Run static code checks:

```bash
uv run ruff check app tests
```

At the final API development checkpoint:

```text
11 tests passed
Ruff: All checks passed
```

Prediction functions are mocked in unit tests where appropriate so that API behaviour can be tested independently of external Open-Meteo availability.

## Docker

Build the production image:

```bash
docker build -t sydney-weather-ml-api .
```

Run the container locally:

```bash
docker run --rm -p 8000:8000 sydney-weather-ml-api
```

Then verify:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy"
}
```

The Docker image uses Python 3.12 and installs dependencies from the locked `uv` environment.

## Deployment

The production API is deployed as a Docker-based Render Web Service.

Deployment workflow:

```text
GitHub main branch
        |
        v
Render Docker build
        |
        v
Production container
        |
        v
Health check (/health)
        |
        v
Public HTTPS API
```

Render supplies the runtime `PORT` environment variable. The Docker container starts Uvicorn using that port and binds to `0.0.0.0`.

## Error Handling

The API provides explicit HTTP responses for common failure modes:

- `422 Unprocessable Entity` — missing or invalid date format
- `400 Bad Request` — expected inference/data errors
- `503 Service Unavailable` — unexpected inference failures, unavailable metadata, or missing model artefacts

## Important Limitations

The service should be interpreted as an assessment ML prototype rather than a safety-critical weather warning system.

In particular:

- Predictions are specific to **Sydney, Australia**.
- Production inference depends on historical weather observations being available from Open-Meteo.
- The CCI models predict a derived comfort index rather than official weather forecasts.
- WHC classes are substantially imbalanced in the historical dataset.
- High and Extreme hazard events are rare.
- The final WHC model improved aggregate evaluation metrics over the tested baselines, but rare High and Extreme events were not successfully detected in the locked 2024–2025 test period.
- The WHC endpoint should therefore **not** be used as a substitute for official severe-weather warnings or other safety-critical decision systems.

For operational weather and hazard decisions, users should consult authoritative meteorological and emergency-warning services.

## Repository Structure

```text
.
├── app/
│   ├── __init__.py
│   └── main.py
├── models/
│   ├── cci_t1.joblib
│   ├── cci_t2.joblib
│   ├── cci_t3.joblib
│   ├── whc_t7.joblib
│   └── metadata.json
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Related Repository

The experiments repository contains the data acquisition, target creation, exploratory data analysis, feature engineering, baseline evaluation, model experiments, production refitting, and custom package development.

https://github.com/TT-Taiki/36120-26SP-AT2-14521304-experiments

## Author

Taiki Komori  
UTS — 36120 Advanced Machine Learning Applications  
Spring 2026