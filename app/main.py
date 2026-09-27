import json
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from weather_ml.modeling.inference import run_production_inference

APP_VERSION = "0.1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
METADATA_PATH = MODELS_DIR / "metadata.json"

app = FastAPI(
    title="Sydney Weather ML API",
    description=(
        "Machine learning API for forecasting the Climate Comfort Index (CCI) "
        "and Weather Hazard Category (WHC) for Sydney, Australia."
    ),
    version=APP_VERSION,
)


def load_metadata() -> dict:
    """Load production model metadata."""
    try:
        with METADATA_PATH.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Model metadata could not be loaded.") from exc


def get_prediction(prediction_date: date) -> dict:
    """Run production inference and convert expected failures to HTTP errors."""
    try:
        return run_production_inference(
            prediction_date.isoformat(),
            models_dir=MODELS_DIR,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is temporarily unavailable.",
        ) from exc


@app.get("/")
def root() -> dict:
    """Describe the API and its available endpoints."""
    return {
        "title": app.title,
        "description": app.description,
        "version": APP_VERSION,
        "github": (
            "https://github.com/TT-Taiki/"
            "36120-26SP-AT2-14521304-api"
        ),
        "endpoints": {
            "health": "/health",
            "comfort_climate": (
                "/predict/index/comfort_climate?date=YYYY-MM-DD"
            ),
            "weather_hazard": (
                "/predict/category/weather_hazard?date=YYYY-MM-DD"
            ),
            "model_metadata": "/model-metadata",
            "docs": "/docs",
        },
        "targets": {
            "comfort_climate": (
                "CCI forecasts for t+1, t+2, and t+3 days"
            ),
            "weather_hazard": "WHC forecast exactly t+7 days",
        },
    }


@app.get("/health")
def health() -> dict:
    """Report API and model-artifact availability."""
    required_files = [
        "cci_t1.joblib",
        "cci_t2.joblib",
        "cci_t3.joblib",
        "whc_t7.joblib",
        "metadata.json",
    ]

    missing = [
        filename
        for filename in required_files
        if not (MODELS_DIR / filename).is_file()
    ]

    if missing:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "missing_model_files": missing,
            },
        )

    return {"status": "healthy"}


@app.get("/model-metadata")
def model_metadata() -> dict:
    """Return metadata for the deployed production models."""
    try:
        return load_metadata()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/predict/index/comfort_climate")
def predict_comfort_climate(
    prediction_date: Annotated[date, Query(alias="date")],
) -> dict:
    """Predict CCI for t+1, t+2, and t+3."""
    result = get_prediction(prediction_date)

    return {
        "input_date": result["input_date"],
        "predictions": result["cci"],
    }


@app.get("/predict/category/weather_hazard")
def predict_weather_hazard(
    prediction_date: Annotated[date, Query(alias="date")],
) -> dict:
    """Predict WHC exactly seven days after the input date."""
    result = get_prediction(prediction_date)

    return {
        "input_date": result["input_date"],
        "prediction": result["whc"],
    }