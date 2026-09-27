from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


MOCK_PREDICTION = {
    "input_date": "2026-09-15",
    "cci": {
        "2026-09-16": 72.075,
        "2026-09-17": 72.238,
        "2026-09-18": 72.240,
    },
    "whc": {
        "prediction_date": "2026-09-22",
        "class_id": 0,
        "class_name": "Low",
    },
}


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()
    assert body["title"] == "Sydney Weather ML API"
    assert body["version"] == "0.1.0"
    assert "endpoints" in body
    assert "targets" in body


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_metadata():
    response = client.get("/model-metadata")

    assert response.status_code == 200

    body = response.json()
    assert isinstance(body, dict)
    assert body


def test_cci_prediction(monkeypatch):
    monkeypatch.setattr(
        "app.main.run_production_inference",
        lambda prediction_date, models_dir: MOCK_PREDICTION,
    )

    response = client.get(
        "/predict/index/comfort_climate",
        params={"date": "2026-09-15"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "input_date": "2026-09-15",
        "predictions": MOCK_PREDICTION["cci"],
    }


def test_whc_prediction(monkeypatch):
    monkeypatch.setattr(
        "app.main.run_production_inference",
        lambda prediction_date, models_dir: MOCK_PREDICTION,
    )

    response = client.get(
        "/predict/category/weather_hazard",
        params={"date": "2026-09-15"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "input_date": "2026-09-15",
        "prediction": MOCK_PREDICTION["whc"],
    }


def test_missing_date_returns_422():
    response = client.get("/predict/index/comfort_climate")

    assert response.status_code == 422


def test_invalid_date_returns_422():
    response = client.get(
        "/predict/index/comfort_climate",
        params={"date": "2026-99-99"},
    )

    assert response.status_code == 422


def test_expected_inference_error_returns_400(monkeypatch):
    def raise_error(prediction_date, models_dir):
        raise ValueError("Requested date is unavailable.")

    monkeypatch.setattr(
        "app.main.run_production_inference",
        raise_error,
    )

    response = client.get(
        "/predict/index/comfort_climate",
        params={"date": "2026-09-15"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Requested date is unavailable."


def test_unexpected_inference_error_returns_503(monkeypatch):
    def raise_error(prediction_date, models_dir):
        raise RuntimeError("Unexpected failure.")

    monkeypatch.setattr(
        "app.main.run_production_inference",
        raise_error,
    )

    response = client.get(
        "/predict/index/comfort_climate",
        params={"date": "2026-09-15"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Prediction service is temporarily unavailable."
    )


def test_health_returns_503_when_model_file_is_missing(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("app.main.MODELS_DIR", tmp_path)

    response = client.get("/health")

    assert response.status_code == 503

    body = response.json()
    assert body["detail"]["status"] == "unhealthy"
    assert "cci_t1.joblib" in body["detail"]["missing_model_files"]


def test_metadata_returns_503_when_metadata_is_missing(
    monkeypatch,
    tmp_path,
):
    missing_metadata = tmp_path / "metadata.json"

    monkeypatch.setattr(
        "app.main.METADATA_PATH",
        missing_metadata,
    )

    response = client.get("/model-metadata")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Model metadata could not be loaded."
    )