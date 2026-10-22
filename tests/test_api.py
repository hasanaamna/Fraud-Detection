
import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

ROOT = Path(__file__).resolve().parents[1]


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["feature_count"] == 30


def test_dashboard_stats():
    response = client.get("/dashboard/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] > 0
    assert 0 <= data["precision"] <= 1
    assert 0 <= data["recall"] <= 1


def test_fraud_trend():
    response = client.get("/dashboard/fraud-trend")

    assert response.status_code == 200
    data = response.json()
    assert data["points"]
    assert data["source"] == "held-out test dataset"


def test_prediction_with_real_sample():
    config = json.loads(
        (ROOT / "models" / "model_config.json").read_text(encoding="utf-8")
    )
    predictions = pd.read_csv(ROOT / "reports" / "test_predictions.csv")
    row = predictions.iloc[0]

    transaction = {
        feature: float(row[feature])
        for feature in config["features"]
    }

    response = client.post(
        "/predict",
        json={"transaction": transaction},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["label"] in ["Fraudulent", "Legitimate"]
    assert 0 <= result["fraud_probability"] <= 1


def test_prediction_rejects_missing_features():
    response = client.post(
        "/predict",
        json={"transaction": {"Time": 10, "Amount": 25}},
    )

    assert response.status_code == 422


def test_explanation_with_real_sample():
    config = json.loads(
        (ROOT / "models" / "model_config.json").read_text(encoding="utf-8")
    )
    predictions = pd.read_csv(ROOT / "reports" / "test_predictions.csv")
    row = predictions.iloc[0]

    transaction = {
        feature: float(row[feature])
        for feature in config["features"]
    }

    response = client.post(
        "/explain",
        json={"transaction": transaction},
    )

    assert response.status_code == 200
    result = response.json()
    assert len(result["top_features"]) > 0
    assert len(result["top_features"]) <= 10
    assert result["explanation_consistent"] is True
