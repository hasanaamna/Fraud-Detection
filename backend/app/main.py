
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PATHS AND MODEL LOADING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "fraud_detection_model.joblib"
CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "test_metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "test_predictions.csv"

if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
    raise RuntimeError(
        "Model files not found. Run python train_model.py first."
    )

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

FEATURES = config["features"]
THRESHOLD = float(config["threshold"])
MODEL_NAME = config["model_name"]


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="SentinelAI Fraud Detection API",
    description="Fraud prediction and analytics for a research dataset.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class TransactionRequest(BaseModel):
    transaction: dict[str, float] = Field(
        ...,
        description="Transaction features matching the training dataset.",
    )


# ============================================================
# SHARED DASHBOARD DATA LOADER
# ============================================================

def load_dashboard_data():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Metrics not found. Run python train_model.py.",
        )

    if not PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Test predictions not found. Rerun python train_model.py.",
        )

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        predictions = pd.read_csv(PREDICTIONS_PATH)

    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load dashboard data: {exc}",
        ) from exc

    required = {
        *FEATURES,
        "actual_class",
        "fraud_probability",
        "predicted_class",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                "Test predictions are missing columns: "
                + ", ".join(sorted(missing))
                + ". Rerun python train_model.py."
            ),
        )

    if predictions.empty:
        raise HTTPException(
            status_code=503,
            detail="The test prediction file is empty.",
        )

    return metrics, predictions


# ============================================================
# BASIC ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "message": "SentinelAI Fraud Detection API is running",
        "model": MODEL_NAME,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "threshold": THRESHOLD,
        "feature_count": len(FEATURES),
    }


# ============================================================
# SINGLE TRANSACTION PREDICTION
# ============================================================

@app.post("/predict")
def predict(request: TransactionRequest):
    transaction = request.transaction

    missing = sorted(set(FEATURES) - set(transaction))
    extra = sorted(set(transaction) - set(FEATURES))

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Transaction features do not match the model.",
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    try:
        values = [float(transaction[name]) for name in FEATURES]
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(
            status_code=422,
            detail="All transaction features must be numeric.",
        ) from exc

    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422,
            detail="All features must contain finite numbers.",
        )

    input_df = pd.DataFrame([values], columns=FEATURES)

    probability = float(model.predict_proba(input_df)[0, 1])
    predicted_class = int(probability >= THRESHOLD)

    if predicted_class == 1:
        risk_level = "HIGH"
        message = "Potential fraud detected. Further review recommended."
    elif probability >= THRESHOLD * 0.5:
        risk_level = "MEDIUM"
        message = "Elevated risk. Additional verification may be appropriate."
    else:
        risk_level = "LOW"
        message = "Low estimated fraud risk."

    return {
        "prediction": predicted_class,
        "label": "Fraudulent" if predicted_class else "Legitimate",
        "fraud_probability": probability,
        "fraud_probability_percent": round(probability * 100, 2),
        "risk_level": risk_level,
        "threshold": THRESHOLD,
        "model": MODEL_NAME,
        "message": message,
    }


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

@app.get("/dashboard/stats")
def dashboard_stats():
    metrics, predictions = load_dashboard_data()

    total = len(predictions)
    actual_frauds = int(predictions["actual_class"].sum())
    predicted_frauds = int(predictions["predicted_class"].sum())

    return {
        "model": metrics["model"],
        "total_transactions": total,
        "actual_frauds": actual_frauds,
        "predicted_frauds": predicted_frauds,
        "fraud_rate": actual_frauds / total if total else 0,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "threshold": metrics["threshold"],
        "true_negatives": metrics["true_negatives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "true_positives": metrics["true_positives"],
    }


# ============================================================
# ACTUAL FRAUD RATE BY ELAPSED-TIME WINDOW
# ============================================================

@app.get("/dashboard/fraud-trend")
def dashboard_fraud_trend():
    _, predictions = load_dashboard_data()

    data = predictions[["Time", "actual_class"]].copy()

    # The Time column represents elapsed seconds, not clock time.
    data["hour"] = (data["Time"].clip(lower=0) // 3600).astype(int)

    grouped = data.groupby("hour").agg(
        transactions=("actual_class", "size"),
        frauds=("actual_class", "sum"),
    )

    grouped["fraud_rate"] = (
        grouped["frauds"] / grouped["transactions"]
    )

    points = [
        {
            "hour": int(hour),
            "transactions": int(row["transactions"]),
            "frauds": int(row["frauds"]),
            "fraud_rate": float(row["fraud_rate"]),
        }
        for hour, row in grouped.iterrows()
    ]

    return {
        "source": "held-out test dataset",
        "time_unit": "elapsed hours",
        "points": points,
    }


# ============================================================
# REAL HELD-OUT TEST TRANSACTIONS
# ============================================================

@app.get("/dashboard/sample-transactions")
def dashboard_sample_transactions(
    limit: int = Query(default=10, ge=1, le=100),
    actual_class: int | None = Query(default=None, ge=0, le=1),
):
    _, predictions = load_dashboard_data()

    data = predictions.copy()

    if actual_class is not None:
        data = data[data["actual_class"] == actual_class]

    data = data.sort_values(
        ["Time", "Amount"],
        kind="stable",
    ).head(limit)

    items = []

    for row in data.to_dict(orient="records"):
        items.append({
            "transaction": {
                feature: float(row[feature])
                for feature in FEATURES
            },
            "time": float(row["Time"]),
            "amount": float(row["Amount"]),
            "actual_class": int(row["actual_class"]),
            "predicted_class": int(row["predicted_class"]),
            "fraud_probability": float(row["fraud_probability"]),
            "decision_threshold": float(
                row.get("decision_threshold", THRESHOLD)
            ),
        })

    return {
        "count": len(items),
        "items": items,
    }
