
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

for directory in (MODEL_DIR, REPORT_DIR, PROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TARGET = "Class"
RANDOM_STATE = 42


def select_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = (
        2 * precision * recall
        / np.maximum(precision + recall, 1e-12)
    )

    return float(thresholds[int(np.argmax(f1_values))])


def evaluate_model(name, model, X, y, threshold):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return metrics, probabilities, predictions


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    if df.isnull().values.any():
        raise ValueError("Dataset contains missing values.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target Class must contain only 0 and 1.")

    original_rows = len(df)
    original_frauds = int(df[TARGET].sum())

    # Exclude exact duplicate rows from modeling.
    # The raw CSV is never modified.
    df = df.drop_duplicates().reset_index(drop=True)

    df.to_csv(PROCESSED_DIR / "modeling_data.csv", index=False)

    print("\nDATA PREPARATION")
    print("-" * 50)
    print(f"Original rows: {original_rows:,}")
    print(f"Rows removed as exact duplicates: {original_rows - len(df):,}")
    print(f"Original fraud cases: {original_frauds:,}")
    print(f"Fraud cases after deduplication: {int(df[TARGET].sum()):,}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Reproducible stratified 70/15/15 split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    for name, labels in [
        ("Training", y_train),
        ("Validation", y_val),
        ("Testing", y_test),
    ]:
        print(
            f"{name}: {len(labels):,} rows | "
            f"Frauds: {int(labels.sum()):,}"
        )

    # Fit preprocessing only on the training partition.
    preprocessing = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), ["Time", "Amount"]),
        ],
        remainder="passthrough",
    )

    models = {
        "LogisticRegression": Pipeline([
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    print("\nTRAINING MODELS")
    print("=" * 60)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        val_probabilities = model.predict_proba(X_val)[:, 1]
        threshold = select_threshold(y_val, val_probabilities)

        val_metrics, _, _ = evaluate_model(
            name, model, X_val, y_val, threshold
        )

        print(f"Validation PR-AUC: {val_metrics['pr_auc']:.4f}")
        print(f"Validation ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"Validation F1: {val_metrics['f1_score']:.4f}")
        print(f"Selected threshold: {threshold:.6f}")

        results.append(val_metrics)
        trained_models[name] = {
            "model": model,
            "threshold": threshold,
        }

    comparison = pd.DataFrame(results).sort_values(
        "pr_auc", ascending=False
    )
    comparison.to_csv(
        REPORT_DIR / "validation_model_comparison.csv",
        index=False,
    )

    best_name = str(comparison.iloc[0]["model"])
    best_model = trained_models[best_name]["model"]
    best_threshold = trained_models[best_name]["threshold"]

    print("\nVALIDATION COMPARISON")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {best_name}")

    test_metrics, test_probabilities, test_predictions = evaluate_model(
        best_name, best_model, X_test, y_test, best_threshold
    )

    print("\nFINAL TEST RESULTS")
    print("=" * 60)
    print(json.dumps(test_metrics, indent=4))
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Legitimate", "Fraudulent"],
            zero_division=0,
        )
    )

    joblib.dump(
        best_model,
        MODEL_DIR / "fraud_detection_model.joblib",
    )

    config = {
        "model_name": best_name,
        "threshold": best_threshold,
        "features": X_train.columns.tolist(),
        "target": TARGET,
        "random_state": RANDOM_STATE,
    }

    with open(
        MODEL_DIR / "model_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=4)

    with open(
        REPORT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(test_metrics, file, indent=4)

    # Retain every feature so real test rows can be sent back
    # through the prediction API.
    predictions_df = X_test.reset_index(drop=True).copy()
    predictions_df["actual_class"] = y_test.to_numpy()
    predictions_df["fraud_probability"] = test_probabilities
    predictions_df["predicted_class"] = test_predictions
    predictions_df["decision_threshold"] = best_threshold

    predictions_df.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False,
    )

    print("\nTRAINING COMPLETED")
    print(f"Saved model: {MODEL_DIR / 'fraud_detection_model.joblib'}")
    print(f"Saved config: {MODEL_DIR / 'model_config.json'}")
    print(f"Saved metrics: {REPORT_DIR / 'test_metrics.json'}")
    print(f"Saved test rows: {REPORT_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
