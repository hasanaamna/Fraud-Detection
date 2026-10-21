
from pathlib import Path
import json

import joblib
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "fraud_detection_model.joblib"
CONFIG_PATH = ROOT / "models" / "model_config.json"
PREDICTIONS_PATH = ROOT / "reports" / "test_predictions.csv"

print("Loading SentinelAI model...")

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

features = config["features"]
df = pd.read_csv(PREDICTIONS_PATH)

# Prefer a correctly detected fraudulent transaction.
candidates = df[
    (df["actual_class"] == 1)
    & (df["predicted_class"] == 1)
]

if candidates.empty:
    candidates = df

sample = candidates.iloc[[0]]
X = sample[features].astype(float)

print("Model:", config["model_name"])
print("Feature count:", len(features))
print("Sample actual class:", int(sample["actual_class"].iloc[0]))
print("Saved fraud probability:",
      float(sample["fraud_probability"].iloc[0]))

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)
explanation = explainer(X)

print("\nSHAP test successful.")
print("Explanation type:", type(explanation).__name__)
print("SHAP values shape:", explanation.values.shape)
print("Base value shape:", explanation.base_values.shape)
print("Feature count explained:", len(explanation.feature_names))
print("\nTop feature contributions:")

values = explanation.values

# Binary classifiers may return one contribution per class.
if values.ndim == 3:
    values = values[0, :, 1]
else:
    values = values[0]

ranking = pd.DataFrame({
    "feature": features,
    "shap_value": values,
})

ranking["absolute_impact"] = ranking["shap_value"].abs()
ranking = ranking.sort_values("absolute_impact", ascending=False)

print(ranking.head(10).to_string(index=False))
