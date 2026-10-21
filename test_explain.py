
import json
from pathlib import Path

import pandas as pd
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parent

config = json.loads(
    (ROOT / "models" / "model_config.json").read_text(encoding="utf-8")
)

df = pd.read_csv(ROOT / "reports" / "test_predictions.csv")

# Choose a known fraudulent transaction that the model detected.
samples = df[
    (df["actual_class"] == 1)
    & (df["predicted_class"] == 1)
]

if samples.empty:
    raise RuntimeError("No correctly detected fraudulent sample found.")

row = samples.iloc[0]
transaction = {feature: float(row[feature]) for feature in config["features"]}

payload = json.dumps({"transaction": transaction}).encode("utf-8")

request = urllib.request.Request(
    "http://127.0.0.1:8000/explain",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))

    print("Status: SUCCESS")
    print("Actual label:", int(row["actual_class"]))
    print("Saved probability:", float(row["fraud_probability"]))
    print("Live probability:", result["fraud_probability"])
    print("Prediction:", result["label"])
    print("Explanation consistent:", result["explanation_consistent"])
    print("\nTop contributing features:")

    for item in result["top_features"]:
        print(
            f'{item["feature"]:>5} | '
            f'SHAP: {item["shap_value"]:+.6f} | '
            f'{item["direction"]}'
        )

except urllib.error.HTTPError as exc:
    print("HTTP error:", exc.code)
    print(exc.read().decode("utf-8"))

except Exception as exc:
    print("Request failed:", repr(exc))
