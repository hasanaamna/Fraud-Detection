# SentinelAI — AI-Powered Credit Card Fraud Detection

**An end-to-end machine learning application for detecting potentially fraudulent credit card transactions, analyzing transaction risk, and explaining model predictions.**

SentinelAI combines a trained Random Forest classifier, a FastAPI backend, and a React + TypeScript dashboard to demonstrate how machine learning can support fraud detection workflows.

<p align="center">

  **Machine Learning · FastAPI · React · TypeScript · SHAP**

</p>

---

## Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions represent only a small fraction of all transactions. A useful model must identify suspicious activity while limiting false alarms.

SentinelAI addresses this problem through a complete ML workflow, from data preparation and model evaluation to API-based predictions and interactive visualization.

### Key Features

- **Fraud Prediction:** Classifies transactions as legitimate or potentially fraudulent.
- **Risk Analysis:** Displays fraud probability and a corresponding risk classification.
- **Interactive Dashboard:** Presents model evaluation metrics and fraud-related visualizations.
- **Transaction Explorer:** Browse sample transactions and filter records by relevant criteria.
- **Explainable AI:** Uses SHAP to identify features contributing to individual predictions.
- **REST API:** Exposes prediction, dashboard, and explanation endpoints through FastAPI.
- **Input Validation:** Checks incoming transaction data and handles invalid inputs.
- **Automated Testing:** Includes API tests to verify core backend functionality.

---

## Model Performance

The final Random Forest model was selected using validation-set performance and evaluated on a held-out test set.

| Metric | Test Result |
|---|---:|
| Accuracy | 99.94% |
| Precision | 84.62% |
| Recall | 77.46% |
| F1 Score | 80.88% |
| ROC-AUC | 93.32% |
| PR-AUC | 78.28% |

### Confusion Matrix

| Actual / Predicted | Legitimate | Fraud |
|---|---:|---:|
| Legitimate | 42,478 | 10 |
| Fraud | 16 | 55 |

**What these results mean**

- **Precision:** Approximately 84.62% of transactions flagged by the model were fraudulent in the test set.
- **Recall:** The model detected approximately 77.46% of fraudulent transactions in the test set.
- **F1 Score:** Balances precision and recall.
- **PR-AUC:** Measures precision-recall performance and is particularly informative for imbalanced classification problems.

The reported results are specific to the benchmark dataset and evaluation procedure. They do not establish performance on live banking transactions.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| UI and Charts | CSS, Recharts, Lucide React |
| Backend | Python, FastAPI, Uvicorn |
| Machine Learning | scikit-learn, NumPy, pandas |
| Model Explainability | SHAP |
| Testing | pytest, FastAPI TestClient |
| Model Persistence | joblib |
| Version Control | Git, GitHub |

---

## System Architecture

```text
Credit Card Transaction Data
             |
             v
    Data Preparation
             |
             v
  Model Training & Evaluation
             |
             v
      Random Forest Model
             |
             v
       FastAPI Backend
        /    |     \
       /     |      \
 Predict  Dashboard  SHAP
       \     |      /
             |
             v
 React + TypeScript Frontend
             |
             v
 Interactive Fraud Analysis
```

The frontend communicates with the backend through HTTP requests. The backend loads the trained model and configuration, validates transaction inputs, generates predictions, and provides explanations.

---

## Project Structure

```text
Fraud-Detection/
├── backend/
│   └── app/
│       └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── models/
│   ├── fraud_detection_model.joblib
│   └── model_config.json
├── notebooks/
├── reports/
│   ├── test_metrics.json
│   ├── validation_model_comparison.csv
│   └── figures/
├── tests/
│   └── test_api.py
├── .gitignore
├── requirements.txt
├── eda.py
├── inspect_dataset.py
├── train_model.py
├── test_explain.py
└── check_shap.py
```

*Generated datasets, virtual environments, and other local artifacts may be excluded from the published repository.*

---

## Getting Started

### Prerequisites

Install the following before running the project:

- Python 3.11 or another Python version compatible with the pinned dependencies
- Node.js and npm
- Git

### 1. Clone the repository

```bash
git clone https://github.com/hasanaamna/Fraud-Detection.git
cd Fraud-Detection
```

If your GitHub repository uses a different name or URL, replace the URL above with the actual repository address.

### 2. Set up the Python environment

On Windows CMD:

```cmd
python -m venv venv
venv\Scripts\activate
```

Install the backend and machine learning dependencies:

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Prepare the model files

The backend expects the following files:

```text
models/fraud_detection_model.joblib
models/model_config.json
```

Make sure these files are available locally before starting the API.

If the model artifact is not included in your GitHub release, generate it by running the training pipeline after preparing the required dataset.

### 4. Prepare the dataset

Place the original dataset at:

```text
data/raw/creditcard.csv
```

The dataset used during development contains transaction features, including `Time`, `Amount`, anonymized components `V1` through `V28`, and the target column `Class`.

The original dataset is not required to be committed to GitHub. Obtain it from its authorized source and follow its applicable terms.

### 5. Start the backend

From the project root, with the virtual environment activated:

```cmd
python -m uvicorn backend.app.main:app --reload
```

The API should be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

### 6. Start the frontend

Open a second terminal:

```cmd
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually:

```text
http://localhost:5173
```

Keep both the frontend and backend running while using the application.

---

## API Endpoints

The application exposes the following endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | API root information |
| `GET /health` | Check API health |
| `POST /predict` | Predict fraud probability and classification |
| `GET /dashboard/stats` | Retrieve dashboard summary statistics |
| `GET /dashboard/fraud-trend` | Retrieve fraud-trend data |
| `GET /dashboard/sample-transactions` | Retrieve sample transaction records |
| `POST /explain` | Generate a SHAP-based prediction explanation |

Use the interactive Swagger documentation at `/docs` to inspect request schemas and test the endpoints.

---

## Explainable AI with SHAP

SentinelAI includes an explanation endpoint to help users understand which input features contributed to a prediction.

The explanation workflow uses SHAP with the trained tree-based model and provides information such as:

- Baseline model output
- Feature contributions
- Most influential features for an individual prediction
- Reconstructed probability and consistency information

The features `V1` through `V28` are anonymized components in the dataset. Their original real-world meanings are not provided, so their contributions should not be interpreted as specific financial behaviors.

SHAP explanations describe model behavior; they do not establish the actual cause of a transaction or prove fraud.

---

## Testing

Run the backend API tests from the project root:

```cmd
python -m pytest tests/test_api.py -v
```

The development test run completed successfully with **6 passing tests**.

The frontend was also manually tested for:

- Dashboard metrics and charts
- Transaction explorer
- Transaction filtering
- Risk analysis and predictions
- Fraud probability and classification
- SHAP explanations
- Invalid-input handling

These are the reported development results; run the tests again after making changes.

---

## Data and Model Reproducibility

The repository includes scripts used during development:

- `inspect_dataset.py` — inspect the input dataset.
- `eda.py` — perform exploratory data analysis.
- `train_model.py` — train and evaluate the fraud detection model.
- `test_explain.py` — check the explanation workflow.
- `check_shap.py` — inspect SHAP-related behavior.

The exact training workflow depends on the dataset and configuration expected by the scripts. Use the same preprocessing and feature ordering as the trained model when reproducing results.

---

## Future Improvements

Potential extensions include:

- Live transaction ingestion and streaming inference
- Additional model comparisons and threshold analysis
- Monitoring for data drift and model performance degradation
- Improved audit logs and prediction history
- Role-based access control and secure authentication
- Deployment with environment-specific configuration
- Performance testing and monitoring under concurrent requests
- Evaluation on representative, appropriately authorized real-world data

These are possible future enhancements, not currently claimed features.

---

## Responsible Use

SentinelAI is an educational and portfolio project demonstrating machine learning for fraud detection.

- Predictions are probabilistic and can be incorrect.
- False positives may affect legitimate transactions.
- The benchmark dataset may not represent current fraud patterns.
- The application has not been established as a production-ready banking or payment security system.
- Do not submit real cardholder data or sensitive financial information to an unapproved deployment.

Production use would require additional security, privacy, monitoring, validation, and operational controls.

---

## Author

**Aamna Hasan**

GitHub: [@hasanaamna](https://github.com/hasanaamna)

---

## License

Choose and add a license before distributing the project. If you intend to publish the code as an open-source project, a `LICENSE` file should state the permissions and conditions clearly.
