# Customer Churn Prediction System

Production-oriented churn prediction system with reproducible training, versioned artifacts, batch scoring, API inference, drift detection, and retraining triggers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Data

Place the raw dataset at:

```
data/raw/Telco-Customer-Churn.csv
```

## Train models

Baseline (logistic regression):

```bash
python -m src.models.train --model logistic
```

Improved (XGBoost):

```bash
python -m src.models.train --model xgboost
```

Training also writes the drift reference file to `data/processed/reference.csv`.

## Start API

```bash
uvicorn src.api.main:app --port 8000
```

POST `/predict` accepts either a single `record` or a list of `records`.

## Batch scoring

```bash
python -m src.batch.score --input data/raw/Telco-Customer-Churn.csv --output reports/preds.csv --model-version latest
```

## Drift report

```bash
python -m src.monitoring.drift --reference data/processed/reference.csv --current data/raw/Telco-Customer-Churn.csv --out reports/drift_report.html --out-json reports/drift_report.json
```

## Retraining triggers

```bash
python -m src.monitoring.triggers --drift reports/drift_report.json --metadata models/<version>/metadata.json
```

## Risk analysis

- Class imbalance: Churn is a minority class, so PR-AUC is the primary metric to reflect minority performance.
- Data drift: Behavior and product mix can shift; a drift report is generated and used as a retraining trigger.
- Training-serving skew: The same clean_data() and preprocessing pipeline are used in training and inference.
- Feature availability: Inference requires the same raw features; schema validation rejects missing or invalid fields.
- Overfitting: Baseline vs XGBoost comparison, CV metrics, and mild tuning reduce overfit risk.

## Tests

```bash
PYTHONPATH=. pytest -q
```
