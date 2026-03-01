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

## Tests

```bash
PYTHONPATH=. pytest -q
```
