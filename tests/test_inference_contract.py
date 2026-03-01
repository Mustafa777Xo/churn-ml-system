import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.pipeline import build_preprocess_pipeline
from src.inference.predict import load_model_and_metadata, predict


def make_clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gender": ["Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "Yes", "No", "Yes"],
            "tenure": [1, 2, 3, 4],
            "PhoneService": ["Yes", "No", "Yes", "No"],
            "MultipleLines": ["No", "No phone service", "Yes", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["Yes", "No", "No internet service", "No"],
            "OnlineBackup": ["No", "Yes", "No internet service", "No"],
            "DeviceProtection": ["No", "Yes", "No internet service", "Yes"],
            "TechSupport": ["No", "Yes", "No internet service", "No"],
            "StreamingTV": ["No", "Yes", "No internet service", "No"],
            "StreamingMovies": ["No", "Yes", "No internet service", "No"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            "MonthlyCharges": [29.85, 56.95, 20.15, 42.3],
            "TotalCharges": [29.85, 1889.5, 20.15, 1840.75],
            "Churn": [0, 1, 0, 1],
        }
    )


def test_inference_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    version = "20260101_000000"
    model_dir = tmp_path / "models" / version
    model_dir.mkdir(parents=True)

    df = make_clean_df()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    preprocess = build_preprocess_pipeline()
    model = LogisticRegression(max_iter=1000)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )
    pipeline.fit(X, y)

    joblib.dump(pipeline, model_dir / "model.joblib")

    metadata = {
        "model_type": "logistic",
        "threshold": 0.5,
        "fn_cost": 5,
        "fp_cost": 1,
        "feature_list": list(X.columns),
        "trained_at": version,
    }
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    model_loaded, metadata_loaded, resolved = load_model_and_metadata(version)
    preds = predict(X.iloc[[0]], model_loaded, metadata_loaded, resolved)

    prob = float(preds.loc[0, "churn_probability"])
    pred = int(preds.loc[0, "churn_prediction"])

    assert 0.0 <= prob <= 1.0
    assert pred in (0, 1)
