import pandas as pd

from src.models import train as train_module


def make_clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
            "Dependents": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
            "tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "PhoneService": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
            "MultipleLines": [
                "No",
                "No phone service",
                "Yes",
                "No phone service",
                "No",
                "No phone service",
                "Yes",
                "No phone service",
                "No",
                "No phone service",
            ],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic", "No", "DSL", "Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["Yes", "No", "No internet service", "No", "Yes", "No internet service", "Yes", "No", "No internet service", "No"],
            "OnlineBackup": ["No", "Yes", "No internet service", "No", "Yes", "No internet service", "No", "Yes", "No internet service", "No"],
            "DeviceProtection": ["No", "Yes", "No internet service", "Yes", "No", "No internet service", "Yes", "No", "No internet service", "Yes"],
            "TechSupport": ["No", "Yes", "No internet service", "No", "Yes", "No internet service", "No", "Yes", "No internet service", "No"],
            "StreamingTV": ["No", "Yes", "No internet service", "No", "Yes", "No internet service", "No", "Yes", "No internet service", "No"],
            "StreamingMovies": ["No", "Yes", "No internet service", "No", "Yes", "No internet service", "No", "Yes", "No internet service", "No"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
            "MonthlyCharges": [29.85, 56.95, 20.15, 42.3, 70.7, 19.95, 99.65, 84.8, 18.95, 55.3],
            "TotalCharges": [29.85, 1889.5, 20.15, 1840.75, 151.65, 19.95, 820.5, 1990.5, 326.8, 1530.6],
            "Churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_train_logistic_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "load_clean_data", lambda training=True: make_clean_df())
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_module, "REPORTS_DIR", tmp_path / "reports")

    result = train_module.train_logistic()
    train_module.save_artifacts(result, "test_version")

    model_path = tmp_path / "models" / "test_version" / "model.joblib"
    metadata_path = tmp_path / "models" / "test_version" / "metadata.json"
    metrics_path = tmp_path / "reports" / "test_version" / "metrics.json"
    threshold_path = tmp_path / "reports" / "test_version" / "threshold_report.csv"

    assert model_path.exists()
    assert metadata_path.exists()
    assert metrics_path.exists()
    assert threshold_path.exists()
