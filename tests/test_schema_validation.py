import pandas as pd
import pytest

from src.data.schema import validate_schema


def make_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["0001-A", "0002-B"],
            "gender": ["Male", "Female"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 12],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "No phone service"],
            "InternetService": ["DSL", "No"],
            "OnlineSecurity": ["Yes", "No internet service"],
            "OnlineBackup": ["No", "No internet service"],
            "DeviceProtection": ["No", "No internet service"],
            "TechSupport": ["No", "No internet service"],
            "StreamingTV": ["No", "No internet service"],
            "StreamingMovies": ["No", "No internet service"],
            "Contract": ["Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 20.15],
            "TotalCharges": ["29.85", "20.15"],
            "Churn": ["No", "Yes"],
        }
    )


def test_missing_column_raises():
    df = make_raw_df().drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df, training=True)


def test_invalid_churn_value_raises():
    df = make_raw_df()
    df.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Invalid values in 'Churn'"):
        validate_schema(df, training=True)
