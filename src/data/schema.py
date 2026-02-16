from typing import Dict, Iterable, Set
import pandas as pd

# We keep customerID here to detect schema drift, even though it is later dropped
REQUIRED_COLUMNS: Set[str] = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}

ALLOWED_VALUES: Dict[str, Set[str]] = {
    "Churn": {"Yes", "No"},
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"No phone service", "No", "Yes"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"No", "Yes", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"No", "Yes", "No internet service"},
    "TechSupport": {"No", "Yes", "No internet service"},
    "StreamingTV": {"No", "Yes", "No internet service"},
    "StreamingMovies": {"No", "Yes", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}

DROP_COLUMNS: Set[str] = {"customerID"}


# Missing column helper
def _missing_columns(df: pd.DataFrame, required: Iterable[str]) -> Set[str]:
    return set(required) - set(df.columns)


def validate_schema(df: pd.DataFrame, training: bool = True) -> None:
    required = set(REQUIRED_COLUMNS)

    if not training:
        required = required - {"Churn"}

    missing = _missing_columns(df, required)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col, allowed in ALLOWED_VALUES.items():
        if col not in df.columns:
            continue
        observed = set(df[col].dropna().unique())
        invalid = observed - allowed

        if invalid:
            raise ValueError(
                f"Invalid values in '{col}': {sorted(invalid)}. "
                f"Allowed: {sorted(allowed)}"
            )
