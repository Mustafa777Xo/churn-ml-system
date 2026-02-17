from pathlib import Path
import pandas as pd

from src.data.schema import DROP_COLUMNS, validate_schema


DEFAULT_RAW_PATH = Path("data/raw/Telco-Customer-Churn.csv")


def load_raw_data(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    validate_schema(df, training=training)

    df = df.copy()

    # TotalCharges comes as string with blanks -> coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Map target only for training
    if training and "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop identifiers
    df = df.drop(columns=list(DROP_COLUMNS), errors="ignore")

    return df


def load_clean_data(path: Path = DEFAULT_RAW_PATH, training:bool = True, save_path: Path | None = None) -> pd.DataFrame: 
    df = load_raw_data(path)
    df = clean_data(df, training=training)

    if save_path is not None:
        df.to_parquet(save_path, index=False)

    return df

