from pathlib import Path
import json
import joblib
import pandas as pd


from src.inference.model_registry import resolve_model_version


REQUIRED_METADATA_KEYS = {"threshold",
                          "model_type", "feature_list", "trained_at"}


def load_model_and_metadata(model_version: str | None):
    version = resolve_model_version(model_version)
    model_dir = Path("models") / version

    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    model = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    missing_keys = REQUIRED_METADATA_KEYS - set(metadata.keys())

    if missing_keys:
        raise ValueError(f"Metadata missing keys: {sorted(missing_keys)}")
    return model, metadata, version


def predict(df: pd.DataFrame, model, metadata, version: str):
    proba = model.predict_proba(df)[:, 1]
    threshold = float(metadata["threshold"])
    pred = (proba >= threshold).astype(int)

    return pd.DataFrame(
        {
            "churn_probability": proba,
            "churn_prediction": pred,
            "threshold": threshold,
            "model_version": version,
        }
    )
