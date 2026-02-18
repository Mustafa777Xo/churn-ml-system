from pathlib import Path
from datetime import datetime
import subprocess
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.load import load_clean_data
from src.features.pipeline import build_preprocess_pipeline


# model path and version
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def _git_sha_short() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None
    
def make_model_version() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sha = _git_sha_short()
    return f"{ts}_{sha}" if sha else ts


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
    }


def train_logistic() -> dict:
    df = load_clean_data(training=True)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocess = build_preprocess_pipeline()
    model =  LogisticRegression(max_iter=1000, solver="lbfgs")

    pipeline = Pipeline(
        steps=[
            ("preprocess",preprocess),
            ("model",model)
        ]
    )

    pipeline.fit(X_train,y_train)

    y_proba = pipeline.predict_proba(X_valid)[:,1]
    metrics = compute_metrics(y_valid, y_proba, threshold=0.5)

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "n_train": len(X_train),
        "n_valid": len(X_valid),
    }

def save_artifacts(result: dict, version: str) -> None:
    model_dir = MODELS_DIR / version
    report_dir = REPORTS_DIR / version
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["pipeline"], model_dir / "model.joblib")
    metrics_path = report_dir / "metrics.json"
    pd.Series(result["metrics"]).to_json(metrics_path, indent=2)





def main():
    version = make_model_version()
    result = train_logistic()
    save_artifacts(result, version)
if __name__ == "__main__":
    main()