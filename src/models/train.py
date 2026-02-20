from pathlib import Path
from datetime import datetime
import subprocess
import joblib
import pandas as pd
import numpy as np
import json

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support
)
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data.load import load_clean_data
from src.features.pipeline import build_preprocess_pipeline

import argparse


# model path and version
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")


def _git_sha_short() -> str | None:
    # git SHA for traceability
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def make_model_version() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sha = _git_sha_short()
    return f"{ts}_{sha}" if sha else ts


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    # Core metrics used for model comparsion
    y_pred = (y_proba >= threshold).astype(int)
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
    }


def build_xgb_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
    )


def crossval_metrics(preprocess, model, X, y, n_splits: int = 3) -> dict:
    # Stratified CV for PR-AUC and ROC_AUC stability
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pr_scores = []
    roc_scores = []
    for train_idx, valid_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]

        fold_pipeline = Pipeline(
            steps=[
                ("preprocess", clone(preprocess)),
                ("model", clone(model)),
            ]
        )
        fold_pipeline.fit(X_tr, y_tr)
        y_proba = fold_pipeline.predict_proba(X_va)[:, 1]

        pr_scores.append(average_precision_score(y_va, y_proba))
        roc_scores.append(roc_auc_score(y_va, y_proba))

    return {
        "pr_auc_mean": float(np.mean(pr_scores)),
        "pr_auc_std": float(np.std(pr_scores)),
        "roc_auc_mean": float(np.mean(roc_scores)),
        "roc_auc_std": float(np.std(roc_scores)),
        "n_splits": int(n_splits),
    }


def train_logistic() -> dict:
    df = load_clean_data(training=True)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocess = build_preprocess_pipeline()
    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_valid)[:, 1]
    metrics = compute_metrics(y_valid, y_proba, threshold=0.5)

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "n_train": len(X_train),
        "n_valid": len(X_valid),
    }


def train_xgboost(cv_folds: int = 3) -> dict:
    df = load_clean_data(training=True)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocess = build_preprocess_pipeline()
    model = build_xgb_model()

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model)
        ]
    )
    cv_metrics = crossval_metrics(
        preprocess, model, X_train, y_train, n_splits=cv_folds)
    # Fit on train split for validation metrics
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_valid)[:, 1]
    metrics = compute_metrics(y_valid, y_proba, threshold=0.5)
    # Final fit on full data for the saved artifact
    pipeline.fit(X, y)
    return {
        "model_type": "xgboost",
        "pipeline": pipeline,
        "metrics": metrics,
        "cv_metrics": cv_metrics,
        "n_train": len(X_train),
        "n_valid": len(X_valid),
    }


def save_feature_importance(pipeline: Pipeline, out_path: Path) -> None:
    # One-hot level importance (document this in README)
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()
    importances = model.feature_importances_
    fi = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    fi.to_csv(out_path, index=False)


def save_artifacts(result: dict, version: str) -> None:
    model_dir = MODELS_DIR / version
    report_dir = REPORTS_DIR / version
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["pipeline"], model_dir / "model.joblib")
    metrics_path = report_dir / "metrics.json"
    pd.Series(result["metrics"]).to_json(metrics_path, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        choices=["logistic", "xgboost"], default="logistic")
    parser.add_argument("--cv-folds", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    version = make_model_version()
    if args.model == "logistic":
        result = train_logistic()
    elif args.model == "xgboost":
        result = train_xgboost(cv_folds=args.cv_folds)
    else:
        raise ValueError("Supported models: logistic, xgboost")
    save_artifacts(result, version)
    if args.model == "xgboost":
        report_dir = REPORTS_DIR / version
        save_feature_importance(
            result["pipeline"], report_dir / "feature_importance.csv")


if __name__ == "__main__":
    main()
