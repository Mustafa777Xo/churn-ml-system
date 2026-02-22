from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def find_optimal_threshold(y_true, y_proba, fn_cost: float = 5.0, fp_cost: float = 1.0, thresholds: Iterable[float] | None = None) -> Tuple[float, pd.DataFrame]:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]).ravel()
        expected_cost = (fn_cost * fn) + (fp_cost * fp)

        rows.append({
            "threshold": float(t),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "expected_cost": float(expected_cost),
        })

    report = pd.DataFrame(rows)
    best_row = report.sort_values(["expected_cost", "threshold"]).iloc[0]
    best_threshold = float(best_row["threshold"])

    return best_threshold, report
