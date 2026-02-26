import argparse
import json

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.data.load import clean_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # load reference (cleaned, pre-one-hot)
    ref = pd.read_parquet(args.reference)

    # load current batch and clean it
    curr_raw = pd.read_csv(args.current)
    curr = clean_data(curr_raw, training=False)

    # Aling columns (defensive)
    if "Churn" in ref.columns:
        ref = ref.drop(columns=["Churn"])
    if "Churn" in curr.columns:
        curr = curr.drop(columns=["Churn"])

    # Create report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr)

    # Save HTML
    report.save_html(args.out)

    # Extract JSON summary
    report_dict = report.as_dict()
    drifted = report_dict["metrics"][0]["result"]["drift_by_columns"]
    drifted_features = [
        k for k, v in drifted.items() if v.get("drift_detected")
    ]
    summary = {
        "drift_detected": len(drifted_features) > 0,
        "n_features": len(drifted),
        "n_drifted": len(drifted_features),
        "share_drifted": len(drifted_features) / max(1, len(drifted)),
        "drifted_features": drifted_features,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
