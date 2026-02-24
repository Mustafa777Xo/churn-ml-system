import argparse
import pandas as pd

from src.data.schema import validate_schema
from src.inference.predict import load_model_and_metadata, predict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-version", default="latest")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    validate_schema(df, training=False)

    customer_id = df["customerID"] if "customerID" in df.columns else None

    feature_df = df.drop(columns=["customerID", "Churn"], errors="ignore")

    model, metadata, version = load_model_and_metadata(args.model_version)
    preds = predict(feature_df, model, metadata, version)

    if customer_id is not None:
        preds.insert(0, "customerID", customer_id.values)
    preds.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
