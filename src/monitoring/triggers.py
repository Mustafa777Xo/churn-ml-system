import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_trained_at(version: str) -> datetime:
    ts = "_".join(version.split("_")[:2])
    return datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)


def should_retrain(
    drift_report_path: Path,
    metadata_path: Path,
    max_age_days: int = 30,
) -> dict:
    reasons = []
    drift = load_json(drift_report_path)
    if drift.get("drift_detected"):
        reasons.append("drift_detected")
    metadata = load_json(metadata_path)
    trained_at = metadata.get("trained_at")
    if trained_at:
        trained_dt = parse_trained_at(trained_at)
        if datetime.now(timezone.utc) - trained_dt > timedelta(days=max_age_days):
            reasons.append("model_age_exceeded")
    return {"should_retrain": len(reasons) > 0, "reasons": reasons}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift", required=True, help="Path to drift_report.json")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--max-age-days", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    decision = should_retrain(
        drift_report_path=Path(args.drift),
        metadata_path=Path(args.metadata),
        max_age_days=args.max_age_days,
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
