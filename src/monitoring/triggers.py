import json
from datetime import datetime, timedelta,  timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_trained_at(version: str) -> datetime:
    # trained_at format: YYYYMMDD_HHMMSS or YYYYMMDD_HHMMSS_sha
    ts = "_".join(version.split("_")[:2])
    return datetime.strptime(ts, "%Y%m%d_%H%M%S")


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
