from pathlib import Path
import re


MODELS_DIR = Path("models")


def _extract_timestamp(version: str) -> str | None:
    # version format: YYYMMDD_HHMMSS_sha
    match = re.match(r"^(\d{8}_\d{6})", version)
    return match.group(1) if match else None


def resolve_model_version(model_version: str | None) -> str:
    if model_version and model_version != "latest":
        return model_version

    candidates = []

    if not MODELS_DIR.exists():
        raise FileNotFoundError("models/ directory not found")
    for p in MODELS_DIR.iterdir():
        if p.is_dir():
            ts = _extract_timestamp(p.name)
            if ts:
                candidates.append((ts, p.name))

    if not candidates:
        raise FileNotFoundError("No model versions found in models/")
    candidates.sort()

    return candidates[-1][1]
