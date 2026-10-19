"""Check the responsiveness record before model admission."""
import json
import math
from pathlib import Path


def calm_monitor(path: Path, now: float, max_age: float = 30.0) -> bool:
    if not math.isfinite(now) or not math.isfinite(max_age) or max_age <= 0:
        return False
    try:
        with path.open() as handle:
            raw = handle.read(65537)
        if len(raw) > 65536:
            return False
        record = json.loads(raw)
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict):
        return False
    timestamp = record.get('timestamp')
    if type(timestamp) not in (int, float):
        return False
    try:
        age = now - timestamp
        valid_age = math.isfinite(timestamp) and math.isfinite(age) and 0 <= age <= max_age
    except (OverflowError, ValueError):
        return False
    return valid_age and record.get('sustained_alert') is False and record.get('reasons') == []
