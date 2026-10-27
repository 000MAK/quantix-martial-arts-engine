from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED = {"athlete_id", "session_id", "sensor_id", "timestamp_ms", "signal", "value", "unit", "source_id", "license_id", "consent_status"}
SIGNAL_UNITS = {
    "heart_rate": {"bpm"},
    "impact_peak": {"g"},
    "accel_x": {"m/s2", "m/s^2"},
    "accel_y": {"m/s2", "m/s^2"},
    "accel_z": {"m/s2", "m/s^2"},
    "gyro_x": {"deg/s", "rad/s"},
    "gyro_y": {"deg/s", "rad/s"},
    "gyro_z": {"deg/s", "rad/s"},
}


def summarize_telemetry(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Validate and summarize consented wearable observations descriptively.

    Sensor acceleration is not a clinical head-impact or concussion measure;
    no injury threshold, cumulative damage, lactate, or medical advice is inferred.
    """
    frame = pd.read_csv(input_csv)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("Telemetry file missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("Telemetry file has no observations")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Every telemetry row requires athlete/session/sensor, timestamp, signal, unit and provenance")
    for col in ("athlete_id", "session_id", "sensor_id", "signal", "unit", "source_id", "license_id", "consent_status"):
        if frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} cannot be blank")
    if not frame.consent_status.astype(str).str.lower().isin({"granted", "authorized"}).all():
        raise ValueError("Processing wearable/health-adjacent records requires documented consent_status=granted/authorized")
    frame["signal"] = frame.signal.astype(str).str.strip().str.lower()
    frame["unit"] = frame.unit.astype(str).str.strip().str.lower()
    unknown = set(frame.signal) - set(SIGNAL_UNITS)
    if unknown:
        raise ValueError("Unsupported signals: " + ", ".join(sorted(unknown)))
    invalid_unit = frame.apply(lambda r: r.unit not in SIGNAL_UNITS[r.signal], axis=1)
    if invalid_unit.any():
        raise ValueError("Signal units do not match the documented units in the telemetry schema")
    frame["timestamp_ms"] = pd.to_numeric(frame.timestamp_ms, errors="raise")
    frame["value"] = pd.to_numeric(frame.value, errors="raise")
    if not frame.value.map(lambda x: pd.notna(x) and abs(float(x)) < float("inf")).all():
        raise ValueError("Telemetry values must be finite numbers")
    if (frame.loc[frame.signal.eq("heart_rate"), "value"] <= 0).any():
        raise ValueError("heart_rate values must be positive")
    keys = ["athlete_id", "session_id", "sensor_id", "signal", "timestamp_ms"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate timestamp for the same athlete/session/sensor/signal")

    groups = ["athlete_id", "session_id", "sensor_id", "signal", "unit"]
    rows = []
    for key, group in frame.groupby(groups, dropna=False, sort=True):
        values = group.value.astype(float)
        rows.append({
            **dict(zip(groups, key)),
            "samples": int(len(group)),
            "duration_seconds": float((group.timestamp_ms.max() - group.timestamp_ms.min()) / 1000.0),
            "mean": float(values.mean()),
            "p95": float(values.quantile(.95)),
            "maximum": float(values.max()),
            "source_ids": "|".join(sorted(group.source_id.astype(str).unique())),
            "license_ids": "|".join(sorted(group.license_id.astype(str).unique())),
            "interpretation_limit": "Descriptive sensor trace only; placement/calibration and sensor validity are unverified. Not a concussion, injury, lactate, or career-risk assessment.",
        })
    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
