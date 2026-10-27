"""Align time-stamped sensor streams (pose, audio, heart rate, events) on one time axis.

Each stream is a CSV with a `timestamp_s` column. Streams are joined onto a base
stream by nearest timestamp within a tolerance; values outside the tolerance stay
empty instead of being interpolated, and each stream keeps a `<name>_lag_s` column
so alignment quality is visible. Devices are rarely clock-synchronised, so each
stream can carry an `offset_s` correction; `estimate_offset` recovers one from
events visible in two streams (e.g. a clap seen on video and heard on audio).

Alignment says nothing about causation, and no stream is used unless its manifest
entry records a source, a licence and granted consent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_STREAM_FIELDS = {"name", "path", "source_id", "license_id", "consent_status"}


def _prepare(frame: pd.DataFrame, name: str, offset_s: float = 0.0) -> pd.DataFrame:
    if "timestamp_s" not in frame.columns:
        raise ValueError(f"Stream '{name}' needs a timestamp_s column")
    data = frame.copy()
    data["timestamp_s"] = pd.to_numeric(data["timestamp_s"], errors="raise") + float(offset_s)
    if data["timestamp_s"].duplicated().any():
        raise ValueError(f"Stream '{name}' has duplicate timestamps")
    return data.sort_values("timestamp_s").reset_index(drop=True)


def align_streams(base: pd.DataFrame, streams: dict[str, pd.DataFrame], tolerance_s: float,
                  offsets_s: dict[str, float] | None = None, base_name: str = "base") -> pd.DataFrame:
    if tolerance_s <= 0:
        raise ValueError("tolerance_s must be positive")
    offsets_s = offsets_s or {}
    fused = _prepare(base, base_name, offsets_s.get(base_name, 0.0))
    for name, frame in streams.items():
        data = _prepare(frame, name, offsets_s.get(name, 0.0))
        value_cols = [c for c in data.columns if c != "timestamp_s"]
        renamed = data.rename(columns={c: f"{name}_{c}" for c in value_cols})
        renamed[f"{name}_timestamp_s"] = renamed["timestamp_s"]
        fused = pd.merge_asof(fused, renamed, on="timestamp_s", direction="nearest", tolerance=tolerance_s)
        fused[f"{name}_lag_s"] = fused[f"{name}_timestamp_s"] - fused["timestamp_s"]
        fused = fused.drop(columns=[f"{name}_timestamp_s"])
    return fused


def estimate_offset(reference_times: np.ndarray, other_times: np.ndarray, max_lag_s: float,
                    step_s: float = 0.01, tolerance_s: float = 0.05) -> dict:
    """Offset to ADD to other_times so its events line up with reference_times.

    Grid search maximising the number of events matched within tolerance; ties are
    broken by the smallest mean absolute residual.
    """
    ref = np.sort(np.asarray(reference_times, dtype=float))
    oth = np.sort(np.asarray(other_times, dtype=float))
    if len(ref) == 0 or len(oth) == 0:
        raise ValueError("Both event lists must be non-empty")
    best = None
    for lag in np.arange(-max_lag_s, max_lag_s + step_s / 2, step_s):
        shifted = oth + lag
        idx = np.clip(np.searchsorted(ref, shifted), 1, len(ref) - 1) if len(ref) > 1 else np.zeros(len(shifted), dtype=int)
        nearest = ref[idx] if len(ref) == 1 else np.where(np.abs(ref[idx] - shifted) < np.abs(ref[idx - 1] - shifted), ref[idx], ref[idx - 1])
        resid = np.abs(nearest - shifted)
        matched = int((resid <= tolerance_s).sum())
        score = (matched, -float(resid[resid <= tolerance_s].mean()) if matched else -np.inf)
        if best is None or score > best[0]:
            best = (score, float(lag))
    (matched, neg_resid), lag = best
    return {"offset_s": round(lag, 6), "matched_events": matched, "events_in_other": int(len(oth)),
            "mean_abs_residual_s": -neg_resid if matched else None}


def fuse_from_manifest(manifest_path: Path, out_csv: Path) -> dict:
    """Manifest: {"tolerance_s": 0.1, "base": {...}, "streams": [{...}, ...]}; each entry needs
    name, path, source_id, license_id, consent_status ('granted') and optional offset_s."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [manifest["base"]] + list(manifest.get("streams", []))
    for entry in entries:
        missing = REQUIRED_STREAM_FIELDS - set(entry)
        if missing:
            raise ValueError(f"Manifest entry {entry.get('name', '?')} is missing: {', '.join(sorted(missing))}")
        if str(entry["consent_status"]).lower() != "granted":
            raise PermissionError(f"Stream '{entry['name']}' does not have granted consent")
    root = manifest_path.parent
    load = lambda e: pd.read_csv((root / e["path"]) if not Path(e["path"]).is_absolute() else Path(e["path"]))
    base = manifest["base"]
    fused = align_streams(load(base), {e["name"]: load(e) for e in manifest.get("streams", [])},
                          float(manifest["tolerance_s"]), {e["name"]: float(e.get("offset_s", 0.0)) for e in entries},
                          base_name=base["name"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(out_csv, index=False)
    coverage = {e["name"]: float(fused[f"{e['name']}_lag_s"].notna().mean()) for e in manifest.get("streams", [])}
    provenance = {e["name"]: {k: e[k] for k in ("source_id", "license_id", "consent_status")} for e in entries}
    return {"rows": int(len(fused)), "tolerance_s": float(manifest["tolerance_s"]), "match_coverage": coverage,
            "provenance": provenance, "output": str(out_csv),
            "note": "Aligned samples only; no interpolation, causal claims or performance scores."}
