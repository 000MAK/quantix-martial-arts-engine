from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REQUIRED = {"event_id", "result_class", "verification_status", "source_id", "license_id"}
NON_FEATURES = REQUIRED | {"bout_id", "athlete_a_id", "athlete_b_id", "fighter_a_id", "fighter_b_id", "split_group", "competition_date", "discipline_id", "ruleset", "division", "sex_class", "competition_level"}


def _prepare(input_csv: Path):
    frame = pd.read_csv(input_csv)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("Match-results table missing columns: " + ", ".join(sorted(missing)))
    if frame.empty or frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Each training row needs event, label, verification status, source and license")
    if not frame.verification_status.astype(str).str.lower().eq("verified").all():
        raise ValueError("Match prediction requires verified outcome labels")
    for col in ("event_id", "source_id", "license_id"):
        if frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} cannot be blank")
    features = [c for c in frame.columns if c.startswith("feature_")]
    if not features:
        raise ValueError("Add numeric, pre-bout-only features with the feature_ prefix")
    for col in features:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        present = frame[col].dropna().astype(float)
        if not np.isfinite(present.to_numpy()).all():
            raise ValueError(f"Feature {col} contains an infinite value")
    if frame[features].isna().all().any():
        raise ValueError("A feature column is entirely missing")
    y = frame.result_class.astype(str).str.strip()
    if y.eq("").any() or y.nunique() < 2:
        raise ValueError("At least two non-empty result classes are required")
    return frame, features, y


def _model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)),
    ])


def train_match_model(input_csv: Path, output_dir: Path, group_col: str = "event_id") -> dict:
    """Train an exploratory multiclass baseline with an event-held-out check.

    Probabilities are uncalibrated model scores. This is not a betting or
    athlete-selection recommendation and cannot predict an exact round reliably.
    """
    frame, features, y = _prepare(input_csv)
    if group_col not in frame:
        raise ValueError(f"Group column {group_col!r} is missing")
    groups = frame[group_col].astype(str)
    if groups.nunique() < 5:
        raise ValueError("At least five independent events are required for an event-grouped holdout baseline")
    splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42)
    train_ix, test_ix = next(splitter.split(frame[features], y, groups))
    train_classes = set(y.iloc[train_ix])
    test_classes = set(y.iloc[test_ix])
    if len(train_classes) < 2:
        raise ValueError("The grouped training partition has fewer than two result classes; add more events/data")
    model = _model()
    model.fit(frame.iloc[train_ix][features], y.iloc[train_ix])
    probabilities = model.predict_proba(frame.iloc[test_ix][features])
    pred = model.predict(frame.iloc[test_ix][features])
    classes = list(model.named_steps["classifier"].classes_)
    metrics = {
        "held_out_events": int(groups.iloc[test_ix].nunique()),
        "train_events": int(groups.iloc[train_ix].nunique()),
        "test_bouts": int(len(test_ix)),
        "accuracy": float(accuracy_score(y.iloc[test_ix], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y.iloc[test_ix], pred)),
        "log_loss": float(log_loss(y.iloc[test_ix], probabilities, labels=classes)),
        "classes_in_holdout": sorted(test_classes),
        "classes_absent_from_holdout": sorted(set(classes) - test_classes),
    }
    final_model = _model()
    final_model.fit(frame[features], y)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output_dir / "model.joblib")
    metadata = {
        "model_type": "standardized logistic-regression multiclass baseline",
        "target": "result_class",
        "feature_columns": features,
        "classes": classes,
        "training_bouts": int(len(frame)),
        "training_events": int(groups.nunique()),
        "group_column": group_col,
        "validation": metrics,
        "probability_status": "uncalibrated; exploratory only",
        "limits": [
            "Only columns prefixed feature_ are used; they must be known before the bout and computed without outcome leakage.",
            "One random event-held-out split is a baseline, not proof of external validity; repeat across seasons/promotions and calibrate on separate data.",
            "Exact finish round, betting odds, and guaranteed winner are not inferred unless explicitly represented by adequate verified labels.",
            "Training data must be licensed and its athlete/event labels verified.",
        ],
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def predict_match(model_dir: Path, features: dict[str, float]) -> dict:
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    columns = meta["feature_columns"]
    missing = set(columns) - set(features)
    if missing:
        raise ValueError("Prediction missing model features: " + ", ".join(sorted(missing)))
    row = pd.DataFrame([{col: float(features[col]) for col in columns}])
    if not np.isfinite(row.to_numpy(dtype=float)).all():
        raise ValueError("Prediction features must be finite numbers")
    model = joblib.load(model_dir / "model.joblib")
    probs = model.predict_proba(row)[0]
    scores = {str(label): float(prob) for label, prob in zip(model.named_steps["classifier"].classes_, probs)}
    return {
        "top_class": max(scores, key=scores.get),
        "uncalibrated_model_scores": scores,
        "probability_status": "uncalibrated; exploratory only",
        "note": "Not betting advice, a guaranteed outcome, or a medical/safety decision.",
    }
