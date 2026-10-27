"""Human-activity baseline on UCI HAR (smartphone inertial features), subject-aware.

UCI HAR's official split puts different volunteers in train and test; this module
checks that, selects a model by GroupKFold over training subjects only, evaluates
once on the held-out subjects, and reports per-subject spread. It writes a model,
metrics and a Hugging Face model card. Daily activities only: this is not a
combat-technique, fatigue or injury model.
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TASK = "general_wearable_activity_classification"
SOURCE = "https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones"
CITATION = ("Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L. & Parra, X. (2013). Human Activity Recognition "
            "Using Smartphones [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K")


def _inner(path: Path) -> zipfile.ZipFile:
    outer = zipfile.ZipFile(path)
    name = next(n for n in outer.namelist() if n.endswith(".zip"))
    return zipfile.ZipFile(io.BytesIO(outer.read(name)))


def load_har(path: Path) -> dict:
    z = _inner(path)
    read = lambda n: z.read(f"UCI HAR Dataset/{n}").decode()
    arrays = {}
    for split in ("train", "test"):
        arrays[f"X_{split}"] = np.loadtxt(io.StringIO(read(f"{split}/X_{split}.txt")))
        arrays[f"y_{split}"] = np.loadtxt(io.StringIO(read(f"{split}/y_{split}.txt")), dtype=int)
        arrays[f"s_{split}"] = np.loadtxt(io.StringIO(read(f"{split}/subject_{split}.txt")), dtype=int)
    labels = dict(line.split() for line in read("activity_labels.txt").strip().splitlines())
    arrays["labels"] = {int(k): v for k, v in labels.items()}
    return arrays


def check_subject_disjoint(s_train: np.ndarray, s_test: np.ndarray) -> None:
    overlap = set(s_train) & set(s_test)
    if overlap:
        raise ValueError(f"Subjects appear in both train and test: {sorted(overlap)}")


def _registry_allows(registry: Path) -> None:
    data = json.loads(registry.read_text(encoding="utf-8"))
    rec = next((d for d in data["datasets"] if d["dataset_id"] == "uci_har_smartphones"), None)
    if not rec or TASK not in rec.get("training_approval", {}).get("approved_tasks", []):
        raise PermissionError(f"uci_har_smartphones is not approved for {TASK}")


def candidates(seed: int) -> dict:
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed),
    }


def train_and_evaluate(d: dict, seed: int = 42, folds: int = 5) -> tuple[object, dict]:
    check_subject_disjoint(d["s_train"], d["s_test"])
    cv = GroupKFold(n_splits=folds)
    cv_scores = {}
    for name, model in candidates(seed).items():
        scores = cross_val_score(model, d["X_train"], d["y_train"], groups=d["s_train"], cv=cv, scoring="f1_macro", n_jobs=1)
        cv_scores[name] = {"mean_macro_f1": float(scores.mean()), "std": float(scores.std()), "folds": scores.round(4).tolist()}
    best = max(cv_scores, key=lambda k: cv_scores[k]["mean_macro_f1"])
    model = candidates(seed)[best].fit(d["X_train"], d["y_train"])
    pred = model.predict(d["X_test"])
    names = [d["labels"][k] for k in sorted(d["labels"])]
    per_subject = {int(s): float(accuracy_score(d["y_test"][d["s_test"] == s], pred[d["s_test"] == s])) for s in np.unique(d["s_test"])}
    acc = np.array(list(per_subject.values()))
    metrics = {
        "dataset": "UCI HAR (Human Activity Recognition Using Smartphones)", "license": "CC-BY-4.0", "task": TASK, "seed": seed,
        "train_subjects": int(len(np.unique(d["s_train"]))), "test_subjects": int(len(np.unique(d["s_test"]))),
        "train_windows": int(len(d["y_train"])), "test_windows": int(len(d["y_test"])), "features": int(d["X_train"].shape[1]),
        "model_selection": {"method": f"GroupKFold({folds}) over training subjects, macro-F1", "candidates": cv_scores, "selected": best},
        "test": {"accuracy": float(accuracy_score(d["y_test"], pred)), "macro_f1": float(f1_score(d["y_test"], pred, average="macro")),
                 "per_class_f1": dict(zip(names, f1_score(d["y_test"], pred, average=None).round(4).tolist())),
                 "confusion_matrix": {"labels": names, "matrix": confusion_matrix(d["y_test"], pred, labels=sorted(d["labels"])).tolist()},
                 "per_subject_accuracy": {"min": float(acc.min()), "median": float(np.median(acc)), "max": float(acc.max()), "by_subject": per_subject}},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return model, metrics


def model_card(m: dict) -> str:
    t = m["test"]
    rows = "\n".join(f"| {k} | {v:.3f} |" for k, v in t["per_class_f1"].items())
    cands = "\n".join(f"| {k} | {v['mean_macro_f1']:.3f} ± {v['std']:.3f} |" for k, v in m["model_selection"]["candidates"].items())
    return f"""---
license: cc-by-4.0
library_name: sklearn
tags:
- tabular-classification
- human-activity-recognition
- sklearn
datasets:
- uci-har
metrics:
- accuracy
- f1
---

# UCI HAR activity classifier (subject-held-out baseline)

Classifies six daily activities (walking, walking upstairs, walking downstairs, sitting, standing, lying) from the 561 precomputed smartphone inertial features of the UCI HAR dataset. Part of the [QMASE research engine](https://github.com/000MAK/quantix-martial-arts-engine).

## Evaluation

The official UCI HAR split keeps volunteers separate: {m['train_subjects']} subjects train ({m['train_windows']} windows), {m['test_subjects']} different subjects test ({m['test_windows']} windows). The model was chosen by {m['model_selection']['method']} without touching the test subjects, then evaluated once.

| Candidate (CV over training subjects) | Macro-F1 |
|---|---|
{cands}

Selected: **{m['model_selection']['selected']}**. Held-out subjects: accuracy **{t['accuracy']:.3f}**, macro-F1 **{t['macro_f1']:.3f}**. Per-subject accuracy ranges from **{t['per_subject_accuracy']['min']:.3f}** to **{t['per_subject_accuracy']['max']:.3f}** (median {t['per_subject_accuracy']['median']:.3f}).

| Activity | F1 |
|---|---|
{rows}

## Limitations

- Daily activities recorded with a waist-mounted smartphone in a lab protocol; not martial-arts techniques, effort, fatigue or injury risk.
- 30 volunteers aged 19–48; performance on other devices, placements or populations is unknown.
- Uses the dataset's precomputed features; raw-signal deep models are a different task.

## Data and licence

UCI HAR is licensed CC BY 4.0. {CITATION}
"""


def train_uci_har(zip_path: Path, registry: Path, out_dir: Path, seed: int = 42) -> dict:
    _registry_allows(registry)
    model, metrics = train_and_evaluate(load_har(zip_path), seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(model_card(metrics), encoding="utf-8")
    return {"out_dir": str(out_dir), "selected": metrics["model_selection"]["selected"],
            "test_accuracy": round(metrics["test"]["accuracy"], 4), "test_macro_f1": round(metrics["test"]["macro_f1"], 4),
            "per_subject_accuracy_range": [round(metrics["test"]["per_subject_accuracy"]["min"], 3), round(metrics["test"]["per_subject_accuracy"]["max"], 3)]}
