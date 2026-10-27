"""Image classifiers from frozen pretrained features (CPU-friendly) on Hugging Face datasets.

A timm backbone turns each image into a feature vector once (cached to .npz); a
logistic-regression head is then chosen by cross-validation on the training split
only and evaluated once on the official test split. Used for the Oxford-IIIT Pet
cat-breed and cat-vs-dog models.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"


def embed_split(dataset_id: str, split: str, cache: Path, backbone: str = BACKBONE, batch: int = 32) -> dict:
    """Features for every image in a split, cached; also returns all non-image columns."""
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return {k: data[k] for k in data.files}
    import timm
    import torch
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    model = timm.create_model(backbone, pretrained=True, num_classes=0).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    feats = []
    with torch.inference_mode():
        for start in range(0, len(ds), batch):
            imgs = ds[start:start + batch]["image"]
            x = torch.stack([transform(im.convert("RGB")) for im in imgs])
            feats.append(model(x).numpy().astype(np.float32))
            print(f"  {split}: {min(start + batch, len(ds))}/{len(ds)}", flush=True)
    out = {"features": np.concatenate(feats)}
    for col in ds.column_names:
        if col != "image":
            out[col] = np.asarray(ds[col])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def train_head(X_train, y_train, X_test, y_test, class_names: list[str], seed: int = 42) -> tuple[object, dict]:
    search = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000)),
        {"logisticregression__C": [0.01, 0.1, 1.0]},
        cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="f1_macro", n_jobs=1)
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    labels = list(range(len(class_names)))
    metrics = {
        "cv": {"best_C": search.best_params_["logisticregression__C"], "best_macro_f1": float(search.best_score_),
               "grid": {str(p["logisticregression__C"]): float(s) for p, s in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])}},
        "test": {"samples": int(len(y_test)), "accuracy": float(accuracy_score(y_test, pred)),
                 "macro_f1": float(f1_score(y_test, pred, average="macro")),
                 "per_class_f1": dict(zip(class_names, f1_score(y_test, pred, average=None, labels=labels).round(4).tolist())),
                 "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist()},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
    }
    return search.best_estimator_, metrics


def save_model(model, metrics: dict, card: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "head.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(card, encoding="utf-8")
