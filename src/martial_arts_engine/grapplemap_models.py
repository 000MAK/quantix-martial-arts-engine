"""Classifiers on GrappleMap keyframes: position family and bottom-player posture.

Features are invariant to where the athletes are and which way they face: all
pairwise joint distances within and between the two players, sorted per pair of
players so that swapping the players does not change them. Labels come from
GrappleMap's own tags (Eelis van der Weegen, public domain). Positions carrying more
than one family tag are left out. Evaluation holds out positions grouped by their
first name word, so close variants of one position are not split across train and
test; GrappleMap is a hand-authored map, so scores describe that map, not footage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FAMILIES = ["standing", "full_guard", "half_guard", "side_control", "butterfly", "turtle", "back", "mount"]
POSTURES = ["bottom_supine", "bottom_turned_in", "bottom_seated"]


def features(keyframe) -> np.ndarray:
    p = np.asarray(keyframe, dtype=float)                       # (2, 23, 3)
    iu = np.triu_indices(23, 1)
    within = [np.linalg.norm(p[k][:, None] - p[k][None], axis=-1)[iu] for k in (0, 1)]
    within = np.sort(np.stack(within), axis=0).ravel()          # order-free across players
    between = np.linalg.norm(p[0][:, None] - p[1][None], axis=-1)
    between = np.minimum(between, between.T).ravel()            # symmetric under player swap
    heights = np.sort(p[:, :, 1], axis=0).ravel()               # joint heights (y up)
    return np.concatenate([within, between, heights])


def label_rows(nodes: list[dict], classes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X, y, groups, names = [], [], [], []
    for n in nodes:
        if not n["name"]:
            continue
        hit = [c for c in classes if c in n["tags"]]
        if len(hit) != 1:
            continue
        X.append(features(n["keyframe"]))
        y.append(classes.index(hit[0]))
        groups.append(n["name"].split()[0].lower())
        names.append(n["name"])
    return np.asarray(X), np.asarray(y), np.asarray(groups), names


def evaluate(nodes: list[dict], classes: list[str], seed: int = 42) -> tuple[object, dict]:
    X, y, groups, _ = label_rows(nodes, classes)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, groups))
    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5)),
        "random_forest": RandomForestClassifier(n_estimators=400, random_state=seed, class_weight="balanced", n_jobs=-1),
    }
    folds = {name: cross_val_score(m, X[tr], y[tr], groups=groups[tr], cv=GroupKFold(5), scoring="f1_macro")
             for name, m in candidates.items()}
    cv = {name: float(s.mean()) for name, s in folds.items()}
    cv_std = {name: float(s.std()) for name, s in folds.items()}
    best = max(cv, key=cv.get)
    model = candidates[best].fit(X[tr], y[tr])
    pred = model.predict(X[te])
    majority = np.bincount(y[tr]).argmax()
    present = sorted(set(y[te]))
    metrics = {
        "classes": classes, "samples": int(len(y)), "train_size": int(len(tr)), "test_size": int(len(te)),
        "class_counts": {classes[k]: int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "split": "GroupShuffleSplit (25% test) on the first word of the position name; selection by GroupKFold(5) on train",
        "cv_macro_f1": cv, "cv_macro_f1_std": cv_std, "selected": best,
        "test": {"accuracy": float(accuracy_score(y[te], pred)), "macro_f1": float(f1_score(y[te], pred, average="macro", labels=present)),
                 "majority_baseline_accuracy": float((y[te] == majority).mean()),
                 "per_class_f1": {classes[k]: float(v) for k, v in zip(present, f1_score(y[te], pred, average=None, labels=present))}},
    }
    return model, metrics


TITLES = {
    "bjj_position_family": ("BJJ position family from a two-person 3D skeleton",
                            "Predicts which of 8 grappling position families a two-person pose belongs to."),
    "bjj_bottom_posture": ("BJJ bottom-player posture from a two-person 3D skeleton",
                           "Predicts the bottom player's posture (supine, turned in, seated)."),
}


def model_card(name: str, m: dict) -> str:
    title, what = TITLES[name]
    b = m["selected"]
    t = m["test"]
    counts = ", ".join(f"{k.replace('bottom_', '')} {v}" for k, v in m["class_counts"].items())
    rows = "\n".join(f"| {k.replace('bottom_', '')} | {v:.2f} |" for k, v in t["per_class_f1"].items())
    return f"""---
license: other
license_name: public-domain-dedication
license_link: https://github.com/Eelis/GrappleMap/blob/master/LICENSE
library_name: sklearn
tags:
- brazilian-jiu-jitsu
- grappling
- pose
- tabular-classification
datasets:
- makactuary/grapplemap-bjj-graph
---

# {title}

{what} Input: one GrappleMap keyframe (2 players x 23 joints x 3D). Features are all within-player and between-player joint distances plus joint heights, so they do not depend on where the athletes are, which way they face, or which player is listed first.

Trained on positions from [GrappleMap](https://github.com/Eelis/GrappleMap) by **Eelis van der Weegen** (public domain), via the [grapplemap-bjj-graph](https://huggingface.co/datasets/makactuary/grapplemap-bjj-graph) dataset. Labels are GrappleMap's own tags; positions with more than one candidate tag are left out ({m['samples']} positions: {counts}).

## Evaluation

GrappleMap contains many variants of the same position, so positions were split by the first word of their name: 25% held out, model selected by GroupKFold(5) on the rest.

| | Macro-F1 |
|---|---|
| Grouped 5-fold CV ({b}), **main estimate** | **{m['cv_macro_f1'][b]:.2f} ± {m['cv_macro_f1_std'][b]:.2f}** |
| Single held-out split ({m['test_size']} positions) | {t['macro_f1']:.2f} (accuracy {t['accuracy']:.2f}; majority-class baseline {t['majority_baseline_accuracy']:.2f}) |

The held-out split happened to be easier than average; read the cross-validated figure first.

| Class | F1 on the held-out split |
|---|---|
{rows}

## Limitations

- GrappleMap is a hand-authored technique map, not recorded matches; the model has not been tested on real footage or on 2-D video poses.
- Small classes and a few hundred positions: estimates vary by split (see the ± above).
- Name-based grouping reduces but does not remove leakage between closely related positions.

## Usage

```python
import joblib
from huggingface_hub import hf_hub_download
model = joblib.load(hf_hub_download("makactuary/{name.replace('_', '-')}", "model.joblib"))
# features(keyframe) from grapplemap_models.py in https://github.com/000MAK/quantix-martial-arts-engine
```

Trained with scikit-learn 1.9.1. Code: `grapplemap_models.py` (MIT).
"""


def run(positions_jsonl: Path, out_root: Path) -> dict:
    import joblib
    nodes = [json.loads(line) for line in positions_jsonl.read_text(encoding="utf-8").splitlines()]
    results = {}
    for name, classes in (("bjj_position_family", FAMILIES), ("bjj_bottom_posture", POSTURES)):
        model, metrics = evaluate(nodes, classes)
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / "model.joblib")
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (out / "README.md").write_text(model_card(name, metrics), encoding="utf-8")
        results[name] = {"selected": metrics["selected"], "n": metrics["samples"], **{k: round(v, 3) for k, v in metrics["test"].items() if isinstance(v, float)}}
    return results
