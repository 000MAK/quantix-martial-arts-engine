#!/usr/bin/env python3
"""Train a sequence-grouped classifier from ViCoS' annotated pose coordinates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "vicos_bjj_positions"
TASK_ID = "bjj_pose_position_classification"
LICENSE_ID = "CC-BY-NC-SA-4.0"
SOURCE_PAGE = "https://www2.vicos.si/resources/jiujitsu/"
DEFAULT_ANNOTATIONS = ROOT / "data" / "raw" / DATASET_ID / "annotations.json"
DEFAULT_REGISTRY = ROOT / "data" / "dataset_registry.json"
DEFAULT_OUTPUT = ROOT / "models" / "vicos_bjj_pose_classifier"
sys.path.insert(0, str(ROOT / "01_Data_Pipelines"))
from import_vicos_positions import _iter_json_array, _sha256


def _torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required. Install it with: "
            "python -m pip install -r requirements-deep-learning.txt"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _verify_registry(path: Path) -> dict[str, object]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("policy") != "deny_by_default":
        raise PermissionError("Dataset registry must retain its deny-by-default policy")
    entries = registry.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("Registry must contain a 'datasets' list")
    matches = [entry for entry in entries if entry.get("dataset_id") == DATASET_ID]
    if len(matches) != 1:
        raise PermissionError(f"Registry must contain exactly one {DATASET_ID!r} entry")
    entry = matches[0]
    approval = entry.get("training_approval", {})
    if TASK_ID not in approval.get("approved_tasks", []):
        raise PermissionError(f"Registry does not approve the {TASK_ID!r} task")
    if approval.get("noncommercial_use_confirmed") is not True:
        raise PermissionError("Non-commercial use must be confirmed before training")
    if entry.get("license_id") != LICENSE_ID:
        raise PermissionError("Registry license does not match the dataset license")
    for field in ("reviewer", "reviewed_at", "evidence_ref"):
        if not str(approval.get(field, "")).strip():
            raise PermissionError(f"Training approval is missing {field}")
    return entry


def _pose_features(pose: object) -> tuple[np.ndarray, bool, np.ndarray, float]:
    flat = np.zeros(51, dtype=np.float32)
    if not isinstance(pose, list) or len(pose) != 17:
        return flat, False, np.zeros(2, dtype=np.float32), 0.0
    try:
        points = np.asarray(pose, dtype=np.float32)
    except (TypeError, ValueError):
        return flat, False, np.zeros(2, dtype=np.float32), 0.0
    if points.shape != (17, 3) or not np.isfinite(points).all():
        return flat, False, np.zeros(2, dtype=np.float32), 0.0

    confidence = np.clip(points[:, 2], 0.0, 1.0)
    visible = confidence > 0
    if not visible.any():
        return flat, False, np.zeros(2, dtype=np.float32), 0.0

    xy = points[:, :2].copy()
    low = xy[visible].min(axis=0)
    high = xy[visible].max(axis=0)
    center = (low + high) / 2.0
    scale = float(max((high - low).max(), 1.0))
    normalized = (xy - center) / scale
    normalized[~visible] = 0.0
    flat.reshape(17, 3)[:, :2] = normalized
    flat.reshape(17, 3)[:, 2] = confidence
    return flat, True, center.astype(np.float32), scale


def _features_for_annotation(row: dict[str, object]) -> tuple[np.ndarray, str, str]:
    pose_features: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    scales: list[float] = []
    present: list[float] = []
    for name in ("pose1", "pose2"):
        features, exists, center, scale = _pose_features(row.get(name))
        pose_features.append(features)
        centers.append(center)
        scales.append(scale)
        present.append(float(exists))

    combined_centers = [centers[i] for i in range(2) if present[i]]
    combined_scales = [scales[i] for i in range(2) if present[i]]
    if combined_centers:
        all_centers = np.stack(combined_centers)
        low, high = all_centers.min(axis=0), all_centers.max(axis=0)
        global_scale = float(max((high - low).max(), max(combined_scales), 1.0))
        global_center = (low + high) / 2.0
    else:
        global_scale = 1.0
        global_center = np.zeros(2, dtype=np.float32)
    relative_centers = [
        (centers[i] - global_center) / global_scale if present[i] else np.zeros(2, dtype=np.float32)
        for i in range(2)
    ]
    relative_scales = [
        min(scales[i] / global_scale, 1.0) if present[i] else 0.0
        for i in range(2)
    ]
    features = np.concatenate([
        pose_features[0],
        pose_features[1],
        np.asarray(present + relative_centers[0].tolist() + relative_centers[1].tolist() + relative_scales, dtype=np.float32),
    ])
    label = str(row.get("position", "")).strip().lower()
    image_id = str(row.get("image", "")).strip().zfill(7)
    if not label or len(image_id) != 7 or not image_id.isdigit():
        raise ValueError(f"Invalid position or image identifier: {label!r}, {image_id!r}")
    return features, label, image_id[:2]


def _split_by_group(labels: np.ndarray, groups: np.ndarray, seed: int):
    indices = np.arange(len(labels))
    best = None
    best_score = -1
    for split_seed in range(seed, seed + 512):
        first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=split_seed)
        train_val, test = next(first.split(indices, labels, groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=split_seed + 10_000)
        train_rel, val_rel = next(second.split(train_val, labels[train_val], groups[train_val]))
        train, val = train_val[train_rel], train_val[val_rel]
        if (
            set(groups[train]) & set(groups[val])
            or set(groups[train]) & set(groups[test])
            or set(groups[val]) & set(groups[test])
        ):
            raise RuntimeError("Sequence leakage detected across pose-model splits")
        classes = set(np.unique(labels))
        covered = [set(labels[part]) for part in (train, val, test)]
        score = 10 * len(covered[0] & classes) + 4 * len(covered[1] & classes)
        score += 4 * len(covered[2] & classes) + 8 * len(set.intersection(*covered))
        score -= abs(len(set(groups[test])) - 3) + abs(len(set(groups[val])) - 3)
        if score > best_score:
            best_score, best = score, (train, val, test)
        if len(set.intersection(*covered)) == len(classes):
            break
    if best is None:
        raise RuntimeError("Could not create sequence-grouped train/validation/test splits")
    return best


def _make_model(nn, feature_count: int, class_count: int):
    class PosePositionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(feature_count, 192),
                nn.LayerNorm(192),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(192, 96),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(96, class_count),
            )

        def forward(self, features):
            return self.network(features)

    return PosePositionModel()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an experimental 18-class BJJ position model from licensed pose annotations."
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--confirm-noncommercial-use", action="store_true")
    args = parser.parse_args()
    if not args.confirm_noncommercial_use:
        parser.error("ViCoS CC BY-NC-SA 4.0 requires --confirm-noncommercial-use")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    registry_entry = _verify_registry(args.registry)
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Pose annotation file not found: {args.annotations}")

    feature_rows: list[np.ndarray] = []
    label_rows: list[str] = []
    group_rows: list[str] = []
    pose_counts = Counter()
    for annotation in _iter_json_array(args.annotations):
        features, label, group = _features_for_annotation(annotation)
        feature_rows.append(features)
        label_rows.append(label)
        group_rows.append(group)
        for pose_id in ("pose1", "pose2"):
            pose = annotation.get(pose_id)
            pose_counts[pose_id] += int(isinstance(pose, list) and len(pose) == 17)

    x = np.stack(feature_rows).astype(np.float32)
    class_names = sorted(set(label_rows))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    y = np.asarray([class_to_index[label] for label in label_rows], dtype=np.int64)
    groups = np.asarray(group_rows, dtype=str)
    if x.shape[1] != 110 or not np.isfinite(x).all():
        raise ValueError("Pose feature matrix must have 110 finite features per sample")

    torch, nn, DataLoader, TensorDataset = _torch()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    train_ix, val_ix, test_ix = _split_by_group(y, groups, args.seed)
    mean = x[train_ix].mean(axis=0, keepdims=True)
    scale = x[train_ix].std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    x = np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def dataset(indices: np.ndarray):
        features = torch.from_numpy(np.ascontiguousarray(x[indices]))
        targets = torch.from_numpy(np.ascontiguousarray(y[indices]))
        return TensorDataset(features, targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(nn, x.shape[1], len(class_names)).to(device)
    train_counts = np.bincount(y[train_ix], minlength=len(class_names))
    represented = train_counts > 0
    weights = np.zeros(len(class_names), dtype=np.float32)
    weights[represented] = len(train_ix) / (int(represented.sum()) * train_counts[represented])
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device)
    )
    loaders = {}
    for offset, (split, indices) in enumerate((
        ("train", train_ix), ("validation", val_ix), ("test", test_ix),
    )):
        generator = torch.Generator()
        generator.manual_seed(args.seed + offset)
        loaders[split] = DataLoader(
            dataset(indices),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            generator=generator,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    history = []
    best_state = None
    best_validation_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for features, targets in loaders["train"]:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        actual, predicted, val_losses = [], [], []
        with torch.no_grad():
            for features, targets in loaders["validation"]:
                logits = model(features.to(device))
                targets_device = targets.to(device)
                val_losses.append(float(criterion(logits, targets_device).cpu()))
                actual.extend(targets.tolist())
                predicted.extend(logits.argmax(dim=1).cpu().tolist())
        validation_loss = float(np.mean(val_losses))
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "validation_accuracy": float(accuracy_score(actual, predicted)),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best model state was saved")
    model.load_state_dict(best_state)
    model.eval()
    test_actual, test_predicted = [], []
    with torch.no_grad():
        for features, targets in loaders["test"]:
            logits = model(features.to(device))
            test_actual.extend(targets.tolist())
            test_predicted.extend(logits.argmax(dim=1).cpu().tolist())

    test_counts = Counter(test_actual)
    class_recalls = {}
    for index, name in enumerate(class_names):
        indices = [i for i, label in enumerate(test_actual) if label == index]
        class_recalls[name] = (
            sum(test_actual[i] == test_predicted[i] for i in indices) / len(indices)
            if indices else None
        )
    supported_recalls = [value for value in class_recalls.values() if value is not None]
    all_class_recalls = [value if value is not None else 0.0 for value in class_recalls.values()]
    majority = max(test_counts, key=test_counts.get)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "vicos_bjj_pose_position_mlp.pt"
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "architecture": "PosePositionMLP",
        "feature_count": int(x.shape[1]),
        "feature_mean": mean.reshape(-1).tolist(),
        "feature_scale": scale.reshape(-1).tolist(),
        "class_names": class_names,
        "dataset_id": DATASET_ID,
        "license_id": LICENSE_ID,
    }, checkpoint_path)
    metrics = {
        "dataset_id": DATASET_ID,
        "task": TASK_ID,
        "run_type": "noncommercial_exploratory_pose_position_classifier",
        "is_research_finding": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_page": SOURCE_PAGE,
        "license_id": LICENSE_ID,
        "license_note": "CC BY-NC-SA 4.0: non-commercial use; attribution and share-alike terms apply.",
        "annotation_sha256": _sha256(args.annotations),
        "samples": int(len(y)),
        "pose_features": int(x.shape[1]),
        "pose_presence": {key: int(value) for key, value in pose_counts.items()},
        "class_names": class_names,
        "class_count": len(class_names),
        "split_group_field": "encoded image/video sequence prefix",
        "split_groups": {
            name: sorted(set(groups[indices].tolist()))
            for name, indices in (("train", train_ix), ("validation", val_ix), ("test", test_ix))
        },
        "split_sample_counts": {
            name: int(len(indices))
            for name, indices in (("train", train_ix), ("validation", val_ix), ("test", test_ix))
        },
        "split_class_counts": {
            name: {class_names[index]: int(count) for index, count in sorted(Counter(y[indices]).items())}
            for name, indices in (("train", train_ix), ("validation", val_ix), ("test", test_ix))
        },
        "history": history,
        "test_metrics": {
            "accuracy": float(accuracy_score(test_actual, test_predicted)),
            "balanced_accuracy_supported_classes": float(np.mean(supported_recalls)),
            "balanced_accuracy_all_18_classes_absent_as_zero": float(np.mean(all_class_recalls)),
            "macro_f1_all_18_classes": float(f1_score(
                test_actual, test_predicted, labels=list(range(len(class_names))),
                average="macro", zero_division=0,
            )),
            "uniform_random_accuracy_reference": 1.0 / len(class_names),
            "majority_class_baseline": {
                "class": class_names[majority],
                "accuracy": float(test_counts[majority] / len(test_actual)),
            },
            "test_classes_with_samples": len(test_counts),
            "test_classes_absent": [
                class_names[index] for index in range(len(class_names)) if index not in test_counts
            ],
            "per_class_recall": class_recalls,
        },
        "limitations": [
            "Exploratory pose-landmark baseline; not a validated production model or athlete evaluation.",
            "Source provides sparse selected frames from six sparring sequences, not continuous video.",
            "Group split uses filename prefixes; those prefixes are not confirmed independent matches or athletes.",
            "Several classes are confined to very few source prefixes, making full-class holdout evaluation impossible.",
            "Position labels describe the labeled frame and are not guard-drop or dwell-time targets.",
            "Results apply only to this limited ViCoS dataset; no unrelated or gated data was ingested.",
            "Use and redistribution must comply with CC BY-NC-SA 4.0.",
        ],
        "checkpoint": str(checkpoint_path),
        "registry_status": registry_entry.get("status"),
    }
    (args.out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "device": str(device),
        "split_samples": metrics["split_sample_counts"],
        "test_metrics": {
            key: value for key, value in metrics["test_metrics"].items()
            if key not in {"per_class_recall", "test_classes_absent"}
        },
        "checkpoint": str(checkpoint_path),
    }, indent=2))


if __name__ == "__main__":
    main()
