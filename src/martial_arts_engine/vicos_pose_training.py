from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from .vicos_bjj import DATASET_URL, LICENSE_ID


DATASET_ID = "vicos_bjj_positions"
TRAINING_TASK = "bjj_pose_position_classification"
FEATURE_NAMES = [
    f"pose{person}_joint{joint}_{axis}"
    for person in (1, 2)
    for joint in range(17)
    for axis in ("x", "y", "confidence")
] + ["pose1_present", "pose2_present"]


def _verify_training_approval(registry_path: Path) -> dict:
    if not registry_path.is_file():
        raise FileNotFoundError(f"Dataset registry not found: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = registry.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("Registry must contain a 'datasets' array")
    matches = [item for item in entries if item.get("dataset_id") == DATASET_ID]
    if len(matches) != 1:
        raise PermissionError(f"Registry must contain exactly one {DATASET_ID!r} record")
    entry = matches[0]
    approval = entry.get("training_approval", {})
    if entry.get("status") != "approved_for_noncommercial_analysis_only":
        raise PermissionError(f"Dataset is not approved for non-commercial analysis: {entry.get('status')!r}")
    if TRAINING_TASK not in approval.get("approved_tasks", []):
        raise PermissionError(f"Dataset is not approved for {TRAINING_TASK!r}")
    if entry.get("license_id") != LICENSE_ID:
        raise PermissionError("Dataset license does not match the reviewed ViCoS license")
    if approval.get("noncommercial_use_confirmed") is not True:
        raise PermissionError("Non-commercial training scope is not confirmed in the registry")
    for field in ("reviewer", "reviewed_at", "evidence_ref", "access_evidence_ref"):
        if not str(approval.get(field, "")).strip():
            raise PermissionError(f"Training approval is missing {field}")
    return entry


def _pose_features(pose: object, row_index: int, field: str) -> np.ndarray:
    if pose is None:
        return np.zeros(51, dtype=np.float32)
    array = np.asarray(pose, dtype=np.float32)
    if array.shape != (17, 3) or not np.isfinite(array).all():
        raise ValueError(f"Annotation {row_index} has invalid {field} pose; expected finite [17,3]")
    return array.reshape(-1)


def load_vicos_pose_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("annotations")
    if not isinstance(payload, list) or not payload:
        raise ValueError("ViCoS annotation input must be a non-empty JSON array")

    features = np.zeros((len(payload), len(FEATURE_NAMES)), dtype=np.float32)
    labels: list[str] = []
    groups: list[str] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Annotation {i} must be a JSON object")
        fields = {str(key).casefold(): value for key, value in item.items()}
        image_id = str(fields.get("image", ""))
        match = re.match(r"(\d{2})", image_id)
        if not match:
            raise ValueError(f"Annotation {i} has invalid sequence image ID {image_id!r}")
        label = fields.get("position")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Annotation {i} has no position class")
        label = label.strip().lower()
        pose1 = fields.get("pose1")
        pose2 = fields.get("pose2")
        if pose1 is None and pose2 is None:
            raise ValueError(f"Annotation {i} has no pose coordinates")
        features[i, :51] = _pose_features(pose1, i, "pose1")
        features[i, 51:102] = _pose_features(pose2, i, "pose2")
        features[i, 102] = float(pose1 is not None)
        features[i, 103] = float(pose2 is not None)
        labels.append(label)
        groups.append(match.group(1))

    return features, np.asarray(labels, dtype=str), np.asarray(groups, dtype=str)


def _grouped_split(
    labels: np.ndarray, groups: np.ndarray, seed_candidates: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    indices = np.arange(len(labels))
    unique_labels = set(labels.tolist())
    best = None
    best_score = None
    for seed in range(seed_candidates):
        first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_val, test = next(first.split(indices, labels, groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1000)
        train_rel, val_rel = next(second.split(
            train_val, labels[train_val], groups[train_val]
        ))
        train = train_val[train_rel]
        val = train_val[val_rel]
        train_classes = set(labels[train].tolist())
        val_classes = set(labels[val].tolist())
        test_classes = set(labels[test].tolist())
        score = (
            int(train_classes == unique_labels),
            len(test_classes),
            len(val_classes),
            len(train_classes),
        )
        if best_score is None or score > best_score:
            best = (train, val, test)
            best_score = score
        if score[0] and score[1] == len(unique_labels) and score[2] == len(unique_labels):
            break
    if best is None or set(labels[best[0]].tolist()) != unique_labels:
        raise ValueError("No sequence-grouped split places every class in training; add labeled sequences")
    train, val, test = best
    split_groups = {
        "train": sorted(set(groups[train].tolist())),
        "validation": sorted(set(groups[val].tolist())),
        "test": sorted(set(groups[test].tolist())),
    }
    if (
        set(split_groups["train"]) & set(split_groups["validation"])
        or set(split_groups["train"]) & set(split_groups["test"])
        or set(split_groups["validation"]) & set(split_groups["test"])
    ):
        raise RuntimeError("Sequence leakage detected across grouped partitions")
    return train, val, test, split_groups


def train_vicos_pose_classifier(
    annotation_path: Path,
    registry_path: Path,
    out_dir: Path,
    *,
    epochs: int = 8,
    batch_size: int = 512,
) -> dict:
    _verify_training_approval(registry_path)
    if epochs < 1:
        raise ValueError("epochs must be at least one")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to train the ViCoS pose classifier") from exc

    x, labels, groups = load_vicos_pose_data(annotation_path)
    classes = sorted(set(labels.tolist()))
    class_to_index = {label: index for index, label in enumerate(classes)}
    y = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    train, val, test, split_groups = _grouped_split(labels, groups)

    mean = x[train].mean(axis=0)
    scale = x[train].std(axis=0)
    scale[scale < 1e-6] = 1.0
    x_scaled = np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class PosePositionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(len(FEATURE_NAMES), 192),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(192, 96),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(96, len(classes)),
            )

        def forward(self, values):
            return self.network(values)

    def make_loader(indices: np.ndarray, shuffle: bool):
        dataset = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(x_scaled[indices])),
            torch.from_numpy(y[indices]),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def fit(
        indices: np.ndarray,
        fit_epochs: int,
        validation_indices: np.ndarray | None = None,
    ):
        model = PosePositionClassifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        loader = make_loader(indices, shuffle=True)
        validation_loader = (
            make_loader(validation_indices, shuffle=False)
            if validation_indices is not None
            else None
        )
        history = []
        for epoch in range(fit_epochs):
            model.train()
            losses = []
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            record = {"epoch": epoch + 1, "training_loss": float(np.mean(losses))}
            if validation_loader is not None:
                model.eval()
                actual_val, predicted_val = [], []
                with torch.no_grad():
                    for bx, by in validation_loader:
                        predicted_val.extend(model(bx.to(device)).argmax(dim=1).cpu().tolist())
                        actual_val.extend(by.tolist())
                record["validation_accuracy"] = float(
                    accuracy_score(actual_val, predicted_val)
                )
            history.append(record)
        return model, history

    eval_model, history = fit(train, epochs, validation_indices=val)
    eval_model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for bx, by in make_loader(test, shuffle=False):
            logits = eval_model(bx.to(device)).cpu()
            predicted.extend(logits.argmax(dim=1).numpy().tolist())
            actual.extend(by.numpy().tolist())

    present_test_classes = sorted(set(actual))
    per_class_recall = [
        float(np.mean(np.asarray(predicted)[np.asarray(actual) == class_index] == class_index))
        for class_index in present_test_classes
    ]
    metrics = {
        "dataset_id": DATASET_ID,
        "task": TRAINING_TASK,
        "run_type": "noncommercial_supervised_pose_position_classification",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "is_universal_score": False,
        "samples": int(len(y)),
        "sequences": int(len(set(groups.tolist()))),
        "classes": classes,
        "feature_count": int(x.shape[1]),
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "split_group_method": "source image ID two-digit sequence prefix",
        "split_sequences": split_groups,
        "sample_counts": {
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
        "history": history,
        "held_out_test_metrics": {
            "accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy_present_test_classes": float(np.mean(per_class_recall)),
            "macro_f1_present_test_classes": float(f1_score(
                actual, predicted, labels=present_test_classes, average="macro", zero_division=0
            )),
            "classes_present_in_test": [classes[index] for index in present_test_classes],
            "classes_absent_from_test": [
                label for index, label in enumerate(classes) if index not in present_test_classes
            ],
        },
        "limitations": [
            "This is one non-commercial ViCoS dataset, not training on all publicly accessible videos.",
            "It predicts the dataset's 18 annotated position classes from precomputed pose coordinates, not RGB video.",
            "The source contains selected frames from a small number of sparring sequences; grouping uses the image-ID sequence prefix.",
            "The class label can refer to athlete 1 or 2; pose1 and pose2 are both provided when available.",
            "The held-out metrics come from a sequence-grouped evaluation model. The saved deployment checkpoint is refit on all annotations.",
            "This model is an exploratory baseline and does not predict guard drops, athlete quality, injury risk, or a universal QMASE score.",
            "CC BY-NC-SA 4.0 applies; verify attribution and share-alike obligations before distributing derived artifacts.",
        ],
    }

    final_model, _ = fit(np.arange(len(y)), epochs)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "vicos_bjj_pose_position_classifier.pt"
    metrics_path = out_dir / "training_metrics.json"
    final_model.eval()
    torch.save({
        "state_dict": final_model.cpu().state_dict(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "feature_names": FEATURE_NAMES,
        "classes": classes,
        "dataset_id": DATASET_ID,
        "task": TRAINING_TASK,
        "license": LICENSE_ID,
        "source_url": DATASET_URL,
        "run_type": metrics["run_type"],
        "trained_samples": int(len(y)),
        "trained_sequence_groups": split_groups["train"] + split_groups["validation"] + split_groups["test"],
    }, checkpoint_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["metrics_file"] = str(metrics_path)
    return metrics


def _read_vicos_registry_approval(registry_path: Path) -> dict:
    if not registry_path.is_file():
        raise FileNotFoundError(f"Dataset registry not found: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = registry.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("Registry must contain a 'datasets' array")
    matches = [item for item in entries if item.get("dataset_id") == DATASET_ID]
    if len(matches) != 1:
        raise PermissionError(f"Registry must contain exactly one {DATASET_ID!r} record")
    entry = matches[0]
    if entry.get("status") != "approved_for_noncommercial_analysis_only":
        raise PermissionError(f"ViCoS analysis is not approved: {entry.get('status')!r}")
    approval = entry.get("image_training_approval", {})
    if "bjj_position_image_classification" not in approval.get("approved_tasks", []):
        raise PermissionError("Registry does not approve ViCoS image-position classification")
    if entry.get("license_id") != LICENSE_ID:
        raise PermissionError("Dataset license does not match the reviewed ViCoS license")
    if approval.get("noncommercial_use_confirmed") is not True:
        raise PermissionError("Non-commercial ViCoS image training is not confirmed")
    for field in ("reviewer", "reviewed_at", "evidence_ref", "scope"):
        if not str(approval.get(field, "")).strip():
            raise PermissionError(f"ViCoS image training approval is missing {field}")
    return entry


def extract_vicos_position_images(
    annotation_path: Path,
    image_archive: Path,
    registry_path: Path,
    out_dir: Path,
) -> dict:
    import zipfile

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from PIL import Image

    entry = _read_vicos_registry_approval(registry_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"ViCoS annotation JSON not found: {annotation_path}")
    if not image_archive.is_file():
        raise FileNotFoundError(f"ViCoS image archive not found: {image_archive}")

    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("ViCoS annotation input must be a non-empty JSON array")
    annotation_by_image: dict[str, str] = {}
    for i, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Annotation {i} must be a JSON object")
        image_id = str(row.get("image", "")).strip()
        label = str(row.get("position", "")).strip()
        if not image_id or not label:
            raise ValueError(f"Annotation {i} requires image and position")
        if image_id in annotation_by_image:
            raise ValueError(f"Duplicate ViCoS image annotation: {image_id!r}")
        annotation_by_image[image_id] = label

    with zipfile.ZipFile(image_archive) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ViCoS image ZIP CRC validation failed at {bad_member!r}")
        names_by_stem: dict[str, str] = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            member_path = Path(info.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe image path in ViCoS ZIP: {info.filename!r}")
            names_by_stem[member_path.stem] = info.filename
        missing = sorted(set(annotation_by_image) - set(names_by_stem))
        if missing:
            raise ValueError(
                f"ViCoS ZIP is missing {len(missing)} annotated images; first: {missing[:5]}"
            )
        images = []
        labels = []
        for image_id, label in annotation_by_image.items():
            try:
                with archive.open(names_by_stem[image_id]) as image_file:
                    image = Image.open(image_file).convert("RGB").resize((96, 96))
                    images.append(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))
            except Exception as exc:
                raise ValueError(f"Could not decode ViCoS image {image_id!r}") from exc
            labels.append(label)

    classes = sorted(set(labels))
    class_to_index = {label: index for index, label in enumerate(classes)}
    y = np.asarray([class_to_index[label] for label in labels], dtype=np.int64)
    groups = np.asarray([
        re.match(r"(\d{2})", image_id).group(1) for image_id in annotation_by_image
    ], dtype=str)
    train, val, test, split_groups = _grouped_split(np.asarray(labels), groups)
    images_array = np.stack(images)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class ImagePositionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(64, len(classes))

        def forward(self, values):
            return self.head(self.features(values).flatten(1))

    image_tensor = torch.from_numpy(images_array.copy()).float().div_(255.0)

    def make_loader(indices: np.ndarray, shuffle: bool):
        dataset = TensorDataset(image_tensor[indices], torch.from_numpy(y[indices]))
        return DataLoader(dataset, batch_size=128, shuffle=shuffle)

    def fit(indices: np.ndarray, fit_epochs: int):
        model = ImagePositionClassifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        loader = make_loader(indices, shuffle=True)
        history = []
        for epoch in range(fit_epochs):
            model.train()
            losses = []
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": epoch + 1, "training_loss": float(np.mean(losses))})
        return model, history

    eval_model, history = fit(train, epochs=8)
    eval_model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for bx, by in make_loader(test, shuffle=False):
            predicted.extend(eval_model(bx.to(device)).argmax(dim=1).cpu().tolist())
            actual.extend(by.tolist())
    present_test_classes = sorted(set(actual))
    predicted_array, actual_array = np.asarray(predicted), np.asarray(actual)
    recalls = [
        float(np.mean(predicted_array[actual_array == class_index] == class_index))
        for class_index in present_test_classes
    ]
    metrics = {
        "dataset_id": DATASET_ID,
        "task": "bjj_rgb_position_image_classification",
        "run_type": "noncommercial_supervised_rgb_position_classification",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "samples": int(len(y)),
        "sequences": int(len(set(groups.tolist()))),
        "classes": classes,
        "image_size": [96, 96],
        "epochs": 8,
        "device": str(device),
        "split_group_method": "source image ID two-digit sequence prefix",
        "split_sequences": split_groups,
        "sample_counts": {
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
        "history": history,
        "held_out_test_metrics": {
            "accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy_present_test_classes": float(np.mean(recalls)),
            "macro_f1_present_test_classes": float(f1_score(
                actual, predicted, labels=present_test_classes, average="macro", zero_division=0
            )),
            "classes_present_in_test": [classes[index] for index in present_test_classes],
            "classes_absent_from_test": [
                label for index, label in enumerate(classes) if index not in present_test_classes
            ],
        },
        "limitations": [
            "Trained on one ViCoS dataset under CC BY-NC-SA 4.0, not all open videos.",
            "Exploratory RGB frame classifier for dataset position labels; not temporal action recognition or guard-drop prediction.",
            "The source contains selected frames from a small number of sparring sequences.",
            "Sequence-grouped held-out metrics come from a sequence-level split and are not an independent external validation.",
            "This model is not a universal QMASE score or athlete evaluation tool.",
        ],
    }

    final_model, _ = fit(np.arange(len(y)), epochs=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "vicos_bjj_rgb_position_classifier.pt"
    metrics_path = out_dir / "training_metrics.json"
    torch.save({
        "state_dict": final_model.cpu().state_dict(),
        "classes": classes,
        "image_size": [96, 96],
        "dataset_id": DATASET_ID,
        "task": metrics["task"],
        "license": entry["license_id"],
        "source_url": entry["source_url"],
        "run_type": metrics["run_type"],
        "trained_samples": int(len(y)),
        "split_sequences": split_groups,
    }, checkpoint_path)
    metrics["attribution"] = (
        "Hudovernik and Skočaj, Video-Based Detection of Combat Positions and "
        "Automatic Scoring in Jiu-jitsu, MMSports 2022."
    )
    metrics["license"] = entry["license_id"]
    metrics["source_url"] = entry["source_url"]
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["metrics_file"] = str(metrics_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics
