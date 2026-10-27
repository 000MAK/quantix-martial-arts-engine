#!/usr/bin/env python3
"""Train an exploratory image classifier on the licensed ViCoS BJJ dataset.

The split is grouped by the source image's encoded sequence prefix to prevent
adjacent sampled frames from leaking across train, validation, and test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import zipfile
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "vicos_bjj_positions"
TASK_ID = "bjj_position_image_classification"
LICENSE_ID = "CC-BY-NC-SA-4.0"
DEFAULT_ANNOTATIONS = ROOT / "outputs" / "vicos_position_analysis" / "position_observations.csv"
DEFAULT_ARCHIVE = ROOT / "data" / "raw" / DATASET_ID / "images.zip.verified"
DEFAULT_REGISTRY = ROOT / "data" / "dataset_registry.json"
DEFAULT_OUTPUT = ROOT / "models" / "vicos_bjj_image_classifier"


def _torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required. Install it with: "
            "python -m pip install -r requirements-deep-learning.txt"
        ) from exc
    return torch, nn, DataLoader, Dataset


def _verify_registry(path: Path) -> dict[str, object]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("policy") != "deny_by_default":
        raise PermissionError("Dataset registry must retain its deny-by-default policy")
    entries = registry.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("Registry must contain a 'datasets' list")
    records = [row for row in entries if row.get("dataset_id") == DATASET_ID]
    if len(records) != 1:
        raise PermissionError(f"Registry must contain exactly one {DATASET_ID!r} entry")
    record = records[0]
    approval = record.get("image_training_approval", {})
    if TASK_ID not in approval.get("approved_tasks", []):
        raise PermissionError(f"Registry does not approve the {TASK_ID!r} task")
    if record.get("license_id") != LICENSE_ID:
        raise PermissionError("Registry license does not match the dataset license")
    for field in ("reviewer", "reviewed_at", "evidence_ref"):
        if not str(approval.get(field, "")).strip():
            raise PermissionError(f"Image training approval is missing {field}")
    if approval.get("noncommercial_use_confirmed") is not True:
        raise PermissionError("Non-commercial license use must be confirmed before training")
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_zip_offsets(archive: zipfile.ZipFile) -> int:
    end_record = zipfile._EndRecData(archive.fp)
    if end_record is None:
        raise zipfile.BadZipFile("Missing ZIP end-of-central-directory record")
    directory_offset = end_record[zipfile._ECD_OFFSET]
    correction = archive.start_dir - directory_offset
    if correction == 0:
        return 0

    image_info = next(
        (
            info for info in archive.infolist()
            if Path(info.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        None,
    )
    if image_info is None:
        raise zipfile.BadZipFile("Image archive contains no supported image files")
    try:
        with archive.open(image_info) as image:
            image.read(1)
        return 0
    except zipfile.BadZipFile:
        original_offset = image_info.header_offset
        image_info.header_offset -= correction
        try:
            with archive.open(image_info) as image:
                image.read(1)
        except zipfile.BadZipFile:
            image_info.header_offset = original_offset
            raise
        for info in archive.infolist():
            if info is not image_info:
                info.header_offset -= correction
        return correction


def _validate_image_members(
    archive: zipfile.ZipFile,
    image_infos: list[zipfile.ZipInfo],
) -> list[dict[str, str]]:
    invalid: list[dict[str, str]] = []
    for info in image_infos:
        try:
            image_bytes = archive.read(info)
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()
        except (zipfile.BadZipFile, zlib.error, OSError) as exc:
            invalid.append({
                "image_id": Path(info.filename).stem,
                "archive_member": info.filename,
                "error": str(exc),
            })
    return invalid


def _split_by_group(labels: np.ndarray, groups: np.ndarray, seed: int = 42):
    indices = np.arange(len(labels))
    best: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
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
            raise RuntimeError("Sequence leakage detected across image splits")

        cover = [set(labels[ix]) for ix in (train, val, test)]
        all_classes = set(np.unique(labels))
        score = 10 * len(cover[0] & all_classes) + 4 * len(cover[1] & all_classes)
        score += 4 * len(cover[2] & all_classes) + 8 * len(set.intersection(*cover))
        score -= abs(len(set(groups[test])) - 3) + abs(len(set(groups[val])) - 3)
        if score > best_score:
            best_score = score
            best = train, val, test
            if len(set.intersection(*cover)) == len(all_classes):
                break
    if best is None:
        raise RuntimeError("Could not create sequence-grouped train/validation/test splits")
    return best


def _cap_per_class(
    indices: np.ndarray,
    labels: np.ndarray,
    max_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_per_class == 0:
        return indices.copy()
    selected: list[int] = []
    for label in sorted(np.unique(labels[indices])):
        members = indices[labels[indices] == label]
        if len(members) > max_per_class:
            members = rng.choice(members, size=max_per_class, replace=False)
        selected.extend(int(index) for index in members)
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def _make_model(nn, num_classes: int):
    class PositionCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=3, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, kernel_size=3, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 96, kernel_size=3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.25),
                nn.Linear(96, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(64, num_classes),
            )

        def forward(self, image):
            return self.classifier(self.features(image))

    return PositionCNN()


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an experimental, sequence-grouped ViCoS BJJ position image classifier."
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-per-class-per-split",
        type=int,
        default=0,
        help="Optional per-class sample cap per split; 0 uses every labeled image.",
    )
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--confirm-noncommercial-use", action="store_true")
    args = parser.parse_args()
    if not args.confirm_noncommercial_use:
        parser.error("ViCoS CC BY-NC-SA 4.0 requires --confirm-noncommercial-use")
    if args.max_per_class_per_split < 0 or args.image_size < 32 or args.batch_size < 1 or args.epochs < 1:
        parser.error("Sample cap must be >= 0, batch/epochs positive, and image size >= 32")

    registry_record = _verify_registry(args.registry)
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Position annotations not found: {args.annotations}")
    if not args.archive.is_file():
        raise FileNotFoundError(
            f"Official image archive not found: {args.archive}. Download it from "
            "https://data.vicos.si/datasets/JuiJuitsu/images.zip"
        )

    annotations = pd.read_csv(args.annotations)
    required = {"image_id", "sequence_id", "position", "label_status", "license_id"}
    absent = sorted(required - set(annotations.columns))
    if absent:
        raise ValueError(f"Annotation CSV is missing columns: {', '.join(absent)}")
    annotations = annotations.loc[annotations["label_status"] == "labeled"].copy()
    if not annotations["license_id"].astype(str).eq(LICENSE_ID).all():
        raise PermissionError("Every training image must retain the approved dataset license provenance")
    if annotations.empty:
        raise ValueError("No labeled observations are available for image training")

    torch, nn, DataLoader, Dataset = _torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    with zipfile.ZipFile(args.archive) as archive:
        zip_offset_correction = _normalize_zip_offsets(archive)
        members: dict[str, str] = {}
        image_infos: list[zipfile.ZipInfo] = []
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = Path(info.filename).stem
            if stem in members:
                raise ValueError(f"Ambiguous duplicate image basename in archive: {stem}")
            members[stem] = info.filename
            image_infos.append(info)

        annotations["image_id"] = annotations["image_id"].astype(str).str.zfill(7)
        filenames = annotations["image_id"].map(members)
        if filenames.isna().any():
            examples = annotations.loc[filenames.isna(), "image_id"].astype(str).head(10).tolist()
            raise FileNotFoundError(
                f"{int(filenames.isna().sum())} labeled images are absent from the archive; examples: {examples}"
            )
        annotations["archive_member"] = filenames
        invalid_images = _validate_image_members(archive, image_infos)
        invalid_ids = {row["image_id"] for row in invalid_images}
        annotations = annotations.loc[~annotations["image_id"].isin(invalid_ids)].reset_index(drop=True)
        if annotations.empty:
            raise ValueError("No decodable labeled images remain after archive validation")
        class_names = sorted(annotations["position"].astype(str).unique())
        class_to_index = {name: index for index, name in enumerate(class_names)}
        labels = annotations["position"].astype(str).map(class_to_index).to_numpy(dtype=np.int64)
        groups = annotations["sequence_id"].astype(str).to_numpy()
        train_ix, val_ix, test_ix = _split_by_group(labels, groups)
        rng = np.random.default_rng(args.seed)
        train_ix = _cap_per_class(train_ix, labels, args.max_per_class_per_split, rng)
        val_ix = _cap_per_class(val_ix, labels, args.max_per_class_per_split, rng)
        test_ix = _cap_per_class(test_ix, labels, args.max_per_class_per_split, rng)
        split_indices = {"train": train_ix, "validation": val_ix, "test": test_ix}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if set(groups[split_indices[left]]) & set(groups[split_indices[right]]):
                raise RuntimeError(f"Sequence leakage detected between {left} and {right}")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        with (args.out_dir / "invalid_images.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=["image_id", "archive_member", "error"])
            writer.writeheader()
            writer.writerows(invalid_images)

        class ArchiveImages(Dataset):
            def __init__(self, indices: np.ndarray):
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, position: int):
                row_index = int(self.indices[position])
                member = str(annotations.iloc[row_index]["archive_member"])
                image_id = str(annotations.iloc[row_index]["image_id"])
                try:
                    with archive.open(member) as image_stream:
                        image = Image.open(io.BytesIO(image_stream.read()))
                        image = ImageOps.exif_transpose(image).convert("RGB")
                        image = image.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
                        array = np.asarray(image, dtype=np.float32) / 255.0
                except (zipfile.BadZipFile, zlib.error, OSError) as exc:
                    raise RuntimeError(f"Failed to decode validated image {image_id!r} ({member})") from exc
                array = (array - 0.5) / 0.5
                tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
                label = class_to_index[str(annotations.iloc[row_index]["position"])]
                return tensor, label

        loaders = {}
        for offset, (split, indices) in enumerate(split_indices.items()):
            generator = torch.Generator()
            generator.manual_seed(args.seed + offset)
            loaders[split] = DataLoader(
                ArchiveImages(indices),
                batch_size=args.batch_size,
                shuffle=(split == "train"),
                num_workers=0,
                generator=generator,
                pin_memory=torch.cuda.is_available(),
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _make_model(nn, len(class_names)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
        train_class_counts = np.bincount(labels[train_ix], minlength=len(class_names))
        class_weights = np.zeros(len(class_names), dtype=np.float32)
        represented = train_class_counts > 0
        class_weights[represented] = (
            len(train_ix) / (int(represented.sum()) * train_class_counts[represented])
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        )
        history: list[dict[str, object]] = []
        best_validation_loss = float("inf")
        best_state = None

        for epoch in range(args.epochs):
            model.train()
            train_losses = []
            for images, target in loaders["train"]:
                images, target = images.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), target)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            model.eval()
            val_actual: list[int] = []
            val_predicted: list[int] = []
            val_losses = []
            with torch.no_grad():
                for images, target in loaders["validation"]:
                    logits = model(images.to(device))
                    target_device = target.to(device)
                    val_losses.append(float(criterion(logits, target_device).cpu()))
                    val_predicted.extend(logits.argmax(dim=1).cpu().tolist())
                    val_actual.extend(target.tolist())
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            history.append({
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": val_loss if np.isfinite(val_loss) else None,
                "validation_accuracy": float(accuracy_score(val_actual, val_predicted)) if val_actual else None,
            })
            print(json.dumps(history[-1]), flush=True)
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if best_state is None:
            raise RuntimeError("No finite validation loss; refusing to write a model checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        test_actual: list[int] = []
        test_predicted: list[int] = []
        with torch.no_grad():
            for images, target in loaders["test"]:
                logits = model(images.to(device))
                test_predicted.extend(logits.argmax(dim=1).cpu().tolist())
                test_actual.extend(target.tolist())
        if not test_actual:
            raise RuntimeError("Sequence-grouped test split is empty")

        per_class = {}
        class_recalls: list[float] = []
        for index, name in enumerate(class_names):
            class_actual = [actual for actual in test_actual if actual == index]
            class_pairs = [(actual, predicted) for actual, predicted in zip(test_actual, test_predicted) if actual == index]
            recall = (
                sum(actual == predicted for actual, predicted in class_pairs) / len(class_pairs)
                if class_pairs else None
            )
            if recall is not None:
                class_recalls.append(recall)
            per_class[name] = {
                "test_samples": len(class_actual),
                "recall": recall,
            }
        test_class_counts = Counter(test_actual)
        majority_class = max(test_class_counts, key=test_class_counts.get)
        majority_baseline_accuracy = test_class_counts[majority_class] / len(test_actual)
        strict_recall = [
            per_class[name]["recall"] if per_class[name]["recall"] is not None else 0.0
            for name in class_names
        ]

    checkpoint_path = args.out_dir / "vicos_bjj_position_cnn.pt"
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "architecture": "PositionCNN",
        "class_names": class_names,
        "image_size": args.image_size,
        "normalization": {"scale": 0.5, "offset": 0.5},
        "dataset_id": DATASET_ID,
        "license_id": LICENSE_ID,
        "run_type": "noncommercial_exploratory_image_classifier",
    }, checkpoint_path)
    metrics = {
        "dataset_id": DATASET_ID,
        "task": TASK_ID,
        "run_type": "noncommercial_exploratory_image_classifier",
        "is_research_finding": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_page": "https://www2.vicos.si/resources/jiujitsu/",
        "license_id": LICENSE_ID,
        "license_note": "CC BY-NC-SA 4.0: non-commercial use; attribution and share-alike terms apply.",
        "paper_citation": "Hudovernik, V. & Skočaj, D. (2022). Video-Based Detection of Combat Positions and Automatic Scoring in Jiu-jitsu. ACM Multimedia Workshop MMSports’22.",
        "annotation_sha256": _sha256(args.annotations),
        "image_archive_sha256": _sha256(args.archive),
        "device": str(device),
        "image_size": args.image_size,
        "class_names": class_names,
        "class_count": len(class_names),
        "archive_image_count": len(members),
        "valid_labeled_image_count": int(len(annotations)),
        "excluded_corrupt_image_count": len(invalid_images),
        "zip_local_header_offset_correction_bytes": zip_offset_correction,
        "split_group_field": "sequence_id (encoded image/video prefix)",
        "split_groups": {
            split: sorted(set(groups[indices].tolist()))
            for split, indices in split_indices.items()
        },
        "split_sample_counts": {
            split: int(len(indices)) for split, indices in split_indices.items()
        },
        "max_per_class_per_split": args.max_per_class_per_split,
        "all_valid_labeled_images_used": args.max_per_class_per_split == 0,
        "split_class_counts": {
            split: {
                class_names[index]: int(count)
                for index, count in sorted(Counter(labels[indices]).items())
            }
            for split, indices in split_indices.items()
        },
        "history": history,
        "test_metrics": {
            "accuracy": float(accuracy_score(test_actual, test_predicted)),
            "balanced_accuracy_supported_classes": float(np.mean(class_recalls)),
            "balanced_accuracy_all_classes_absent_as_zero": float(np.mean(strict_recall)),
            "macro_f1_all_classes": float(f1_score(
                test_actual,
                test_predicted,
                labels=list(range(len(class_names))),
                average="macro",
                zero_division=0,
            )),
            "uniform_random_accuracy_reference": 1.0 / len(class_names),
            "majority_class_baseline": {
                "class": class_names[majority_class],
                "accuracy": float(majority_baseline_accuracy),
            },
            "test_classes_with_samples": int(sum(count > 0 for count in test_class_counts.values())),
            "test_classes_absent": [
                class_names[index]
                for index in range(len(class_names))
                if test_class_counts.get(index, 0) == 0
            ],
            "per_class": per_class,
        },
        "limitations": [
            "Exploratory image-classification baseline; not a validated production model.",
            "Random split unit is the source sequence prefix, not an athlete or confirmed match; camera views may be related.",
            "Source labels describe positions, not complete match state trajectories or exact dwell durations.",
            "ViCoS reports six sparring sequences and manually labeled positions in selected frames.",
            "The optional per-class cap may subsample each sequence-grouped split.",
            "The 18 labels include athlete-orientation variants and should not be collapsed without a protocol.",
            "No external dataset or independent athlete cohort was used.",
            "Model use and redistribution must comply with CC BY-NC-SA 4.0.",
        ],
        "checkpoint": str(checkpoint_path),
        "invalid_images_manifest": str(args.out_dir / "invalid_images.csv"),
        "registry_status": registry_record.get("status"),
    }
    (args.out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({
        "device": str(device),
        "split_samples": metrics["split_sample_counts"],
        "test_metrics": {
            "accuracy": metrics["test_metrics"]["accuracy"],
            "balanced_accuracy_all_classes": metrics["test_metrics"]["balanced_accuracy_all_classes_absent_as_zero"],
            "macro_f1": metrics["test_metrics"]["macro_f1_all_classes"],
            "majority_baseline_accuracy": metrics["test_metrics"]["majority_class_baseline"]["accuracy"],
        },
        "checkpoint": str(checkpoint_path),
        "warning": "Exploratory small-CNN baseline; not a research conclusion or production model.",
    }, indent=2))


if __name__ == "__main__":
    _main()
