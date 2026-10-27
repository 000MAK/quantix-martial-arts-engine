#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consent-gated temporal skeleton classifier for QMASE.

Dataset contract for a real run (all arrays share sample order):
  sequences_X.npy: float array [samples, timesteps, features]
  labels_y.npy:    binary guard-drop labels [samples] or [samples, 1]
  match_ids.npy:   group IDs [samples] (match/session; never individual frames)
  sample_manifest.csv: one row/sample with sample_id, dataset_id,
                       consent_status, training_permission, source_id, license_id

`--smoke-test` uses generated data only to verify tensor/model/checkpoint flow.
It never writes to the real processed-data or production-model directories.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_TASK = "guard_drop_temporal_binary_v1"


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


def _make_model(nn, feature_count: int):
    class GuardSequenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm1 = nn.LSTM(feature_count, 96, batch_first=True)
            self.norm = nn.LayerNorm(96)
            self.dropout = nn.Dropout(0.25)
            self.lstm2 = nn.LSTM(96, 48, batch_first=True)
            self.head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, 1))

        def forward(self, x):
            x, _ = self.lstm1(x)
            x = self.dropout(self.norm(x))
            x, _ = self.lstm2(x)
            return self.head(x[:, -1, :]).squeeze(-1)

    return GuardSequenceModel()


def _split_by_group(groups: np.ndarray, seed: int = 42):
    indices = np.arange(len(groups))
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val, test = next(first.split(indices, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    tr_rel, val_rel = next(second.split(train_val, groups=groups[train_val]))
    train, val = train_val[tr_rel], train_val[val_rel]
    if set(groups[train]) & set(groups[val]) or set(groups[train]) & set(groups[test]) or set(groups[val]) & set(groups[test]):
        raise RuntimeError("Group leakage detected between train/validation/test partitions")
    return train, val, test


def _verify_registry(registry_path: Path, dataset_id: str):
    if not registry_path.is_file():
        raise FileNotFoundError(f"Dataset registry not found: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = registry.get("datasets")
    if not isinstance(entries, list):
        raise ValueError("Registry must contain a 'datasets' array")
    matches = [entry for entry in entries if entry.get("dataset_id") == dataset_id]
    if len(matches) != 1:
        raise PermissionError(f"Registry must contain exactly one record for dataset_id={dataset_id!r}")
    entry = matches[0]
    approval = entry.get("training_approval", {})
    if entry.get("status") != "approved_for_training":
        raise PermissionError(f"Dataset {dataset_id!r} is not approved_for_training (status={entry.get('status')!r})")
    if TRAINING_TASK not in approval.get("approved_tasks", []):
        raise PermissionError(f"Dataset {dataset_id!r} is not approved for task {TRAINING_TASK!r}")
    for field in ("source_url", "license_id"):
        if not str(entry.get(field, "")).strip():
            raise ValueError(f"Registry record {dataset_id!r} requires non-empty {field}")
    for field in ("reviewer", "reviewed_at", "evidence_ref"):
        if not str(approval.get(field, "")).strip():
            raise PermissionError(f"Training approval for {dataset_id!r} is missing {field}")
    if entry.get("access_type") in {"gated", "manual_approval", "restricted"} and not approval.get("access_evidence_ref"):
        raise PermissionError(f"Restricted dataset {dataset_id!r} needs an access approval evidence reference")
    return entry


def _load_real(data_dir: Path, registry_path: Path, dataset_id: str):
    registry_entry = _verify_registry(registry_path, dataset_id)
    files = [data_dir / "sequences_X.npy", data_dir / "labels_y.npy", data_dir / "match_ids.npy"]
    missing = [str(p) for p in files if not p.is_file()]
    manifest_path = data_dir / "sample_manifest.csv"
    if not manifest_path.is_file():
        missing.append(str(manifest_path))
    if missing:
        raise FileNotFoundError("Real training data is incomplete:\n  " + "\n  ".join(missing))

    x = np.load(files[0], allow_pickle=False)
    y = np.load(files[1], allow_pickle=False).reshape(-1)
    groups = np.load(files[2], allow_pickle=False).reshape(-1).astype(str)
    manifest = pd.read_csv(manifest_path)
    required = {"sample_id", "dataset_id", "consent_status", "training_permission", "source_id", "license_id"}
    absent = sorted(required - set(manifest.columns))
    if absent:
        raise ValueError(f"sample_manifest.csv missing columns: {', '.join(absent)}")
    if not (len(x) == len(y) == len(groups) == len(manifest)):
        raise ValueError("X, y, match_ids, and manifest must have the same sample count/order")
    status = manifest["consent_status"].astype(str).str.strip().str.lower()
    permitted = manifest["training_permission"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "granted"})
    if not status.eq("granted").all() or not permitted.all():
        raise PermissionError("Training stopped: every sample must have granted consent and explicit training_permission")
    if not manifest["dataset_id"].astype(str).eq(dataset_id).all():
        raise PermissionError("Every sample must reference the selected registry dataset_id")
    if manifest[["source_id", "license_id"]].isna().any().any() or (manifest[["source_id", "license_id"]].astype(str).apply(lambda c: c.str.strip().eq("")).any().any()):
        raise ValueError("Every sample requires non-empty source_id and license_id provenance")
    if not manifest["license_id"].astype(str).eq(str(registry_entry["license_id"])).all():
        raise PermissionError("Per-sample license_id does not match the approved dataset registry record")
    return x, y, groups, manifest, registry_entry


def _run(x, y, groups, out_dir: Path, *, smoke: bool, epochs: int, batch_size: int = 16):
    torch, nn, DataLoader, TensorDataset = _torch()
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y).reshape(-1)
    groups = np.asarray(groups).reshape(-1).astype(str)
    if x.ndim != 3 or x.shape[0] != len(y) or len(y) != len(groups):
        raise ValueError("Expected X=[samples,timesteps,features] and aligned y/groups")
    if x.shape[1] < 2 or x.shape[2] < 1 or not np.isfinite(x).all():
        raise ValueError("Input sequences must have >=2 timesteps and finite features")
    unique = set(np.unique(y).tolist())
    if not unique.issubset({0, 1, 0.0, 1.0}) or len(unique) != 2:
        raise ValueError("This baseline expects both binary labels 0 and 1 (guard-drop event)")
    if len(np.unique(groups)) < 5:
        raise ValueError("Need at least five distinct match/session groups for leakage-safe splits")

    train, val, test = _split_by_group(groups)
    if set(np.unique(y[train])) != {0, 1}:
        raise ValueError("Training partition must contain both classes; add sessions or revise the sampling plan")
    mean = x[train].mean(axis=(0, 1), keepdims=True)
    scale = x[train].std(axis=(0, 1), keepdims=True)
    scale[scale < 1e-6] = 1.0
    x = np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)

    def dataset(ix):
        xx = torch.from_numpy(np.ascontiguousarray(x[ix]))
        yy = torch.from_numpy(np.asarray(y[ix], dtype=np.float32))
        return TensorDataset(xx, yy)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _make_model(nn, x.shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(dataset(train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset(val), batch_size=batch_size)
    test_loader = DataLoader(dataset(test), batch_size=batch_size)
    history = []

    for epoch in range(epochs):
        model.train()
        losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_y, val_p = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx.to(device)).cpu()
                val_p.extend((torch.sigmoid(logits) >= 0.5).int().tolist())
                val_y.extend(by.int().tolist())
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_accuracy": float(accuracy_score(val_y, val_p)) if val_y else None,
        })

    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            logits = model(bx.to(device)).cpu()
            predicted.extend((torch.sigmoid(logits) >= 0.5).int().tolist())
            actual.extend(by.int().tolist())
    metrics = {
        "run_type": "synthetic_architecture_smoke_test" if smoke else "consented_labeled_training",
        "is_research_finding": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "samples": int(len(y)),
        "timesteps": int(x.shape[1]),
        "features_per_timestep": int(x.shape[2]),
        "group_counts": {"train": int(len(set(groups[train]))), "validation": int(len(set(groups[val]))), "test": int(len(set(groups[test])))},
        "history": history,
        "test_metrics": {
            "accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "f1": float(f1_score(actual, predicted, zero_division=0)),
        },
        "warning": "Synthetic smoke-test metrics are not evidence of martial-arts performance." if smoke else "Evaluate externally before operational use.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / ("lstm_smoke_only.pt" if smoke else "guard_sequence_lstm.pt")
    torch.save({
        "state_dict": model.cpu().state_dict(),
        "feature_mean": mean.reshape(-1).tolist(),
        "feature_scale": scale.reshape(-1).tolist(),
        "feature_count": int(x.shape[2]),
        "timesteps": int(x.shape[1]),
        "label_definition": "binary guard-drop event (0/1)",
        "run_type": metrics["run_type"],
    }, model_path)
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics, model_path


def main():
    parser = argparse.ArgumentParser(description="Train or smoke-test the consent-gated QMASE temporal pose model.")
    parser.add_argument("--smoke-test", action="store_true", help="Use ephemeral synthetic tensors to verify LSTM flow only")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--registry", type=Path, default=PROJECT_ROOT / "data" / "dataset_registry.json")
    parser.add_argument("--dataset-id", type=str, default=None, help="Required for real training; must be approved in the dataset registry")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    if args.smoke_test:
        rng = np.random.default_rng(20261027)
        n, t, f = 60, 30, 99
        x = rng.normal(size=(n, t, f)).astype(np.float32)
        y = np.tile(np.array([0, 1], dtype=np.float32), n // 2)
        groups = np.repeat([f"smoke-session-{i:02d}" for i in range(12)], 5)
        out_dir = args.out_dir or PROJECT_ROOT / "outputs" / "architecture_smoke"
        metrics, model_path = _run(x, y, groups, out_dir, smoke=True, epochs=min(args.epochs, 2))
        print("SMOKE TEST ONLY: synthetic data; no athletic-performance claim.")
    else:
        if not args.dataset_id:
            parser.error("--dataset-id is required for real training")
        x, y, groups, _manifest, _registry_entry = _load_real(args.data_dir, args.registry, args.dataset_id)
        out_dir = args.out_dir or PROJECT_ROOT / "models" / "guard_sequence_lstm"
        metrics, model_path = _run(x, y, groups, out_dir, smoke=False, epochs=args.epochs)
        print("Trained on consented, labeled local data. Validate on an independent cohort before use.")
    print(f"Run type: {metrics['run_type']}")
    print(f"Device: {metrics['device']}; groups: {metrics['group_counts']}")
    print(f"Checkpoint: {model_path}")
    print(f"Metrics: {out_dir / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
