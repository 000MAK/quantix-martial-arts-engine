"""Batch pose/kinematic extraction for the owner's own training videos (resumable).

Consent: recorded once per run from the owner's attestation (the owner plus
adult training partners who agreed to analysis). Videos are processed locally;
nothing leaves the machine. Output per video: outputs/own_videos/<source>/<stem>/.
A video whose analysis_report.json already exists is skipped, so the run can be
stopped and restarted. Duplicate files (same size and first 4 MB) are skipped.

This extracts descriptive features only. It does not train a model: there are no
labels, and MediaPipe person indices are not stable identities across frames.

Usage:
  python 01_Data_Pipelines/process_owned_videos.py --source kaplan="D:/kaplan kickboks" --source sampiyon="F:/sampiyon_kickboksmuaythai"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mts", ".mkv"}
CONSENT_NOTE = "Owner and adult training partners agreed to local analysis (attested by the owner, 2026-10-27)."


def _key(path: Path) -> str:
    with path.open("rb") as handle:
        return f"{path.stat().st_size}-{hashlib.sha256(handle.read(4 << 20)).hexdigest()[:16]}"


def collect(sources: dict[str, Path]) -> list[tuple[str, Path]]:
    seen, items = set(), []
    for name, root in sources.items():
        for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file()):
            key = _key(path)
            if key not in seen:
                seen.add(key)
                items.append((name, path))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="name=folder")
    parser.add_argument("--model-asset", type=Path, default=ENGINE / "models" / "mediapipe" / "pose_landmarker_full.task")
    parser.add_argument("--out-root", type=Path, default=ENGINE / "outputs" / "own_videos")
    parser.add_argument("--every-n-frames", type=int, default=5)
    parser.add_argument("--discipline-id", default="ma-003-kickboxing")
    args = parser.parse_args()

    sources = {}
    for item in args.source:
        name, _, folder = item.partition("=")
        sources[name.strip()] = Path(folder.strip().strip('"'))
    if not args.model_asset.exists():
        sys.exit(f"Pose model not found: {args.model_asset}")

    items = collect(sources)
    args.out_root.mkdir(parents=True, exist_ok=True)
    log_path = args.out_root / "progress.csv"
    new_log = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        if new_log:
            writer.writerow(["finished_at_utc", "source", "video", "out_dir", "status", "seconds", "consent_note"])
        for i, (name, path) in enumerate(items, 1):
            out_dir = args.out_root / name / path.stem
            if (out_dir / "analysis_report.json").exists():
                continue
            print(f"[{i}/{len(items)}] {path}", flush=True)
            started = time.time()
            proc = subprocess.run(
                [sys.executable, str(ENGINE / "main.py"), "analyze-video", "--video", str(path),
                 "--model-asset", str(args.model_asset), "--discipline-id", args.discipline_id,
                 "--sequence-id", f"{name}-{path.stem}", "--consent-status", "granted",
                 "--source-id", "owned-training-recording", "--license-id", "owned-original-recording",
                 "--every-n-frames", str(args.every_n_frames), "--out-dir", str(out_dir)],
                cwd=ENGINE, capture_output=True, text=True)
            status = "ok" if proc.returncode == 0 else f"failed: {(proc.stderr or proc.stdout).strip().splitlines()[-1][:200] if (proc.stderr or proc.stdout).strip() else proc.returncode}"
            writer.writerow([datetime.now(timezone.utc).isoformat(), name, str(path), str(out_dir), status,
                             round(time.time() - started, 1), CONSENT_NOTE])
            log.flush()
    print(f"Done. {len(items)} unique videos; log: {log_path}")


if __name__ == "__main__":
    main()
