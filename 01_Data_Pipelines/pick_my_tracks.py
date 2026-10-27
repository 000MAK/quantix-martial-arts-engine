"""Interactive "who is me" picker for tracked training videos.

For each tracked video without a pick, opens a contact sheet of the main person
tracks (A, B, C) and asks which one is the owner. Earlier picks suggest a letter
from torso colours; press Enter to accept it. Stop any time with q; rerun to resume.

Usage:  python 01_Data_Pipelines/pick_my_tracks.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE / "src"))
from martial_arts_engine.identity import apply_choice, render_sheet, suggest  # noqa: E402

ROOT = ENGINE / "outputs" / "own_videos"
MIN_SUGGEST_SIMILARITY = 0.6


def video_paths() -> dict[str, Path]:
    log = ROOT / "progress.csv"
    with log.open(encoding="utf-8") as handle:
        return {str(Path(r["out_dir"]).resolve()).lower(): Path(r["video"]) for r in csv.DictReader(handle) if r["status"] == "ok"}


def my_histograms() -> list[np.ndarray]:
    hists = []
    for f in ROOT.rglob("me_track.json"):
        h = json.loads(f.read_text(encoding="utf-8")).get("torso_hist")
        if h:
            hists.append(np.asarray(h))
    return hists


def main() -> None:
    videos = video_paths()
    todo = sorted(d.parent for d in ROOT.rglob("tracked_landmarks.csv") if not (d.parent / "me_track.json").exists())
    print(f"{len(todo)} tracked videos without a pick. Run option 10 (track people) first for newly processed videos.")
    for n, video_dir in enumerate(todo, 1):
        video = videos.get(str(video_dir.resolve()).lower())
        if video is None or not video.exists():
            print(f"[{n}/{len(todo)}] {video_dir.name}: source video not found, skipped")
            continue
        info_path = video_dir / "identity" / "candidates.json"
        if not info_path.exists():
            render_sheet(video_dir, video)
        info = json.loads(info_path.read_text(encoding="utf-8"))
        letters = [c["letter"] for c in info["candidates"]]
        if not letters:
            print(f"[{n}/{len(todo)}] {video_dir.name}: no usable tracks, skipped")
            continue
        os.startfile(video_dir / "identity" / "sheet.jpg")
        guess, sim = suggest(info["candidates"], my_histograms())
        hint = f"  Enter = {guess} (colour match {sim:.2f})" if guess and sim >= MIN_SUGGEST_SIMILARITY else ""
        answer = input(f"[{n}/{len(todo)}] {video_dir.name}: which is you? {'/'.join(letters)}, n = not in video, s = skip, q = quit{hint}: ").strip().upper()
        if answer == "" and hint:
            answer = guess
        if answer == "Q":
            break
        if answer in ("S", ""):
            continue
        result = apply_choice(video_dir, None if answer == "N" else answer if answer in letters else None)
        print(f"    saved: track {result['me_track_id']} ({result['my_rows']} of your frames kept)")


if __name__ == "__main__":
    main()
