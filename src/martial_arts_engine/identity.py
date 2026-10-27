"""Mark which person track in a video is the owner ("who is me").

For each tracked video folder this renders a contact sheet: the largest tracks,
each shown on one clear frame with its body outlined and labelled A, B, C. The
owner picks a letter; the choice is stored in `me_track.json` and
`my_landmarks.csv` keeps only that track, without frames flagged
identity_uncertain. A colour histogram of the torso lets earlier picks suggest a
track in later videos (same clothes within a session); a suggestion is never
applied without confirmation.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LETTERS = "ABC"
TORSO = (11, 12, 23, 24)


def candidate_tracks(tracked: pd.DataFrame, max_tracks: int = 3, min_detections: int = 5) -> list[int]:
    counts = tracked.groupby("track_id").size().sort_values(ascending=False)
    return [int(t) for t, n in counts.items() if n >= min_detections][:max_tracks]


def representative_row(tracked: pd.DataFrame, track_id: int) -> pd.Series:
    """A clear observation near the middle of the track: certain identity, high torso visibility."""
    g = tracked[tracked["track_id"] == track_id]
    clear = g[~g["identity_uncertain"].astype(bool)] if "identity_uncertain" in g else g
    pool = clear if len(clear) else g
    vis = pool[[f"pose_{i}_visibility" for i in TORSO]].mean(axis=1)
    middle = pool["timestamp_ms"].median()
    score = vis - 0.1 * (pool["timestamp_ms"] - middle).abs() / max(1.0, float(pool["timestamp_ms"].max() - pool["timestamp_ms"].min()))
    return pool.loc[score.idxmax()]


def _box(row: pd.Series, width: int, height: int) -> tuple[int, int, int, int]:
    xs = np.array([row[f"pose_{i}_x"] for i in range(33)]) * width
    ys = np.array([row[f"pose_{i}_y"] for i in range(33)]) * height
    pad = 0.08 * height
    return (int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad)), int(min(width - 1, xs.max() + pad)), int(min(height - 1, ys.max() + pad)))


def read_frame(video: Path, frame_index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def torso_histogram(frame: np.ndarray, row: pd.Series) -> np.ndarray | None:
    h, w = frame.shape[:2]
    xs = [row[f"pose_{i}_x"] * w for i in TORSO]
    ys = [row[f"pose_{i}_y"] * h for i in TORSO]
    x0, x1, y0, y1 = int(max(0, min(xs))), int(min(w, max(xs))), int(max(0, min(ys))), int(min(h, max(ys)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).flatten()
    return hist / max(hist.sum(), 1e-9)


def render_sheet(video_dir: Path, video: Path, out_height: int = 360) -> dict:
    """Write identity/sheet.jpg and identity/candidates.json for one tracked video folder."""
    tracked = pd.read_csv(video_dir / "tracked_landmarks.csv")
    tracks = candidate_tracks(tracked)
    panels, candidates = [], []
    for letter, track_id in zip(LETTERS, tracks):
        row = representative_row(tracked, track_id)
        frame = read_frame(video, int(row["frame_index"]))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = _box(row, w, h)
        hist = torso_histogram(frame, row)
        shown = frame.copy()
        cv2.rectangle(shown, (x0, y0), (x1, y1), (0, 140, 255), max(3, h // 150))
        scale = out_height / h
        shown = cv2.resize(shown, (int(w * scale), out_height))
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 44), (20, 20, 20), -1)
        cv2.putText(shown, f"{letter}  track {track_id}  ({int((tracked.track_id == track_id).sum())} frames)", (10, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panels.append(shown)
        candidates.append({"letter": letter, "track_id": track_id, "frame_index": int(row["frame_index"]),
                           "torso_hist": hist.tolist() if hist is not None else None})
    out = video_dir / "identity"
    out.mkdir(exist_ok=True)
    if panels:
        cv2.imwrite(str(out / "sheet.jpg"), np.hstack(panels))
    (out / "candidates.json").write_text(json.dumps({"video": str(video), "candidates": candidates}), encoding="utf-8")
    return {"video_dir": str(video_dir), "candidates": len(candidates)}


def suggest(candidates: list[dict], my_hists: list[np.ndarray]) -> tuple[str | None, float]:
    """Letter whose torso colours best match earlier picks (Bhattacharyya similarity), with that similarity."""
    if not my_hists:
        return None, 0.0
    best, best_sim = None, 0.0
    for c in candidates:
        if c["torso_hist"] is None:
            continue
        h = np.asarray(c["torso_hist"], dtype=np.float32)
        sim = max(1.0 - cv2.compareHist(h, m.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA) for m in my_hists)
        if sim > best_sim:
            best, best_sim = c["letter"], float(sim)
    return best, best_sim


def apply_choice(video_dir: Path, letter: str | None) -> dict:
    """Store the owner's pick. letter None = the owner does not appear among the candidates."""
    info = json.loads((video_dir / "identity" / "candidates.json").read_text(encoding="utf-8"))
    chosen = next((c for c in info["candidates"] if c["letter"] == letter), None) if letter else None
    record = {"me_track_id": chosen["track_id"] if chosen else None, "letter": letter,
              "torso_hist": chosen["torso_hist"] if chosen else None, "picked_by": "owner"}
    (video_dir / "me_track.json").write_text(json.dumps(record), encoding="utf-8")
    rows = 0
    if chosen:
        tracked = pd.read_csv(video_dir / "tracked_landmarks.csv")
        mine = tracked[(tracked["track_id"] == chosen["track_id"]) & ~tracked["identity_uncertain"].astype(bool)]
        mine.to_csv(video_dir / "my_landmarks.csv", index=False)
        rows = len(mine)
    return {"me_track_id": record["me_track_id"], "my_rows": rows}
