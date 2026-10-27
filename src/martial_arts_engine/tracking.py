"""Stable person identities across frames for MediaPipe pose output.

MediaPipe's `person_index` is only an index within one frame: when two athletes
cross or clinch, index 0 can be either of them. This module links detections into
tracks frame by frame using keypoint distance and optimal (Hungarian) assignment,
so a swapped index does not swap the identity.

Limits: matching uses geometry only, no appearance. When two bodies overlap
heavily (constant in grappling) the assignment can still be wrong; those rows are
flagged `identity_uncertain` and should be excluded from per-athlete metrics.
Which track is a given athlete is not inferred; it has to be supplied.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

N_LANDMARKS = 33
MIN_VISIBILITY = 0.5


def _keypoints(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(people, 33, 2) coordinates in height units (aspect-corrected) and (people, 33) visibility."""
    aspect = (frame["image_width"] / frame["image_height"]).to_numpy()[:, None]
    xs = frame[[f"pose_{i}_x" for i in range(N_LANDMARKS)]].to_numpy(dtype=float) * aspect
    ys = frame[[f"pose_{i}_y" for i in range(N_LANDMARKS)]].to_numpy(dtype=float)
    vis = frame[[f"pose_{i}_visibility" for i in range(N_LANDMARKS)]].to_numpy(dtype=float)
    return np.stack([xs, ys], axis=-1), vis


TORSO = (11, 12, 23, 24)  # shoulders and hips


def _torso(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Torso centre and torso length (shoulder midpoint to hip midpoint), in height units."""
    shoulders = points[[11, 12]].mean(axis=0)
    hips = points[[23, 24]].mean(axis=0)
    return points[list(TORSO)].mean(axis=0), max(float(np.linalg.norm(shoulders - hips)), 0.02)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """Torso-centre displacement measured in torso lengths, so the threshold scales with apparent size.
    Limbs are ignored: in striking they move far between sampled frames while the torso does not."""
    (ca, la), (cb, lb) = _torso(a), _torso(b)
    return float(np.linalg.norm(ca - cb) / ((la + lb) / 2))


def _box(points: np.ndarray, vis: np.ndarray) -> np.ndarray:
    keep = vis >= MIN_VISIBILITY
    p = points[keep] if keep.sum() >= 4 else points
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _stitch(data: pd.DataFrame, stitch_gap_ms: float, stitch_distance: float) -> pd.DataFrame:
    """Join a track that ends to one that starts shortly after, nearby, without overlapping in time.
    Fragments arise when a person leaves the (at most two) detections MediaPipe returns per frame."""
    pts, _ = _keypoints(data)
    spans = []
    for tid, idx in data.groupby("track_id").groups.items():
        rows = np.asarray(idx)
        order = rows[np.argsort(data.loc[rows, "timestamp_ms"].to_numpy())]
        spans.append({"id": tid, "start": data.at[order[0], "timestamp_ms"], "end": data.at[order[-1], "timestamp_ms"],
                      "first": pts[order[0]], "last": pts[order[-1]]})
    spans.sort(key=lambda s: s["start"])
    mapping = {s["id"]: s["id"] for s in spans}
    chain_end: dict[int, dict] = {}  # root id -> last span in that chain
    for s in spans:
        best, best_d = None, stitch_distance
        for root, tail in chain_end.items():
            gap = s["start"] - tail["end"]
            if 0 < gap <= stitch_gap_ms:
                d = _distance(tail["last"], s["first"])
                if d <= best_d:
                    best, best_d = root, d
        if best is None:
            chain_end[s["id"]] = s
        else:
            mapping[s["id"]] = best
            chain_end[best] = s
    data["track_id"] = data["track_id"].map(mapping)
    return data


def assign_tracks(poses: pd.DataFrame, max_distance: float = 1.0, max_gap_ms: float = 1500.0,
                  overlap_iou: float = 0.3, stitch_gap_ms: float = 4000.0, stitch_distance: float = 1.5) -> pd.DataFrame:
    """Add `track_id` and `identity_uncertain` to a pose_landmarks table.

    max_distance: torso-centre displacement allowed between consecutive observations, in torso
    lengths; max_gap_ms: how long a track may go unseen. A second pass stitches fragments that
    end and restart within stitch_gap_ms and stitch_distance torso lengths.
    """
    data = poses.sort_values(["timestamp_ms", "person_index"]).reset_index(drop=True).copy()
    track_ids = np.full(len(data), -1, dtype=int)
    uncertain = np.zeros(len(data), dtype=bool)
    tracks: dict[int, dict] = {}
    next_id = 0
    for ts, idx in data.groupby("timestamp_ms", sort=True).groups.items():
        rows = np.asarray(idx)
        pts, vis = _keypoints(data.loc[rows])
        live = [t for t, s in tracks.items() if ts - s["ts"] <= max_gap_ms]
        assigned = {}
        if live:
            cost = np.array([[_distance(tracks[t]["pts"], pts[j]) for j in range(len(rows))] for t in live])
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] <= max_distance:
                    assigned[c] = live[r]
        for j in range(len(rows)):
            if j not in assigned:
                assigned[j] = next_id
                next_id += 1
            t = assigned[j]
            tracks[t] = {"pts": pts[j], "vis": vis[j], "ts": ts}
            track_ids[rows[j]] = t
        if len(rows) >= 2:
            boxes = [_box(pts[j], vis[j]) for j in range(len(rows))]
            if max(_iou(boxes[a], boxes[b]) for a in range(len(rows)) for b in range(a + 1, len(rows))) >= overlap_iou:
                uncertain[rows] = True
    data["track_id"] = track_ids
    data["identity_uncertain"] = uncertain
    return _stitch(data, stitch_gap_ms, stitch_distance) if stitch_gap_ms > 0 else data


def track_summary(tracked: pd.DataFrame) -> pd.DataFrame:
    g = tracked.groupby("track_id")
    return pd.DataFrame({
        "detections": g.size(),
        "first_ms": g["timestamp_ms"].min(),
        "last_ms": g["timestamp_ms"].max(),
        "uncertain_share": g["identity_uncertain"].mean().round(3),
        "mean_hip_x": ((g["pose_23_x"].mean() + g["pose_24_x"].mean()) / 2).round(3),
    }).reset_index().sort_values("detections", ascending=False)


def track_video_dir(video_dir: Path) -> dict:
    """Write tracked_landmarks.csv and track_summary.csv; add track_id to guard_proxy_frames.csv if present."""
    poses = pd.read_csv(video_dir / "pose_landmarks.csv")
    if poses.empty:
        return {"video_dir": str(video_dir), "status": "no_pose_rows"}
    tracked = assign_tracks(poses)
    tracked.to_csv(video_dir / "tracked_landmarks.csv", index=False)
    summary = track_summary(tracked)
    summary.to_csv(video_dir / "track_summary.csv", index=False)
    guard_path = video_dir / "guard_proxy_frames.csv"
    if guard_path.exists():
        guard = pd.read_csv(guard_path)
        keys = tracked[["timestamp_ms", "person_index", "track_id", "identity_uncertain"]]
        guard.drop(columns=[c for c in ("track_id", "identity_uncertain") if c in guard.columns]).merge(
            keys, on=["timestamp_ms", "person_index"], how="left").to_csv(video_dir / "guard_proxy_tracked.csv", index=False)
    top = summary.head(2)
    return {"video_dir": str(video_dir), "status": "ok", "tracks": int(len(summary)),
            "main_tracks_share_of_detections": round(float(top["detections"].sum() / len(tracked)), 3),
            "uncertain_share": round(float(tracked["identity_uncertain"].mean()), 3)}
