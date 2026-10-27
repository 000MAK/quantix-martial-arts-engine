"""Demo material from one tracked training video: a skeleton clip and a one-page card.

Everything shown is measured from the video's pose output; there are no scores,
grades or readiness verdicts. Intensity is the speed of wrists and ankles in
torso lengths per second (camera-plane, 2-D), which reflects how much the limbs
move, not effort, power or fatigue.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LIMBS = (15, 16, 27, 28)  # wrists and ankles
EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (0, 11), (0, 12)]
COLOURS = [(0, 140, 255), (60, 200, 255), (120, 220, 120), (255, 120, 200)]  # BGR; first two match the chart (#ff8c00, #ffc83c)


def intensity_timeline(tracked: pd.DataFrame, track_id: int, window_s: float = 2.0) -> pd.DataFrame:
    """Rolling median of wrist/ankle speed (torso lengths per second) for one track."""
    g = tracked[(tracked["track_id"] == track_id) & ~tracked["identity_uncertain"].astype(bool)].sort_values("timestamp_ms")
    if len(g) < 3:
        return pd.DataFrame(columns=["t_s", "intensity"])
    aspect = (g["image_width"] / g["image_height"]).to_numpy()
    torso = np.hypot((g[["pose_11_x", "pose_12_x"]].mean(axis=1) - g[["pose_23_x", "pose_24_x"]].mean(axis=1)) * aspect,
                     g[["pose_11_y", "pose_12_y"]].mean(axis=1) - g[["pose_23_y", "pose_24_y"]].mean(axis=1)).clip(lower=0.02).to_numpy()
    t = g["timestamp_ms"].to_numpy() / 1000.0
    moves = []
    for i in LIMBS:
        x = g[f"pose_{i}_x"].to_numpy() * aspect
        y = g[f"pose_{i}_y"].to_numpy()
        moves.append(np.hypot(np.diff(x), np.diff(y)))
    dt = np.diff(t)
    speed = np.median(np.vstack(moves), axis=0) / ((torso[1:] + torso[:-1]) / 2) / np.where(dt > 0, dt, np.nan)
    out = pd.DataFrame({"t_s": t[1:], "intensity": speed}).dropna()
    window = max(1, int(round(window_s / max(np.median(dt), 1e-3))))
    out["intensity"] = out["intensity"].rolling(window, center=True, min_periods=1).median()
    return out


def _blur_outside(frame: np.ndarray, rows: pd.DataFrame, pad_x: float = 0.03, pad_y: float = 0.07) -> np.ndarray:
    """Blur everything except the tracked athletes, so bystanders in the background are not identifiable."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for _, row in rows.iterrows():
        xs = np.array([row[f"pose_{i}_x"] for i in range(33)]) * w
        ys = np.array([row[f"pose_{i}_y"] for i in range(33)]) * h
        px, py = pad_x * w, pad_y * h
        cv2.rectangle(mask, (int(xs.min() - px), int(ys.min() - py)), (int(xs.max() + px), int(ys.max() + py)), 255, -1)
    k = max(31, (min(h, w) // 12) | 1)
    blurred = cv2.GaussianBlur(frame, (k, k), 0)
    return np.where(mask[:, :, None] == 255, frame, blurred)


def render_clip(video: Path, tracked: pd.DataFrame, track_ids: list[int], out_mp4: Path, max_seconds: float = 15.0,
                blur_bystanders: bool = True) -> Path:
    """Overlay each track's skeleton in its own colour; detections are held for up to 0.3 s between samples.
    With blur_bystanders (default) everything outside the tracked athletes is blurred; frames with no
    recent detection are blurred entirely."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 720 / h if h > 720 else 1.0
    size = (int(w * scale), int(h * scale))
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    by_frame = {f: g for f, g in tracked[tracked["track_id"].isin(track_ids)].groupby("frame_index")}
    sampled = sorted(by_frame)
    last, idx = None, 0
    for frame_index in range(int(fps * max_seconds)):
        ok, frame = cap.read()
        if not ok:
            break
        while idx < len(sampled) and sampled[idx] <= frame_index:
            last, idx = sampled[idx], idx + 1
        frame = cv2.resize(frame, size)
        recent = last is not None and frame_index - last <= int(0.3 * fps)
        if blur_bystanders:
            frame = _blur_outside(frame, by_frame[last]) if recent else cv2.GaussianBlur(frame, (51, 51), 0)
        if recent:
            for _, row in by_frame[last].iterrows():
                colour = COLOURS[track_ids.index(int(row["track_id"])) % len(COLOURS)]
                pts = {i: (int(row[f"pose_{i}_x"] * size[0]), int(row[f"pose_{i}_y"] * size[1])) for i in range(33)}
                for a, b in EDGES:
                    cv2.line(frame, pts[a], pts[b], colour, 3, cv2.LINE_AA)
                for i in LIMBS:
                    cv2.circle(frame, pts[i], 6, colour, -1, cv2.LINE_AA)
                label = f"Athlete {'AB'[track_ids.index(int(row['track_id']))] if track_ids.index(int(row['track_id'])) < 2 else row['track_id']}"
                cv2.putText(frame, label, (pts[0][0] - 50, max(30, pts[0][1] - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()
    return out_mp4


def _timeline_png(timelines: dict[str, pd.DataFrame], out_png: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=120)
    for (label, tl), colour in zip(timelines.items(), ["#ff8c00", "#ffc83c", "#78dc78"]):
        ax.plot(tl["t_s"], tl["intensity"], label=label, color=colour, linewidth=2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("wrist/ankle speed\n(torso lengths / s)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def build_card(video_dir: Path, video: Path, out_dir: Path, max_tracks: int = 2) -> dict:
    tracked = pd.read_csv(video_dir / "tracked_landmarks.csv")
    counts = tracked.groupby("track_id").size().sort_values(ascending=False)
    track_ids = [int(t) for t in counts.index[:max_tracks]]
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = render_clip(video, tracked, track_ids, out_dir / "tracking_demo.mp4")
    timelines = {f"Athlete {'AB'[i]}": intensity_timeline(tracked, t) for i, t in enumerate(track_ids)}
    png = _timeline_png(timelines, out_dir / "intensity_timeline.png")
    guard = video_dir / "guard_proxy_tracked.csv"
    guard_df = pd.read_csv(guard) if guard.exists() else pd.DataFrame()
    rows = []
    for i, t in enumerate(track_ids):
        g = tracked[tracked["track_id"] == t]
        gp = guard_df[(guard_df.get("track_id") == t)] if len(guard_df) else pd.DataFrame()
        valid = gp["guard_proxy_both_hands_near_head"].dropna() if len(gp) else pd.Series(dtype=float)
        tl = timelines[f"Athlete {'AB'[i]}"]
        rows.append({"athlete": f"Athlete {'AB'[i]}", "visible_s": round((g["timestamp_ms"].max() - g["timestamp_ms"].min()) / 1000, 1),
                     "hands_near_head": f"{valid.mean():.0%} of {len(valid)} measurable frames" if len(valid) else "not measurable",
                     "median_intensity": round(float(tl["intensity"].median()), 2) if len(tl) else None})
    duration = tracked["timestamp_ms"].max() / 1000
    img = base64.b64encode(png.read_bytes()).decode()
    table = "".join(f"<tr><td>{r['athlete']}</td><td>{r['visible_s']} s</td><td>{html.escape(r['hands_near_head'])}</td><td>{r['median_intensity']}</td></tr>" for r in rows)
    card = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QMASE session card</title><style>
body{{margin:0;background:#0f1724;color:#e8eef7;font-family:Segoe UI,system-ui,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:28px 18px}} h1{{font-size:26px;margin:0 0 4px}} .sub{{color:#9fb0c6;margin:0 0 20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:18px}}
.k{{background:#16233a;border:1px solid #263a5a;border-radius:12px;padding:14px}} .k b{{display:block;font-size:24px}} .k span{{color:#9fb0c6;font-size:14px}}
table{{width:100%;border-collapse:collapse;margin:8px 0 18px}} td,th{{padding:8px;border-bottom:1px solid #263a5a;text-align:left}}
img,video{{width:100%;border-radius:10px;background:#000}} .note{{color:#9fb0c6;font-size:14px;line-height:1.5}}
</style></head><body><main>
<h1>QMASE session card</h1><p class="sub">{html.escape(video.name)} · measured from pose estimation · no scores or grades</p>
<div class="grid"><div class="k"><b>{duration:.1f} s</b><span>footage analysed</span></div>
<div class="k"><b>{len(track_ids)}</b><span>athletes tracked separately</span></div>
<div class="k"><b>{len(tracked):,}</b><span>pose detections</span></div></div>
<h2>Tracking</h2><video src="tracking_demo.mp4" controls muted loop playsinline></video>
<h2>Movement intensity</h2><img alt="Wrist and ankle speed over time for each athlete" src="data:image/png;base64,{img}">
<table><tr><th>Athlete</th><th>Visible</th><th>Both hands near head</th><th>Median intensity</th></tr>{table}</table>
<p class="note"><b>What this measures:</b> body keypoints from a single camera, linked into one track per athlete. Intensity is wrist and ankle speed in torso lengths per second, a 2-D camera measure of how much the limbs move. <b>What it does not measure:</b> power, effort, fatigue, technique quality or skill level. "Hands near head" is a simple geometric proxy, not a judgement of guard discipline.</p>
</main></body></html>"""
    (out_dir / "session_card.html").write_text(card, encoding="utf-8")
    (out_dir / "session_card.json").write_text(json.dumps({"video": str(video), "tracks": track_ids, "rows": rows}, indent=2), encoding="utf-8")
    return {"card": str(out_dir / "session_card.html"), "clip": str(clip), "rows": rows}
