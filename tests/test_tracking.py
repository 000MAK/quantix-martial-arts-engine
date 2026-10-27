import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from martial_arts_engine.tracking import assign_tracks, track_summary, track_video_dir


def _person(frame, ts, person_index, cx, cy=0.5, scale=0.1):
    rng = np.random.default_rng(frame * 10 + person_index)
    offsets = rng.uniform(-1, 1, (33, 2)) * scale
    rec = {"video_id": "v", "frame_index": frame, "timestamp_ms": ts, "person_index": person_index,
           "image_width": 100, "image_height": 100}
    base = np.random.default_rng(int(cx * 1000)).uniform(-1, 1, (33, 2)) * scale  # stable body shape per person
    # realistic torso: shoulders above hips, torso length 0.2 frame heights
    base[11], base[12], base[23], base[24] = (-0.05, -0.15), (0.05, -0.15), (-0.04, 0.05), (0.04, 0.05)
    for i in range(33):
        rec[f"pose_{i}_x"] = cx + base[i, 0] + offsets[i, 0] * 0.05
        rec[f"pose_{i}_y"] = cy + base[i, 1] + offsets[i, 1] * 0.05
        rec[f"pose_{i}_z"] = 0.0
        rec[f"pose_{i}_visibility"] = 0.9
    return rec


def _two_people(frames=20, swap=True, gap_at=None):
    rows = []
    for f in range(frames):
        if gap_at and gap_at[0] <= f < gap_at[1]:
            continue
        left, right = 0.2 + 0.005 * f, 0.8 - 0.005 * f
        order = (right, left) if (swap and f % 2) else (left, right)
        rows += [_person(f, f * 100, 0, order[0]), _person(f, f * 100, 1, order[1])]
    return pd.DataFrame(rows)


class TrackingTests(unittest.TestCase):
    def test_swapped_person_index_keeps_identity(self):
        tracked = assign_tracks(_two_people())
        self.assertEqual(tracked["track_id"].nunique(), 2)
        for _, g in tracked.groupby("track_id"):
            hip_x = (g["pose_23_x"] + g["pose_24_x"]) / 2
            self.assertTrue((hip_x < 0.5).all() or (hip_x > 0.5).all())
        self.assertFalse(tracked["identity_uncertain"].any())

    def test_long_gap_starts_new_tracks_unless_stitched(self):
        people = _two_people(frames=40, swap=False, gap_at=(8, 30))  # 2.2 s gap
        self.assertEqual(assign_tracks(people, max_gap_ms=500, stitch_gap_ms=0)["track_id"].nunique(), 4)
        stitched = assign_tracks(people, max_gap_ms=500, stitch_gap_ms=4000)
        self.assertEqual(stitched["track_id"].nunique(), 2)
        for _, g in stitched.groupby("track_id"):  # each stitched track stays on one side
            hip_x = (g["pose_23_x"] + g["pose_24_x"]) / 2
            self.assertTrue((hip_x < 0.5).all() or (hip_x > 0.5).all())

    def test_overlap_is_flagged(self):
        rows = [_person(0, 0, 0, 0.5), _person(0, 0, 1, 0.52)]
        tracked = assign_tracks(pd.DataFrame(rows))
        self.assertTrue(tracked["identity_uncertain"].all())

    def test_video_dir_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            poses = _two_people(frames=6)
            poses.to_csv(d / "pose_landmarks.csv", index=False)
            pd.DataFrame({"timestamp_ms": poses["timestamp_ms"], "person_index": poses["person_index"],
                          "guard_proxy_both_hands_near_head": 1.0}).to_csv(d / "guard_proxy_frames.csv", index=False)
            result = track_video_dir(d)
            self.assertEqual(result["tracks"], 2)
            guard = pd.read_csv(d / "guard_proxy_tracked.csv")
            self.assertTrue(guard["track_id"].notna().all())
            self.assertEqual(len(track_summary(pd.read_csv(d / "tracked_landmarks.csv"))), 2)


if __name__ == "__main__":
    unittest.main()
