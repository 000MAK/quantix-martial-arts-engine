import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from martial_arts_engine.identity import apply_choice, candidate_tracks, render_sheet, suggest


def _video(path: Path, frames: int = 6) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for _ in range(frames):
        img = np.zeros((120, 160, 3), np.uint8)
        img[20:100, 10:60] = (0, 0, 200)    # red shirt on the left
        img[20:100, 100:150] = (200, 0, 0)  # blue shirt on the right
        writer.write(img)
    writer.release()


def _rows(track_id, cx, frames, uncertain=False):
    rows = []
    for f in range(frames):
        rec = {"frame_index": f, "timestamp_ms": f * 100, "person_index": 0, "track_id": track_id, "identity_uncertain": uncertain}
        for i in range(33):
            rec[f"pose_{i}_x"], rec[f"pose_{i}_y"], rec[f"pose_{i}_visibility"] = cx, 0.5, 0.9
        for i, (dx, dy) in zip((11, 12, 23, 24), ((-0.1, -0.25), (0.1, -0.25), (-0.1, 0.25), (0.1, 0.25))):
            rec[f"pose_{i}_x"], rec[f"pose_{i}_y"] = cx + dx, 0.5 + dy
        rows.append(rec)
    return rows


class IdentityTests(unittest.TestCase):
    def test_candidates_ranked_and_small_tracks_dropped(self):
        t = pd.DataFrame(_rows(1, 0.2, 6) + _rows(2, 0.78, 8) + _rows(3, 0.5, 2))
        self.assertEqual(candidate_tracks(t), [2, 1])

    def test_sheet_choice_and_suggestion(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _video(d / "v.avi")
            t = pd.DataFrame(_rows(1, 0.22, 6) + _rows(2, 0.78, 6) + _rows(2, 0.78, 1, uncertain=True))
            t.to_csv(d / "tracked_landmarks.csv", index=False)
            self.assertEqual(render_sheet(d, d / "v.avi")["candidates"], 2)
            self.assertTrue((d / "identity" / "sheet.jpg").exists())
            info = json.loads((d / "identity" / "candidates.json").read_text())
            red = next(c for c in info["candidates"] if c["track_id"] == 1)
            result = apply_choice(d, red["letter"])
            self.assertEqual(result["me_track_id"], 1)
            mine = pd.read_csv(d / "my_landmarks.csv")
            self.assertTrue((mine["track_id"] == 1).all() and not mine["identity_uncertain"].any())
            # earlier pick of the red shirt suggests the red track again
            letter, sim = suggest(info["candidates"], [np.asarray(red["torso_hist"])])
            self.assertEqual(letter, red["letter"])
            self.assertGreater(sim, 0.9)

    def test_not_me(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _video(d / "v.avi")
            pd.DataFrame(_rows(1, 0.22, 6)).to_csv(d / "tracked_landmarks.csv", index=False)
            render_sheet(d, d / "v.avi")
            self.assertEqual(apply_choice(d, None), {"me_track_id": None, "my_rows": 0})
            self.assertFalse((d / "my_landmarks.csv").exists())


if __name__ == "__main__":
    unittest.main()
