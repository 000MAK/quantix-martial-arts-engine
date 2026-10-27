import unittest

import numpy as np
import pandas as pd

from martial_arts_engine.demo_card import _blur_outside, intensity_timeline


def _track(speed_per_s, n=20, dt_ms=200):
    rows = []
    for k in range(n):
        rec = {"track_id": 1, "identity_uncertain": False, "timestamp_ms": k * dt_ms, "image_width": 100, "image_height": 100}
        for i in range(33):
            rec[f"pose_{i}_x"], rec[f"pose_{i}_y"] = 0.5, 0.5
        rec["pose_11_y"] = rec["pose_12_y"] = 0.3   # shoulders
        rec["pose_23_y"] = rec["pose_24_y"] = 0.5   # hips: torso length 0.2
        for i in (15, 16, 27, 28):                  # limbs move right at speed_per_s frame-heights per second
            rec[f"pose_{i}_x"] = 0.1 + speed_per_s * k * dt_ms / 1000
        rows.append(rec)
    return pd.DataFrame(rows)


class DemoCardTests(unittest.TestCase):
    def test_intensity_in_torso_lengths_per_second(self):
        tl = intensity_timeline(_track(0.1), 1)
        self.assertAlmostEqual(float(tl["intensity"].median()), 0.5, places=6)  # 0.1 / 0.2
        self.assertTrue(intensity_timeline(_track(0.1), 99).empty)

    def test_blur_keeps_athlete_and_changes_background(self):
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        row = _track(0.0, n=1).iloc[[0]].copy()
        for i in range(33):
            row[f"pose_{i}_x"], row[f"pose_{i}_y"] = 0.5, 0.5
        out = _blur_outside(frame, row, pad_x=0.05, pad_y=0.05)
        self.assertTrue(np.array_equal(out[50, 50], frame[50, 50]))   # athlete area untouched
        self.assertFalse(np.array_equal(out[5:15, 5:15], frame[5:15, 5:15]))  # background blurred


if __name__ == "__main__":
    unittest.main()
