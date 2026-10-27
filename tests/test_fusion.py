import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from martial_arts_engine.fusion import align_streams, estimate_offset, fuse_from_manifest


class FusionTests(unittest.TestCase):
    def test_nearest_join_with_tolerance_and_lag(self):
        pose = pd.DataFrame({"timestamp_s": [0.0, 1.0, 2.0, 3.0], "wrist_x": [1, 2, 3, 4]})
        hr = pd.DataFrame({"timestamp_s": [0.02, 1.9, 10.0], "bpm": [80, 95, 120]})
        fused = align_streams(pose, {"hr": hr}, tolerance_s=0.15)
        self.assertEqual(fused["hr_bpm"].iloc[0], 80)
        self.assertTrue(np.isnan(fused["hr_bpm"].iloc[1]))
        self.assertEqual(fused["hr_bpm"].iloc[2], 95)
        self.assertTrue(np.isnan(fused["hr_bpm"].iloc[3]))
        self.assertAlmostEqual(fused["hr_lag_s"].iloc[2], -0.1, places=9)

    def test_offset_is_applied(self):
        pose = pd.DataFrame({"timestamp_s": [5.0], "x": [1]})
        audio = pd.DataFrame({"timestamp_s": [2.0], "rms": [0.7]})
        fused = align_streams(pose, {"audio": audio}, tolerance_s=0.01, offsets_s={"audio": 3.0})
        self.assertEqual(fused["audio_rms"].iloc[0], 0.7)

    def test_estimate_offset_recovers_clock_shift(self):
        rng = np.random.default_rng(0)
        video_events = np.sort(rng.uniform(0, 300, 25))
        audio_events = video_events - 1.37 + rng.normal(0, 0.01, 25)
        est = estimate_offset(video_events, audio_events, max_lag_s=5.0)
        self.assertAlmostEqual(est["offset_s"], 1.37, delta=0.02)
        self.assertEqual(est["matched_events"], 25)

    def test_rejects_duplicates_and_missing_consent(self):
        with self.assertRaises(ValueError):
            align_streams(pd.DataFrame({"timestamp_s": [0.0, 0.0]}), {}, 0.1)
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame({"timestamp_s": [0.0, 1.0], "x": [1, 2]}).to_csv(Path(tmp) / "pose.csv", index=False)
            pd.DataFrame({"timestamp_s": [0.0], "bpm": [70]}).to_csv(Path(tmp) / "hr.csv", index=False)
            base = {"name": "pose", "path": "pose.csv", "source_id": "own", "license_id": "own", "consent_status": "granted"}
            hr = {"name": "hr", "path": "hr.csv", "source_id": "own", "license_id": "own", "consent_status": "pending"}
            manifest = Path(tmp) / "m.json"
            manifest.write_text(json.dumps({"tolerance_s": 0.1, "base": base, "streams": [hr]}), encoding="utf-8")
            with self.assertRaises(PermissionError):
                fuse_from_manifest(manifest, Path(tmp) / "out.csv")
            hr["consent_status"] = "granted"
            manifest.write_text(json.dumps({"tolerance_s": 0.1, "base": base, "streams": [hr]}), encoding="utf-8")
            result = fuse_from_manifest(manifest, Path(tmp) / "out.csv")
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["match_coverage"]["hr"], 0.5)


if __name__ == "__main__":
    unittest.main()
