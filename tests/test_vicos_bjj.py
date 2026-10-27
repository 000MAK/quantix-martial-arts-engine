import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from martial_arts_engine.vicos_bjj import (
    analyze_vicos_positions,
    read_vicos_annotations,
    write_vicos_analysis,
)


class VicosBjjTests(unittest.TestCase):
    def test_reads_source_labels_and_extracts_observed_transitions(self):
        annotations = [
            {"image": "0100001", "frame": 1, "position": "closed_guard1"},
            {"image": "0100002", "frame": 2, "position": "closed_guard1"},
            {"image": "0100010", "frame": 10, "position": "side_control2"},
            {"Image": "0200001.png", "Frame": 1, "Position": "mount1"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.json"
            path.write_text(json.dumps(annotations), encoding="utf-8")
            observations = read_vicos_annotations(path)

        self.assertEqual(observations["sequence_id"].tolist(), [
            "vicos-01", "vicos-01", "vicos-01", "vicos-02"
        ])
        self.assertEqual(observations["state"].tolist(), [
            "closed_guard", "closed_guard", "side_control", "mount"
        ])
        result = analyze_vicos_positions(observations)
        self.assertEqual(len(result["transitions"]), 1)
        self.assertEqual(result["transitions"].iloc[0]["from_state"], "closed_guard")
        self.assertEqual(result["transitions"].iloc[0]["to_state"], "side_control")
        self.assertAlmostEqual(
            result["episodes"].iloc[0]["observed_label_span_seconds"], 1 / 30
        )
        self.assertEqual(
            result["episodes"].iloc[-1]["terminal_right_censored"], True
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = write_vicos_analysis(observations, Path(temp_dir) / "analysis")
            self.assertTrue(outputs["transition_matrix"].is_file())
            metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["analysis"], "descriptive_bjj_position_transitions")
            self.assertTrue(any("not validated physical dwell times" in item
                                for item in metadata["interpretation"]))

    def test_rejects_duplicate_sequence_frame_annotations(self):
        observations = pd.DataFrame([
            {"sequence_id": "vicos-01", "frame_index": 1, "state": "mount"},
            {"sequence_id": "vicos-01", "frame_index": 1, "state": "back"},
        ])
        with self.assertRaisesRegex(ValueError, "exactly one position label"):
            analyze_vicos_positions(observations)

    def test_rejects_unknown_position_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.json"
            path.write_text(json.dumps([
                {"Image": "0100001.png", "Frame": 1, "Position": "unknown"}
            ]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unrecognized ViCoS"):
                read_vicos_annotations(path)


if __name__ == "__main__":
    unittest.main()
