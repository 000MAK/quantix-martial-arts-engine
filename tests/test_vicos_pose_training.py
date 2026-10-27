import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from martial_arts_engine.vicos_pose_training import (
    TRAINING_TASK,
    _grouped_split,
    _verify_training_approval,
    load_vicos_pose_data,
)


class VicosPoseTrainingTests(unittest.TestCase):
    def test_loads_pose_slots_and_keeps_source_sequence_groups(self):
        annotation = [{
            "image": "0300001",
            "frame": 1,
            "position": "mount1",
            "pose1": [[1, 2, 0.9]] * 17,
        }, {
            "image": "0300002",
            "frame": 2,
            "position": "mount2",
            "pose2": [[3, 4, 0.8]] * 17,
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "annotations.json"
            path.write_text(json.dumps(annotation), encoding="utf-8")
            features, labels, groups = load_vicos_pose_data(path)

        self.assertEqual(features.shape, (2, 104))
        self.assertEqual(labels.tolist(), ["mount1", "mount2"])
        self.assertEqual(groups.tolist(), ["03", "03"])
        self.assertEqual(features[:, 102:].tolist(), [[1.0, 0.0], [0.0, 1.0]])

    def test_grouped_split_has_no_sequence_leakage_and_all_classes_in_train(self):
        labels = np.repeat(["a", "b"], 80)
        groups = np.repeat(np.tile([f"{i:02}" for i in range(8)], 2), 10)
        train, val, test, split_groups = _grouped_split(labels, groups)
        self.assertFalse(set(groups[train]) & set(groups[val]))
        self.assertFalse(set(groups[train]) & set(groups[test]))
        self.assertFalse(set(groups[val]) & set(groups[test]))
        self.assertEqual(set(labels[train]), {"a", "b"})
        self.assertEqual(set(split_groups), {"train", "validation", "test"})

    def test_registry_requires_task_specific_approval(self):
        registry = {
            "datasets": [{
                "dataset_id": "vicos_bjj_positions",
                "status": "approved_for_noncommercial_analysis_only",
                "license_id": "CC-BY-NC-SA-4.0",
                "training_approval": {
                    "approved_tasks": [TRAINING_TASK],
                    "reviewer": "test",
                    "reviewed_at": "2026-10-27",
                    "evidence_ref": "https://example.test/dataset",
                    "access_evidence_ref": "https://example.test/annotations",
                    "noncommercial_use_confirmed": True,
                },
            }]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "registry.json"
            path.write_text(json.dumps(registry), encoding="utf-8")
            self.assertEqual(_verify_training_approval(path)["dataset_id"], "vicos_bjj_positions")
            registry["datasets"][0]["training_approval"]["approved_tasks"] = []
            path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaises(PermissionError):
                _verify_training_approval(path)

if __name__ == "__main__":
    unittest.main()
