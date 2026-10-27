import unittest

import numpy as np

from martial_arts_engine.grapplemap_models import FAMILIES, features, label_rows


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(-0.6, 0.6, (2, 23, 3))
    p[..., 1] = rng.uniform(0.0, 1.6, (2, 23))
    return p


class GrappleMapModelTests(unittest.TestCase):
    def test_features_ignore_rotation_translation_and_player_order(self):
        p = _frame(1)
        q = p.copy()
        c, s = np.cos(0.9), np.sin(0.9)
        x, z = q[..., 0].copy(), q[..., 2].copy()
        q[..., 0], q[..., 2] = c * x - s * z + 0.4, s * x + c * z - 0.3
        self.assertTrue(np.allclose(features(p), features(q)))
        self.assertTrue(np.allclose(features(p), features(p[::-1])))

    def test_only_single_family_positions_are_labelled(self):
        nodes = [
            {"name": "mount high", "tags": ["mount", "top_kneeling"], "keyframe": _frame(2).tolist()},
            {"name": "mount to back", "tags": ["mount", "back"], "keyframe": _frame(3).tolist()},
            {"name": None, "tags": ["mount"], "keyframe": _frame(4).tolist()},
            {"name": "standing neutral", "tags": ["standing"], "keyframe": _frame(5).tolist()},
        ]
        X, y, groups, names = label_rows(nodes, FAMILIES)
        self.assertEqual(names, ["mount high", "standing neutral"])
        self.assertEqual([FAMILIES[i] for i in y], ["mount", "standing"])
        self.assertEqual(list(groups), ["mount", "standing"])


if __name__ == "__main__":
    unittest.main()
