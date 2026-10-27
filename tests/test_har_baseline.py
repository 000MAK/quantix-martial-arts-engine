import unittest

import numpy as np

from martial_arts_engine.har_baseline import check_subject_disjoint, model_card, train_and_evaluate


def _data(seed=0):
    rng = np.random.default_rng(seed)
    def split(subjects):
        X, y, s = [], [], []
        for subj in subjects:
            for label in (1, 2, 3):
                X.append(rng.normal(label * 2.0, 0.5, (20, 6)))
                y += [label] * 20
                s += [subj] * 20
        return np.vstack(X), np.array(y), np.array(s)
    Xtr, ytr, str_ = split(range(1, 11))
    Xte, yte, ste = split(range(11, 15))
    return {"X_train": Xtr, "y_train": ytr, "s_train": str_, "X_test": Xte, "y_test": yte, "s_test": ste,
            "labels": {1: "A", 2: "B", 3: "C"}}


class HarBaselineTests(unittest.TestCase):
    def test_overlapping_subjects_are_rejected(self):
        with self.assertRaises(ValueError):
            check_subject_disjoint(np.array([1, 2, 3]), np.array([3, 4]))

    def test_selection_uses_training_subjects_and_reports_per_subject(self):
        model, m = train_and_evaluate(_data(), folds=3)
        self.assertIn(m["model_selection"]["selected"], ("logistic_regression", "random_forest"))
        self.assertGreater(m["test"]["accuracy"], 0.95)
        self.assertEqual(set(m["test"]["per_subject_accuracy"]["by_subject"]), {11, 12, 13, 14})
        card = model_card(m)
        self.assertIn("license: cc-by-4.0", card)
        self.assertIn("Limitations", card)


if __name__ == "__main__":
    unittest.main()
