import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from martial_arts_engine.semi_markov import (
    analyze_vicos_semi_markov,
    fit_dwell_models,
    jump_chain,
    position_episodes,
    simulate_semi_markov,
)


def _obs():
    labels = ["standing"] * 3 + ["mount1"] * 2 + ["standing"] * 2
    return pd.DataFrame({
        "sequence_id": ["00"] * 7, "frame": range(1, 8),
        "timestamp_ms_approx": [0, 1000, 2000, 3000, 4000, 5000, 6000],
        "position": labels, "label_status": ["labeled"] * 7,
    })


class SemiMarkovTests(unittest.TestCase):
    def test_episodes_and_censoring(self):
        ep = position_episodes(_obs())
        self.assertEqual(ep["position"].tolist(), ["standing", "mount1", "standing"])
        self.assertEqual(ep["duration_s"].tolist(), [3.0, 2.0, 1.0])
        self.assertEqual(ep["observed"].tolist(), [True, True, False])
        self.assertEqual(ep["left_censored"].tolist(), [True, False, False])
        counts, probs = jump_chain(ep)
        self.assertEqual(counts.loc["standing", "mount1"], 1)
        self.assertEqual(probs.loc["mount1", "standing"], 1.0)
        self.assertEqual(np.diag(counts.to_numpy()).sum(), 0)

    def test_recovers_weibull_shape_with_censoring(self):
        rng = np.random.default_rng(1)
        d = 20.0 * rng.weibull(2.0, 400)
        cens = rng.uniform(0, 60, 400)
        fit = fit_dwell_models(np.minimum(d, cens), d <= cens)
        self.assertAlmostEqual(fit["weibull"]["shape_k"], 2.0, delta=0.3)
        lo, hi = fit["weibull"]["shape_k_95ci_profile"]
        self.assertTrue(lo < 2.0 < hi)
        self.assertLess(fit["likelihood_ratio_p"], 0.001)

    def test_exponential_data_is_not_called_weibull(self):
        rng = np.random.default_rng(2)
        fit = fit_dwell_models(rng.exponential(10.0, 400), np.ones(400, dtype=bool))
        lo, hi = fit["weibull"]["shape_k_95ci_profile"]
        self.assertTrue(lo < 1.0 < hi)

    def test_session_rate_mixture_biases_shared_shape_down(self):
        rng = np.random.default_rng(3)
        fast, slow = rng.exponential(2.0, 300), rng.exponential(60.0, 300)
        d = np.concatenate([fast, slow])
        group = np.array(["a"] * 300 + ["b"] * 300)
        shared = fit_dwell_models(d, np.ones(600, dtype=bool))
        per = fit_dwell_models(d, np.ones(600, dtype=bool), group)
        self.assertLess(shared["weibull"]["shape_k"], 0.8)
        self.assertAlmostEqual(per["weibull"]["shape_k"], 1.0, delta=0.12)

    def test_simulator_and_registry_gate(self):
        probs = pd.DataFrame([[0, 1], [1, 0]], index=["a", "b"], columns=["a", "b"], dtype=float)
        path = simulate_semi_markov(probs, 2.0, 5.0, "a", 200.0, seed=0)
        self.assertTrue((path["position"].to_numpy()[1:] != path["position"].to_numpy()[:-1]).all())
        self.assertAlmostEqual(path["start_s"].iloc[-1] + path["duration_s"].iloc[-1], 200.0, places=6)
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "obs.csv"
            _obs().to_csv(csv, index=False)
            registry = Path(tmp) / "registry.json"
            registry.write_text(json.dumps({"datasets": []}), encoding="utf-8")
            with self.assertRaises(PermissionError):
                analyze_vicos_semi_markov(csv, registry, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
