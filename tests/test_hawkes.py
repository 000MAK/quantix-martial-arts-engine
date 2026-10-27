import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from martial_arts_engine.hawkes import (
    analyze_vicos_hawkes,
    fit_hawkes,
    hawkes_loglik,
    poisson_loglik,
    simulate_hawkes,
    vicos_change_events,
)


class HawkesTests(unittest.TestCase):
    def test_zero_branching_matches_poisson_likelihood(self):
        seq = [(np.array([1.0, 4.0, 9.0]), 0.0, 10.0)]
        ll_p, mu = poisson_loglik(seq)
        self.assertAlmostEqual(hawkes_loglik(mu, 0.0, 1.0, seq), ll_p, places=9)

    def test_recovers_simulated_self_excitation(self):
        sequences = [(simulate_hawkes(0.05, 0.6, 0.5, 3000.0, seed=s), 0.0, 3000.0) for s in range(4)]
        fit = fit_hawkes(sequences, n_starts=6)
        self.assertAlmostEqual(fit["branching_ratio"], 0.6, delta=0.15)
        self.assertAlmostEqual(fit["mu_per_s"], 0.05, delta=0.02)

    def test_poisson_data_gives_low_branching(self):
        rng = np.random.default_rng(3)
        sequences = [(np.sort(rng.uniform(0, 3000.0, 150)), 0.0, 3000.0) for _ in range(4)]
        fit = fit_hawkes(sequences, n_starts=6)
        self.assertLess(fit["branching_ratio"], 0.2)

    def test_rate_differences_between_sequences_are_not_self_excitation(self):
        rng = np.random.default_rng(5)
        sequences = [(np.sort(rng.uniform(0, 600.0, 120)), 0.0, 600.0)] + [
            (np.sort(rng.uniform(0, 600.0, 3)), 0.0, 600.0) for _ in range(4)
        ]
        shared = fit_hawkes(sequences, n_starts=6)
        per_seq = fit_hawkes(sequences, n_starts=6, per_sequence_mu=True)
        self.assertGreater(shared["branching_ratio"], 0.5)
        self.assertLess(per_seq["branching_ratio"], 0.2)

    def test_change_events_and_registry_gate(self):
        obs = pd.DataFrame({
            "sequence_id": ["00"] * 5 + ["03"] * 3,
            "frame": [1, 2, 3, 4, 5, 1, 2, 3],
            "timestamp_ms_approx": [0, 1000, 2000, 3000, 4000, 0, 1000, 2000],
            "position": ["standing", "standing", "mount1", "mount1", "standing", "turtle1", "turtle1", "turtle1"],
            "label_status": ["labeled"] * 8,
        })
        events, sequences, ids = vicos_change_events(obs)
        self.assertEqual(ids, ["00", "03"])
        self.assertEqual(events["time_s"].tolist(), [2.0, 4.0])
        self.assertEqual(len(sequences[1][0]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "obs.csv"
            obs.to_csv(csv, index=False)
            registry = Path(tmp) / "registry.json"
            registry.write_text(json.dumps({"datasets": [{"dataset_id": "vicos_bjj_positions", "analysis_approval": {"approved_tasks": []}}]}), encoding="utf-8")
            with self.assertRaises(PermissionError):
                analyze_vicos_hawkes(csv, registry, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
