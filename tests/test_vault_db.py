import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from martial_arts_engine.vault_db import ingest_outputs, query


class VaultDbTests(unittest.TestCase):
    def _outputs(self, root: Path) -> None:
        (root / "vicos_semi_markov").mkdir(parents=True)
        (root / "vicos_semi_markov" / "semi_markov_summary.json").write_text(
            json.dumps({"dataset_id": "vicos_bjj_positions", "license_id": "CC-BY-NC-SA-4.0", "task": "t"}), encoding="utf-8")
        pd.DataFrame({"sequence_id": ["00", "00"], "position": ["standing", "mount1"], "next_position": ["mount1", None],
                      "start_s": [0.0, 3.0], "duration_s": [3.0, 2.0], "observed": [True, False],
                      "left_censored": [True, False]}).to_csv(root / "vicos_semi_markov" / "position_episodes.csv", index=False)

    def test_ingest_is_idempotent_and_queryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, db = Path(tmp) / "outputs", Path(tmp) / "vault.db"
            self._outputs(root)
            first = ingest_outputs(db, root)
            self.assertEqual(first["semi_markov"]["detail_rows"], 2)
            self.assertEqual(first["hawkes"], "not_found")
            self.assertEqual(ingest_outputs(db, root)["semi_markov"], "already_ingested")
            result = query(db, "SELECT position, observed FROM position_episodes ORDER BY start_s")
            self.assertEqual(result["position"].tolist(), ["standing", "mount1"])
            self.assertEqual(result["observed"].tolist(), [1, 0])
            self.assertEqual(query(db, "SELECT COUNT(*) AS n FROM analysis_runs")["n"].iloc[0], 1)

    def test_query_connection_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, db = Path(tmp) / "outputs", Path(tmp) / "vault.db"
            self._outputs(root)
            ingest_outputs(db, root)
            with self.assertRaises(Exception):
                query(db, "DELETE FROM analysis_runs")
            self.assertEqual(query(db, "SELECT COUNT(*) AS n FROM analysis_runs")["n"].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
