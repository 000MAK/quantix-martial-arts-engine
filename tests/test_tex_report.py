import json
import tempfile
import unittest
from pathlib import Path

from martial_arts_engine.tex_report import render_report, tex_escape, write_report

HAWKES = {
    "dataset_id": "vicos_bjj_positions", "task": "bjj_position_transition_analysis",
    "license_id": "CC-BY-NC-SA-4.0", "sessions_used": ["00", "11"], "position_changes": 96,
    "total_window_s": 1778.5, "primary_model": "per_session_baseline",
    "model_note": "Shared baseline is a trap.", "likelihood_ratio_note": "Boundary case.",
    "models": {"per_session_baseline": {
        "poisson": {"aic": 577.2, "ks_rescaled_p": 6.8e-05},
        "hawkes": {"aic": 581.2, "branching_ratio": 0.0, "excitation_timescale_s": 120.0, "ks_rescaled_p": 6.8e-05},
    }},
    "limitations": ["Two athletes & six sessions (50% from one)."],
}
SEMI = {
    "dataset_id": "vicos_bjj_positions", "task": "bjj_position_transition_analysis",
    "license_id": "CC-BY-NC-SA-4.0", "sessions_used": ["00", "11"], "primary_model": "per_session_scale",
    "interpretation_guide": "k > 1 means rising hazard.",
    "dwell_models": {"per_session_scale": {
        "episodes_used": 96, "observed_exits": 90, "likelihood_ratio_p": 3.35e-08,
        "exponential": {"aic": 520.7},
        "weibull": {"aic": 492.2, "shape_k": 1.6, "shape_k_95ci_profile": [1.38, 1.77]},
    }},
    "limitations": ["Two athletes & six sessions (50% from one)."],
}


class TexReportTests(unittest.TestCase):
    def test_escape(self):
        self.assertEqual(tex_escape("a_b & 50% #1"), r"a\_b \& 50\% \#1")

    def test_render_uses_only_supplied_values(self):
        tex = render_report(HAWKES, SEMI)
        for expected in ("577.2", "581.2", "492.2", "[1.38, 1.77]", "6.8e-05", "CC-BY-NC-SA-4.0", "Exploratory"):
            self.assertIn(expected, tex)
        self.assertEqual(tex.count(r"\item"), 1)  # duplicate limitations merged
        self.assertIn(r"50\%", tex)
        self.assertTrue(tex.strip().endswith(r"\end{document}"))

    def test_requires_some_input(self):
        with self.assertRaises(ValueError):
            render_report(None, None)

    def test_write_tex_without_compiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = Path(tmp) / "h.json"
            h.write_text(json.dumps(HAWKES), encoding="utf-8")
            result = write_report(Path(tmp) / "out", h, None, compile_pdf=False)
            self.assertTrue(Path(result["tex"]).exists())
            self.assertEqual(result["pdf_status"], "not_requested")


if __name__ == "__main__":
    unittest.main()
