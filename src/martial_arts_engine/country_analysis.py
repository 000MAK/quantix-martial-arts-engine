from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {
    "entry_id", "athlete_id", "discipline_id", "event_id", "competition_year", "era",
    "ruleset", "division", "sex_class", "competition_level", "represented_country_code",
    "nationality_country_code", "country_code_system", "placement", "medal", "bouts",
    "wins", "losses", "draws", "verification_status", "source_id", "license_id",
}
STRATA = ["discipline_id", "era", "ruleset", "division", "sex_class", "competition_level"]


def _event_bootstrap_rate(frame: pd.DataFrame, numerator: str, denominator: str, reps: int, seed: int = 42):
    clusters = [g[[numerator, denominator]].to_numpy(float) for _, g in frame.groupby("event_id")]
    clusters = [c for c in clusters if len(c)]
    if len(clusters) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(reps):
        chosen = rng.integers(0, len(clusters), len(clusters))
        sample = np.concatenate([clusters[i] for i in chosen], axis=0)
        denom = float(sample[:, 1].sum())
        if denom:
            estimates.append(float(sample[:, 0].sum() / denom))
    if not estimates:
        return None, None
    return float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))


def analyze_country_results(input_csv: Path, output_csv: Path, bootstrap_reps: int = 1000) -> pd.DataFrame:
    """Summarize results separately by represented country and recorded nationality.

    No composite country score is created. Historical country/entity codes are
    retained as supplied and never silently merged (for example, USSR -> RUS).
    """
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be at least 1")
    frame = pd.read_csv(input_csv)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("Country-results table missing columns: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("Country results contain no athlete-event rows")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError("Country rows require complete event, country, result and provenance fields")
    for col in ("entry_id", "athlete_id", "discipline_id", "event_id", "era", "ruleset", "division", "sex_class", "competition_level", "represented_country_code", "nationality_country_code", "country_code_system", "verification_status", "source_id", "license_id"):
        if frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Country-results field {col} cannot be blank")
    if not frame.verification_status.astype(str).str.lower().eq("verified").all():
        raise ValueError("Country analysis requires verification_status=verified for every row")
    for col in ("competition_year", "placement", "bouts", "wins", "losses", "draws"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if frame.duplicated("entry_id").any():
        raise ValueError("entry_id must identify one athlete entry in one event/discipline/ruleset/division")
    if (frame[["bouts", "wins", "losses", "draws"]] < 0).any().any():
        raise ValueError("Bout, win, loss and draw counts cannot be negative")
    if (frame.placement < 1).any() or (frame["competition_year"] < 1800).any():
        raise ValueError("Placement must be positive and competition_year must be a plausible year")
    for col in ("bouts", "wins", "losses", "draws", "placement", "competition_year"):
        if not np.equal(frame[col], np.floor(frame[col])).all():
            raise ValueError(f"{col} must contain integer values")
    if (frame.wins + frame.losses + frame.draws > frame.bouts).any():
        raise ValueError("wins + losses + draws cannot exceed bouts")
    allowed_medals = {"gold", "silver", "bronze", "none"}
    frame["medal"] = frame.medal.astype(str).str.strip().str.lower()
    if not set(frame.medal).issubset(allowed_medals):
        raise ValueError("medal must be gold, silver, bronze, or none")
    expected_place = {"gold": 1, "silver": 2, "bronze": 3}
    mismatched = frame.medal.map(expected_place).notna() & frame.medal.map(expected_place).ne(frame.placement)
    if mismatched.any():
        raise ValueError("Medal label conflicts with placement (gold=1, silver=2, bronze=3)")
    frame["medal_count"] = frame.medal.ne("none").astype(int)
    frame["gold_count"] = frame.medal.eq("gold").astype(int)
    frame["win_denominator"] = frame.wins + frame.losses + frame.draws
    frame["country_code_system"] = frame.country_code_system.astype(str).str.strip().str.upper()

    summaries = []
    for basis, country_col in (("represented_country", "represented_country_code"), ("nationality", "nationality_country_code")):
        for key, group in frame.groupby(STRATA + ["country_code_system", country_col], dropna=False, sort=True):
            dims = dict(zip(STRATA + ["country_code_system", "country_code"], key))
            entries = len(group)
            medals = int(group.medal_count.sum())
            wins = int(group.wins.sum())
            decisions = int(group.win_denominator.sum())
            medal_ci = _event_bootstrap_rate(group.assign(_medal_den=1), "medal_count", "_medal_den", bootstrap_reps)
            gold_ci = _event_bootstrap_rate(group.assign(_entry_den=1), "gold_count", "_entry_den", bootstrap_reps)
            win_ci = _event_bootstrap_rate(group, "wins", "win_denominator", bootstrap_reps)
            summaries.append({
                "country_basis": basis, **dims,
                "athlete_entries": entries,
                "unique_athletes": int(group.athlete_id.nunique()),
                "events": int(group.event_id.nunique()),
                "medals": medals,
                "gold_medals": int(group.gold_count.sum()),
                "medal_rate_pct": 100 * medals / entries,
                "medal_rate_event_cluster_boot95_low_pct": 100 * medal_ci[0] if medal_ci[0] is not None else None,
                "medal_rate_event_cluster_boot95_high_pct": 100 * medal_ci[1] if medal_ci[1] is not None else None,
                "gold_rate_pct": 100 * int(group.gold_count.sum()) / entries,
                "gold_rate_event_cluster_boot95_low_pct": 100 * gold_ci[0] if gold_ci[0] is not None else None,
                "gold_rate_event_cluster_boot95_high_pct": 100 * gold_ci[1] if gold_ci[1] is not None else None,
                "bouts": int(group.bouts.sum()),
                "wins": wins,
                "bout_win_rate_pct": 100 * wins / decisions if decisions else None,
                "bout_win_rate_event_cluster_boot95_low_pct": 100 * win_ci[0] if win_ci[0] is not None else None,
                "bout_win_rate_event_cluster_boot95_high_pct": 100 * win_ci[1] if win_ci[1] is not None else None,
                "interpretation": "Separate observed totals and rates; no composite country score. Country attribution depends on the declared code field and source.",
            })
    result = pd.DataFrame(summaries)
    result.sort_values(["country_basis"] + STRATA + ["gold_medals", "medals", "medal_rate_pct"], ascending=[True] + [True] * len(STRATA) + [False, False, False], inplace=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
