"""Semi-Markov dwell-time model for observed position episodes.

A semi-Markov process separates *where* the next position goes (an embedded
jump chain without self-loops) from *how long* a position is held (a dwell-time
distribution). The question tested here: is holding time memoryless
(exponential, i.e. a plain continuous-time Markov chain), or does the hazard of
leaving change with time already spent in the position (Weibull shape k != 1)?
k > 1 means leaving becomes more likely the longer a position is held, which is
what a minimum hold time would produce.

Censoring: the last episode of each sequence is right-censored (recording ended).
The first episode is left-censored (its start is unknown) and is excluded from
fitting. Durations come from sampled annotation frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .hawkes import VICOS_SESSION_VIEWS, _registry_allows


def position_episodes(observations: pd.DataFrame, views: tuple[str, ...] = VICOS_SESSION_VIEWS) -> pd.DataFrame:
    data = observations[observations["label_status"] == "labeled"].copy()
    data["sequence_id"] = data["sequence_id"].astype(str).str.zfill(2)
    data = data[data["sequence_id"].isin(views)].sort_values(["sequence_id", "frame"])
    rows = []
    for seq, g in data.groupby("sequence_id", sort=True):
        t = g["timestamp_ms_approx"].to_numpy(dtype=float) / 1000.0
        pos = g["position"].astype(str).to_numpy()
        starts = np.concatenate([[0], np.flatnonzero(pos[1:] != pos[:-1]) + 1])
        for j, s in enumerate(starts):
            last = j == len(starts) - 1
            end_time = t[-1] if last else t[starts[j + 1]]
            rows.append({
                "sequence_id": seq, "position": pos[s],
                "next_position": None if last else pos[starts[j + 1]],
                "start_s": t[s], "duration_s": end_time - t[s],
                "observed": not last, "left_censored": j == 0,
            })
    return pd.DataFrame(rows)


def jump_chain(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    moves = episodes.dropna(subset=["next_position"])
    states = sorted(set(episodes["position"]) | set(moves["next_position"]))
    counts = pd.crosstab(moves["position"], moves["next_position"]).reindex(index=states, columns=states, fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return counts, probs


def _weibull_loglik(k: float, scales: np.ndarray, durations: np.ndarray, observed: np.ndarray, group: np.ndarray) -> float:
    lam = scales[group]
    z = durations / lam
    log_surv = -(z ** k)
    log_pdf = np.log(k) - np.log(lam) + (k - 1.0) * np.log(z) + log_surv
    return float(np.sum(np.where(observed, log_pdf, log_surv)))


def fit_dwell_models(durations: np.ndarray, observed: np.ndarray, group: np.ndarray | None = None) -> dict:
    """Censored maximum-likelihood fits; group gives each sequence its own scale/rate."""
    durations = np.asarray(durations, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    keep = durations > 0
    durations, observed = durations[keep], observed[keep]
    group = np.zeros(len(durations), dtype=int) if group is None else np.unique(np.asarray(group)[keep], return_inverse=True)[1]
    n_groups = int(group.max()) + 1

    # Exponential with one rate per group has a closed form.
    events = np.bincount(group, weights=observed.astype(float), minlength=n_groups)
    exposure = np.bincount(group, weights=durations, minlength=n_groups)
    rates = np.where(events > 0, events / exposure, 0.0)
    exp_ll = float(np.sum(np.where(events > 0, events * np.log(np.where(rates > 0, rates, 1.0)), 0.0) - rates * exposure))

    def neg(x: np.ndarray) -> float:
        return -_weibull_loglik(float(np.exp(x[0])), np.exp(x[1:]), durations, observed, group)

    start = np.concatenate([[0.0], np.log(np.where(rates > 0, 1.0 / np.maximum(rates, 1e-9), exposure.clip(min=1.0)))])
    best = optimize.minimize(neg, start, method="L-BFGS-B", bounds=[(-3.0, 3.0)] + [(-5.0, 12.0)] * n_groups)
    k = float(np.exp(best.x[0]))
    wb_ll = -float(best.fun)

    # Profile-likelihood 95% interval for the shared shape k.
    def profile(log_k: float) -> float:
        inner = optimize.minimize(lambda s: -_weibull_loglik(float(np.exp(log_k)), np.exp(s), durations, observed, group),
                                  best.x[1:], method="L-BFGS-B", bounds=[(-5.0, 12.0)] * n_groups)
        return -float(inner.fun)

    cutoff = wb_ll - stats.chi2.ppf(0.95, 1) / 2.0
    grid = np.linspace(best.x[0] - 2.0, best.x[0] + 2.0, 81)
    inside = [g for g in grid if profile(g) >= cutoff]
    ci = (float(np.exp(min(inside))), float(np.exp(max(inside)))) if inside else (None, None)

    lr = 2.0 * (wb_ll - exp_ll)
    return {
        "episodes_used": int(len(durations)), "observed_exits": int(observed.sum()),
        "right_censored": int((~observed).sum()), "groups": n_groups,
        "exponential": {"loglik": exp_ll, "aic": 2 * n_groups - 2 * exp_ll},
        "weibull": {"shape_k": k, "shape_k_95ci_profile": ci, "scales_s": np.exp(best.x[1:]).tolist(),
                    "loglik": wb_ll, "aic": 2 * (n_groups + 1) - 2 * wb_ll, "converged": bool(best.success)},
        "likelihood_ratio_stat": lr, "likelihood_ratio_p": float(stats.chi2.sf(max(lr, 0.0), 1)),
    }


def simulate_semi_markov(jump_probs: pd.DataFrame, shape_k: float, scale_s: float, start: str,
                         horizon_s: float, seed: int = 0) -> pd.DataFrame:
    """Simulate a path: Weibull(shape_k, scale_s) holding times and jumps from jump_probs."""
    rng = np.random.default_rng(seed)
    states = list(jump_probs.index)
    rows, t, state = [], 0.0, start
    while t < horizon_s:
        hold = scale_s * rng.weibull(shape_k)
        rows.append({"position": state, "start_s": t, "duration_s": min(hold, horizon_s - t), "observed": t + hold < horizon_s})
        t += hold
        p = jump_probs.loc[state].to_numpy(dtype=float)
        if p.sum() == 0:
            break
        state = states[int(rng.choice(len(states), p=p / p.sum()))]
    return pd.DataFrame(rows)


def analyze_vicos_semi_markov(observations_csv: Path, registry: Path, out_dir: Path) -> dict:
    _registry_allows(registry, "vicos_bjj_positions", "bjj_position_transition_analysis")
    episodes = position_episodes(pd.read_csv(observations_csv, dtype={"sequence_id": str}))
    usable = episodes[~episodes["left_censored"]]
    counts, probs = jump_chain(episodes)
    shared = fit_dwell_models(usable["duration_s"].to_numpy(), usable["observed"].to_numpy())
    per_session = fit_dwell_models(usable["duration_s"].to_numpy(), usable["observed"].to_numpy(), usable["sequence_id"].to_numpy())
    by_state = []
    for state, g in usable.groupby("position", sort=True):
        obs = g.loc[g["observed"], "duration_s"]
        by_state.append({"position": state, "episodes": int(len(g)), "observed_exits": int(len(obs)),
                         "median_observed_s": float(obs.median()) if len(obs) else None,
                         "min_observed_s": float(obs.min()) if len(obs) else None})
    summary = {
        "dataset_id": "vicos_bjj_positions", "license_id": "CC-BY-NC-SA-4.0",
        "task": "bjj_position_transition_analysis", "is_research_finding": False,
        "sessions_used": list(VICOS_SESSION_VIEWS), "episodes_total": int(len(episodes)),
        "episodes_excluded_left_censored": int(episodes["left_censored"].sum()),
        "dwell_models": {"shared_scale": shared, "per_session_scale": per_session},
        "primary_model": "per_session_scale",
        "interpretation_guide": "Weibull shape k > 1: the hazard of leaving rises with time held (consistent with a minimum hold time); k = 1: memoryless (exponential / Markov); k < 1: leaving becomes less likely the longer a position is held. The shared-scale fit mixes sessions with very different rates, which biases k downward.",
        "by_position": by_state,
        "limitations": [
            "Two athletes and six sparring sessions; exploratory, not generalizable to BJJ in general.",
            "One camera view per session; durations come from sampled annotation frames.",
            "First episode per session excluded (unknown start); last episode right-censored.",
            "Position-specific dwell models are not fitted: too few complete episodes per position.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(out_dir / "position_episodes.csv", index=False)
    counts.to_csv(out_dir / "jump_chain_counts.csv")
    probs.to_csv(out_dir / "jump_chain_probabilities.csv")
    (out_dir / "semi_markov_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
