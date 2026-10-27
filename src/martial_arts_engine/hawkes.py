"""Exponential-kernel Hawkes (self-exciting) model for position-change timing.

Question answered: does one observed position change make further changes more
likely shortly afterwards (clustering / "momentum"), beyond a constant-rate
Poisson process?

Intensity:  lambda(t) = mu + n * beta * sum_{t_i < t} exp(-beta * (t - t_i))
  mu    baseline rate of changes (per second)
  n     branching ratio: expected number of changes directly triggered by one change
  beta  decay rate (per second); 1/beta is the excitation time scale

Limitations: a Hawkes fit cannot distinguish self-excitation from phases with a
higher underlying rate (an inhomogeneous Poisson process); event times come from
sampled annotation frames, so each change time is an upper bound on the true time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

# Decay rates outside this range describe frame-level noise (fast) or a constant
# offset over a whole session (slow) rather than excitation.
BETA_BOUNDS = (1.0 / 120.0, 5.0)
# One camera view per ViCoS sparring session (prefix groups 0-2, 3-5, 6-8, 9-10, 11-13, 14-15).
VICOS_SESSION_VIEWS = ("00", "03", "06", "09", "11", "14")

Sequence = tuple[np.ndarray, float, float]  # (event times, window start, window end) in seconds


def _unpack(x: np.ndarray) -> tuple[float, float, float]:
    mu = float(np.exp(x[0]))
    n = float(1.0 / (1.0 + np.exp(-x[1])))
    beta = float(np.exp(x[2]))
    return mu, n, beta


def hawkes_loglik(mu: float | np.ndarray, n: float, beta: float, sequences: list[Sequence]) -> float:
    """mu may be one shared baseline or one baseline per sequence."""
    mus = np.broadcast_to(np.asarray(mu, dtype=float), (len(sequences),))
    total = 0.0
    for mu, (times, start, end) in zip(mus, sequences):
        t = np.asarray(times, dtype=float) - start
        horizon = end - start
        if len(t) == 0:
            total -= mu * horizon
            continue
        excitation = np.zeros(len(t))
        decay = np.exp(-beta * np.diff(t))
        for i in range(1, len(t)):
            excitation[i] = decay[i - 1] * (1.0 + excitation[i - 1])
        intensity = mu + n * beta * excitation
        compensator = mu * horizon + n * np.sum(1.0 - np.exp(-beta * (horizon - t)))
        total += float(np.sum(np.log(intensity)) - compensator)
    return total


def poisson_loglik(sequences: list[Sequence], per_sequence: bool = False) -> tuple[float, float | np.ndarray]:
    counts = np.array([len(s[0]) for s in sequences], dtype=float)
    horizons = np.array([s[2] - s[1] for s in sequences], dtype=float)
    if not per_sequence:
        mu = counts.sum() / horizons.sum()
        return float(counts.sum() * np.log(mu) - mu * horizons.sum()), float(mu)
    mus = counts / horizons
    terms = np.where(counts > 0, counts * np.log(np.where(mus > 0, mus, 1.0)), 0.0) - counts
    return float(terms.sum()), mus


def fit_hawkes(sequences: list[Sequence], n_starts: int = 12, seed: int = 0, per_sequence_mu: bool = False) -> dict:
    """Maximum likelihood fit. With per_sequence_mu each sequence gets its own baseline,
    so between-sequence rate differences are not mistaken for self-excitation."""
    rng = np.random.default_rng(seed)
    k = len(sequences) if per_sequence_mu else 1
    _, mu_hat = poisson_loglik(sequences, per_sequence=per_sequence_mu)
    mu0 = np.maximum(np.atleast_1d(mu_hat), 1e-4)
    lo, hi = np.log(BETA_BOUNDS[0]), np.log(BETA_BOUNDS[1])

    def objective(x: np.ndarray) -> float:
        mu = np.exp(x[:k])
        n = 1.0 / (1.0 + np.exp(-x[k]))
        value = hawkes_loglik(mu if per_sequence_mu else float(mu[0]), n, float(np.exp(x[k + 1])), sequences)
        return -value if np.isfinite(value) else 1e12

    bounds = [(float(np.log(m)) - 8, float(np.log(m)) + 3) for m in mu0] + [(-12.0, 12.0), (lo, hi)]
    best = None
    for _ in range(n_starts):
        start = np.concatenate([np.log(mu0 * rng.uniform(0.2, 1.0, k)), [rng.uniform(-3.0, 2.0), rng.uniform(lo, hi)]])
        result = optimize.minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result
    mu = np.exp(best.x[:k])
    n = float(1.0 / (1.0 + np.exp(-best.x[k])))
    beta = float(np.exp(best.x[k + 1]))
    return {"mu_per_s": mu.tolist() if per_sequence_mu else float(mu[0]), "branching_ratio": n,
            "beta_per_s": beta, "excitation_timescale_s": 1.0 / beta, "loglik": -float(best.fun),
            "n_params": k + 2, "converged": bool(best.success)}


def rescaled_intervals(sequences: list[Sequence], mu: float | np.ndarray, n: float = 0.0, beta: float = 1.0) -> np.ndarray:
    """Time-rescaling theorem: compensator increments between events are Exp(1) if the model is correct."""
    mus = np.broadcast_to(np.asarray(mu, dtype=float), (len(sequences),))
    out = []
    for mu, (times, start, _) in zip(mus, sequences):
        t = np.asarray(times, dtype=float) - start
        if len(t) < 2:
            continue
        cum = np.array([mu * ti + n * np.sum(1.0 - np.exp(-beta * (ti - t[:i]))) for i, ti in enumerate(t)])
        out.extend(np.diff(cum))
    return np.asarray(out)


def simulate_hawkes(mu: float, n: float, beta: float, horizon: float, seed: int = 0) -> np.ndarray:
    """Ogata thinning; used by tests to check parameter recovery."""
    rng = np.random.default_rng(seed)
    events: list[float] = []
    t = 0.0
    while True:
        bound = mu + n * beta * np.sum(np.exp(-beta * (t - np.asarray(events)))) + n * beta
        t += rng.exponential(1.0 / bound)
        if t >= horizon:
            return np.asarray(events)
        current = mu + n * beta * np.sum(np.exp(-beta * (t - np.asarray(events))))
        if rng.uniform() <= current / bound:
            events.append(t)


def vicos_change_events(observations: pd.DataFrame, views: tuple[str, ...] = VICOS_SESSION_VIEWS) -> tuple[pd.DataFrame, list[Sequence], list[str]]:
    data = observations[observations["label_status"] == "labeled"].copy()
    data["sequence_id"] = data["sequence_id"].astype(str).str.zfill(2)
    data = data[data["sequence_id"].isin(views)].sort_values(["sequence_id", "frame"])
    rows, sequences, ids = [], [], []
    for seq, g in data.groupby("sequence_id", sort=True):
        t = g["timestamp_ms_approx"].to_numpy(dtype=float) / 1000.0
        pos = g["position"].astype(str).to_numpy()
        change = np.flatnonzero(pos[1:] != pos[:-1]) + 1
        for i in change:
            rows.append({"sequence_id": seq, "time_s": t[i], "state_from": pos[i - 1],
                         "state_to": pos[i], "timing_uncertainty_s": t[i] - t[i - 1]})
        sequences.append((t[change], float(t[0]), float(t[-1])))
        ids.append(seq)
    return pd.DataFrame(rows), sequences, ids


def _registry_allows(registry: Path, dataset_id: str, task: str) -> None:
    data = json.loads(registry.read_text(encoding="utf-8"))
    record = next((d for d in data["datasets"] if d["dataset_id"] == dataset_id), None)
    approved = (record or {}).get("analysis_approval", {}).get("approved_tasks", [])
    if task not in approved:
        raise PermissionError(f"{dataset_id} is not approved for {task} in {registry}")


def analyze_vicos_hawkes(observations_csv: Path, registry: Path, out_dir: Path) -> dict:
    _registry_allows(registry, "vicos_bjj_positions", "bjj_position_transition_analysis")
    events, sequences, ids = vicos_change_events(pd.read_csv(observations_csv, dtype={"sequence_id": str}))
    models = {}
    for label, per_seq in (("shared_baseline", False), ("per_session_baseline", True)):
        p_ll, p_mu = poisson_loglik(sequences, per_sequence=per_seq)
        h = fit_hawkes(sequences, per_sequence_mu=per_seq)
        k = len(sequences) if per_seq else 1
        models[label] = {
            "poisson": {"mu_per_s": np.atleast_1d(p_mu).tolist() if per_seq else p_mu, "loglik": p_ll, "aic": 2 * k - 2 * p_ll,
                        "ks_rescaled_p": float(stats.kstest(rescaled_intervals(sequences, p_mu), "expon").pvalue)},
            "hawkes": {**h, "aic": 2 * h["n_params"] - 2 * h["loglik"],
                       "ks_rescaled_p": float(stats.kstest(rescaled_intervals(sequences, np.asarray(h["mu_per_s"]), h["branching_ratio"], h["beta_per_s"]), "expon").pvalue)},
            "likelihood_ratio_stat": 2.0 * (h["loglik"] - p_ll),
        }
    per_session = []
    for seq_id, seq in zip(ids, sequences):
        gaps = np.diff(seq[0])
        fit = fit_hawkes([seq], n_starts=6) if len(seq[0]) >= 10 else None
        per_session.append({"sequence_id": seq_id, "changes": int(len(seq[0])), "window_s": seq[2] - seq[1],
                            "inter_change_cv": float(gaps.std() / gaps.mean()) if len(gaps) > 1 else None,
                            "branching_ratio": fit["branching_ratio"] if fit else None,
                            "excitation_timescale_s": fit["excitation_timescale_s"] if fit else None})
    summary = {
        "dataset_id": "vicos_bjj_positions", "license_id": "CC-BY-NC-SA-4.0",
        "task": "bjj_position_transition_analysis", "is_research_finding": False,
        "sessions_used": list(VICOS_SESSION_VIEWS), "position_changes": int(len(events)),
        "total_window_s": float(sum(s[2] - s[1] for s in sequences)),
        "models": models,
        "primary_model": "per_session_baseline",
        "model_note": "The shared-baseline fit attributes between-session rate differences to self-excitation; the per-session-baseline comparison is the one to interpret.",
        "likelihood_ratio_note": "n=0 lies on the parameter boundary and beta is unidentified under it, so the chi-square reference is only approximate; compare AIC and the KS checks.",
        "changes_with_timing_gap_over_1s": int((events["timing_uncertainty_s"] > 1.0).sum()),
        "per_session": per_session,
        "limitations": [
            "Two athletes and six sparring sessions; not generalizable to athletes, belts or competition.",
            "Self-excitation cannot be distinguished from rate changes caused by match phase.",
            "Change times come from sampled labeled frames and are upper bounds on the true change times.",
            "Orientation variants (e.g. back1/back2) count as distinct positions, as in the source labels.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_dir / "position_change_events.csv", index=False)
    (out_dir / "hawkes_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
