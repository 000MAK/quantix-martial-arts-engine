# 04 · Stochastic Models

`src/martial_arts_engine/transitions.py` estimates transition counts/probabilities and observed dwell summaries from event logs. The dwell summaries are descriptive and censored at sequence ends; they are not, by themselves, a fitted semi-Markov model. `modeling.py` contains group-aware scikit-learn action-label baselines. `mixed_models.py` provides an optional Bayesian binomial mixed-effects model with an athlete random intercept, not an individual ability score. `demo-markov` is synthetic and marked as such.

Verified attempt-level takedown logs produce success/defense proportions with Wilson intervals in `performance_metrics.py`; unresolved attempts are reported separately and excluded from the denominator.

## Hawkes (self-exciting) position-change model

`hawkes.py` / `python main.py fit-vicos-hawkes` asks whether one position change makes further changes more likely shortly afterwards, compared with a constant-rate Poisson process. It uses one camera view per ViCoS session (prefixes 00, 03, 06, 09, 11, 14) so the same moments are not counted twice, checks the registry approval for `bjj_position_transition_analysis`, and reports AIC, a likelihood-ratio statistic and time-rescaling KS checks.

Both a shared-baseline and a per-session-baseline fit are reported. The shared-baseline fit is a known trap: session 11 holds 80 of 96 changes, so one busy session plus five quiet ones looks like strong self-excitation (n ≈ 0.93). With a separate baseline per session, the branching ratio is ≈ 0 and the Poisson model has the lower AIC: these data show no evidence of momentum-like clustering. The KS check rejects both models; changes are more regular than Poisson (inter-change CV ≈ 0.73 in session 11), which suggests a minimum hold time and motivates a renewal / semi-Markov model next. Two athletes and six sessions: exploratory, not a finding about BJJ in general.
