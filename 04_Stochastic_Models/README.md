# 04 · Stochastic Models

`src/martial_arts_engine/transitions.py` estimates transition counts/probabilities and observed dwell summaries from event logs. The dwell summaries are descriptive and censored at sequence ends; they are not, by themselves, a fitted semi-Markov model. `modeling.py` contains group-aware scikit-learn action-label baselines. `mixed_models.py` provides an optional Bayesian binomial mixed-effects model with an athlete random intercept, not an individual ability score. `demo-markov` is synthetic and marked as such.

Verified attempt-level takedown logs produce success/defense proportions with Wilson intervals in `performance_metrics.py`; unresolved attempts are reported separately and excluded from the denominator.
