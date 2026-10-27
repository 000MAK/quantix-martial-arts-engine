# 04 · Stochastic Models

`src/martial_arts_engine/transitions.py` estimates transition counts/probabilities and observed dwell summaries from event logs. The dwell summaries are descriptive and censored at sequence ends; they are not, by themselves, a fitted semi-Markov model. `modeling.py` contains group-aware scikit-learn action-label baselines. `mixed_models.py` provides an optional Bayesian binomial mixed-effects model with an athlete random intercept, not an individual ability score. `demo-markov` is synthetic and marked as such.

Verified attempt-level takedown logs produce success/defense proportions with Wilson intervals in `performance_metrics.py`; unresolved attempts are reported separately and excluded from the denominator.

## Hawkes (self-exciting) position-change model

`hawkes.py` / `python main.py fit-vicos-hawkes` asks whether one position change makes further changes more likely shortly afterwards, compared with a constant-rate Poisson process. It uses one camera view per ViCoS session (prefixes 00, 03, 06, 09, 11, 14) so the same moments are not counted twice, checks the registry approval for `bjj_position_transition_analysis`, and reports AIC, a likelihood-ratio statistic and time-rescaling KS checks.

Both a shared-baseline and a per-session-baseline fit are reported. The shared-baseline fit is a known trap: session 11 holds 80 of 96 changes, so one busy session plus five quiet ones looks like strong self-excitation (n ≈ 0.93). With a separate baseline per session, the branching ratio is ≈ 0 and the Poisson model has the lower AIC: these data show no evidence of momentum-like clustering. The KS check rejects both models; changes are more regular than Poisson (inter-change CV ≈ 0.73 in session 11), which suggests a minimum hold time and motivates a renewal / semi-Markov model next. Two athletes and six sessions: exploratory, not a finding about BJJ in general.

## Semi-Markov dwell-time model

`semi_markov.py` / `python main.py fit-vicos-semi-markov` splits position flow into an embedded jump chain (where the next position goes, no self-loops) and a dwell-time distribution (how long a position is held). It fits censored exponential (memoryless, i.e. Markov) and Weibull models by maximum likelihood, with a profile-likelihood 95% interval for the Weibull shape k. The first episode of each session is excluded (unknown start) and the last is right-censored.

ViCoS result (one camera view per session, 96 usable episodes, 90 observed exits): with a separate scale per session, k = 1.60 (95% CI 1.38–1.77) and the Weibull model beats the exponential (AIC 492.2 vs 520.7, LR p ≈ 3e-8). The hazard of leaving a position rises with time held, consistent with a minimum hold time and with the regularity the Hawkes KS check picked up. The result is stable when restricted to sessions with ≥3 exits (k = 1.58), to session 11 alone (k = 1.56) and within single positions there (standing 1.61, takedown1 2.00, takedown2 2.28). A shared-scale fit gives k = 0.66: mixing sessions with very different rates biases k downward.

Caveat: 80 of the 96 episodes come from session 11 (standing and takedown exchanges); ground positions have too few episodes for their own models. Two athletes, exploratory.
