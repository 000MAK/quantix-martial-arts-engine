# 05 · Risk & Actuarial

`src/martial_arts_engine/survival.py` fits an exploratory Cox proportional hazards model to an eligible injury follow-up cohort. The blank `injury_followup_template.csv` is a schema only. Do not infer discipline-level risk from empty or non-comparable cohorts; define exposure, follow-up, injury events, censoring, confounders and cohort before fitting. Install `requirements-survival.txt`.

The video dashboard does not produce an injury hazard index. Pose angles alone do not measure joint loading or establish prospective injury risk. An athlete-level mixed-effects logistic baseline is available separately for labeled binary outcomes; it needs repeated, well-defined observations and does not yield a personal ability rating.
