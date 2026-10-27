from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

def fit_athlete_random_intercept_glmm(csv_path: Path, output_json: Path, outcome: str='success', athlete_col: str='athlete_id', discipline_col: str='discipline_id', rule_col: str='rule_set') -> dict:
    """Fit an exploratory Bayesian mixed-effects logistic model.

    Athlete random intercepts account for clustered repeated observations;
    they are not standalone athlete ability scores or causal effects.
    """
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as exc:
        raise RuntimeError('Install requirements-mixed-models.txt to fit this optional GLMM.') from exc
    df=pd.read_csv(csv_path)
    required={outcome,athlete_col,discipline_col}
    missing=required-set(df.columns)
    if missing: raise ValueError('GLMM input is missing columns: '+', '.join(sorted(missing)))
    df=df.dropna(subset=list(required)).copy()
    df[outcome]=pd.to_numeric(df[outcome],errors='raise')
    if not set(df[outcome].unique()).issubset({0,1}): raise ValueError(f'{outcome} must be binary 0/1')
    if df[outcome].nunique()!=2: raise ValueError('Both outcome classes (0 and 1) must be present')
    if df[athlete_col].nunique()<5: raise ValueError('At least five athletes are required for this exploratory grouped model; substantially more are preferable')
    if df[discipline_col].nunique()<2: raise ValueError('At least two disciplines are required for comparative fixed effects')
    if rule_col in df and df[rule_col].nunique(dropna=True)>1:
        formula=f'{outcome} ~ C({discipline_col}) + C({rule_col})'
    else:
        formula=f'{outcome} ~ C({discipline_col})'
    random={'athlete_intercept':f'0 + C({athlete_col})'}
    model=BinomialBayesMixedGLM.from_formula(formula,random,df)
    fit=model.fit_vb()
    summary=str(fit.summary())
    result={'model':'Bayesian binomial mixed-effects logistic regression (variational Bayes)','formula':formula,'random_effect':'athlete-level intercept','n_observations':len(df),'n_athletes':int(df[athlete_col].nunique()),'n_disciplines':int(df[discipline_col].nunique()),'outcome_prevalence':float(df[outcome].mean()),'summary':summary,'caveat':'Exploratory association adjusted for athlete clustering; not causal, not an individual ability score, and not interpretable across unmatched rulesets or sampling designs.'}
    output_json.parent.mkdir(parents=True,exist_ok=True);output_json.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result
