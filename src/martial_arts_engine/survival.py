from __future__ import annotations
from pathlib import Path
import pandas as pd

def fit_injury_cox(csv_path:Path,out_path:Path)->dict:
    try:
        from lifelines import CoxPHFitter
    except ImportError as e:
        raise RuntimeError('Survival analysis requires requirements-survival.txt (lifelines).') from e
    df=pd.read_csv(csv_path)
    required={'followup_days','event_injury','age_years','training_hours_week','prior_injury','discipline_id'}
    missing=required-set(df.columns)
    if missing: raise ValueError('Injury follow-up table missing: '+', '.join(sorted(missing)))
    if df.event_injury.sum()<5: raise ValueError('At least five observed injury events are required for this exploratory Cox fit; more are strongly recommended.')
    model_df=pd.get_dummies(df[list(required)],columns=['discipline_id'],drop_first=True,dtype=float)
    fitter=CoxPHFitter();fitter.fit(model_df,duration_col='followup_days',event_col='event_injury',robust=True)
    out_path.parent.mkdir(parents=True,exist_ok=True);fitter.summary.to_csv(out_path)
    return {'n':len(df),'events':int(df.event_injury.sum()),'concordance_index':float(fitter.concordance_index_),'summary_path':str(out_path),'warning':'Exploratory association only; requires a well-defined cohort, censoring, confounder review, and adequate event count.'}
