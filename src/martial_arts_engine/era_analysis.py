from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

REQUIRED={'bout_id','event_id','discipline_id','competition_year','era','ruleset','division','sex_class','competition_level','duration_seconds','attempts','successful_attempts','finish','transition_count','unique_techniques','annotation_status','annotation_protocol','source_id','license_id'}
STRATA=['discipline_id','ruleset','division','sex_class','competition_level']

def _wilson(k: int,n: int,z: float=1.959963984540054):
    if not n:return None,None
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.,c-h),min(1.,c+h)

def _cluster_bootstrap_mean(frame: pd.DataFrame, column: str, reps: int=1000, seed: int=42):
    clusters=[g[column].dropna().to_numpy(float) for _,g in frame.groupby('event_id')]
    clusters=[x for x in clusters if len(x)]
    if len(clusters)<2:return None,None
    rng=np.random.default_rng(seed);means=np.empty(reps)
    for i in range(reps):
        ix=rng.integers(0,len(clusters),len(clusters));vals=np.concatenate([clusters[j] for j in ix]);means[i]=float(np.mean(vals))
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))

def _cluster_bootstrap_ratio(frame: pd.DataFrame, numerator: str, denominator: str, reps: int=1000, seed: int=42):
    clusters=[g[[numerator,denominator]].to_numpy(float) for _,g in frame.groupby('event_id')]
    clusters=[x for x in clusters if len(x)]
    if len(clusters)<2:return None,None
    rng=np.random.default_rng(seed);ratios=[]
    for _ in range(reps):
        ix=rng.integers(0,len(clusters),len(clusters));sample=np.concatenate([clusters[j] for j in ix],axis=0)
        den=float(sample[:,1].sum())
        if den>0:ratios.append(float(sample[:,0].sum()/den))
    if not ratios:return None,None
    return float(np.quantile(ratios,.025)),float(np.quantile(ratios,.975))

def compare_eras(input_csv: Path, output_csv: Path, report_path: Path|None=None, bootstrap_reps: int=1000) -> pd.DataFrame:
    if bootstrap_reps<1:raise ValueError('bootstrap_reps must be at least 1')
    df=pd.read_csv(input_csv);missing=REQUIRED-set(df.columns)
    if missing:raise ValueError('Bout-metric table missing: '+', '.join(sorted(missing)))
    if df.empty:raise ValueError('No bout data supplied; cannot compare eras')
    if not df.annotation_status.astype(str).str.lower().eq('verified').all() or df.annotation_protocol.isna().any():
        raise ValueError('Era analysis requires verified bout metrics with a documented annotation protocol')
    if df[list(REQUIRED)].isna().any().any():raise ValueError('Era-analysis rows need complete bout, ruleset, annotation and source/license fields')
    for col in ('competition_year','duration_seconds','attempts','successful_attempts','finish','transition_count','unique_techniques'):
        df[col]=pd.to_numeric(df[col],errors='raise')
    if (df.duration_seconds<=0).any() or (df.attempts<0).any() or (df.successful_attempts<0).any() or (df.successful_attempts>df.attempts).any():
        raise ValueError('Durations/attempt counts invalid or successful_attempts exceeds attempts')
    if not set(df.finish.unique()).issubset({0,1}):raise ValueError('finish must be 0/1')
    if df.duplicated(['discipline_id','bout_id']).any():raise ValueError('Provide one verified summary row per bout and discipline')
    df['attempt_rate_per_min']=df.attempts/(df.duration_seconds/60)
    df['transition_rate_per_min']=df.transition_count/(df.duration_seconds/60)
    out=[]
    for key,group in df.groupby(STRATA+['era'],dropna=False,sort=True):
        keys=dict(zip(STRATA+['era'],key));n=len(group);finish=int(group.finish.sum());finish_lo,finish_hi=_wilson(finish,n)
        attempts=int(group.attempts.sum());successes=int(group.successful_attempts.sum());success_lo,success_hi=_wilson(successes,attempts)
        event_count=int(group.event_id.nunique())
        group=group.assign(_finish_denominator=1)
        finish_cluster_ci=_cluster_bootstrap_ratio(group,'finish','_finish_denominator',bootstrap_reps)
        success_cluster_ci=_cluster_bootstrap_ratio(group,'successful_attempts','attempts',bootstrap_reps)
        attempt_rate_ci=_cluster_bootstrap_mean(group,'attempt_rate_per_min',bootstrap_reps)
        transition_ci=_cluster_bootstrap_mean(group,'transition_rate_per_min',bootstrap_reps)
        out.append({**keys,'bouts':n,'events':event_count,'finish_rate_pct':100*finish/n,'finish_wilson95_low_pct':100*finish_lo,'finish_wilson95_high_pct':100*finish_hi,'finish_event_cluster_boot95_low_pct':100*finish_cluster_ci[0] if finish_cluster_ci[0] is not None else None,'finish_event_cluster_boot95_high_pct':100*finish_cluster_ci[1] if finish_cluster_ci[1] is not None else None,'technical_success_rate_pct':100*successes/attempts if attempts else None,'technical_success_wilson95_low_pct':100*success_lo if success_lo is not None else None,'technical_success_wilson95_high_pct':100*success_hi if success_hi is not None else None,'technical_success_event_cluster_boot95_low_pct':100*success_cluster_ci[0] if success_cluster_ci[0] is not None else None,'technical_success_event_cluster_boot95_high_pct':100*success_cluster_ci[1] if success_cluster_ci[1] is not None else None,'attempts_per_min_mean':float(group.attempt_rate_per_min.mean()),'attempts_per_min_cluster_boot95_low':attempt_rate_ci[0],'attempts_per_min_cluster_boot95_high':attempt_rate_ci[1],'transitions_per_min_mean':float(group.transition_rate_per_min.mean()),'transitions_per_min_cluster_boot95_low':transition_ci[0],'transitions_per_min_cluster_boot95_high':transition_ci[1],'mean_unique_techniques_per_bout':float(group.unique_techniques.mean()),'limitations':'Descriptive era profile; not a universal quality ranking. Wilson intervals treat bouts/attempts as independent; event-cluster bootstrap intervals are also supplied when at least two events are available. Compare only within identical stratum; account for athlete strength, rule changes, bout selection and recording/annotation shifts.'})
    result=pd.DataFrame(out);output_csv.parent.mkdir(parents=True,exist_ok=True);result.to_csv(output_csv,index=False)
    if report_path:
        report_path.parent.mkdir(parents=True,exist_ok=True)
        cols=['discipline_id','era','ruleset','division','sex_class','competition_level','bouts','events','finish_rate_pct','finish_event_cluster_boot95_low_pct','finish_event_cluster_boot95_high_pct','technical_success_rate_pct','technical_success_event_cluster_boot95_low_pct','technical_success_event_cluster_boot95_high_pct','attempts_per_min_mean','transitions_per_min_mean','mean_unique_techniques_per_bout']
        view=result.sort_values(STRATA+['era'])[cols]
        visible='\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in view.itertuples(index=False,name=None)]) if not view.empty else 'No rows.'
        report_path.write_text('# Era comparison\n\nThis report contains descriptive, within-discipline/ruleset/weight/sex/competition-level summaries. It does not select a universal best era. A result depends on the declared outcome and comparable coverage. Era boundaries are supplied in the data and must be justified per sport/ruleset.\n\n'+visible+'\n',encoding='utf-8')
    return result
