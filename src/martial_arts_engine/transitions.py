from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED={'sequence_id','timestamp_ms','discipline_id','state_from','state_to'}

def read_event_log(path:Path)->pd.DataFrame:
    frame=pd.read_csv(path)
    missing=REQUIRED-set(frame.columns)
    if missing:
        raise ValueError('Missing required event-log columns: '+', '.join(sorted(missing)))
    frame=frame.dropna(subset=['sequence_id','timestamp_ms','discipline_id','state_from','state_to']).copy()
    frame['timestamp_ms']=pd.to_numeric(frame['timestamp_ms'],errors='raise')
    frame['sequence_id']=frame['sequence_id'].astype(str)
    frame['discipline_id']=frame['discipline_id'].astype(str)
    frame['state_from']=frame['state_from'].astype(str).str.strip().str.lower()
    frame['state_to']=frame['state_to'].astype(str).str.strip().str.lower()
    return frame.sort_values(['discipline_id','sequence_id','timestamp_ms']).reset_index(drop=True)

def analyze_transitions(events:pd.DataFrame, discipline_id:str|None=None)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    data=events if discipline_id is None else events[events.discipline_id==discipline_id]
    if data.empty: raise ValueError(f'No events found for discipline: {discipline_id}')
    counts=pd.crosstab(data['state_from'],data['state_to'])
    states=sorted(set(data.state_from)|set(data.state_to))
    counts=counts.reindex(index=states,columns=states,fill_value=0)
    matrix=counts.div(counts.sum(axis=1).replace(0,pd.NA),axis=0).fillna(0.0)
    episodes=[]
    group_cols=['sequence_id']
    if 'athlete_id' in data.columns: group_cols.append('athlete_id')
    for group_key,group in data.groupby(group_cols,dropna=False,sort=False):
        g=group.sort_values('timestamp_ms')
        # Event rows encode a state transition at timestamp_ms. Dwell time is
        # the interval until the next observed event; the final state is censored.
        for i,row in enumerate(g.itertuples(index=False)):
            next_row=next(g.iloc[i+1:].itertuples(index=False),None)
            state=row.state_to
            duration=max(0.0,(float(next_row.timestamp_ms)-float(row.timestamp_ms))/1000.0) if next_row else None
            episodes.append({'sequence_id':str(getattr(row,'sequence_id')),'state':state,'duration_seconds':duration,'event_observed':next_row is not None,'left_censored':i==0})
    dwell=pd.DataFrame(episodes)
    summaries=[]
    for state,g in dwell.groupby('state',sort=True):
        observed=g.loc[g.event_observed & ~g.left_censored,'duration_seconds'].dropna()
        summaries.append({'state':state,'episodes':int(len(g)),'observed_durations':int(len(observed)),'right_censored':int((~g.event_observed).sum()),'median_seconds':float(observed.median()) if len(observed) else None,'p90_seconds':float(observed.quantile(.9)) if len(observed) else None})
    return counts,matrix,pd.DataFrame(summaries)

def synthetic_demo(seed:int=42,n_sessions:int=8,steps:int=90)->pd.DataFrame:
    import numpy as np
    rng=np.random.default_rng(seed)
    states=['standing','clinch','closed_guard','half_guard','side_control','mount','back_control','submission_attempt']
    p=np.array([
      [.48,.32,.12,.02,.02,.01,.01,.02], [.18,.16,.28,.12,.12,.05,.03,.04],
      [.02,.05,.48,.20,.08,.07,.03,.05], [.02,.03,.10,.42,.25,.08,.05,.06],
      [.01,.02,.04,.05,.42,.22,.16,.09], [.01,.01,.02,.02,.20,.38,.22,.15],
      [.01,.01,.01,.02,.06,.10,.55,.25], [.02,.02,.04,.05,.15,.20,.22,.35]])
    p=p/p.sum(axis=1,keepdims=True)
    rows=[]
    for s in range(n_sessions):
        state=0; t=0
        for _ in range(steps):
            next_state=int(rng.choice(len(states),p=p[state]))
            rows.append({'sequence_id':f'SYN-{s:03d}','timestamp_ms':t,'discipline_id':'ma-027-brazilian-jiu-jitsu-no-gi','athlete_id':f'athlete-{s%4}','state_from':states[state],'state_to':states[next_state],'action_label':states[next_state],'outcome':'simulated'})
            state=next_state; t+=int(rng.integers(700,9000))
    return pd.DataFrame(rows)
