from __future__ import annotations
from pathlib import Path
from html import escape
import json
import numpy as np
import pandas as pd

def guard_position_proxy(pose_csv: Path, output_csv: Path, max_head_distance_shoulder_width: float=1.35,
                        max_wrist_below_nose_shoulder_width: float=.35, visibility_threshold: float=.5,
                        source_id: str|None=None, license_id: str|None=None, consent_status: str|None=None) -> dict:
    """Estimate how often both wrists remain near the head in the camera plane.

    This geometric proxy does not detect gloves, opponent-relative coverage,
    guard quality, or defensive effectiveness.
    """
    df=pd.read_csv(pose_csv)
    required=['video_id','timestamp_ms','person_index']+[f'pose_{i}_{axis}' for i in (0,11,12,15,16) for axis in ('x','y','visibility')]
    missing=set(required)-set(df.columns)
    if missing: raise ValueError('Pose file lacks landmarks for guard proxy: '+', '.join(sorted(missing)))
    records=[]
    for row in df.to_dict('records'):
        vis=[pd.to_numeric(row.get(f'pose_{i}_visibility'),errors='coerce') for i in (0,11,12,15,16)]
        coords=[pd.to_numeric(row.get(f'pose_{i}_{axis}'),errors='coerce') for i in (0,11,12,15,16) for axis in ('x','y')]
        valid=all(pd.notna(v) and float(v)>=visibility_threshold for v in vis) and all(pd.notna(v) and np.isfinite(float(v)) for v in coords)
        result={'video_id':row['video_id'],'timestamp_ms':row['timestamp_ms'],'person_index':row['person_index'],'source_id':source_id,'license_id':license_id,'consent_status':consent_status,'valid_frame':bool(valid),'guard_proxy_both_hands_near_head':np.nan}
        if valid:
            shoulder_width=float(np.hypot(float(row['pose_11_x'])-float(row['pose_12_x']),float(row['pose_11_y'])-float(row['pose_12_y'])))
            if shoulder_width>1e-5:
                nx,ny=float(row['pose_0_x']),float(row['pose_0_y'])
                hand_ok=[]
                for wrist in (15,16):
                    wx,wy=float(row[f'pose_{wrist}_x']),float(row[f'pose_{wrist}_y'])
                    distance=float(np.hypot(wx-nx,wy-ny))/shoulder_width
                    above_limit=wy <= ny+max_wrist_below_nose_shoulder_width*shoulder_width
                    hand_ok.append(distance<=max_head_distance_shoulder_width and above_limit)
                result['guard_proxy_both_hands_near_head']=float(all(hand_ok))
            else:
                result['valid_frame']=False
        records.append(result)
    out=pd.DataFrame(records);output_csv.parent.mkdir(parents=True,exist_ok=True);out.to_csv(output_csv,index=False)
    by_person=[]
    for (video_id,person_index),group in out.groupby(['video_id','person_index'],dropna=False):
        measured=group.loc[group.valid_frame,'guard_proxy_both_hands_near_head'].dropna()
        by_person.append({'video_id':video_id,'person_index':int(person_index),'proxy_retention_pct':float(measured.mean()*100) if len(measured) else None,'valid_frames':int(len(measured)),'total_frames':int(len(group))})
    return {'metric':'two-hand head-proximity guard proxy','by_person':by_person,'thresholds':{'max_wrist_to_nose_distance_in_shoulder_widths':max_head_distance_shoulder_width,'max_wrist_below_nose_in_shoulder_widths':max_wrist_below_nose_shoulder_width,'minimum_landmark_visibility':visibility_threshold},'interpretation':'Camera-plane heuristic only; not glove detection, guard discipline ground truth, or defensive success. person_index is detector order, not persistent identity tracking.'}

def coordination_windows(pose_csv: Path, kinematics_csv: Path, windows_csv: Path, output_csv: Path, max_lag_steps: int=3) -> pd.DataFrame:
    """Cross-correlate torso and elbow/knee angular change with effector speed.

    Requires verified strike windows. Reports camera-plane coupling and sample-
    resolution lag, not a biomechanical coordination score.
    """
    windows=pd.read_csv(windows_csv);need={'video_id','start_ms','end_ms','actor_person_index','effector','annotation_status','annotation_protocol','source_id','license_id'}
    missing=need-set(windows.columns)
    if missing: raise ValueError('Strike-window file missing: '+', '.join(sorted(missing)))
    pose=pd.read_csv(pose_csv);kin=pd.read_csv(kinematics_csv)
    video_ids=set(pose.video_id.astype(str).unique())
    windows=windows[windows.video_id.astype(str).isin(video_ids)].copy()
    if windows.empty: raise ValueError('No verified strike windows match this analyzed video_id')
    if not windows.annotation_status.astype(str).str.lower().eq('verified').all() or windows.annotation_protocol.isna().any():
        raise ValueError('Coordination analysis requires verified strike windows and an annotation protocol')
    if windows[list(need)].isna().any().any(): raise ValueError('Strike-window labels must include source/license provenance')
    merged=pose.merge(kin,on=['video_id','frame_index','timestamp_ms','person_index'],suffixes=('','_kin'))
    rows=[]
    for w in windows.to_dict('records'):
        start,end=float(w['start_ms']),float(w['end_ms'])
        if end<=start: raise ValueError('Each strike window must have end_ms > start_ms')
        eff=str(w['effector']).lower()
        if eff not in {'hand','foot'}: raise ValueError("effector must be 'hand' or 'foot'")
        actor=merged[(merged.video_id.astype(str)==str(w['video_id']))&(merged.person_index==int(w['actor_person_index']))&(merged.timestamp_ms>=start)&(merged.timestamp_ms<=end)].sort_values('timestamp_ms').copy()
        row={'video_id':str(w['video_id']),'start_ms':start,'end_ms':end,'actor_person_index':int(w['actor_person_index']),'effector':eff,'action_type':w.get('action_type'),'source_id':w['source_id'],'license_id':w['license_id'],'annotation_protocol':w['annotation_protocol'],'sample_count':len(actor),'torso_lag_ms':None,'torso_correlation':None,'joint_lag_ms':None,'joint_correlation':None,'sample_resolution_ms':None}
        if len(actor)>=5:
            dt=actor.timestamp_ms.diff().median()/1000
            if pd.notna(dt) and dt>0:
                sx=actor.pose_12_x.to_numpy(float)-actor.pose_11_x.to_numpy(float);sy=actor.pose_12_y.to_numpy(float)-actor.pose_11_y.to_numpy(float)
                hx=actor.pose_24_x.to_numpy(float)-actor.pose_23_x.to_numpy(float);hy=actor.pose_24_y.to_numpy(float)-actor.pose_23_y.to_numpy(float)
                torso=np.unwrap(np.arctan2(sy,sx)-np.arctan2(hy,hx));torso_rate=np.abs(np.gradient(torso,dt))
                speed_cols=['left_wrist_speed_px_s','right_wrist_speed_px_s'] if eff=='hand' else ['left_ankle_speed_px_s','right_ankle_speed_px_s']
                available=[c for c in speed_cols if c in actor]
                if available:
                    effector=actor[available].max(axis=1,skipna=True).to_numpy(float)
                    torso_lag,torso_corr=_lagged_correlation(torso_rate,effector,max_lag_steps)
                    joint_cols=['left_elbow_angle_deg','right_elbow_angle_deg'] if eff=='hand' else ['left_knee_angle_deg','right_knee_angle_deg']
                    joint_cols=[c for c in joint_cols if c in actor]
                    if joint_cols:
                        joint_angle=actor[joint_cols].mean(axis=1,skipna=True).to_numpy(float)
                        joint_rate=np.abs(np.gradient(np.unwrap(np.deg2rad(joint_angle)),dt))
                        joint_lag,joint_corr=_lagged_correlation(joint_rate,effector,max_lag_steps)
                    else:joint_lag,joint_corr=None,None
                    row.update({'torso_lag_ms':int(round(torso_lag*dt*1000)) if torso_lag is not None else None,'torso_correlation':torso_corr,'joint_lag_ms':int(round(joint_lag*dt*1000)) if joint_lag is not None else None,'joint_correlation':joint_corr,'sample_resolution_ms':int(round(dt*1000))})
        rows.append(row)
    result=pd.DataFrame(rows);output_csv.parent.mkdir(parents=True,exist_ok=True);result.to_csv(output_csv,index=False);return result

def _wilson(successes: int, total: int, z: float=1.959963984540054) -> tuple[float|None,float|None]:
    if total<=0: return None,None
    p=successes/total;den=1+z*z/total;center=(p+z*z/(2*total))/den;half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return max(0.0,center-half),min(1.0,center+half)

def _lagged_correlation(driver: np.ndarray, response: np.ndarray, max_lag_steps: int) -> tuple[int|None,float|None]:
    best=None
    for lag in range(-max_lag_steps,max_lag_steps+1):
        if lag<0:x,y=driver[-lag:],response[:lag]
        elif lag>0:x,y=driver[:-lag],response[lag:]
        else:x,y=driver,response
        valid=np.isfinite(x)&np.isfinite(y)
        if valid.sum()<4 or np.std(x[valid])<1e-9 or np.std(y[valid])<1e-9:continue
        corr=float(np.corrcoef(x[valid],y[valid])[0,1])
        if best is None or abs(corr)>abs(best[1]):best=(lag,corr)
    return best if best is not None else (None,None)

def _rate_bars(frame: pd.DataFrame|None) -> str:
    if frame is None or frame.empty:return ''
    w=680;row_h=38;h=24+row_h*len(frame);parts=[f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Takedown and defense rates with Wilson intervals">']
    for i,row in frame.reset_index(drop=True).iterrows():
        y=18+i*row_h;label=f"{row['side']} · {row['athlete_id']} · {row['attempt_type']}"
        parts.append(f'<text x="4" y="{y+14}" class="axis-label">{escape(label[:44])}</text><rect x="330" y="{y}" width="250" height="16" rx="4" fill="#34445a"/>')
        if pd.notna(row['rate_pct']):
            x=330;scale=2.5;rate=float(row['rate_pct']);lo=float(row['wilson_95_lower_pct']);hi=float(row['wilson_95_upper_pct'])
            parts.append(f'<rect x="{x}" y="{y}" width="{rate*scale:.1f}" height="16" rx="4" fill="#35b9ad"/><line x1="{x+lo*scale:.1f}" y1="{y+8}" x2="{x+hi*scale:.1f}" y2="{y+8}" stroke="#f4c76b" stroke-width="3"/><text x="590" y="{y+13}" class="axis-label">{rate:.1f}%</text>')
    parts.append('</svg>');return ''.join(parts)

def summarize_takedown_attempts(attempts_csv: Path, output_csv: Path, discipline_id: str, sequence_id: str) -> pd.DataFrame:
    df=pd.read_csv(attempts_csv)
    required={'attempt_id','sequence_id','discipline_id','timestamp_ms','attacker_id','defender_id','attempt_type','outcome','annotation_status','annotation_protocol','rule_set','source_id','license_id'}
    missing=required-set(df.columns)
    if missing: raise ValueError('Takedown attempt log missing: '+', '.join(sorted(missing)))
    df=df[(df.discipline_id.astype(str)==str(discipline_id))&(df.sequence_id.astype(str)==str(sequence_id))].copy()
    if df.empty: raise ValueError('No takedown attempt rows match the analyzed discipline and sequence')
    if not df.annotation_status.astype(str).str.lower().eq('verified').all() or df.annotation_protocol.isna().any():
        raise ValueError('Takedown rates require verified attempt outcomes and a documented annotation protocol')
    if df.attempt_id.duplicated().any(): raise ValueError('attempt_id must uniquely identify one attempt row')
    if df[list(required)].isna().any().any(): raise ValueError('Matched takedown records cannot contain blank required fields')
    for col in required-{'timestamp_ms'}:
        if df[col].astype(str).str.strip().eq('').any(): raise ValueError(f'Matched takedown records contain an empty {col}')
    df['timestamp_ms']=pd.to_numeric(df.timestamp_ms,errors='raise')
    if (df.timestamp_ms<0).any(): raise ValueError('takedown timestamps must be non-negative')
    allowed={'completed','defended','unresolved'}
    df['outcome']=df.outcome.astype(str).str.lower()
    if not set(df.outcome).issubset(allowed): raise ValueError('outcome values must be completed, defended, or unresolved')
    resolved=df[df.outcome.isin({'completed','defended'})]
    rows=[]
    def summarize(group,side,identity_col,metric_name,success_value):
        for key,g in group.groupby([identity_col,'attempt_type','rule_set'],dropna=False):
            identity,attempt_type,rule_set=key;success=int((g.outcome==success_value).sum());n=len(g);lo,hi=_wilson(success,n)
            unresolved=int(((df[identity_col]==identity)&(df.attempt_type==attempt_type)&(df.rule_set==rule_set)&(df.outcome=='unresolved')).sum())
            rows.append({'side':side,'athlete_id':identity,'attempt_type':attempt_type,'rule_set':rule_set,'sequence_id':sequence_id,'discipline_id':discipline_id,'source_ids':'|'.join(sorted(g.source_id.astype(str).unique())),'license_ids':'|'.join(sorted(g.license_id.astype(str).unique())),'annotation_protocols':'|'.join(sorted(g.annotation_protocol.astype(str).unique())),'metric':metric_name,'successes':success,'resolved_attempts':n,'unresolved_excluded':unresolved,'rate_pct':100*success/n if n else None,'wilson_95_lower_pct':100*lo if lo is not None else None,'wilson_95_upper_pct':100*hi if hi is not None else None})
    summarize(resolved,'attacker','attacker_id','takedown_success_rate','completed')
    summarize(resolved,'defender','defender_id','takedown_defense_rate','defended')
    result=pd.DataFrame(rows);output_csv.parent.mkdir(parents=True,exist_ok=True);result.to_csv(output_csv,index=False);return result

def write_performance_dashboard(output_html: Path, guard: dict, coordination: pd.DataFrame|None, takedowns: pd.DataFrame|None) -> None:
    def table(frame):
        if frame is None or frame.empty:return '<p>Not available: required verified annotations were not supplied.</p>'
        return frame.to_html(index=False,border=0,classes='data')
    guard_rows=guard.get('by_person',[])
    if not guard_rows: guard_text='Not available: usable head/wrist landmarks were not found.';guard_visual=''
    else:
        guard_text='Per-detector-person summaries; these person indices are not identity tracks.'
        bars=[]
        for person in guard_rows:
            gp=person['proxy_retention_pct']
            if gp is None: continue
            y=18+len(bars)*55
            bars.append(f'<text x="8" y="{y+14}" class="axis-label">person index {person["person_index"]} · n={person["valid_frames"]}</text><rect x="8" y="{y+22}" width="500" height="16" rx="4" fill="#34445a"/><rect x="8" y="{y+22}" width="{float(gp)*5:.1f}" height="16" rx="4" fill="#35b9ad"/><text x="520" y="{y+35}" class="axis-label">{float(gp):.1f}%</text>')
        guard_visual=f'<svg viewBox="0 0 600 {max(55,18+len(bars)*55)}" class="chart" role="img" aria-label="Per-person guard proxy retention, not scores">{"".join(bars)}</svg>' if bars else ''
    coordination_text=table(coordination)
    takedown_text=table(takedowns)
    td_visual=_rate_bars(takedowns)
    doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>QMASE Performance Metrics</title><style>body{{font:16px/1.5 system-ui;background:#101827;color:#e7edf5;max-width:1100px;margin:36px auto;padding:0 20px}}section{{background:#182538;border:1px solid #2b3b51;border-radius:14px;padding:20px;margin:16px 0;overflow:auto}}h1{{font-size:36px}}h2{{font-size:21px}}.caution{{border-left:4px solid #f1a33c;padding:12px;background:#251f18}}.data{{border-collapse:collapse;width:100%;font-size:14px}}.data th,.data td{{text-align:left;border-bottom:1px solid #35475d;padding:8px}}.data th{{color:#65d6cd}}.chart{{width:100%;max-width:720px;background:#111d2d;border-radius:8px}}.axis-label{{fill:#aebdd0;font-size:11px}}</style><main><h1>QMASE · Performance Metrics</h1><p class="caution">Measurement dashboard, not a validated scoring engine or medical assessment. Rates require verified human labels. No overall score is calculated.</p><section><h2>Athlete · Guard-position proxy</h2><p>{guard_text}</p>{guard_visual}<p>Definition: both detected wrists satisfy configured image-plane distance/height thresholds relative to the nose and shoulder width. Occlusion, viewing angle and detector identity swaps can change the result.</p></section><section><h2>Athlete / Coach · Coordination windows</h2><p>Separate cross-correlations compare hand/foot image speed with camera-plane shoulder/hip axis change and elbow/knee angle change, within verified strike windows. Lags are quantized at sampled-frame resolution; they are not millisecond-accurate biomechanics or a 0–100 score.</p>{coordination_text}</section><section><h2>Coach · Takedown success and defense</h2><p>Attempt-level rates use verified completed/defended labels. Unresolved attempts are excluded; Wilson 95% intervals show uncertainty. The comparison is valid only under compatible annotation and ruleset protocols.</p>{td_visual}{takedown_text}</section><section><h2>Performance staff</h2><p>Image-space motion does not measure force vectors, spinal load, knee shear force, or future injury probability. No such clinical outputs are produced.</p></section></main></html>'''
    output_html.parent.mkdir(parents=True,exist_ok=True);output_html.write_text(doc,encoding='utf-8')
