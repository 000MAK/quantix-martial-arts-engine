from __future__ import annotations
from pathlib import Path
from html import escape
import json
import numpy as np
import pandas as pd

def _line_svg(title: str, values: list[float], color: str, y_label: str) -> str:
    vals=np.asarray(values,dtype=float);vals=vals[np.isfinite(vals)]
    if len(vals)<2: return '<p>Not enough valid samples to plot.</p>'
    w,h=720,250;left,top,right,bottom=58,32,18,42
    lo,hi=float(vals.min()),float(vals.max());span=max(hi-lo,1e-6)
    points=[]
    for i,val in enumerate(vals):
        x=left+i*(w-left-right)/(len(vals)-1);y=top+(hi-val)*(h-top-bottom)/span
        points.append(f'{x:.1f},{y:.1f}')
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}" class="chart">'
       f'<text x="{left}" y="22" class="chart-title">{escape(title)}</text>'
       f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" class="axis"/>'
       f'<text x="{left}" y="{top+12}" class="axis-label">max {hi:.1f}</text><text x="{left}" y="{h-bottom-4}" class="axis-label">min {lo:.1f}</text>'
       f'<text x="8" y="{h/2}" class="axis-label" transform="rotate(-90 8 {h/2})">{escape(y_label)}</text>'
       f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>'
       f'<text x="{left}" y="{h-12}" class="axis-label">sample 1</text><text x="{w-right-70}" y="{h-12}" class="axis-label">sample {len(vals)}</text></svg>')

def _heatmap_svg(matrix: pd.DataFrame, title: str) -> str:
    if matrix.empty: return '<p>No annotated transition events supplied.</p>'
    states=list(matrix.index);n=len(states);cell=max(22,min(48,580//max(n,1)));left,top=155,55;w=left+cell*n+16;h=top+cell*n+24
    parts=[f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}" class="chart"><text x="8" y="22" class="chart-title">{escape(title)}</text>']
    arr=matrix.to_numpy(dtype=float);maxv=float(np.nanmax(arr)) if arr.size else 0
    for j,state in enumerate(states):
        parts.append(f'<text x="{left+j*cell+cell/2:.1f}" y="48" text-anchor="middle" class="axis-label">{escape(str(state)[:10])}</text>')
    for i,state in enumerate(states):
        parts.append(f'<text x="{left-8}" y="{top+i*cell+cell*.65:.1f}" text-anchor="end" class="axis-label">{escape(str(state)[:20])}</text>')
        for j in range(n):
            val=float(arr[i,j]);alpha=.08+.82*(val/maxv if maxv else 0)
            parts.append(f'<rect x="{left+j*cell}" y="{top+i*cell}" width="{cell-1}" height="{cell-1}" fill="rgba(30,170,170,{alpha:.2f})"><title>{escape(str(state))} → {escape(str(states[j]))}: {val:.3f}</title></rect>')
    parts.append('</svg>');return ''.join(parts)

def write_analysis_report(path: Path, payload: dict, kinematics: pd.DataFrame, heart_rate: pd.DataFrame|None, transition_matrix: pd.DataFrame|None, audio_summary: dict|None=None) -> None:
    pose_svg='<p>Pose observations unavailable.</p>'
    if not kinematics.empty:
        col='left_knee_angle_deg'
        if col not in kinematics or kinematics[col].notna().sum()<2:
            col=next((c for c in kinematics if c.endswith('_angle_deg') and kinematics[c].notna().sum()>=2),None)
        if col:
            plots=[]
            if 'person_index' in kinematics:
                for person,group in kinematics.groupby('person_index',dropna=False):
                    if group[col].notna().sum()>=2: plots.append(f'<h3>Detector person index {escape(str(person))}</h3>'+_line_svg(f'Camera-plane joint angle · {col.replace("_angle_deg","")}',group[col].tolist(),'#f1a33c','degrees'))
            if plots: pose_svg=''.join(plots)+'<p>Detector person index is not persistent identity tracking across video frames.</p>'
            else: pose_svg=_line_svg(f'Camera-plane joint angle · {col.replace("_angle_deg","")}',kinematics[col].tolist(),'#f1a33c','degrees')
    hr_svg='<p>No heart-rate sensor data supplied; gas-tank and oxygen-debt estimates are unavailable.</p>'
    if heart_rate is not None and not heart_rate.empty:
        hr_svg=_line_svg('Heart-rate sensor trace',heart_rate['heart_rate_bpm'].tolist(),'#e85b78','beats/minute')
    audio_text='No audio feature summary available.'
    if audio_summary:
        payload['audio_summary']={k:(float(v) if isinstance(v,(np.floating,)) else v) for k,v in audio_summary.items() if k in {'duration_seconds','rms_mean','rms_std','centroid_mean','centroid_std','mel_energy_mean'}}
        audio_text='Extracted RMS, spectral-centroid and mel-energy summaries. These are signal descriptors, not impact-quality or voice-command classifications.'
        audio_text += '<br><code>'+escape(json.dumps(payload['audio_summary'],ensure_ascii=False))+'</code>'
    trans_svg=_heatmap_svg(transition_matrix,'Observed transition probabilities') if transition_matrix is not None else '<p>Provide annotated event labels to build this view. Video pose features are not automatically treated as tactical position labels.</p>'
    score='Unavailable: no validated, cross-discipline scoring protocol or comparable labeled dataset.'
    payload['overall_score']=None;payload['overall_score_status']=score
    payload['interpretation_limits']=[
      'Pose angles are 2-D camera projections; joint speed is image-space pixels/second, not calibrated physical speed.',
      'Pose landmarks alone do not identify techniques, force, leverage, or tissue loading.',
      'Acoustic summaries do not classify impact quality or coach commands.',
      'No injury probability, diagnosis, or universal discipline ranking is produced.',
      'Transition results require supplied human-verified event labels and remain descriptive.'
    ]
    path.parent.mkdir(parents=True,exist_ok=True)
    (path.parent/'analysis_report.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    perf_link=escape(str(payload.get('performance_dashboard','performance_dashboard.html')))
    cards=[('Athlete · workload','Sensor-derived heart-rate trace only; no inferred energy percentage.',hr_svg),('Coach · tactical flow','Requires human-verified position/action event labels.',trans_svg),('Audio · signal summary',audio_text,'<p>Audio interpretation is intentionally limited to measured signal features.</p>'),('Performance staff · movement screen','Camera-plane joint geometry for review; not an injury-risk score.',pose_svg),('Four requested performance metrics','See the metrics dashboard for the guard proxy, annotated coordination windows, and attempt-level takedown rates.',f'<p><a href="{perf_link}" style="color:#65d6cd">Open performance metrics dashboard</a></p>'),('Cross-discipline score',escape(score),'<p>Scoring remains disabled until outcome definitions, comparable data, validation and uncertainty criteria are specified.</p>')]
    body=''.join(f'<section class="card"><h2>{h}</h2><p>{d}</p>{visual}</section>' for h,d,visual in cards)
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>QMASE Video Analysis</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#101827;color:#e7edf5;font:16px/1.5 system-ui,sans-serif}}main{{max-width:1100px;margin:36px auto;padding:0 20px}}h1{{font-size:clamp(28px,5vw,42px)}}h2{{font-size:20px;margin:0 0 8px}}.muted{{color:#9aabc0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}.card{{background:#182538;border:1px solid #2b3b51;border-radius:14px;padding:18px;overflow:hidden}}.chart{{width:100%;height:auto;background:#111d2d;border-radius:8px}}.chart-title{{fill:#d8e5f2;font-size:14px}}.axis,.axis-label{{stroke:#8799ad;fill:#aebdd0;font-size:10px}}.notice{{border-left:4px solid #f1a33c;padding:12px 16px;background:#251f18;margin:18px 0}}footer{{color:#91a2b7;font-size:13px;margin:24px 0}}</style><main><h1>QMASE · Video Analysis</h1><p class="muted">Research dashboard · {escape(payload.get('discipline_name',''))} · {escape(payload.get('video_id',''))}</p><div class="notice"><b>Interpretation:</b> automated measurements are feature summaries, not coaching instructions, medical advice, or validated skill scores.</div><div class="grid">{body}</div><footer>Generated locally. Raw video is not embedded in this report. Review consent, source rights and measurement limitations before sharing.</footer></main></html>'''
    path.write_text(html,encoding='utf-8')
