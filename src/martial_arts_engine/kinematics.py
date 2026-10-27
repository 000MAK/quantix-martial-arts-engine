from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd

def _angle(a, b, c):
    if any(v is None for v in (a,b,c)) or any(not np.isfinite(v).all() for v in (a,b,c)):
        return np.nan
    u=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    v=np.asarray(c,dtype=float)-np.asarray(b,dtype=float)
    denom=np.linalg.norm(u)*np.linalg.norm(v)
    if denom<1e-9: return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/denom,-1,1))))

def derive_kinematics(pose_csv: Path, output_csv: Path, visibility_threshold: float=.35) -> pd.DataFrame:
    """Derive 2-D projected joint angles and image-space endpoint speeds.

    Angles are camera-plane geometry; speeds are pixels/second and are not
    physical velocities without camera calibration and depth handling.
    """
    df=pd.read_csv(pose_csv)
    keypoints={i:[f'pose_{i}_{axis}' for axis in ('x','y','visibility')] for i in (11,12,13,14,15,16,23,24,25,26,27,28)}
    missing=[col for cols in keypoints.values() for col in cols if col not in df]
    if missing:
        raise ValueError('Pose CSV is missing required landmarks: '+', '.join(missing[:8]))
    rows=[]
    joints={'left_elbow':(11,13,15),'right_elbow':(12,14,16),'left_shoulder':(23,11,13),'right_shoulder':(24,12,14),'left_knee':(23,25,27),'right_knee':(24,26,28),'left_hip':(11,23,25),'right_hip':(12,24,26)}
    for row in df.to_dict('records'):
        rec={k:row.get(k) for k in ('video_id','frame_index','timestamp_ms','person_index','image_width','image_height')}
        for name,(ia,ib,ic) in joints.items():
            ids=(ia,ib,ic)
            vis=[]
            for i in ids:
                value=pd.to_numeric(row.get(f'pose_{i}_visibility',0),errors='coerce')
                vis.append(float(value) if pd.notna(value) else np.nan)
            if any(not np.isfinite(v) or v<visibility_threshold for v in vis):
                rec[name+'_angle_deg']=np.nan
                continue
            points=[]
            for i in ids:
                x=float(row[f'pose_{i}_x']);y=float(row[f'pose_{i}_y'])
                points.append((x,y))
            rec[name+'_angle_deg']=_angle(*points)
        for label,idx in [('left_wrist',15),('right_wrist',16),('left_ankle',27),('right_ankle',28)]:
            rec[label+'_x_px']=float(row[f'pose_{idx}_x'])*float(row.get('image_width',1) or 1)
            rec[label+'_y_px']=float(row[f'pose_{idx}_y'])*float(row.get('image_height',1) or 1)
        rows.append(rec)
    result=pd.DataFrame(rows)
    if not result.empty:
        result=result.sort_values(['video_id','person_index','timestamp_ms']).reset_index(drop=True)
        grouped=result.groupby(['video_id','person_index'],sort=False)
        dt=grouped['timestamp_ms'].diff().div(1000).replace(0,np.nan)
        for label in ('left_wrist','right_wrist','left_ankle','right_ankle'):
            dx=grouped[label+'_x_px'].diff();dy=grouped[label+'_y_px'].diff()
            result[label+'_speed_px_s']=np.sqrt(dx.pow(2)+dy.pow(2)).div(dt)
    output_csv.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output_csv,index=False)
    return result
