from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED={'video_id','discipline_id','competition_name','event_id','bout_id','competition_date','ruleset','division','sex_class','competition_level','source_id','source_url','license_id','license_scope','media_rights_status','ml_training_permission','athlete_data_basis','video_path','annotation_protocol','split_group'}

def validate_pro_video_manifest(path: Path) -> dict:
    """Validate only locally held, explicitly ML-cleared professional footage."""
    manifest_path=Path(path).resolve();df=pd.read_csv(manifest_path)
    missing=REQUIRED-set(df.columns)
    if missing: raise ValueError('Professional-video manifest missing columns: '+', '.join(sorted(missing)))
    if df.empty: raise ValueError('Manifest contains no video rows; no model can be trained yet')
    if df[list(REQUIRED)].isna().any().any(): raise ValueError('Manifest rows may not have blank required provenance or split fields')
    for col in REQUIRED-{'competition_date'}:
        if df[col].astype(str).str.strip().eq('').any(): raise ValueError(f'Manifest has blank {col} values')
    if not df.media_rights_status.astype(str).str.lower().isin({'cleared','licensed','owned'}).all():
        raise ValueError('Every video needs media_rights_status cleared/licensed/owned and documented evidence')
    if not df.ml_training_permission.astype(str).str.lower().isin({'granted','authorized'}).all():
        raise ValueError('A viewable video is not automatically training data; explicit ML-training permission is required')
    if df.license_scope.astype(str).str.strip().eq('').any() or df.athlete_data_basis.astype(str).str.strip().eq('').any():
        raise ValueError('Record license scope and the athlete-data processing basis for every video')
    df['competition_date']=pd.to_datetime(df.competition_date,errors='raise',utc=True)
    df['video_path_resolved']=df.video_path.map(lambda value:str((manifest_path.parent/str(value)).resolve()) if not Path(str(value)).is_absolute() else str(Path(str(value)).resolve()))
    missing_files=[p for p in df.video_path_resolved if not Path(p).is_file()]
    if missing_files: raise FileNotFoundError(f'{len(missing_files)} manifest video files are missing; first: {missing_files[0]}')
    # A bout must never cross train/validation/test groups, even when there are
    # multiple camera angles or clips for that same bout.
    groups=df.groupby(['event_id','bout_id']).split_group.nunique()
    if (groups>1).any(): raise ValueError('Clips from a single bout must share one split_group to prevent leakage')
    return {'rows':len(df),'disciplines':int(df.discipline_id.nunique()),'events':int(df.event_id.nunique()),'bouts':int(df.bout_id.nunique()),'all_media_rights_cleared':True,'all_ml_permissions_documented':True,'video_paths_exist':True,'split_groups':int(df.split_group.nunique()),'note':'Validation checks recorded fields and local paths; it does not independently verify a license or legal basis.'}

def process_pro_video_manifest(path: Path, model_asset: Path, output_dir: Path, every_n_frames: int=5, max_frames: int=0) -> pd.DataFrame:
    """Extract licensed local video features; this step does not auto-label actions or train a classifier."""
    validate_pro_video_manifest(path)
    manifest_path=Path(path).resolve();df=pd.read_csv(manifest_path)
    df['video_path_resolved']=df.video_path.map(lambda value:str((manifest_path.parent/str(value)).resolve()) if not Path(str(value)).is_absolute() else str(Path(str(value)).resolve()))
    from .video_pipeline import run_video_analysis
    results=[];pose_frames=[];kinematic_frames=[]
    for row in df.to_dict('records'):
        safe_id=''.join(c.lower() if c.isalnum() else '-' for c in str(row['video_id'])).strip('-') or 'video'
        safe_discipline=''.join(c.lower() if c.isalnum() else '-' for c in str(row['discipline_id'])).strip('-')
        dest=Path(output_dir)/safe_discipline/safe_id
        try:
            info=run_video_analysis(Path(row['video_path_resolved']),Path(model_asset).resolve(),str(row['discipline_id']),str(row['bout_id']),'authorized',str(row['source_id']),str(row['license_id']),dest,every_n_frames,max_frames,video_id=str(row['video_id']))
            meta={key:row[key] for key in ('video_id','discipline_id','competition_name','event_id','bout_id','competition_date','ruleset','division','sex_class','competition_level','source_id','source_url','license_id','license_scope','media_rights_status','ml_training_permission','athlete_data_basis','annotation_protocol','split_group')}
            pose_frame=pd.read_csv(dest/'pose_landmarks.csv')
            for key,value in meta.items():pose_frame[key]=value
            pose_frame['action_label']=pd.NA
            pose_frames.append(pose_frame)
            kin_frame=pd.read_csv(dest/'kinematic_features.csv')
            for key,value in meta.items():kin_frame[key]=value
            kin_frame['action_label']=pd.NA
            kinematic_frames.append(kin_frame)
            results.append({'video_id':row['video_id'],'discipline_id':row['discipline_id'],'event_id':row['event_id'],'bout_id':row['bout_id'],'status':'features_extracted','pose_rows':info['pose_person_frame_rows'],'report':str(dest/'analysis_report.html')})
        except Exception as exc:
            results.append({'video_id':row['video_id'],'discipline_id':row['discipline_id'],'event_id':row['event_id'],'bout_id':row['bout_id'],'status':'processing_error','error':f'{type(exc).__name__}: {exc}'})
    result=pd.DataFrame(results);summary_path=Path(output_dir)/'batch_processing_summary.csv';summary_path.parent.mkdir(parents=True,exist_ok=True);result.to_csv(summary_path,index=False)
    if pose_frames:pd.concat(pose_frames,ignore_index=True).to_csv(Path(output_dir)/'pose_features_unlabeled.csv',index=False)
    if kinematic_frames:pd.concat(kinematic_frames,ignore_index=True).to_csv(Path(output_dir)/'kinematic_features_unlabeled.csv',index=False)
    return result
