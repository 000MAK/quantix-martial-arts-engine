from __future__ import annotations
from pathlib import Path
import json
import shutil
import subprocess
import pandas as pd
from .catalog import load_catalog
from .vision import extract_pose_csv
from .kinematics import derive_kinematics
from .events import read_event_log
from .transitions import analyze_transitions
from .analysis_report import write_analysis_report

def _extract_audio(video: Path, target: Path) -> tuple[Path|None,str]:
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if not ffmpeg:
        return None,'ffmpeg was not found; pass --audio-path or install requirements-audio.txt (includes a bundled ffmpeg executable).'
    proc=subprocess.run([ffmpeg,'-nostdin','-hide_banner','-loglevel','error','-y','-i',str(video),'-vn','-ac','1','-ar','16000',str(target)],capture_output=True,text=True)
    if proc.returncode!=0:
        return None,'Audio track could not be extracted: '+(proc.stderr.strip()[-500:] or 'ffmpeg returned an error')
    return target,'Audio track extracted for feature summaries only.'

def run_video_analysis(video_path: Path, model_asset: Path, discipline_id: str, sequence_id: str,
        consent_status: str, source_id: str, license_id: str, output_dir: Path,
        every_n_frames: int=5, max_frames: int=0, audio_path: Path|None=None,
        heart_rate_csv: Path|None=None, event_log_csv: Path|None=None, sensor_consent_status: str|None=None,
        audio_source_id: str|None=None, audio_license_id: str|None=None, audio_consent_status: str|None=None,
        sensor_source_id: str|None=None, sensor_license_id: str|None=None,
        strike_windows_csv: Path|None=None, takedown_attempts_csv: Path|None=None, video_id: str|None=None) -> dict:
    """Run local pose/feature extraction and emit an honest, limitation-aware report."""
    video_path=Path(video_path).resolve();model_asset=Path(model_asset).resolve();output_dir=Path(output_dir).resolve()
    if consent_status.strip().lower() not in {'granted','authorized'}:
        raise ValueError('Video analysis requires documented consent_status=granted (or authorized).')
    if not source_id.strip() or not license_id.strip():
        raise ValueError('Provide source_id and license_id/rights basis for every analyzed video.')
    if not video_path.is_file(): raise FileNotFoundError(video_path)
    catalog=load_catalog();entry=next((x for x in catalog if x['discipline_id']==discipline_id),None)
    if entry is None: raise ValueError(f'Unknown discipline_id: {discipline_id}')
    output_dir.mkdir(parents=True,exist_ok=True)
    pose_csv=output_dir/'pose_landmarks.csv';kinematics_csv=output_dir/'kinematic_features.csv'
    video_id=video_id or video_path.stem
    n_people_frames=extract_pose_csv(video_path,model_asset,pose_csv,every_n_frames,max_frames,video_id)
    kinematics=derive_kinematics(pose_csv,kinematics_csv)
    payload={'pipeline':'QMASE local video research pipeline','pipeline_version':'0.1.0','video_id':video_id,'discipline_id':discipline_id,'discipline_name':entry['name'],'sequence_id':sequence_id,'source_id':source_id,'license_id':license_id,'consent_status':'authorized','pose_person_frame_rows':n_people_frames,'pose_status':'features_extracted' if n_people_frames else 'no_pose_detected','kinematics_status':'2d_camera_projection_features','audio_status':'not_processed','heart_rate_status':'not_supplied','transition_status':'event_labels_not_supplied','generated_files':['pose_landmarks.csv','kinematic_features.csv','analysis_report.html','analysis_report.json']}
    # Video-audio extraction is optional and never converts acoustics into an
    # impact-quality or coach-command label.
    candidate_audio=Path(audio_path).resolve() if audio_path else None
    if audio_path and (not audio_source_id or not audio_license_id or not audio_consent_status or audio_consent_status.strip().lower() not in {'granted','authorized'}):
        raise ValueError('A separately supplied audio file requires its own source ID, license/rights ID and consent status.')
    extracted_audio=False
    if candidate_audio is None:
        candidate_audio,payload['audio_status']=_extract_audio(video_path,output_dir/'audio_track.wav')
        extracted_audio=candidate_audio is not None
    audio_summary=None
    if candidate_audio is not None and candidate_audio.exists():
        manifest=output_dir/'audio_manifest.csv'
        try:
            from .audio import extract_audio_manifest
            pd.DataFrame([{'session_id':sequence_id,'discipline_id':discipline_id,'split_group':sequence_id,'audio_path':str(candidate_audio),'source_id':audio_source_id or source_id,'license_id':audio_license_id or license_id,'consent_status':audio_consent_status or consent_status}]).to_csv(manifest,index=False)
            features=extract_audio_manifest(manifest,output_dir/'audio_features.csv')
            audio_summary=features.iloc[0].to_dict() if not features.empty else None
            payload['audio_status']='acoustic_features_extracted' if audio_summary else 'no_audio_features'
            payload['generated_files'] += ['audio_features.csv']
        except (RuntimeError,ImportError) as exc:
            payload['audio_status']='optional_audio_dependencies_missing'
            payload['audio_note']=str(exc)
        finally:
            manifest.unlink(missing_ok=True)
            if extracted_audio: candidate_audio.unlink(missing_ok=True)
    if heart_rate_csv:
        hr=pd.read_csv(heart_rate_csv)
        required={'timestamp_s','heart_rate_bpm'}
        if required-set(hr.columns): raise ValueError('Heart-rate input requires timestamp_s and heart_rate_bpm columns')
        if sensor_source_id: hr['source_id']=sensor_source_id
        if sensor_license_id: hr['license_id']=sensor_license_id
        if sensor_consent_status: hr['sensor_consent_status']=sensor_consent_status
        for field in ('source_id','license_id','sensor_consent_status'):
            if field not in hr or hr[field].isna().any() or (hr[field].astype(str).str.strip()=='').any():
                raise ValueError(f'Heart-rate input requires populated {field} provenance')
        if not hr.sensor_consent_status.astype(str).str.lower().isin({'granted','authorized'}).all():
            raise ValueError('Heart-rate processing requires separate documented sensor_consent_status=granted (or authorized).')
        hr['timestamp_s']=pd.to_numeric(hr.timestamp_s,errors='raise');hr['heart_rate_bpm']=pd.to_numeric(hr.heart_rate_bpm,errors='raise')
        if (hr.heart_rate_bpm<=0).any(): raise ValueError('heart_rate_bpm values must be positive')
        hr['consent_status']='authorized'
        hr=hr.sort_values('timestamp_s').reset_index(drop=True)
        hr.to_csv(output_dir/'heart_rate_review.csv',index=False)
        payload['heart_rate_status']='sensor_trace_available';payload['heart_rate_summary']={'samples':len(hr),'median_bpm':float(hr.heart_rate_bpm.median()),'p90_bpm':float(hr.heart_rate_bpm.quantile(.9)),'max_bpm':float(hr.heart_rate_bpm.max())}
        payload['generated_files'] += ['heart_rate_review.csv']
    else: hr=None
    matrix=None
    if event_log_csv:
        events=read_event_log(event_log_csv)
        matched=events[(events.discipline_id==discipline_id)&(events.sequence_id==str(sequence_id))]
        if matched.empty: raise ValueError('No annotated events match both the selected discipline_id and sequence_id')
        if 'annotation_status' not in matched or not matched.annotation_status.astype(str).str.lower().eq('verified').all():
            raise ValueError('Tactical transition output requires annotation_status=verified on every matched event row')
        if 'annotation_protocol' not in matched or matched.annotation_protocol.isna().any():
            raise ValueError('Tactical transition output requires a documented annotation_protocol on each matched event row')
        if not {'source_id','license_id'}.issubset(matched.columns) or matched[['source_id','license_id']].isna().any().any():
            raise ValueError('Tactical event annotations require source_id and license_id provenance')
        counts,matrix,dwell=analyze_transitions(matched)
        counts.to_csv(output_dir/'transition_counts.csv');matrix.to_csv(output_dir/'transition_matrix.csv');dwell.to_csv(output_dir/'dwell_summary.csv',index=False)
        payload['transition_status']='human_annotated_events_analyzed';payload['transition_event_rows']=len(matched)
        payload['generated_files'] += ['transition_counts.csv','transition_matrix.csv','dwell_summary.csv']
    from .performance_metrics import guard_position_proxy,coordination_windows,summarize_takedown_attempts,write_performance_dashboard
    guard=guard_position_proxy(pose_csv,output_dir/'guard_proxy_frames.csv',source_id=source_id,license_id=license_id,consent_status='authorized')
    payload['guard_position_proxy']=guard;payload['generated_files'].append('guard_proxy_frames.csv')
    coordination=None;takedown_rates=None
    if strike_windows_csv:
        coordination=coordination_windows(pose_csv,kinematics_csv,strike_windows_csv,output_dir/'coordination_windows.csv')
        payload['coordination_windows']=len(coordination);payload['generated_files'].append('coordination_windows.csv')
    if takedown_attempts_csv:
        takedown_rates=summarize_takedown_attempts(takedown_attempts_csv,output_dir/'takedown_rates.csv',discipline_id,sequence_id)
        payload['takedown_rate_groups']=len(takedown_rates);payload['generated_files'].append('takedown_rates.csv')
    performance_dashboard=output_dir/'performance_dashboard.html'
    write_performance_dashboard(performance_dashboard,guard,coordination,takedown_rates)
    payload['performance_dashboard']=performance_dashboard.name;payload['generated_files'].append(performance_dashboard.name)
    report_path=output_dir/'analysis_report.html'
    write_analysis_report(report_path,payload,kinematics,hr,matrix,audio_summary)
    return payload
