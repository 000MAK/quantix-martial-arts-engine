from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def extract_audio_manifest(manifest:Path,out_csv:Path,sample_rate:int=16000,n_mfcc:int=20)->pd.DataFrame:
    try:
        import librosa
    except ImportError as e:
        raise RuntimeError('Audio features require requirements-audio.txt (librosa + soundfile).') from e
    table=pd.read_csv(manifest)
    needed={'session_id','discipline_id','split_group','audio_path'}
    missing=needed-set(table.columns)
    if missing: raise ValueError('Audio manifest is missing: '+', '.join(sorted(missing)))
    rows=[]
    for row in table.itertuples(index=False):
        audio_path=Path(row.audio_path)
        if not audio_path.is_absolute(): audio_path=manifest.parent/audio_path
        y,sr=librosa.load(audio_path,sr=sample_rate,mono=True)
        if len(y)==0: raise ValueError(f'Empty audio file: {audio_path}')
        mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_mfcc)
        spec=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=64)
        rms=librosa.feature.rms(y=y)[0]
        centroid=librosa.feature.spectral_centroid(y=y,sr=sr)[0]
        rec={'session_id':row.session_id,'discipline_id':row.discipline_id,'split_group':row.split_group,'action_label':getattr(row,'label',None),'source_id':getattr(row,'source_id',None),'license_id':getattr(row,'license_id',None),'consent_status':getattr(row,'consent_status',None),'duration_seconds':len(y)/sr,'rms_mean':float(np.mean(rms)),'rms_std':float(np.std(rms)),'centroid_mean':float(np.mean(centroid)),'centroid_std':float(np.std(centroid)),'mel_energy_mean':float(np.mean(spec))}
        for i in range(n_mfcc): rec[f'mfcc_{i}_mean']=float(np.mean(mfcc[i])); rec[f'mfcc_{i}_std']=float(np.std(mfcc[i]))
        rows.append(rec)
    result=pd.DataFrame(rows); out_csv.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out_csv,index=False); return result
