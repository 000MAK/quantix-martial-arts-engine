from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

ID_COLUMNS={'sequence_id','session_id','athlete_id','video_id','clip_id','timestamp_ms','frame_index','audio_path','video_path','annotator_id','license_id','consent_status','split_group','state_to','outcome'}

def train_event_models(frame:pd.DataFrame,out_dir:Path,target:str='action_label',group_col:str='sequence_id')->dict:
    if target not in frame: raise ValueError(f'Target column not found: {target}')
    if group_col not in frame: raise ValueError(f'Group column not found: {group_col}; group-aware split is required')
    df=frame.dropna(subset=[target,group_col]).copy()
    groups=df[group_col].astype(str)
    if groups.nunique()<3: raise ValueError('Need at least 3 distinct session/sequence groups for a group-aware holdout')
    candidates=[c for c in df.columns if c not in ID_COLUMNS|{target,'outcome'} and df[c].nunique(dropna=True)>1]
    if not candidates: raise ValueError('No usable feature columns found')
    cat=[c for c in candidates if not pd.api.types.is_numeric_dtype(df[c])]
    num=[c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    split=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42)
    train_idx,test_idx=next(split.split(df[candidates],df[target],groups))
    Xtr,Xte=df.iloc[train_idx][candidates],df.iloc[test_idx][candidates]
    ytr,yte=df.iloc[train_idx][target].astype(str),df.iloc[test_idx][target].astype(str)
    if ytr.nunique()<2: raise ValueError('Training partition has fewer than two classes; collect more independent sessions')
    pre=ColumnTransformer([
      ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
      ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)
    ])
    models={'logistic_regression':LogisticRegression(max_iter=1000,class_weight='balanced'),'random_forest':RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',random_state=42,n_jobs=-1,min_samples_leaf=2)}
    out_dir.mkdir(parents=True,exist_ok=True); reports={}
    for name,estimator in models.items():
        pipe=Pipeline([('preprocess',pre),('model',estimator)]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
        report={'accuracy':float(accuracy_score(yte,pred)),'balanced_accuracy':float(balanced_accuracy_score(yte,pred)),'classification_report':classification_report(yte,pred,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(yte,pred,labels=sorted(set(ytr)|set(yte))).tolist(),'labels':sorted(set(ytr)|set(yte)),'test_groups':sorted(groups.iloc[test_idx].unique().tolist()),'train_groups':sorted(groups.iloc[train_idx].unique().tolist()),'feature_columns':candidates,'target':target,'split_unit':group_col}
        reports[name]=report; joblib.dump(pipe,out_dir/f'{name}.joblib')
    (out_dir/'metrics.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    return reports
