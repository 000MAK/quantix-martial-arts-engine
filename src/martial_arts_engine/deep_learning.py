from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report

def train_pose_mlp(csv_path:Path,out_dir:Path,target:str='action_label',group_col:str='split_group',epochs:int=30)->dict:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader,TensorDataset
    except ImportError as e:
        raise RuntimeError('Pose MLP requires requirements-deep-learning.txt (PyTorch).') from e
    frame=pd.read_csv(csv_path)
    pose_cols=[c for c in frame.columns if c.startswith('pose_') and pd.api.types.is_numeric_dtype(frame[c])]
    if not pose_cols: raise ValueError('No numeric pose_* feature columns found. Extract poses, add labels, and provide split_group.')
    if target not in frame or group_col not in frame: raise ValueError(f'Expected target={target} and group column={group_col}')
    frame=frame.dropna(subset=[target,group_col]).copy(); groups=frame[group_col].astype(str); labels=frame[target].astype(str)
    idx=np.arange(len(frame)); outer=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42); trval,te=next(outer.split(idx,labels,groups))
    inner=GroupShuffleSplit(n_splits=1,test_size=.1875,random_state=43); tr_rel,val_rel=next(inner.split(trval,labels.iloc[trval],groups.iloc[trval])); tr=trval[tr_rel]; va=trval[val_rel]
    classes=sorted(labels.iloc[tr].unique()); class_ix={c:i for i,c in enumerate(classes)}
    if len(classes)<2: raise ValueError('Need at least two action classes in training groups')
    x=frame[pose_cols].to_numpy(dtype=np.float32); mu=x[tr].mean(0); sd=x[tr].std(0); sd[sd<1e-6]=1
    x=np.nan_to_num((x-mu)/sd,nan=0.0,posinf=0.0,neginf=0.0)
    def ds(ix):
        keep=[i for i in ix if labels.iloc[i] in class_ix]
        yy=np.array([class_ix[labels.iloc[i]] for i in keep],dtype=np.int64)
        return TensorDataset(torch.from_numpy(x[keep]),torch.from_numpy(yy)),keep
    train_set,_=ds(tr); val_set,val_keep=ds(va); test_set,test_keep=ds(te)
    if not len(test_keep): raise ValueError('No held-out test labels are represented in training; collect more sessions or revise the group split')
    torch.manual_seed(42); device='cuda' if torch.cuda.is_available() else 'cpu'
    model=nn.Sequential(nn.Linear(len(pose_cols),256),nn.ReLU(),nn.Dropout(.25),nn.Linear(256,128),nn.ReLU(),nn.Dropout(.15),nn.Linear(128,len(classes))).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); lossf=nn.CrossEntropyLoss(); history=[]
    for epoch in range(epochs):
        model.train(); losses=[]
        for bx,by in DataLoader(train_set,batch_size=64,shuffle=True):
            bx,by=bx.to(device),by.to(device); opt.zero_grad(); loss=lossf(model(bx),by); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        model.eval(); hit=total=0
        with torch.no_grad():
            for bx,by in DataLoader(val_set,batch_size=128):
                pred=model(bx.to(device)).argmax(1).cpu(); hit+=(pred==by).sum().item(); total+=len(by)
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)) if losses else None,'validation_accuracy':hit/total if total else None})
    def evaluate(ds_):
        model.eval(); ys=[];preds=[]
        with torch.no_grad():
            for bx,by in DataLoader(ds_,batch_size=128):
                ys.extend(by.numpy().tolist());preds.extend(model(bx.to(device)).argmax(1).cpu().numpy().tolist())
        return {'accuracy':float(accuracy_score(ys,preds)),'balanced_accuracy':float(balanced_accuracy_score(ys,preds)),'classification_report':classification_report(ys,preds,output_dict=True,zero_division=0)}
    metrics={'model':'pose_mlp','device':device,'classes':classes,'feature_count':len(pose_cols),'split_group':group_col,'train_groups':sorted(groups.iloc[tr].unique().tolist()),'validation_groups':sorted(groups.iloc[va].unique().tolist()),'test_groups':sorted(groups.iloc[te].unique().tolist()),'history':history,'test':evaluate(test_set)}
    out_dir.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.cpu().state_dict(),'mean':mu.tolist(),'scale':sd.tolist(),'features':pose_cols,'classes':classes},out_dir/'pose_mlp.pt'); (out_dir/'pose_mlp_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');return metrics
