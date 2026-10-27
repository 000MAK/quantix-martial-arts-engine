from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
CATALOG_PATH=ROOT/'data'/'martial_arts_catalog.json'

def load_catalog(path:Path=CATALOG_PATH)->list[dict]:
    payload=json.loads(path.read_text(encoding='utf-8'))
    entries=payload['entries']
    id_field='discipline_id' if entries and 'discipline_id' in entries[0] else 'id'
    ids=[x[id_field] for x in entries]
    if payload.get('total_disciplines')!=100 or len(entries)!=100:
        raise ValueError('Catalog must contain exactly 100 entries')
    if len(ids)!=len(set(ids)):
        raise ValueError('Catalog IDs must be unique')
    return entries

def catalog_frame(path:Path=CATALOG_PATH)->pd.DataFrame:
    frame=pd.DataFrame(load_catalog(path))
    frame['modalities']=frame['modalities'].map('|'.join)
    return frame

def export_catalog(path:Path, fmt:str='csv')->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if fmt=='json':
        path.write_text(CATALOG_PATH.read_text(encoding='utf-8'),encoding='utf-8')
    else:
        catalog_frame().to_csv(path,index=False)
