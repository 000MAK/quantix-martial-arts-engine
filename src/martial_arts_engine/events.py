from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED = {'sequence_id', 'timestamp_ms', 'discipline_id', 'state_from', 'state_to'}

def read_event_log(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError('Missing required event-log columns: ' + ', '.join(sorted(missing)))
    frame = frame.dropna(subset=list(REQUIRED)).copy()
    frame['timestamp_ms'] = pd.to_numeric(frame['timestamp_ms'], errors='raise')
    if (frame['timestamp_ms'] < 0).any():
        raise ValueError('timestamp_ms must be non-negative')
    for col in ('sequence_id', 'discipline_id', 'state_from', 'state_to'):
        frame[col] = frame[col].astype(str).str.strip()
        if (frame[col] == '').any():
            raise ValueError(f'{col} contains an empty value')
    if 'action_label' not in frame:
        frame['action_label'] = frame['state_to']
    return frame.sort_values(['discipline_id', 'sequence_id', 'timestamp_ms']).reset_index(drop=True)
