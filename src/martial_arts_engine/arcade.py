from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def import_arcade_jsonl(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Normalize game telemetry into event-log columns; rows are synthetic gameplay by default."""
    rows=[]
    with input_path.open(encoding='utf-8-sig') as stream:
        for line_no,line in enumerate(stream,1):
            if not line.strip():
                continue
            try:
                item=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON on line {line_no}: {exc}') from exc
            aliases={
                'sequence_id': item.get('sequence_id',item.get('game_id')),
                'timestamp_ms': item.get('timestamp_ms'),
                'discipline_id': item.get('discipline_id'),
                'state_from': item.get('state_from',item.get('from_state')),
                'state_to': item.get('state_to',item.get('to_state')),
                'action_label': item.get('action_label',item.get('action')),
                'outcome': item.get('outcome',item.get('result')),
                'source_system': item.get('source_system','arcade_game'),
                'synthetic': item.get('synthetic',True)
            }
            required=('sequence_id','timestamp_ms','discipline_id','state_from','state_to')
            missing=[key for key in required if aliases[key] is None]
            if missing:
                raise ValueError(f'Line {line_no} missing required telemetry fields: {", ".join(missing)}')
            aliases['timestamp_ms']=int(aliases['timestamp_ms'])
            rows.append(aliases)
    frame=pd.DataFrame(rows)
    if not frame.empty:
        frame=frame.sort_values(['discipline_id','sequence_id','timestamp_ms']).reset_index(drop=True)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output_path,index=False)
    return frame
