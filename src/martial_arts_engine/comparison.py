from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED = {'discipline_id','task','metric_name','metric_value','unit','sample_size','uncertainty_lower','uncertainty_upper','protocol_id','test_context'}

def compare_disciplines(path: Path, task: str, metric: str, protocol_id: str, test_context: str, higher_is_better: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError('Evaluation file missing columns: ' + ', '.join(sorted(missing)))
    df = df[(df.task == task) & (df.metric_name == metric) & (df.protocol_id == protocol_id) & (df.test_context == test_context)].copy()
    if df.empty:
        raise ValueError('No rows match the requested task, metric, protocol_id and test_context')
    if df.unit.nunique() != 1:
        raise ValueError('Cannot compare rows with different units')
    for col in ('metric_value','sample_size','uncertainty_lower','uncertainty_upper'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[['metric_value','sample_size','uncertainty_lower','uncertainty_upper']].isna().any().any():
        raise ValueError('All compared rows need numeric metric values, sample sizes and uncertainty bounds')
    if (df.sample_size <= 0).any() or (df.uncertainty_lower > df.uncertainty_upper).any():
        raise ValueError('Sample sizes must be positive and lower uncertainty bounds must not exceed upper bounds')
    if df.discipline_id.duplicated().any():
        raise ValueError('Provide one precomputed estimate per discipline for this exact comparison context')
    df['rank'] = df.metric_value.rank(method='min', ascending=not higher_is_better).astype(int)
    best = df.loc[df['rank'] == 1]
    best_lo = best.uncertainty_lower.min()
    best_hi = best.uncertainty_upper.max()
    # This flag only reports interval overlap with the point-estimate leader;
    # it is not a formal test of statistical significance.
    df['interval_overlaps_point_leader'] = (df.uncertainty_lower <= best_hi) & (df.uncertainty_upper >= best_lo)
    cols = ['rank','discipline_id','metric_value','unit','sample_size','uncertainty_lower','uncertainty_upper','interval_overlaps_point_leader']
    return df.sort_values(['rank','discipline_id'])[cols].reset_index(drop=True)
