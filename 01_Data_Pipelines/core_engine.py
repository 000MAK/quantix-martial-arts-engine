"""Deterministic catalog/schema access for the Global Martial Arts platform.

This module deliberately does not fabricate engagement, injury, or performance
values. Those fields remain missing until supported by licensed observations.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'data' / 'martial_arts_catalog.json'

class GlobalMartialArtsEngine:
    def __init__(self, base_path: str | Path = ROOT):
        self.base_path = Path(base_path)
        self.catalog_path = self.base_path / 'data' / 'martial_arts_catalog.json'

    def load_catalog(self) -> list[dict]:
        payload = json.loads(self.catalog_path.read_text(encoding='utf-8'))
        entries = payload['entries']
        ids = [item['discipline_id'] for item in entries]
        if len(entries) != 100 or len(ids) != len(set(ids)):
            raise ValueError('Catalog must have 100 rows with unique discipline_id values')
        return entries

    def generate_global_schema(self) -> pd.DataFrame:
        """Return 100 schema rows with unobserved measurements left null."""
        frame = pd.DataFrame(self.load_catalog())
        frame['Avg_Engagement_Time_Sec'] = pd.NA
        frame['Injury_Risk_Coefficient'] = pd.NA
        frame['Dominant_Modality'] = pd.NA
        frame['analysis_profile'] = frame['category'].map({
            'striking_standup': 'pose+audio+events',
            'grappling_ground': 'pose+transitions+events',
            'hybrid_modern': 'multimodal+transitions',
            'traditional_weapons_forms': 'pose+form+audio'
        })
        return frame

    def export_schema(self, output: str | Path | None = None) -> Path:
        target = Path(output) if output else self.base_path / 'data' / 'global_schema.csv'
        target.parent.mkdir(parents=True, exist_ok=True)
        self.generate_global_schema().to_csv(target, index=False)
        return target

if __name__ == '__main__':
    engine = GlobalMartialArtsEngine()
    frame = engine.generate_global_schema()
    print(f'Catalog rows: {len(frame)}; measured performance/injury values: 0')
    print(engine.export_schema())
