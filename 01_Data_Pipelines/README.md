# 01 · Data Pipelines

Owns the 100-row taxonomy, versioned input contracts, source/license provenance and event-log validation. `core_engine.py` builds a deterministic 100-row schema; all unmeasured metrics remain null. Templates live in `templates/`; source video/audio belongs under `data/raw/` only when rights and consent permit it.

Data contracts: `templates/event_log_template.csv`, `video_manifest_template.csv`, and `audio_manifest_template.csv`. The executable validator and readers live in `src/martial_arts_engine/events.py`.
