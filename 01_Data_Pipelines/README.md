# 01 · Data Pipelines

Owns the 100-row taxonomy, versioned input contracts, source/license provenance and event-log validation. `core_engine.py` builds a deterministic 100-row schema; all unmeasured metrics remain null. Templates live in `templates/`; source video/audio belongs under `data/raw/` only when rights and consent permit it.

Data contracts: `templates/event_log_template.csv`, `video_manifest_template.csv`, and `audio_manifest_template.csv`. The executable validator and readers live in `src/martial_arts_engine/events.py`.

## Multimodal stream alignment

`python main.py fuse-streams --manifest my_manifest.json` aligns time-stamped CSV streams (pose, audio features, heart rate, event labels) onto one time axis (`fusion.py`). Copy `templates/fusion_manifest_template.json`; every stream needs `source_id`, `license_id` and `consent_status: granted`, otherwise it is refused. Samples further apart than `tolerance_s` stay empty (no interpolation) and each stream gets a `<name>_lag_s` column. Use `offset_s` to correct device clock offsets; `estimate_offset` can recover one from events visible in two streams (e.g. a clap on video and audio). No dataset currently in the registry contains co-recorded video, audio and heart rate, so this is tested on fixtures only.

## Local analysis store

`python main.py db-ingest` loads the Hawkes and semi-Markov outputs into `outputs/quantix_vault.db` (SQLite, standard library; git-ignored). Each summary is stored once, keyed by its SHA-256. `python main.py db-query --sql "SELECT ..."` opens the database read-only.
