# 06 · Arcade Integration

`arcade_event_template.csv` defines an adapter contract for exporting QUANTIX BJJ or other game telemetry into the shared event-log format. `python main.py import-arcade --input GAME_EVENTS.jsonl` converts JSONL events. Rows default to `synthetic=true`; never mix game telemetry with competition observations without a source-type field and explicit analysis filter.
