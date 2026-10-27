"""Local SQLite store for analysis outputs (standard library only, no server).

`ingest_outputs` loads Hawkes and semi-Markov results into tables; each summary is
stored once (keyed by its SHA-256), so re-running ingestion is idempotent.
`query` opens the database read-only, so ad-hoc SQL cannot modify stored results.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id INTEGER PRIMARY KEY,
    analysis TEXT NOT NULL,
    dataset_id TEXT, license_id TEXT, task TEXT,
    summary_sha256 TEXT NOT NULL UNIQUE,
    ingested_at TEXT NOT NULL,
    summary_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS position_change_events (
    run_id INTEGER NOT NULL REFERENCES analysis_runs(run_id),
    sequence_id TEXT, time_s REAL, state_from TEXT, state_to TEXT, timing_uncertainty_s REAL
);
CREATE TABLE IF NOT EXISTS position_episodes (
    run_id INTEGER NOT NULL REFERENCES analysis_runs(run_id),
    sequence_id TEXT, position TEXT, next_position TEXT,
    start_s REAL, duration_s REAL, observed INTEGER, left_censored INTEGER
);
"""

# analysis name -> (summary file, detail CSV, detail table)
SOURCES = {
    "hawkes": ("vicos_hawkes/hawkes_summary.json", "vicos_hawkes/position_change_events.csv", "position_change_events"),
    "semi_markov": ("vicos_semi_markov/semi_markov_summary.json", "vicos_semi_markov/position_episodes.csv", "position_episodes"),
}


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def ingest_outputs(db_path: Path, outputs_root: Path) -> dict:
    report = {}
    conn = connect(db_path)
    try:
        _ingest(conn, outputs_root, report)
        conn.commit()
    finally:
        # sqlite3's context manager only commits; closing releases the Windows file lock.
        conn.close()
    return report


def _ingest(conn: sqlite3.Connection, outputs_root: Path, report: dict) -> None:
    for analysis, (summary_rel, detail_rel, table) in SOURCES.items():
        summary_path = outputs_root / summary_rel
        if not summary_path.exists():
            report[analysis] = "not_found"
            continue
        raw = summary_path.read_text(encoding="utf-8")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if conn.execute("SELECT 1 FROM analysis_runs WHERE summary_sha256 = ?", (digest,)).fetchone():
            report[analysis] = "already_ingested"
            continue
        summary = json.loads(raw)
        cur = conn.execute(
            "INSERT INTO analysis_runs (analysis, dataset_id, license_id, task, summary_sha256, ingested_at, summary_json) VALUES (?,?,?,?,?,?,?)",
            (analysis, summary.get("dataset_id"), summary.get("license_id"), summary.get("task"), digest,
             datetime.now(timezone.utc).isoformat(), raw))
        run_id = cur.lastrowid
        rows = 0
        detail_path = outputs_root / detail_rel
        if detail_path.exists():
            detail = pd.read_csv(detail_path, dtype={"sequence_id": str})
            for col in ("observed", "left_censored"):
                if col in detail.columns:
                    detail[col] = detail[col].astype(str).eq("True").astype(int)
            detail.insert(0, "run_id", run_id)
            detail.to_sql(table, conn, if_exists="append", index=False)
            rows = len(detail)
        report[analysis] = {"run_id": run_id, "detail_rows": rows}


def query(db_path: Path, sql: str) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()
