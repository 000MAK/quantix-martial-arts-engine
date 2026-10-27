#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence-aware one-command report orchestrator for the 100 MA Engine.

It never creates synthetic performance scores. Empty inputs are reported as
missing coverage; provided inputs are validated and analyzed by the existing
engine modules. Video extraction is opt-in because it requires local footage,
explicit rights metadata and a MediaPipe model asset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from martial_arts_engine.catalog import load_catalog


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if pd.isna(value):
        return None
    if isinstance(value, (Path,)):
        return str(value)
    return value


def table_html(frame: pd.DataFrame, empty: str = "No verified observations supplied.") -> str:
    if frame.empty:
        return f'<p class="muted">{html.escape(empty)}</p>'
    return frame.to_html(index=False, border=0, classes="data", na_rep="—", escape=True)


def write_report(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "master_orchestrator_report.json"
    html_path = output_dir / "master_orchestrator_report.html"
    json_path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    coverage = pd.DataFrame(payload.get("coverage", []))
    eras = pd.DataFrame(payload.get("era_comparison", []))
    countries = pd.DataFrame(payload.get("country_summary", []))
    transitions = pd.DataFrame(payload.get("transition_summary", []))
    status_rows = pd.DataFrame(payload.get("modules", []))
    category_counts = coverage.groupby("category", dropna=False).size().to_dict() if not coverage.empty else {}
    bars = "".join(
        f'<div class="bar-row"><span>{html.escape(str(name))}</span><div class="track"><i style="width:{100 * count / max(category_counts.values(), default=1):.0f}%"></i></div><b>{count}</b></div>'
        for name, count in category_counts.items()
    ) or '<p class="muted">Catalog unavailable.</p>'
    summary = payload.get("summary", {})
    warnings = "".join(f"<li>{html.escape(str(item))}</li>" for item in payload.get("limitations", []))
    doc = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quantix Martial Arts Engine — Run Report</title><style>
:root{{--bg:#07111f;--panel:#101e31;--line:#223650;--text:#e7eef8;--muted:#9cb0c8;--cyan:#53d8e8;--orange:#ff9b54}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 85% 0,#183251 0,transparent 36%),var(--bg);color:var(--text);font:15px/1.55 system-ui,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:32px 20px 60px}}h1,h2{{line-height:1.2}}h1{{font-size:clamp(30px,5vw,48px);margin:8px 0}}h2{{font-size:20px;margin:0 0 15px}}p{{color:var(--muted)}}.eyebrow{{color:var(--cyan);font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:26px 0}}.card,.panel{{background:linear-gradient(145deg,#12243a,var(--panel));border:1px solid var(--line);border-radius:16px;padding:18px}}.card strong{{display:block;font-size:30px;color:var(--cyan)}}.card span{{color:var(--muted);font-size:13px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;margin:14px 0}}.panel{{margin:14px 0;overflow:auto}}.data{{border-collapse:collapse;width:100%;font-size:13px}}.data th,.data td{{border-bottom:1px solid var(--line);padding:9px;text-align:left;white-space:nowrap}}.data th{{color:var(--cyan)}}.bar-row{{display:grid;grid-template-columns:minmax(100px,180px) 1fr 30px;align-items:center;gap:12px;margin:10px 0;color:var(--muted)}}.track{{height:9px;border-radius:8px;background:#21344a;overflow:hidden}}.track i{{display:block;height:100%;background:linear-gradient(90deg,var(--cyan),var(--orange));border-radius:8px}}.muted{{color:var(--muted)}}li{{margin:8px 0;color:var(--muted)}}code{{color:#bdeff4}}footer{{border-top:1px solid var(--line);margin-top:30px;padding-top:16px;color:var(--muted);font-size:12px}}
</style></head><body><main>
<div class="eyebrow">Quantix · 100 Martial Arts Analytics</div><h1>Master run report</h1>
<p>Generated {html.escape(str(payload.get('generated_at_utc','')))} · Evidence-aware summary. Scores remain unavailable until measured data and a validated scoring protocol exist.</p>
<section class="cards"><div class="card"><strong>{summary.get('catalog_disciplines',0)}</strong><span>catalog entries</span></div><div class="card"><strong>{summary.get('video_rows',0)}</strong><span>rights-documented video rows</span></div><div class="card"><strong>{summary.get('annotated_bouts',0)}</strong><span>verified bout summaries</span></div><div class="card"><strong>{summary.get('event_rows',0)}</strong><span>verified transition events</span></div><div class="card"><strong>{summary.get('country_summary_rows',0)}</strong><span>country summary strata</span></div></section>
<div class="grid"><section class="panel"><h2>Catalog coverage by group</h2>{bars}<p>Bar length shows catalog entry counts, not quality or performance.</p></section><section class="panel"><h2>Module execution</h2>{table_html(status_rows, 'No modules were run.')}</section></div>
<section class="panel"><h2>Transition analysis</h2>{table_html(transitions)}</section>
<section class="panel"><h2>Era comparison</h2>{table_html(eras, 'No verified era metrics supplied; no era winner is inferred.')}</section>
<section class="panel"><h2>Country results by discipline and context</h2>{table_html(countries, 'No verified championship entries supplied; no country ranking is inferred.')}</section>
<section class="panel"><h2>Coverage by discipline</h2>{table_html(coverage)}</section>
<section class="panel"><h2>Interpretation limits</h2><ul>{warnings}</ul></section>
<footer>Generated locally by master_orchestrator.py. No random rankings, fabricated observations, or universal QMASE score are produced.</footer></main></body></html>'''
    html_path.write_text(doc, encoding="utf-8")
    return json_path, html_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    from martial_arts_engine.era_analysis import compare_eras
    from martial_arts_engine.pro_video_data import process_pro_video_manifest, validate_pro_video_manifest
    from martial_arts_engine.transitions import analyze_transitions

    catalog = load_catalog()
    catalog_frame = pd.DataFrame(catalog)
    coverage = catalog_frame[[c for c in ("discipline_id", "name", "category") if c in catalog_frame]].copy()
    coverage = coverage.rename(columns={"name": "discipline_name"})
    coverage["rights_documented_video_rows"] = 0
    coverage["verified_event_rows"] = 0
    coverage["coverage_status"] = "not_provided"
    modules: list[dict[str, Any]] = []
    transition_summary: list[dict[str, Any]] = []
    era_records: list[dict[str, Any]] = []
    country_records: list[dict[str, Any]] = []
    video_rows = 0
    annotated_bouts = 0
    event_rows = 0

    if args.video_manifest:
        try:
            validation = validate_pro_video_manifest(args.video_manifest)
            video_rows = validation["rows"]
            manifest = pd.read_csv(args.video_manifest)
            counts = manifest.groupby("discipline_id").video_id.nunique().to_dict()
            coverage["rights_documented_video_rows"] = coverage.discipline_id.map(counts).fillna(0).astype(int)
            coverage.loc[coverage.rights_documented_video_rows.gt(0), "coverage_status"] = "manifest_validated_features_not_run"
            modules.append({"module": "professional_video_manifest", "status": "validated", "details": validation})
            if args.model_asset:
                processed = process_pro_video_manifest(args.video_manifest, args.model_asset, args.output_dir / "pro_features", args.every_n_frames, args.max_frames)
                failures = int((~processed.status.eq("features_extracted")).sum()) if not processed.empty else 0
                modules.append({"module": "professional_video_feature_extraction", "status": "completed_with_failures" if failures else "completed", "failed_videos": failures, "details": processed.to_dict(orient="records")})
                good_ids = set(processed.loc[processed.status.eq("features_extracted"), "video_id"].astype(str)) if not processed.empty else set()
                for discipline, part in manifest.groupby("discipline_id"):
                    successful = int(part.video_id.astype(str).isin(good_ids).sum())
                    if successful:
                        coverage.loc[coverage.discipline_id.eq(discipline), "coverage_status"] = "features_extracted_unlabeled"
            else:
                modules.append({"module": "professional_video_feature_extraction", "status": "skipped", "details": "Pass --model-asset to extract pose and kinematic features."})
        except Exception as exc:
            modules.append({"module": "professional_video_pipeline", "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    else:
        modules.append({"module": "professional_video_pipeline", "status": "no_input", "details": "No manifest supplied; no video processed."})

    if args.event_log:
        try:
            events = pd.read_csv(args.event_log)
            required = {"sequence_id", "timestamp_ms", "discipline_id", "state_from", "state_to"}
            missing = required - set(events.columns)
            if missing:
                raise ValueError("Event log missing columns: " + ", ".join(sorted(missing)))
            event_rows = len(events)
            for discipline, part in events.groupby("discipline_id"):
                counts, matrix, dwell = analyze_transitions(part)
                slug = "".join(c.lower() if c.isalnum() else "-" for c in str(discipline)).strip("-")
                out = args.output_dir / "transitions"
                out.mkdir(parents=True, exist_ok=True)
                counts.to_csv(out / f"{slug}_transition_counts.csv")
                matrix.to_csv(out / f"{slug}_transition_matrix.csv")
                dwell.to_csv(out / f"{slug}_dwell_summary.csv", index=False)
                transition_summary.append({"discipline_id": discipline, "sequences": int(part.sequence_id.nunique()), "event_rows": len(part), "states": len(matrix), "transition_matrix_csv": str(out / f"{slug}_transition_matrix.csv"), "dwell_summary_csv": str(out / f"{slug}_dwell_summary.csv")})
                coverage.loc[coverage.discipline_id.eq(discipline), "verified_event_rows"] = len(part)
                coverage.loc[coverage.discipline_id.eq(discipline), "coverage_status"] = "event_log_supplied"
            modules.append({"module": "transition_matrix_and_dwell_summary", "status": "completed", "disciplines": len(transition_summary), "event_rows": event_rows})
        except Exception as exc:
            modules.append({"module": "transition_matrix_and_dwell_summary", "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    else:
        modules.append({"module": "transition_matrix_and_dwell_summary", "status": "no_input", "details": "No annotated event log supplied."})

    if args.era_metrics:
        try:
            era_path = args.output_dir / "era_comparison.csv"
            report_path = args.output_dir / "era_comparison.md"
            eras = compare_eras(args.era_metrics, era_path, report_path, args.bootstrap_reps)
            era_records = eras.to_dict(orient="records")
            annotated_bouts = int(eras.bouts.sum()) if not eras.empty else 0
            modules.append({"module": "stratified_era_comparison", "status": "completed", "strata": len(eras), "verified_bouts": annotated_bouts, "csv": str(era_path), "report": str(report_path), "details": "Descriptive within-stratum comparison; no universal best era selected."})
        except Exception as exc:
            modules.append({"module": "stratified_era_comparison", "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    else:
        modules.append({"module": "stratified_era_comparison", "status": "no_input", "details": "No verified bout-metric table supplied."})

    if args.country_results:
        try:
            from martial_arts_engine.country_analysis import analyze_country_results
            country_path = args.output_dir / "country_dominance.csv"
            country = analyze_country_results(args.country_results, country_path, args.bootstrap_reps)
            country_records = country.to_dict(orient="records")
            modules.append({"module": "country_dominance_by_stratum", "status": "completed", "summary_rows": len(country), "csv": str(country_path), "details": "Represented-country and nationality views are separate; no composite score or universal ranking."})
        except Exception as exc:
            modules.append({"module": "country_dominance_by_stratum", "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    else:
        modules.append({"module": "country_dominance_by_stratum", "status": "no_input", "details": "No verified athlete-event championship table supplied."})

    coverage["coverage_status"] = coverage.apply(
        lambda r: "video_and_events" if r.rights_documented_video_rows and r.verified_event_rows else r.coverage_status if r.coverage_status == "features_extracted_unlabeled" else "video_manifest" if r.rights_documented_video_rows else "events_only" if r.verified_event_rows else r.coverage_status,
        axis=1,
    )
    limitations = [
        "Catalog entries are identifiers, not a claim that every entry is a distinct or directly comparable sport.",
        "No random performance values are generated. No global QMASE score or discipline ranking is calculated because no validated weighting protocol and complete measured data exist.",
        "Video feature extraction requires local footage, recorded rights/consent fields and a MediaPipe .task model. Field validation does not independently verify legal rights.",
        "Pose and kinematic features are not automatic technique labels. Human-verified annotations are required for transition, takedown and era metrics.",
        "An era comparison is descriptive and stratified; conclusions depend on rules, division, sex class, competition level, sample coverage and annotation quality.",
        "Country aggregates are descriptive: represented team/NOC and recorded nationality remain separate, and event rates may still reflect repeated-athlete dependence and unequal access to competition.",
    ]
    return {
        "project": "Quantix 100 Martial Arts Analytics Engine",
        "report_type": "evidence_aware_master_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {"catalog_disciplines": len(catalog), "video_rows": video_rows, "annotated_bouts": annotated_bouts, "event_rows": event_rows, "country_summary_rows": len(country_records)},
        "modules": modules,
        "coverage": coverage.to_dict(orient="records"),
        "transition_summary": transition_summary,
        "era_comparison": era_records,
        "country_summary": country_records,
        "limitations": limitations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the evidence-aware 100-discipline report pipeline.")
    parser.add_argument("--video-manifest", type=Path, help="Validated manifest of local footage with explicit rights/ML permissions.")
    parser.add_argument("--model-asset", type=Path, help="MediaPipe Pose Landmarker .task asset; enables video feature extraction.")
    parser.add_argument("--event-log", type=Path, help="Human-verified event log for empirical transition matrices.")
    parser.add_argument("--era-metrics", type=Path, help="Verified one-row-per-bout metrics for stratified era summaries.")
    parser.add_argument("--country-results", type=Path, help="Verified championship entries with representation and nationality recorded separately.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "master_run")
    parser.add_argument("--every-n-frames", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    args = parser.parse_args()
    if args.model_asset and not args.video_manifest:
        parser.error("--model-asset requires --video-manifest")
    if args.every_n_frames < 1 or args.max_frames < 0 or args.bootstrap_reps < 1:
        parser.error("frame intervals and bootstrap repetitions must be positive; max frames cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    payload = run(args)
    json_path, html_path = write_report(payload, args.output_dir)
    print(f"Master report written:\n  JSON: {json_path}\n  HTML: {html_path}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    errors = [m for m in payload["modules"] if m.get("status") == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
