from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from martial_arts_engine.catalog import load_catalog,catalog_frame,export_catalog
from martial_arts_engine.events import read_event_log
from martial_arts_engine.transitions import analyze_transitions,synthetic_demo
from martial_arts_engine.vicos_bjj import read_vicos_annotations,write_vicos_analysis
from martial_arts_engine.vicos_pose_training import train_vicos_pose_classifier
from martial_arts_engine.modeling import train_event_models
from martial_arts_engine.comparison import compare_disciplines

ROOT=Path(__file__).resolve().parent

def main()->None:
    ap=argparse.ArgumentParser(description='100-discipline martial-arts analytics research scaffold')
    sub=ap.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('catalog',help='List or export the 100-discipline taxonomy');c.add_argument('--format',choices=['table','csv','json'],default='table');c.add_argument('--out',type=Path)
    v=sub.add_parser('validate-events',help='Validate an event log against the shared schema');v.add_argument('--input',type=Path,required=True)
    d=sub.add_parser('demo-markov',help='Run a synthetic-only transition/dwell-time demonstration');d.add_argument('--out',type=Path,default=ROOT/'outputs'/'synthetic_demo');d.add_argument('--sessions',type=int,default=8);d.add_argument('--steps',type=int,default=90)
    a=sub.add_parser('analyze-transitions',help='Create per-discipline empirical transition matrices and dwell summaries');a.add_argument('--input',type=Path,required=True);a.add_argument('--out',type=Path,default=ROOT/'outputs'/'transitions')
    vb=sub.add_parser('analyze-vicos-bjj',help='Summarize observed ViCoS BJJ position transitions (descriptive, no training)');vb.add_argument('--input',type=Path,required=True);vb.add_argument('--out',type=Path,default=ROOT/'outputs'/'vicos_bjj_positions');vb.add_argument('--fps',type=float,default=30.0)
    hk=sub.add_parser('fit-vicos-hawkes',help='Fit a self-exciting (Hawkes) model to ViCoS position-change timing vs. a Poisson baseline');hk.add_argument('--input',type=Path,default=ROOT/'outputs'/'vicos_position_analysis'/'position_observations.csv');hk.add_argument('--registry',type=Path,default=ROOT/'data'/'dataset_registry.json');hk.add_argument('--out',type=Path,default=ROOT/'outputs'/'vicos_hawkes')
    sm=sub.add_parser('fit-vicos-semi-markov',help='Fit censored exponential vs. Weibull dwell-time models and the jump chain to ViCoS position episodes');sm.add_argument('--input',type=Path,default=ROOT/'outputs'/'vicos_position_analysis'/'position_observations.csv');sm.add_argument('--registry',type=Path,default=ROOT/'data'/'dataset_registry.json');sm.add_argument('--out',type=Path,default=ROOT/'outputs'/'vicos_semi_markov')
    rp=sub.add_parser('build-tex-report',help='Render Hawkes/semi-Markov summaries into a LaTeX technical report (PDF via pdflatex if installed)');rp.add_argument('--hawkes',type=Path,default=ROOT/'outputs'/'vicos_hawkes'/'hawkes_summary.json');rp.add_argument('--semi-markov',type=Path,default=ROOT/'outputs'/'vicos_semi_markov'/'semi_markov_summary.json');rp.add_argument('--out',type=Path,default=ROOT/'outputs'/'reports');rp.add_argument('--no-pdf',action='store_true')
    fu=sub.add_parser('fuse-streams',help='Align consented pose/audio/heart-rate CSV streams on one time axis from a manifest');fu.add_argument('--manifest',type=Path,required=True);fu.add_argument('--out',type=Path,default=ROOT/'outputs'/'fused_streams.csv')
    di=sub.add_parser('db-ingest',help='Load Hawkes/semi-Markov outputs into the local SQLite store (idempotent)');di.add_argument('--db',type=Path,default=ROOT/'outputs'/'quantix_vault.db');di.add_argument('--outputs',type=Path,default=ROOT/'outputs')
    dq=sub.add_parser('db-query',help='Run a read-only SQL query against the local SQLite store');dq.add_argument('--db',type=Path,default=ROOT/'outputs'/'quantix_vault.db');dq.add_argument('--sql',required=True)
    tp=sub.add_parser('track-people',help='Link MediaPipe detections into stable person tracks for analysed video folders (adds track_id; flags overlap frames)');tp.add_argument('--root',type=Path,default=ROOT/'outputs'/'own_videos',help='A video output folder, or a root searched for pose_landmarks.csv');tp.add_argument('--skip-done',action='store_true')
    dc=sub.add_parser('demo-card',help='Build a session card (skeleton clip with bystanders blurred, intensity timeline) from one tracked video');dc.add_argument('--video-dir',type=Path,required=True);dc.add_argument('--video',type=Path,required=True);dc.add_argument('--out',type=Path,default=None)
    vt=sub.add_parser('train-vicos-bjj-pose',help='Train a non-commercial position classifier from licensed ViCoS pose annotations');vt.add_argument('--input',type=Path,default=ROOT/'data'/'raw'/'vicos_bjj_positions'/'annotations.json');vt.add_argument('--registry',type=Path,default=ROOT/'data'/'dataset_registry.json');vt.add_argument('--out-dir',type=Path,default=ROOT/'models'/'vicos_bjj_pose_position');vt.add_argument('--epochs',type=int,default=8);vt.add_argument('--batch-size',type=int,default=512)
    m=sub.add_parser('train-actions',help='Fit grouped-split scikit-learn action-label baselines');m.add_argument('--input',type=Path,required=True);m.add_argument('--out',type=Path,default=ROOT/'models'/'action_classifiers');m.add_argument('--target',default='action_label');m.add_argument('--group-col',default='sequence_id')
    au=sub.add_parser('extract-audio',help='Extract MFCC and acoustic summary features from a licensed audio manifest');au.add_argument('--manifest',type=Path,required=True);au.add_argument('--out',type=Path,default=ROOT/'outputs'/'audio_features.csv')
    po=sub.add_parser('extract-pose',help='Extract MediaPipe pose landmarks from one consented video');po.add_argument('--video',type=Path,required=True);po.add_argument('--model-asset',type=Path,required=True);po.add_argument('--out',type=Path,required=True);po.add_argument('--every-n-frames',type=int,default=5);po.add_argument('--max-frames',type=int,default=0)
    dl=sub.add_parser('train-pose-mlp',help='Train a PyTorch MLP on labeled pose features with session-grouped splits');dl.add_argument('--input',type=Path,required=True);dl.add_argument('--out',type=Path,default=ROOT/'models'/'pose_mlp');dl.add_argument('--target',default='action_label');dl.add_argument('--group-col',default='split_group');dl.add_argument('--epochs',type=int,default=30)
    sx=sub.add_parser('fit-injury-survival',help='Fit an exploratory Cox model to an eligible injury follow-up cohort');sx.add_argument('--input',type=Path,required=True);sx.add_argument('--out',type=Path,default=ROOT/'outputs'/'cox_summary.csv')
    cp=sub.add_parser('compare-disciplines',help='Compare estimates only within one matched task/protocol/context');cp.add_argument('--input',type=Path,required=True);cp.add_argument('--task',required=True);cp.add_argument('--metric',required=True);cp.add_argument('--protocol-id',required=True);cp.add_argument('--test-context',required=True);cp.add_argument('--lower-is-better',action='store_true');cp.add_argument('--out',type=Path,default=ROOT/'outputs'/'discipline_comparison.csv')
    ar=sub.add_parser('import-arcade',help='Normalize game telemetry JSONL to the shared event schema');ar.add_argument('--input',type=Path,required=True);ar.add_argument('--out',type=Path,default=ROOT/'outputs'/'arcade_events.csv')
    va=sub.add_parser('analyze-video',help='Run consent-gated local video feature extraction and write a limitations-aware report');va.add_argument('--video',type=Path,required=True);va.add_argument('--video-id');va.add_argument('--model-asset',type=Path,required=True);va.add_argument('--discipline-id',required=True);va.add_argument('--sequence-id',required=True);va.add_argument('--consent-status',required=True);va.add_argument('--source-id',required=True);va.add_argument('--license-id',required=True);va.add_argument('--out-dir',type=Path,default=ROOT/'outputs'/'video_analysis');va.add_argument('--audio-path',type=Path);va.add_argument('--audio-source-id');va.add_argument('--audio-license-id');va.add_argument('--audio-consent-status');va.add_argument('--heart-rate-csv',type=Path);va.add_argument('--sensor-consent-status');va.add_argument('--sensor-source-id');va.add_argument('--sensor-license-id');va.add_argument('--event-log-csv',type=Path);va.add_argument('--strike-windows-csv',type=Path);va.add_argument('--takedown-attempts-csv',type=Path);va.add_argument('--every-n-frames',type=int,default=5);va.add_argument('--max-frames',type=int,default=0)
    gl=sub.add_parser('fit-athlete-glmm',help='Fit an exploratory athlete-random-intercept binary mixed model to labeled events');gl.add_argument('--input',type=Path,required=True);gl.add_argument('--out',type=Path,default=ROOT/'outputs'/'athlete_glmm.json');gl.add_argument('--outcome',default='success');gl.add_argument('--athlete-col',default='athlete_id');gl.add_argument('--discipline-col',default='discipline_id');gl.add_argument('--rule-col',default='rule_set')
    pv=sub.add_parser('validate-pro-manifest',help='Check local professional-video files, ML rights fields and bout-level split groups');pv.add_argument('--input',type=Path,required=True)
    pp=sub.add_parser('process-pro-manifest',help='Extract features from explicitly rights-cleared professional videos; does not auto-label or train');pp.add_argument('--input',type=Path,required=True);pp.add_argument('--model-asset',type=Path,required=True);pp.add_argument('--out-dir',type=Path,default=ROOT/'outputs'/'pro_features');pp.add_argument('--every-n-frames',type=int,default=5);pp.add_argument('--max-frames',type=int,default=0)
    ea=sub.add_parser('compare-eras',help='Summarize verified bout metrics by era only within matched discipline/ruleset/division strata');ea.add_argument('--input',type=Path,required=True);ea.add_argument('--out',type=Path,default=ROOT/'outputs'/'era_comparison.csv');ea.add_argument('--report',type=Path,default=ROOT/'outputs'/'era_comparison.md');ea.add_argument('--bootstrap-reps',type=int,default=1000)
    co=sub.add_parser('analyze-countries',help='Compare verified championship outcomes by represented country and separately recorded nationality');co.add_argument('--input',type=Path,required=True);co.add_argument('--out',type=Path,default=ROOT/'outputs'/'country_dominance.csv');co.add_argument('--bootstrap-reps',type=int,default=1000)
    wt=sub.add_parser('summarize-telemetry',help='Validate and summarize consented wearable sensor traces without clinical risk inference');wt.add_argument('--input',type=Path,required=True);wt.add_argument('--out',type=Path,default=ROOT/'outputs'/'telemetry_summary.csv')
    tm=sub.add_parser('train-match-model',help='Train an exploratory multiclass model with event-grouped holdout validation');tm.add_argument('--input',type=Path,required=True);tm.add_argument('--out',type=Path,default=ROOT/'models'/'match_predictor');tm.add_argument('--group-col',default='event_id')
    args=ap.parse_args()
    if args.cmd=='catalog':
        df=catalog_frame()
        if args.format=='table': print(df[['ordinal','discipline_id','name','category']].to_string(index=False))
        else:
            target=args.out or ROOT/'data'/('martial_arts_catalog.'+args.format);export_catalog(target,args.format);print(f'Wrote {target}')
        return
    if args.cmd=='validate-events':
        df=read_event_log(args.input);print(json.dumps({'rows':len(df),'disciplines':int(df.discipline_id.nunique()),'sequences':int(df.sequence_id.nunique()),'columns':list(df.columns)},indent=2));return
    if args.cmd=='demo-markov':
        df=synthetic_demo(n_sessions=args.sessions,steps=args.steps);args.out.mkdir(parents=True,exist_ok=True);df.to_csv(args.out/'synthetic_event_log.csv',index=False)
        counts,matrix,dwell=analyze_transitions(df);counts.to_csv(args.out/'transition_counts.csv');matrix.to_csv(args.out/'transition_matrix.csv');dwell.to_csv(args.out/'dwell_summary.csv',index=False)
        print('SYNTHETIC DATA ONLY - no athlete or competition observations');print(f'Wrote demonstration outputs to {args.out}');return
    if args.cmd=='analyze-transitions':
        df=read_event_log(args.input);args.out.mkdir(parents=True,exist_ok=True)
        for discipline,g in df.groupby('discipline_id'):
            counts,matrix,dwell=analyze_transitions(g)
            slug=''.join(c.lower() if c.isalnum() else '-' for c in discipline).strip('-')
            counts.to_csv(args.out/f'{slug}_transition_counts.csv');matrix.to_csv(args.out/f'{slug}_transition_matrix.csv');dwell.to_csv(args.out/f'{slug}_dwell_summary.csv',index=False)
        print(f'Analyzed {len(df)} event rows across {df.discipline_id.nunique()} discipline IDs. Outputs: {args.out}');return
    if args.cmd=='fit-vicos-hawkes':
        from martial_arts_engine.hawkes import analyze_vicos_hawkes
        print(json.dumps(analyze_vicos_hawkes(args.input,args.registry,args.out),indent=2,ensure_ascii=False));return
    if args.cmd=='fit-vicos-semi-markov':
        from martial_arts_engine.semi_markov import analyze_vicos_semi_markov
        print(json.dumps(analyze_vicos_semi_markov(args.input,args.registry,args.out),indent=2,ensure_ascii=False));return
    if args.cmd=='build-tex-report':
        from martial_arts_engine.tex_report import write_report
        print(json.dumps(write_report(args.out,args.hawkes,args.semi_markov,compile_pdf=not args.no_pdf),indent=2));return
    if args.cmd=='fuse-streams':
        from martial_arts_engine.fusion import fuse_from_manifest
        print(json.dumps(fuse_from_manifest(args.manifest,args.out),indent=2,ensure_ascii=False));return
    if args.cmd=='db-ingest':
        from martial_arts_engine.vault_db import ingest_outputs
        print(json.dumps(ingest_outputs(args.db,args.outputs),indent=2));return
    if args.cmd=='db-query':
        from martial_arts_engine.vault_db import query
        print(query(args.db,args.sql).to_string(index=False));return
    if args.cmd=='track-people':
        from martial_arts_engine.tracking import track_video_dir
        dirs=sorted({p.parent for p in args.root.rglob('pose_landmarks.csv')})
        results=[track_video_dir(d) for d in dirs if not (args.skip_done and (d/'track_summary.csv').exists())]
        ok=[r for r in results if r['status']=='ok']
        print(json.dumps({'video_dirs':len(dirs),'tracked_now':len(ok),'median_uncertain_share':float(pd.Series([r['uncertain_share'] for r in ok]).median()) if ok else None,'note':'Rows flagged identity_uncertain (overlapping bodies) should be excluded from per-athlete metrics; which track is which athlete must be supplied.'},indent=2));return
    if args.cmd=='demo-card':
        from martial_arts_engine.demo_card import build_card
        print(json.dumps(build_card(args.video_dir,args.video,args.out or ROOT/'outputs'/'demo_cards'/args.video_dir.name),indent=2,ensure_ascii=False));return
    if args.cmd=='analyze-vicos-bjj':
        observations=read_vicos_annotations(args.input)
        outputs=write_vicos_analysis(observations,args.out,args.fps)
        print(json.dumps({
            'observations':len(observations),
            'sequences':int(observations.sequence_id.nunique()),
            'outputs':{key:str(path) for key,path in outputs.items()},
            'notice':'Descriptive sampled-label analysis only; not model training or true dwell-time measurement.',
        },indent=2,ensure_ascii=False))
        return
    if args.cmd=='train-vicos-bjj-pose':
        report=train_vicos_pose_classifier(args.input,args.registry,args.out_dir,epochs=args.epochs,batch_size=args.batch_size)
        print(json.dumps(report,indent=2,ensure_ascii=False))
        return
    if args.cmd=='train-actions':
        report=train_event_models(pd.read_csv(args.input),args.out,args.target,args.group_col);print(json.dumps({k:{'accuracy':v['accuracy'],'balanced_accuracy':v['balanced_accuracy']} for k,v in report.items()},indent=2));return
    if args.cmd=='extract-audio':
        from martial_arts_engine.audio import extract_audio_manifest
        result=extract_audio_manifest(args.manifest,args.out);print(f'Wrote {len(result)} feature rows to {args.out}');return
    if args.cmd=='extract-pose':
        from martial_arts_engine.vision import extract_pose_csv
        n=extract_pose_csv(args.video,args.model_asset,args.out,args.every_n_frames,args.max_frames);print(f'Wrote {n} pose/person rows to {args.out}');return
    if args.cmd=='train-pose-mlp':
        from martial_arts_engine.deep_learning import train_pose_mlp
        result=train_pose_mlp(args.input,args.out,args.target,args.group_col,args.epochs);print(json.dumps({'test':result['test'],'classes':result['classes'],'device':result['device']},indent=2));return
    if args.cmd=='fit-injury-survival':
        from martial_arts_engine.survival import fit_injury_cox
        print(json.dumps(fit_injury_cox(args.input,args.out),indent=2));return
    if args.cmd=='compare-disciplines':
        report=compare_disciplines(args.input,args.task,args.metric,args.protocol_id,args.test_context,not args.lower_is_better)
        args.out.parent.mkdir(parents=True,exist_ok=True);report.to_csv(args.out,index=False);print(report.to_string(index=False));print(f'Wrote comparison to {args.out}');return
    if args.cmd=='import-arcade':
        from martial_arts_engine.arcade import import_arcade_jsonl
        report=import_arcade_jsonl(args.input,args.out);print(f'Normalized {len(report)} game events to {args.out}; source rows are synthetic gameplay by default.');return
    if args.cmd=='analyze-video':
        from martial_arts_engine.video_pipeline import run_video_analysis
        report=run_video_analysis(args.video,args.model_asset,args.discipline_id,args.sequence_id,args.consent_status,args.source_id,args.license_id,args.out_dir,args.every_n_frames,args.max_frames,args.audio_path,args.heart_rate_csv,args.event_log_csv,args.sensor_consent_status,args.audio_source_id,args.audio_license_id,args.audio_consent_status,args.sensor_source_id,args.sensor_license_id,args.strike_windows_csv,args.takedown_attempts_csv,args.video_id)
        print(json.dumps(report,indent=2,ensure_ascii=False));print(f"Dashboard: {args.out_dir/'analysis_report.html'}");return
    if args.cmd=='fit-athlete-glmm':
        from martial_arts_engine.mixed_models import fit_athlete_random_intercept_glmm
        result=fit_athlete_random_intercept_glmm(args.input,args.out,args.outcome,args.athlete_col,args.discipline_col,args.rule_col)
        print(json.dumps({k:v for k,v in result.items() if k!='summary'},indent=2,ensure_ascii=False));print(f'Wrote model summary: {args.out}');return
    if args.cmd=='validate-pro-manifest':
        from martial_arts_engine.pro_video_data import validate_pro_video_manifest
        print(json.dumps(validate_pro_video_manifest(args.input),indent=2,ensure_ascii=False));return
    if args.cmd=='process-pro-manifest':
        from martial_arts_engine.pro_video_data import process_pro_video_manifest
        result=process_pro_video_manifest(args.input,args.model_asset,args.out_dir,args.every_n_frames,args.max_frames)
        print(result.to_string(index=False));print(f'Feature batch summary: {args.out_dir / "batch_processing_summary.csv"}');return
    if args.cmd=='compare-eras':
        from martial_arts_engine.era_analysis import compare_eras
        result=compare_eras(args.input,args.out,args.report,args.bootstrap_reps)
        print(result.to_string(index=False));print(f'Wrote era summary: {args.out}; report: {args.report}');return
    if args.cmd=='analyze-countries':
        from martial_arts_engine.country_analysis import analyze_country_results
        result=analyze_country_results(args.input,args.out,args.bootstrap_reps)
        print(result.to_string(index=False));print(f'Wrote country summaries: {args.out}');return
    if args.cmd=='summarize-telemetry':
        from martial_arts_engine.wearable_telemetry import summarize_telemetry
        result=summarize_telemetry(args.input,args.out)
        print(result.to_string(index=False));print(f'Wrote descriptive telemetry summary: {args.out}');return
    if args.cmd=='train-match-model':
        from martial_arts_engine.match_prediction import train_match_model
        result=train_match_model(args.input,args.out,args.group_col)
        print(json.dumps(result,indent=2,ensure_ascii=False));print(f'Wrote exploratory model: {args.out}');return

if __name__=='__main__':main()
