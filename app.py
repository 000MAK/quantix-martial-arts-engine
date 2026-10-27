from __future__ import annotations

from pathlib import Path
import json
import sys
import uuid

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
st.set_page_config(page_title="Quantix MA Engine", page_icon="🥋", layout="wide")

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 88% 0%, #173554 0, #091321 42%); color:#edf4ff}
.block-container {max-width: 1280px; padding-top: 2rem}
[data-testid="stMetric"] {background:#112238; border:1px solid #26415d; padding:1rem; border-radius:14px}
</style>
""", unsafe_allow_html=True)

st.title("🥋 Quantix · Martial Arts Analytics")
st.caption("Local research dashboard · provenance-aware inputs · descriptive measurements, not a validated universal QMASE score")
st.warning("Use this app only on a trusted local machine. Uploaded video, extracted pose files, and reports are written under outputs/dashboard_video_runs and remain there until you remove that run folder.")

tab_home, tab_mine, tab_video, tab_sensor, tab_predict = st.tabs(["Overview", "My videos", "Video analysis", "Wearable telemetry", "Match model"])

with tab_mine:
    own = ROOT / "outputs" / "own_videos"
    log = own / "progress.csv"
    if not log.is_file():
        st.info("No batch run yet. Start it from the command hub (option 8).")
    else:
        runs = pd.read_csv(log)
        picks = [json.loads(p.read_text(encoding="utf-8")) for p in own.rglob("me_track.json")]
        c1, c2, c3 = st.columns(3)
        c1.metric("Videos processed", f"{int((runs.status == 'ok').sum())} / 710")
        c2.metric("Failed", int((runs.status != "ok").sum()))
        c3.metric("You picked in", f"{sum(p.get('me_track_id') is not None for p in picks)} videos")
        rows = []
        for video_dir in sorted({p.parent for p in own.rglob("track_summary.csv")}):
            summary = pd.read_csv(video_dir / "track_summary.csv")
            me = json.loads((video_dir / "me_track.json").read_text(encoding="utf-8")) if (video_dir / "me_track.json").is_file() else None
            mine = video_dir / "my_landmarks.csv"
            rows.append({"video": video_dir.name, "tracks": len(summary),
                         "you": "not picked" if me is None else ("not in video" if me.get("me_track_id") is None else f"track {me['me_track_id']}"),
                         "your frames": sum(1 for _ in mine.open(encoding="utf-8")) - 1 if mine.is_file() else 0})
        st.subheader("Tracked videos")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Descriptive pose data only: no labels, no scores. Pick yourself per video with command hub option 11.")

with tab_home:
    report = ROOT / "outputs" / "master_run" / "master_orchestrator_report.json"
    if report.is_file():
        data = json.loads(report.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        cols = st.columns(5)
        for col, key, label in zip(cols, ["catalog_disciplines", "video_rows", "annotated_bouts", "event_rows", "country_summary_rows"], ["Catalog entries", "Rights-documented videos", "Verified bouts", "Transition events", "Country strata"]):
            col.metric(label, summary.get(key, 0))
        st.subheader("Pipeline status")
        st.dataframe(pd.DataFrame(data.get("modules", [])), width="stretch", hide_index=True)
    else:
        st.info("No master run report yet. Run `python master_orchestrator.py` from the project folder to create it.")
    st.markdown("**No invented findings:** without verified input data, the dashboard shows coverage gaps rather than rankings or scores.")

with tab_video:
    st.subheader("Consent-gated local pose and kinematics extraction")
    st.write("This extracts MediaPipe pose landmarks and camera-plane features. It does not auto-label techniques or calculate a QMASE total score.")
    try:
        from martial_arts_engine.catalog import load_catalog
        catalog = load_catalog()
        discipline_map = {f"{row['name']} ({row['discipline_id']})": row['discipline_id'] for row in catalog}
    except Exception as exc:
        discipline_map = {}
        st.error(f"Could not load discipline catalog: {exc}")
    video = st.file_uploader("Select an authorized video", type=["mp4", "mov", "avi", "mkv"], key="video_upload")
    model_asset = st.text_input("MediaPipe Pose Landmarker .task path", value=str(ROOT / "pose_landmarker.task"))
    discipline_label = st.selectbox("Discipline", list(discipline_map), disabled=not discipline_map)
    c1, c2 = st.columns(2)
    source_id = c1.text_input("Video source ID")
    license_id = c2.text_input("License / rights record ID")
    consent = st.checkbox("I confirm I have permission to process this video and the required athlete consent.")
    every_n = st.number_input("Process every Nth frame", min_value=1, max_value=120, value=5)
    run_video = st.button("Analyze video", type="primary", disabled=not (video and source_id.strip() and license_id.strip() and consent and discipline_map))
    if run_video:
        from martial_arts_engine.video_pipeline import run_video_analysis
        model_path = Path(model_asset).expanduser()
        if not model_path.is_file():
            st.error("Model asset file was not found. Download a compatible Pose Landmarker model and check its license.")
        else:
            run_id = uuid.uuid4().hex[:12]
            run_dir = ROOT / "outputs" / "dashboard_video_runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            suffix = Path(video.name).suffix.lower() or ".mp4"
            video_path = run_dir / ("authorized_video" + suffix)
            video_path.write_bytes(video.getvalue())
            try:
                with st.spinner("Extracting pose and review features…"):
                    result = run_video_analysis(video_path, model_path, discipline_map[discipline_label], run_id, "granted", source_id.strip(), license_id.strip(), run_dir / "analysis", int(every_n), video_id=run_id)
                st.success("Feature extraction complete")
                st.json(result)
                st.caption(f"Local artifacts and uploaded video: {run_dir}")
                report_path = run_dir / "analysis" / "analysis_report.html"
                if report_path.is_file():
                    st.download_button("Download analysis report", report_path.read_bytes(), file_name=f"{run_id}_analysis_report.html", mime="text/html")
            except Exception as exc:
                st.error(f"Analysis failed: {type(exc).__name__}: {exc}")

with tab_sensor:
    st.subheader("Wearable telemetry summary")
    st.write("Expected tidy CSV columns are listed in `01_Data_Pipelines/templates/wearable_telemetry_template.csv`. The summary is descriptive; it does not infer concussion, lactate, or cumulative career damage.")
    sensor_file = st.file_uploader("Upload authorized telemetry CSV", type=["csv"], key="telemetry_upload")
    if sensor_file and st.button("Summarize telemetry"):
        from tempfile import TemporaryDirectory
        from martial_arts_engine.wearable_telemetry import summarize_telemetry
        try:
            with TemporaryDirectory(prefix="quantix_dashboard_telemetry_", dir=ROOT / "outputs") as tmp:
                temp_input = Path(tmp) / "input.csv"
                temp_output = Path(tmp) / "summary.csv"
                temp_input.write_bytes(sensor_file.getvalue())
                summary = summarize_telemetry(temp_input, temp_output)
                st.dataframe(summary, width="stretch", hide_index=True)
                st.download_button("Download summary CSV", temp_output.read_bytes(), file_name="telemetry_summary.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Telemetry rejected: {type(exc).__name__}: {exc}")

with tab_predict:
    st.subheader("Exploratory match outcome model")
    st.write("Train first with verified, licensed bout data. Prediction scores are uncalibrated and are not betting advice, guaranteed results, or exact-round forecasts.")
    model_dir = ROOT / "models" / "match_predictor"
    meta_path = model_dir / "model_metadata.json"
    if not meta_path.is_file():
        st.info("No trained match model is available. See the match prediction template and train-match-model CLI command.")
    else:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        st.caption(f"Classes: {', '.join(metadata['classes'])} · validation: event-grouped holdout · {metadata['probability_status']}")
        st.write("Enter feature values using the same scale and definition as the model training table.")
        feature_text = st.text_area("Feature JSON", value="{}", height=160, placeholder='{"feature_strike_accuracy_delta": 0.12}')
        if st.button("Generate exploratory scores"):
            try:
                from martial_arts_engine.match_prediction import predict_match
                prediction = predict_match(model_dir, json.loads(feature_text))
                st.metric("Most likely class", prediction["top_class"])
                st.bar_chart(pd.Series(prediction["uncalibrated_model_scores"], name="Uncalibrated score"))
                st.caption(prediction["note"])
            except Exception as exc:
                st.error(f"Prediction unavailable: {type(exc).__name__}: {exc}")

st.divider()
st.caption("Optional local API: `uvicorn api:app --host 127.0.0.1 --port 8000`. Keep the host bound to localhost; do not expose personal video or health-adjacent data to a public network.")
