# Global Martial Arts Analytics Platform · 100 MA Engine

[![QMASE CI](https://github.com/000MAK/quantix-martial-arts-engine/actions/workflows/qmase_ci.yml/badge.svg)](https://github.com/000MAK/quantix-martial-arts-engine/actions/workflows/qmase_ci.yml) [![License: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

Local research scaffold for the **100 numbered entries** supplied by the user. It combines shared data contracts with optional computer vision/pose extraction, acoustic features, transition analysis, supervised-learning baselines, injury survival analysis and an arcade telemetry contract.

See [MODEL_CATALOG.md](MODEL_CATALOG.md) for the implemented-versus-proposed model inventory and research prerequisites. The catalog explicitly does not claim that roadmap methods are deployed or integrated with BIOGAS.

## Evidence rules

- The catalogue preserves the supplied numbering and four groups of 25. A row ID is a catalogue identifier, not evidence that every row is a distinct, standardized or directly comparable martial art. Some names repeat or overlap; the catalogue flags several for review rather than silently rewriting the source list.
- The global schema is deterministic. Engagement time, injury risk and performance results stay blank until measured from eligible data. **No random values are presented as findings.**
- No raw videos, audio, tournament logs or athlete health data are bundled. The included synthetic Markov demo is for pipeline demonstration only.
- “Best” is task- and population-specific. Do not compare a kata score, a competition win rate, a pose-classification score and an injury rate as one universal ranking. The comparison command only ranks matching task/metric/protocol/test-context rows and requires reported uncertainty bounds.

## Architecture

```text
100_MA_Engine/
├── 01_Data_Pipelines/       catalog, schema generator, input templates
├── 02_Computer_Vision/      MediaPipe pose extraction + pose baseline notes
├── 03_Acoustic_Engine/      Librosa feature extraction notes
├── 04_Stochastic_Models/    transition matrices, dwell summaries, ML baselines
├── 05_Risk_Actuarial/      injury follow-up schema and Cox model notes
├── 06_Arcade_Integration/  game telemetry event contract
├── data/                    catalog, source registry, empty schemas
├── src/martial_arts_engine/ shared executable modules
├── models/                  generated model artifacts (Git ignored)
└── outputs/                 generated analyses (Git ignored)
```

Domain code is centralized in `src/martial_arts_engine/`; numbered folders document module ownership and hold domain-specific schemas, avoiding divergent duplicate implementations.

## Setup on F: (PowerShell)

```powershell
cd F:\Quantix_Master_Vault\100_MA_Engine
$env:PIP_CACHE_DIR = "$PWD\.pip-cache"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py catalog
python main.py demo-markov

# Descriptive pilot from manually downloaded ViCoS annotation JSON (no video download or model training)
python main.py analyze-vicos-bjj --input data\vicos\annotations.json --out outputs\vicos_bjj_positions

# Non-commercial pose-position baseline on the locally available ViCoS annotations (registry-gated)
python main.py train-vicos-bjj-pose --epochs 8

# One-command evidence-aware JSON + HTML run report (does not invent scores)
python master_orchestrator.py

# Local, consent-gated analysis for one video (MediaPipe task model required)
python main.py analyze-video --video sparring.mp4 --model-asset pose_landmarker.task --discipline-id ma-027-brazilian-jiu-jitsu-no-gi --sequence-id session-001 --consent-status granted --source-id owned-recording --license-id owned-original-recording --out-dir outputs\session-001

# Optional verified labels activate coordination and attempt-level TD rates
python main.py analyze-video --video sparring.mp4 --model-asset pose_landmarker.task --discipline-id ma-027-brazilian-jiu-jitsu-no-gi --sequence-id session-001 --consent-status granted --source-id owned-recording --license-id owned-original-recording --strike-windows-csv VERIFIED_STRIKE_WINDOWS.csv --takedown-attempts-csv VERIFIED_TAKEDOWN_ATTEMPTS.csv --out-dir outputs\session-001

# Optional athlete-clustered binary model (requires labeled repeated observations)
python main.py fit-athlete-glmm --input LABELED_EVENTS.csv --out outputs\athlete_glmm.json

# Rights-cleared local professional footage: validate, then extract features
python main.py validate-pro-manifest --input 01_Data_Pipelines\templates\my_pro_videos.csv
python main.py process-pro-manifest --input 01_Data_Pipelines\templates\my_pro_videos.csv --model-asset pose_landmarker.task --out-dir outputs\pro_features
python main.py train-pose-mlp --input LABELED_POSE_FEATURES.csv --group-col split_group

# Descriptive era comparison from verified, bout-level metrics
python main.py compare-eras --input VERIFIED_BOUT_METRICS.csv --out outputs\era_comparison.csv --report outputs\era_comparison.md
```

The virtual environment and pip cache are located on F:. The project does not copy source datasets to C:. This assumes a Python runtime is already installed and available through the Windows Python launcher.

Optional dependencies (same F: pip cache):

```powershell
python -m pip install -r requirements-vision.txt
python -m pip install -r requirements-audio.txt
python -m pip install -r requirements-deep-learning.txt
python -m pip install -r requirements-survival.txt
python -m pip install -r requirements-mixed-models.txt
```

The core install needs Python, NumPy, Pandas, scikit-learn and joblib. Optional components are isolated so the catalog and event analysis do not require GPU or computer-vision packages.

## Main workflows

`master_orchestrator.py` writes `outputs/master_run/master_orchestrator_report.json` and `.html`. To include actual local professional video, verified transitions, and a stratified era comparison in one run, supply the corresponding inputs:

```powershell
python master_orchestrator.py --video-manifest RIGHTS_CLEARED_VIDEOS.csv --model-asset pose_landmarker.task --event-log VERIFIED_EVENTS.csv --era-metrics VERIFIED_BOUT_METRICS.csv --country-results VERIFIED_CHAMPIONSHIP_ENTRIES.csv
```

The video manifest must pass the provenance and local-file checks. Pose extraction produces reviewable, unlabeled features; it does not identify techniques or train a classifier. The report lists missing coverage and does not calculate random “QMASE” scores or rank the 100 catalog entries. A validated scoring specification and measured comparable observations are prerequisites for such a ranking.

```powershell
# List/verify the 100-entry catalog
python main.py catalog
python 01_Data_Pipelines\core_engine.py

# Build event transitions and observed dwell-time summaries
python main.py validate-events --input 01_Data_Pipelines\templates\event_log_template.csv
python main.py analyze-transitions --input PATH_TO_ANNOTATED_EVENTS.csv
python main.py demo-markov

# Group-aware supervised baseline
python main.py train-actions --input LABELED_FEATURES.csv --group-col sequence_id

# Optional pose, audio and survival workflows (see the domain READMEs)
python main.py extract-pose --video VIDEO.mp4 --model-asset pose_landmarker.task --out outputs\pose.csv
python main.py extract-audio --manifest 01_Data_Pipelines\templates\audio_manifest_template.csv
python main.py train-pose-mlp --input LABELED_POSE.csv --group-col split_group
python main.py fit-injury-survival --input 05_Risk_Actuarial\injury_followup_template.csv

# Temporal LSTM architecture smoke test (synthetic only; not a performance result)
python 03_Models\train_pipeline.py --smoke-test

# ViCoS BJJ position transitions (non-commercial CC BY-NC-SA 4.0; annotations only)
python 01_Data_Pipelines\import_vicos_positions.py --confirm-noncommercial-use

# Exploratory image classifier (sequence-prefix grouped split; all decodable labeled images by default)
python 03_Models\train_vicos_position_image.py --confirm-noncommercial-use --epochs 1 --image-size 64 --batch-size 128

# Real temporal training additionally requires a registry record approved for this exact task
python 03_Models\train_pipeline.py --data-dir data\processed --dataset-id APPROVED_DATASET_ID --epochs 10

# Normalize a game telemetry JSONL export to the common event-log schema
python main.py import-arcade --input GAME_EVENTS.jsonl

# Compare only matched task/protocol/context results with confidence bounds
python main.py compare-disciplines --input EVALUATIONS.csv --task action-recognition --metric balanced_accuracy --protocol-id pose-v1 --test-context held-out-athletes

# Country outcome summaries; representation and nationality are kept separate
python main.py analyze-countries --input VERIFIED_CHAMPIONSHIP_ENTRIES.csv --out outputs\country_dominance.csv

# Consented wearable traces; descriptive only
python main.py summarize-telemetry --input AUTHORIZED_TELEMETRY.csv --out outputs\telemetry_summary.csv

# Exploratory bout outcome baseline; requires verified, licensed, multi-event labels
python main.py train-match-model --input VERIFIED_MATCH_FEATURES.csv --out models\match_predictor --group-col event_id
```

The provided templates contain headers only. Fill them with properly sourced observations and annotate `source_id`, `license_id`, consent, ruleset, split group and annotation protocol. `split_group` should identify the highest dependency unit (often athlete, bout or source video) to prevent train/test leakage.

`analyze-video` is a local CLI workflow (not a public upload server). It writes pose landmarks, camera-plane kinematic features, optional acoustic summaries, optional heart-rate review, optional transitions from verified annotations, and a local HTML/JSON report. Installing `requirements-audio.txt` provides librosa and a bundled ffmpeg executable; pass `--audio-path` with its own `--audio-source-id`, `--audio-license-id`, and `--audio-consent-status granted` when the audio file is separate from the video. Heart-rate input requires a separate sensor consent plus source and rights metadata in the CSV template or CLI options. Tactical transitions require an event CSV whose matching rows have `annotation_status=verified` and a recorded `annotation_protocol`. Processing time depends on video length, frame sampling and hardware; no fixed “seconds” guarantee is made.

For heart-rate input, use `01_Data_Pipelines/templates/heart_rate_template.csv`. For event labels, use `event_log_template.csv`; strike windows use `strike_windows_template.csv`; takedown outcomes use `takedown_attempts_template.csv`. All templates are blank. Populate annotation, source/license and consent fields before analysis.

The `analyze-vicos-bjj` command reads the public ViCoS annotation JSON supplied locally and writes observed position episodes, transitions, an empirical transition matrix, and sampled-label span summaries. It does not download the dataset, train a model, or enable the separate guard-drop LSTM. Position labels are collapsed across athlete-side suffixes. Because ViCoS contains selected labeled frames from six sparring sequences rather than continuous frame-by-frame event logs, transitions may be missed and frame-span summaries are not validated physical dwell times. Outputs are descriptive pilot results only. The dataset is CC BY-NC-SA 4.0; retain attribution and follow non-commercial/share-alike requirements before distributing data or derived materials.

`train-vicos-bjj-pose` is a separate, registry-gated exploratory baseline that predicts the ViCoS 18-class position label from its precomputed `pose1`/`pose2` coordinates; it is not an RGB-video model. It uses sequence-prefix grouped train/validation/test evaluation, then refits the saved checkpoint on all annotations. Metrics retain the grouped holdout result and coverage limitations. It is not training on every open dataset, and does not alter the consent-gated guard-drop LSTM approval. Outputs and the checkpoint carry the ViCoS CC BY-NC-SA 4.0 attribution/license metadata.

The four metric outputs are explicitly scoped: **coordination** reports separate camera-plane cross-correlations between shoulder/hip axis change and hand/foot speed, plus elbow/knee angle change and hand/foot speed, with frame-sampled lags inside verified strike windows; these are not a 0-100 skill score. **Guard** is the percentage of valid frames where both wrist landmarks satisfy documented head-proximity thresholds; it is a heuristic proxy, not glove/guard recognition. **Takedown success** is completed attempts divided by resolved completed-or-defended attempts; **defense** is defended attempts over that same denominator. Both include Wilson intervals, separate rule sets, and exclude unresolved attempts. Labels must be verified by a human using a consistent annotation protocol. Pose detector person indices are not automatically mapped to athlete IDs in attempt logs.

The report deliberately does not calculate “gas-tank percentage,” oxygen debt, strike quality, a coach command, injury hazard, or an overall QMASE score. Sensor heart-rate traces are shown as heart rate, not inferred energy. Pose angles are 2-D camera projections and endpoint speeds are pixels/second without camera calibration. Detector `person_index` is not a guaranteed persistent athlete identity across frames. The kinematic panel is a review aid, not a medical or injury prediction tool.

## Professional footage and era research

`data/pro_video_coverage.csv` records the current state: no professional footage or annotations were supplied for any of the 100 catalogue entries. `pro_video_manifest_template.csv` is the intake contract. It requires a local file, event/bout/ruleset metadata, source and license scope, explicit ML-training permission, athlete-data processing basis, annotation protocol, and one split group per bout. `process-pro-manifest` extracts features from footage that passes these checks; it **does not auto-label actions or train a classifier**. After verified labels have been added to extracted features, `train-pose-mlp` can fit the current frame-feature baseline, with bouts kept together across splits. It is not a temporal action-recognition model and training across 100 disciplines has not happened.

`data/dataset_registry.json` is deny-by-default. It records candidate source, access type, license, task compatibility, and review evidence. None of the current records is approved for guard-drop model training. `03_Models/train_pipeline.py` checks the registry status, task-specific approval, review evidence, and per-sample consent/provenance before a real training run. The real-data manifest requires `sample_id`, `dataset_id`, `consent_status`, `training_permission`, `source_id`, and `license_id`, with one row aligned to every array sample. It also requires aligned `sequences_X.npy`, `labels_y.npy`, and `match_ids.npy`. Train/validation/test partitions are split by match/session group. Smoke mode uses only ephemeral synthetic arrays and writes under `outputs/architecture_smoke/`; its checkpoint and scores are explicitly marked synthetic-only. No real temporal model has been trained because the project does not yet contain eligible guard-drop sequences and annotations.

`01_Data_Pipelines/import_vicos_positions.py` converts the official ViCoS BJJ annotation JSON into position observations, observed label-transition counts/probabilities, and same-label observation-span summaries. It requires `--confirm-noncommercial-use`, validates the analysis-specific registry approval, and does not download the image archive or train an image classifier. The source is licensed CC BY-NC-SA 4.0; provide attribution and follow share-alike terms for redistributed adaptations. The source page reports six sparring sequences, while annotation filenames encode 16 distinct sequence prefixes; these prefixes are not assumed to be independent matches. Labels cover selected frames rather than continuous video, so frame-gap transitions can hide intermediate positions and same-label spans are not exact dwell-time estimates. Do not treat these descriptive outputs as a universal BJJ model or performance result.

`03_Models/train_vicos_position_image.py` trains a small experimental CNN directly from the official image ZIP and annotation CSV without unpacking the full archive. The default archive is the separately CRC-verified `data/raw/vicos_bjj_positions/images.zip.verified`; it requires a task-specific registry approval and `--confirm-noncommercial-use`, splits by encoded sequence prefix, validates the image archive, and uses every decodable labeled image by default; corrupt members are listed in `models/vicos_bjj_image_classifier/invalid_images.csv`. `--max-per-class-per-split N` optionally caps image reads for a faster pilot. The group IDs are not verified athlete or match identities; results are exploratory and must not be presented as production performance. Image files and trained weights remain local under `data/raw/` and `models/`; external redistribution requires compliance with CC BY-NC-SA 4.0.

The initial CPU baseline used one epoch at 64x64 and all 120,273 decodable labeled images (six archive members failed CRC/decompression checks). Its sequence-prefix-held-out accuracy was **3.69%**, versus an **18.67%** majority-class baseline; validation accuracy was 0.64%. Three of 18 labels were absent from the test split, and fixed-18-class balanced accuracy was 1.25%. This is a failed exploratory baseline, not a useful classifier, research finding, or QMASE score. The checkpoint and detailed metrics are retained for reproducibility only; do not use them to evaluate athletes or publish model capability claims.

For era comparisons, `pro_bout_metrics_template.csv` expects verified one-row-per-bout metrics. `compare-eras` reports finish rates, annotated action success, action/transition tempo, and technique diversity within discipline, ruleset, division, sex class, and competition level strata. Outputs include Wilson proportion intervals plus event-cluster bootstrap intervals when at least two events are available; tempo intervals bootstrap events as clusters. Wilson intervals treat bouts/attempts as independent, so use the cluster intervals when interpreting event-level samples. These are descriptive profiles, not a universal winner or an athlete-quality causal effect. Set and justify era boundaries per sport/ruleset in `era_definitions_template.csv`; account for rule changes, athlete/opponent strength, weight class, selection, camera coverage, and annotation drift before drawing conclusions.

The repository does not scrape or download broadcast footage. UCF101 is a general action-recognition dataset, not a 100-discipline professional-fight archive; it contains 13,320 videos across 101 classes and its maintainers warn against splitting related clips across train/test. UFC's terms restrict automated access/copying, and IBJJF's media guidance requires approval for recording at events. Those sources do not grant machine-learning rights; get explicit rights for each video before processing. See [UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/), [UFC Terms of Use](https://www.ufc.com/news/terms-use), and [IBJJF media guidance](https://help.ibjjf.com/media/).

## Wearables, match modeling, and local dashboard

- `wearable_telemetry.py` consumes the tidy `wearable_telemetry_template.csv` format, requires consent and provenance on each sample, and emits session/sensor/signal descriptive summaries. Supported signal/unit pairs include `heart_rate`/`bpm`, `impact_peak`/`g`, and `accel_x|y|z`/`m/s2`. Impact values are recorded as supplied by the sensor; this pipeline does not validate sensor placement, diagnose concussion, estimate lactate, set medical cut-offs, or forecast cumulative career damage.
- `match_prediction.py` trains a logistic-regression multiclass baseline only from verified rows and numeric `feature_` columns. Use consistent `result_class` values such as `A_KO`, `A_SUBMISSION`, `A_DECISION`, `B_KO`, `B_SUBMISSION`, `B_DECISION`, or `DRAW`; only classes present in your data can be estimated. It holds out whole events for a first evaluation, then fits a final artifact. Output probabilities are uncalibrated model scores, not betting advice or guaranteed outcomes. Template features must be pre-bout measurements; don't include post-bout leakage. Exact finish-round predictions require adequately sampled, separately labeled outcomes and a dedicated validation plan.
- `app.py` is an optional Streamlit interface for the existing local video pipeline, telemetry summaries and a trained predictor. Uploaded videos and extracted pose files persist below `outputs/dashboard_video_runs/`; remove run folders when no longer needed. It does not render a universal QMASE score.
- `api.py` provides local FastAPI health, telemetry-summary and trained-model prediction endpoints. Bind it to `127.0.0.1`; no authentication or public deployment hardening is included. Uploads to the telemetry endpoint are kept only in a temporary directory for the request.

Install the optional local UI/API dependencies and start the interfaces:

```powershell
python -m pip install -r requirements-dashboard.txt
python -m streamlit run app.py --server.address 127.0.0.1
# In a second terminal, if API routes are needed:
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

## Country and historical dominance analysis

`country_competition_results_template.csv` accepts one verified athlete-entry row per event context. `represented_country_code` (the team/NOC shown in that competition) and `nationality_country_code` are analyzed as separate views; do not infer either from birthplace, gym, ethnicity, or an athlete's current residence. Provide `country_code_system` and retain historically distinct NOC/entity codes as they appeared in the source; the analysis does not silently merge USSR, Unified Team, Russia, or successor states. The module reports entries, athlete counts, medals, golds, medal rate and bout win rate within discipline, era, ruleset, division, sex class and competition level. It supplies event-cluster bootstrap intervals when there are at least two events. It creates no composite country score or universal “best nation” ranking. Selection into competitions, number of entries, event format, historical borders, rule changes and source completeness remain relevant limitations.

## Modeling limits

- **Computer vision:** MediaPipe extracts body pose landmarks. It does not classify a technique. The included MLP is a baseline over labeled pose rows, not a temporal CNN/ViT and not validated across all 100 entries.
- **Audio:** Librosa extracts MFCC and acoustic summaries. Those are not validated impact detections, speech transcripts or technique labels.
- **Transitions:** Observed edge counts yield a descriptive Markov transition matrix. Dwell summaries use intervals between recorded events and mark terminal intervals as censored. A formal semi-Markov model needs a defined event process, enough sequences and uncertainty analysis.
- **Injury:** Cox coefficients are exploratory associations, sensitive to cohort selection, exposure, event definitions, censoring, confounding and sample size. They do not establish causal or individual risk.
- **Athlete mixed effects:** `fit-athlete-glmm` offers an optional Bayesian binomial mixed-effects baseline with an athlete random intercept. It adjusts for clustered observations; the random intercept is not an individual ability score.
- **Comparison:** Training footage and competition data differ in camera, rules, athlete level and annotation. Report dataset coverage, per-discipline sample sizes, group-aware evaluation and uncertainty. Where a discipline lacks data, leave its estimate unavailable.

## Source and licensing register

`data/sources.json` lists candidate benchmark data and visual-study references; `data/dataset_registry.json` is the authorization gate. Additional candidates verified against their official pages:

- **HMDB51** is the most immediately accessible general-action candidate: its official page lists 6,849 clips, classes such as punch, kick someone, fencing and sword fight, and states CC BY 4.0. It is not BJJ-specific. Verify that the license applies to the actual clip files and intended derivatives before adding any training approval.
- **UCF101** has 13,320 YouTube-sourced clips and classes including Punch, boxing-bag drills, fencing, sumo and tai chi. Its official page does not state a reuse/model-training license, so it remains blocked pending rights review.
- **AVA / AVA-Kinetics** provides person/time-localized action labels including martial art, fight/hit a person, grab a person and kick a person. Its annotations/video IDs do not grant blanket rights to the underlying third-party videos.
- **NTU RGB+D 120** includes RGB and 3D skeleton data with punch/slap and kicking labels, but requires account approval and an academic non-commercial release agreement. Redistribution, derivative dataset creation and commercial use need explicit ROSE Lab permission.
- **UCF-Crime** has a coarse surveillance Fighting label, not sport-technique labels; the official page does not state training rights. **RWF-2000** is similarly only violent/non-violent surveillance data, and its repository currently says the video files are unavailable.
- **COCO Person Keypoints** provides static people/keypoint annotations for generic pose-detector work, not combat or temporal labels. Check image-level terms and attribution before use.
- **Penn Action** has 2,326 RGB sequences across 15 fitness/sports actions with frame-level 13-joint annotations; it can help test temporal pose tooling, but has no martial-arts classes and its project page does not specify a model-training license.
- **ESC-50** contains 2,000 five-second labeled sound clips (e.g. breathing, clapping and footsteps), but no combat-specific sound labels. Its dataset-level license is CC BY-NC 3.0 and each source clip has provenance details.
- **FSD50K** offers 51,197 human-labeled audio clips across 200 sound classes. Clip licenses vary (including CC0, CC-BY, CC-BY-NC and CC Sampling+), so any training subset must be filtered against per-clip rights; commercial use requires author contact.
- **AudioSet** provides labels for 2,084,320 ten-second YouTube-derived segments plus precomputed audio embeddings. It does not provide blanket rights to download or train on the underlying recordings and is not combat-sound-specific.
- **AMASS** unifies 15 marker-based motion-capture sources into a shared SMPL representation (40+ hours, 11,000+ motions); registration and each constituent dataset's conditions apply. It is generic motion, not martial-arts technique ground truth.
- **CMU Graphics Lab Motion Capture Database** provides 2,605 trials in broad categories including human interaction and sports. Its official terms allow use in products but prohibit reselling the data directly or in converted form; use provided archives rather than crawling and include the requested acknowledgment.
- **PhysioNet Wearable Exam Stress** includes EDA, heart rate, BVP, skin temperature, inter-beat interval and accelerometer recordings from exam sessions. It can exercise the sensor pipeline but is not athlete data and cannot validate a sports-fatigue model.

These are candidates, not approvals: each new dataset record has empty `approved_tasks` and a blocked status. Do not mix their broad action labels or skeleton formats into the ViCoS BJJ position model without task-specific label mapping, provenance review and sequence/subject-aware validation. A video being viewable or downloadable online does not grant permission to train on or redistribute it. Obtain consent and required approvals before processing identifiable training or health data. Keep sensitive footage and injury records private.

## Published artifacts

| Artifact | Where | Notes |
|---|---|---|
| UCI HAR activity baseline | [makactuary/uci-har-activity-baseline](https://huggingface.co/makactuary/uci-har-activity-baseline) | Subject-held-out evaluation: accuracy 0.955, macro-F1 0.954 on 9 unseen volunteers (`train-uci-har`) |
| GrappleMap BJJ position-transition graph | [makactuary/grapplemap-bjj-graph](https://huggingface.co/datasets/makactuary/grapplemap-bjj-graph) | 725 positions, 1,485 transitions, decoded from Eelis van der Weegen's public-domain [GrappleMap](https://github.com/Eelis/GrappleMap) (`export-grapplemap`) |

## Known evaluation limitation (ViCoS)

The 16 ViCoS sequence prefixes correspond to roughly six sparring sessions recorded from two or three camera views (prefix groups 0-2, 3-5, 6-8, 9-10, 11-13, 14-15 have near-identical label counts). Prefix-grouped splits therefore place different views of the same moments in train and test, so reported pose-baseline scores are optimistic. Sessions also contain disjoint position subsets (e.g. turtle only in 9-10, back control only in 14-15), so a session-held-out 18-class evaluation is not possible with this dataset alone. An RGB CNN trained on these splits learned camera/background cues rather than positions and performed below the majority-class baseline.

## License

Source code in this repository is released under the [MIT License](LICENSE). Dataset and model rights are separate: no ViCoS images, annotations, derived outputs or trained checkpoints are included, and any such material remains under its own CC BY-NC-SA 4.0 terms. This repository does not grant rights to external materials.
